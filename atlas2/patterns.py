"""Chart-pattern detectors on daily bars.

Each detector inspects the recent window of an indicator-annotated DataFrame and
returns PatternHit objects with entry / stop / target levels:

- bull_flag              (continuation, bullish)
- cup_and_handle         (continuation, bullish)
- inverse_head_shoulders (reversal, bullish)
- double_bottom          (reversal, bullish)
- ascending_triangle     (continuation, bullish)
- box_breakout           (consolidation near highs, bullish)
- head_shoulders_top     (reversal, bearish - warning signal)
- double_top             (reversal, bearish - warning signal)

Targets use the classic measured-move rule for each pattern; stops sit under the
pattern's structural low (or above the structural high for bearish patterns).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import find_pivots


@dataclass
class PatternHit:
    pattern: str
    direction: str  # "bullish" | "bearish"
    entry: float    # breakout level that activates the trade
    stop: float
    target: float
    confidence: float  # 0..1
    status: str     # "forming" | "triggered"
    start_idx: int
    end_idx: int
    key_points: list = field(default_factory=list)  # [(bar_idx, price), ...] for charting
    note: str = ""

    @property
    def risk_reward(self) -> float | None:
        risk = self.entry - self.stop if self.direction == "bullish" else self.stop - self.entry
        reward = self.target - self.entry if self.direction == "bullish" else self.entry - self.target
        if risk <= 0:
            return None
        return round(reward / risk, 2)


def _status(df: pd.DataFrame, entry: float, direction: str = "bullish") -> str | None:
    """Classify setup vs. fresh breakout; None if the move already ran away."""
    close = float(df["close"].iloc[-1])
    if direction == "bullish":
        if close < entry:
            return "forming"
        if close <= entry * 1.03:
            return "triggered"
        return None
    if close > entry:
        return "forming"
    if close >= entry * 0.97:
        return "triggered"
    return None


def detect_bull_flag(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 45:
        return []
    hits = []
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    # flag: last 4..15 bars; pole: the <=25 bars before it
    for flag_len in range(4, 16):
        pole_end = n - flag_len
        window = c[max(0, pole_end - 25) : pole_end]
        if len(window) < 8:
            continue
        pole_low_rel = int(window.argmin())
        pole_low = float(window[pole_low_rel])
        pole_high = float(h[pole_end - 1])
        pole_gain = pole_high / pole_low - 1
        if pole_gain < 0.18:
            continue
        flag_high = float(h[pole_end:].max())
        flag_low = float(l[pole_end:].min())
        pole_range = pole_high - pole_low
        if pole_range <= 0 or flag_high > pole_high * 1.02:
            continue
        retrace = (pole_high - flag_low) / pole_range
        if retrace > 0.5 or retrace < 0.02:
            continue
        drift = c[-1] <= c[pole_end] * 1.01  # sideways/down, not still running
        if not drift:
            continue
        entry = flag_high * 1.002
        stop = flag_low * 0.99
        target = entry + pole_range * 0.7  # conservative measured move
        status = _status(df, entry)
        if status is None:
            continue
        conf = min(1.0, 0.45 + pole_gain * 0.8 + (0.5 - retrace) * 0.4)
        pole_start_idx = max(0, pole_end - 25) + pole_low_rel
        hits.append(
            PatternHit(
                "bull_flag", "bullish", round(entry, 4), round(stop, 4), round(target, 4),
                round(conf, 2), status, pole_start_idx, n - 1,
                key_points=[(pole_start_idx, pole_low), (pole_end - 1, pole_high), (n - 1, flag_low)],
                note=f"pole +{pole_gain*100:.0f}% then {flag_len}-bar flag retracing {retrace*100:.0f}%",
            )
        )
        break  # first (shortest) valid flag wins
    return hits


def detect_cup_and_handle(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 80:
        return []
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    hits = []
    for base_len in (140, 110, 80, 55):
        if n < base_len + 5:
            continue
        seg_h = h[-base_len:]
        seg_l = l[-base_len:]
        left_rim = float(seg_h[: base_len // 4].max())
        cup_low_rel = int(seg_l.argmin())
        cup_low = float(seg_l[cup_low_rel])
        depth = 1 - cup_low / left_rim
        if not (0.12 <= depth <= 0.40):
            continue
        if not (base_len // 5 <= cup_low_rel <= base_len * 4 // 5):
            continue  # low should sit in the middle of the base
        right_rim = float(seg_h[base_len * 3 // 4 :].max())
        if abs(right_rim / left_rim - 1) > 0.06:
            continue
        # handle: last 5..20 bars, shallow pullback in the upper half of the cup
        handle_len = min(20, base_len // 4)
        handle_low = float(l[-handle_len:].min())
        handle_high = float(h[-handle_len:].max())
        handle_depth = 1 - handle_low / handle_high
        if handle_depth > 0.15 or handle_low < cup_low + (left_rim - cup_low) * 0.5:
            continue
        entry = max(right_rim, handle_high) * 1.002
        stop = handle_low * 0.99
        target = entry * (1 + depth * 0.75)
        status = _status(df, entry)
        if status is None:
            continue
        conf = min(1.0, 0.5 + (0.40 - depth) * 0.8 + (0.15 - handle_depth))
        base_start = n - base_len
        hits.append(
            PatternHit(
                "cup_and_handle", "bullish", round(entry, 4), round(stop, 4), round(target, 4),
                round(conf, 2), status, base_start, n - 1,
                key_points=[
                    (base_start, left_rim),
                    (base_start + cup_low_rel, cup_low),
                    (n - handle_len, handle_high),
                    (n - 1, handle_low),
                ],
                note=f"{base_len}-bar cup {depth*100:.0f}% deep, handle {handle_depth*100:.0f}%",
            )
        )
        break
    return hits


def _neckline_between(df: pd.DataFrame, i1: int, i2: int) -> float | None:
    if i2 - i1 < 2:
        return None
    return float(df["high"].to_numpy()[i1 : i2 + 1].max())


def detect_inverse_head_shoulders(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 60:
        return []
    _, pl = find_pivots(df, order=4)
    pl = [i for i in pl if i >= n - 130]
    if len(pl) < 3:
        return []
    lows = df["low"].to_numpy()
    hits = []
    for a in range(len(pl) - 2):
        ls_i, head_i, rs_i = pl[a], pl[a + 1], pl[a + 2]
        if rs_i < n - 40:  # right shoulder must be recent
            continue
        ls, head, rs = lows[ls_i], lows[head_i], lows[rs_i]
        if not (head < ls * 0.97 and head < rs * 0.97):
            continue
        if abs(rs / ls - 1) > 0.08:
            continue
        neck1 = _neckline_between(df, ls_i, head_i)
        neck2 = _neckline_between(df, head_i, rs_i)
        if neck1 is None or neck2 is None:
            continue
        neckline = max(neck1, neck2)
        entry = neckline * 1.002
        stop = float(rs) * 0.985
        target = entry + (neckline - float(head))
        status = _status(df, entry)
        if status is None:
            continue
        symmetry = 1 - abs(rs / ls - 1) / 0.08
        depth = (neckline - head) / neckline
        conf = min(1.0, 0.45 + symmetry * 0.25 + min(depth, 0.25))
        hits.append(
            PatternHit(
                "inverse_head_shoulders", "bullish", round(entry, 4), round(stop, 4),
                round(target, 4), round(conf, 2), status, ls_i, n - 1,
                key_points=[(ls_i, float(ls)), (head_i, float(head)), (rs_i, float(rs))],
                note=f"neckline {neckline:.2f}, head {float(head):.2f}",
            )
        )
    return hits[-1:]  # most recent formation only


def detect_head_shoulders_top(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 60:
        return []
    ph, _ = find_pivots(df, order=4)
    ph = [i for i in ph if i >= n - 130]
    if len(ph) < 3:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    hits = []
    for a in range(len(ph) - 2):
        ls_i, head_i, rs_i = ph[a], ph[a + 1], ph[a + 2]
        if rs_i < n - 40:
            continue
        ls, head, rs = highs[ls_i], highs[head_i], highs[rs_i]
        if not (head > ls * 1.03 and head > rs * 1.03):
            continue
        if abs(rs / ls - 1) > 0.08:
            continue
        neckline = float(min(lows[ls_i : rs_i + 1].min(), lows[rs_i:].min()))
        entry = neckline * 0.998  # breakdown level
        stop = float(rs) * 1.015
        target = entry - (float(head) - neckline)
        status = _status(df, entry, "bearish")
        if status is None:
            continue
        conf = min(1.0, 0.45 + (1 - abs(rs / ls - 1) / 0.08) * 0.3)
        hits.append(
            PatternHit(
                "head_shoulders_top", "bearish", round(entry, 4), round(stop, 4),
                round(target, 4), round(conf, 2), status, ls_i, n - 1,
                key_points=[(ls_i, float(ls)), (head_i, float(head)), (rs_i, float(rs))],
                note=f"neckline {neckline:.2f}",
            )
        )
    return hits[-1:]


def detect_double_bottom(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 50:
        return []
    _, pl = find_pivots(df, order=4)
    pl = [i for i in pl if i >= n - 110]
    if len(pl) < 2:
        return []
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    hits = []
    for a in range(len(pl) - 1):
        i1, i2 = pl[a], pl[a + 1]
        if i2 < n - 35 or i2 - i1 < 15:
            continue
        l1, l2 = lows[i1], lows[i2]
        if abs(l2 / l1 - 1) > 0.03:
            continue
        interim_high = float(highs[i1 : i2 + 1].max())
        if interim_high < l1 * 1.05:
            continue
        entry = interim_high * 1.002
        stop = float(min(l1, l2)) * 0.985
        target = entry + (interim_high - float(min(l1, l2)))
        status = _status(df, entry)
        if status is None:
            continue
        conf = min(1.0, 0.5 + (0.03 - abs(l2 / l1 - 1)) * 10)
        hits.append(
            PatternHit(
                "double_bottom", "bullish", round(entry, 4), round(stop, 4), round(target, 4),
                round(conf, 2), status, i1, n - 1,
                key_points=[(i1, float(l1)), (i2, float(l2))],
                note=f"lows {float(l1):.2f}/{float(l2):.2f}, neckline {interim_high:.2f}",
            )
        )
    return hits[-1:]


def detect_double_top(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 50:
        return []
    ph, _ = find_pivots(df, order=4)
    ph = [i for i in ph if i >= n - 110]
    if len(ph) < 2:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    hits = []
    for a in range(len(ph) - 1):
        i1, i2 = ph[a], ph[a + 1]
        if i2 < n - 35 or i2 - i1 < 15:
            continue
        h1, h2 = highs[i1], highs[i2]
        if abs(h2 / h1 - 1) > 0.03:
            continue
        interim_low = float(lows[i1 : i2 + 1].min())
        if interim_low > h1 * 0.95:
            continue
        entry = interim_low * 0.998
        stop = float(max(h1, h2)) * 1.015
        target = entry - (float(max(h1, h2)) - interim_low)
        status = _status(df, entry, "bearish")
        if status is None:
            continue
        conf = min(1.0, 0.5 + (0.03 - abs(h2 / h1 - 1)) * 10)
        hits.append(
            PatternHit(
                "double_top", "bearish", round(entry, 4), round(stop, 4), round(target, 4),
                round(conf, 2), status, i1, n - 1,
                key_points=[(i1, float(h1)), (i2, float(h2))],
                note=f"tops {float(h1):.2f}/{float(h2):.2f}, support {interim_low:.2f}",
            )
        )
    return hits[-1:]


def detect_ascending_triangle(df: pd.DataFrame) -> list[PatternHit]:
    n = len(df)
    if n < 50:
        return []
    ph, pl = find_pivots(df, order=3)
    ph = [i for i in ph if i >= n - 70]
    pl = [i for i in pl if i >= n - 70]
    if len(ph) < 2 or len(pl) < 2:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    resistance = float(highs[ph].max())
    flat = [i for i in ph if highs[i] >= resistance * 0.975]
    if len(flat) < 2 or flat[-1] < n - 40:
        return []
    recent_pl = pl[-3:]
    rising = all(lows[recent_pl[i + 1]] > lows[recent_pl[i]] for i in range(len(recent_pl) - 1))
    if not rising or len(recent_pl) < 2:
        return []
    last_low = float(lows[recent_pl[-1]])
    if last_low < resistance * 0.85:
        return []
    entry = resistance * 1.002
    stop = last_low * 0.99
    height = resistance - float(lows[recent_pl[0]])
    target = entry + height
    status = _status(df, entry)
    if status is None:
        return []
    conf = min(1.0, 0.45 + 0.1 * len(flat) + 0.1 * (len(recent_pl) - 1))
    return [
        PatternHit(
            "ascending_triangle", "bullish", round(entry, 4), round(stop, 4), round(target, 4),
            round(conf, 2), status, min(flat[0], recent_pl[0]), n - 1,
            key_points=[(i, float(highs[i])) for i in flat[:3]] + [(i, float(lows[i])) for i in recent_pl],
            note=f"flat top {resistance:.2f} touched {len(flat)}x, rising lows",
        )
    ]


def detect_darvas_box(df: pd.DataFrame) -> list[PatternHit]:
    """Darvas box: near 52-week highs, a box top untouched for 3+ bars, then a
    box bottom untouched for 3+ bars, quiet volume inside, breakout above the top
    (ideally on a volume surge). Stop under the box bottom, classic Darvas style."""
    n = len(df)
    if n < 60 or "pct_off_high" not in df.columns:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    # Box top: the highest high in the recent window, CONFIRMED by the next
    # 3 bars all staying below it, and never exceeded again inside the box.
    window = min(50, n - 5)
    seg_start = n - window
    top_rel = int(highs[seg_start : n - 3].argmax())
    top_idx = seg_start + top_rel
    top = float(highs[top_idx])
    if top_idx > n - 13:  # box needs >= ~10 bars of life after the top forms
        return []
    if highs[top_idx + 1 : top_idx + 4].max() >= top:
        return []  # top not confirmed by 3 quieter bars
    # The top must hold until (at most) a fresh breakout in the last 2 bars —
    # if it was exceeded earlier, the breakout is stale and the box is spent.
    older = highs[top_idx + 4 : n - 2]
    if len(older) and older.max() > top * 1.002:
        return []

    # Box bottom: lowest low after the top, confirmed by 3 bars holding above it.
    bot_rel = int(lows[top_idx + 1 : n].argmin())
    bot_idx = top_idx + 1 + bot_rel
    bottom = float(lows[bot_idx])
    if bot_idx > n - 4:
        return []  # bottom too recent to be confirmed
    if lows[bot_idx + 1 : min(bot_idx + 4, n)].min() <= bottom:
        return []
    if lows[bot_idx + 1 :].min() < bottom * 0.998:
        return []  # bottom later undercut -> box broken downward

    height = top / bottom - 1
    if height <= 0.01 or height > 0.13:
        return []
    pct_off = float(df["pct_off_high"].iloc[-1])
    if pct_off < -12:  # Darvas demanded stocks pressing their highs
        return []

    entry = top * 1.002
    stop = bottom * 0.99
    target = entry * (1 + max(0.10, height * 1.5))
    status = _status(df, entry)
    if status is None:
        return []

    conf = 0.5 + (0.13 - height) * 2 + (12 + pct_off) / 120
    vol_note = ""
    vol = df["volume"].to_numpy()
    vol_avg = df["vol_avg20"].to_numpy() if "vol_avg20" in df.columns else None
    if vol_avg is not None and vol_avg[-1] == vol_avg[-1] and vol_avg[-1] > 0:
        inside = vol[max(top_idx, n - 10) : n - 1]
        if len(inside) and inside.mean() < vol_avg[-1] * 0.9:
            conf += 0.08
            vol_note = ", quiet volume inside box"
        if status == "triggered":
            if vol[-1] >= vol_avg[-1] * 1.3:
                conf += 0.12
                vol_note = ", volume surge on breakout"
            elif vol[-1] < vol_avg[-1] * 0.8:
                conf -= 0.15
                vol_note = ", but breakout volume is weak"
    conf = max(0.1, min(1.0, conf))
    return [
        PatternHit(
            "darvas_box", "bullish", round(entry, 4), round(stop, 4), round(target, 4),
            round(conf, 2), status, top_idx, n - 1,
            key_points=[(top_idx, top), (bot_idx, bottom)],
            note=(f"box top {top:.2f} / bottom {bottom:.2f} ({height*100:.1f}% tall, "
                  f"{n-1-top_idx} bars), {pct_off:.1f}% off 52w high{vol_note}"),
        )
    ]


BULLISH_DETECTORS = [
    detect_bull_flag,
    detect_cup_and_handle,
    detect_inverse_head_shoulders,
    detect_double_bottom,
    detect_ascending_triangle,
    detect_darvas_box,
]
BEARISH_DETECTORS = [detect_head_shoulders_top, detect_double_top]


def detect_all(df: pd.DataFrame) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for det in BULLISH_DETECTORS + BEARISH_DETECTORS:
        try:
            hits.extend(det(df))
        except Exception:
            continue
    return hits
