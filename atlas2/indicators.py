"""Technical indicators computed on a daily-bar DataFrame (open/high/low/close/volume)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with indicator columns appended."""
    out = df.copy()
    c = out["close"]
    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()
    out["sma200"] = c.rolling(200).mean()
    out["ema21"] = c.ewm(span=21, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)

    prev_close = c.shift()
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    out["vol_avg20"] = out["volume"].rolling(20).mean()
    out["dollar_vol20"] = (c * out["volume"]).rolling(20).mean()
    out["high_52w"] = out["high"].rolling(252, min_periods=60).max()
    out["low_52w"] = out["low"].rolling(252, min_periods=60).min()
    out["pct_off_high"] = (c / out["high_52w"] - 1) * 100
    out["ret_1m"] = c.pct_change(21) * 100
    out["ret_3m"] = c.pct_change(63) * 100
    out["ret_6m"] = c.pct_change(126) * 100
    return out


def relative_strength_3m(df: pd.DataFrame, bench: pd.DataFrame) -> float | None:
    """Stock 3-month return minus benchmark 3-month return, in percent points."""
    if len(df) < 64 or len(bench) < 64:
        return None
    try:
        s = float(df["close"].iloc[-1] / df["close"].iloc[-64] - 1) * 100
        b = float(bench["close"].iloc[-1] / bench["close"].iloc[-64] - 1) * 100
        return s - b
    except Exception:
        return None


def find_pivots(df: pd.DataFrame, order: int = 4) -> tuple[list[int], list[int]]:
    """Local swing highs/lows: bar i is a pivot high if high[i] is the max of
    the surrounding `order` bars on each side (analogous for lows).
    Returns (pivot_high_indices, pivot_low_indices)."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    ph, pl = [], []
    for i in range(order, n - order):
        window_h = highs[i - order : i + order + 1]
        window_l = lows[i - order : i + order + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            ph.append(i)
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            pl.append(i)
    return ph, pl
