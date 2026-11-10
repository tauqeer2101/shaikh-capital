"""Operations: what keeps the scheduled jobs running, honest and recoverable.

Every scheduled command (launchd: the scan at 07:30, the monitor every hour, the
dashboard always on) goes through this module:

- run_lock(name): an exclusive, non-blocking file lock (fcntl.flock on
  data/locks/<name>.lock) so a second scan - SCAN.command double-clicked during
  the 07:30 run, or the monitor's self-heal - never runs alongside the first.
  The OS releases it when the process ends, however it ends.
- job(name): records every run in data/health.json (start, end, ok / failed with
  the error, duration, failures in a row) and sends a macOS notification when a
  run fails. The dashboard (/api/health) and `doctor` read that file.
- run_daily_scan(markets): every market, the Comeback scan and the Nasdaq signal
  run isolated - one failure never stops the others - after the daily backup;
  the command exits 1 when any part failed.
- self_heal(): the hourly monitor starts the scan itself when none succeeded today
  by 10:00 on a weekday (the Mac was off or asleep at 07:30, or the run failed),
  at most twice a day, never while a scan is running.
- backup_daily(): a consistent copy of the trade journal (sqlite's backup API,
  after an integrity check), the alert log and the paper-trading state into
  data/backups/<day>/; the last 30 days are kept. A journal that fails the
  integrity check is NOT copied (it must not push out the good copies).
  These copies sit on the same disk: Time Machine is what survives a dead disk
  (doctor warns when it finds no Time Machine backup).
- stamped_output() / rotate_log(): time-stamped log lines; logs rotated at 5 MB,
  3 old ones kept.
- doctor(): checks all of the above and prints OK / WARN / FAIL lines
  (python -m atlas2 doctor; exit 1 on any FAIL).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LAUNCHD_LABELS = ("com.atlas2.serve", "com.atlas2.scan", "com.atlas2.monitor")
DASHBOARD_URL = "http://127.0.0.1:8899"
BACKUP_KEEP_DAYS = 30
BACKUP_FILES = ("alerts.jsonl", "leverage_paper.json")
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_KEEP = 3
HEAL_AFTER_HOUR = 10
HEAL_MAX_PER_DAY = 2
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _health_path() -> Path:
    return DATA / "health.json"


def _lock_dir() -> Path:
    d = DATA / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> datetime:
    return datetime.now()


# --------------------------------------------------------------------------- notify

def notify(title: str, message: str) -> None:
    """macOS notification; never raises. Quotes and backslashes are removed so the
    text cannot break out of the AppleScript string."""
    clean = lambda s: str(s).replace("\\", "/").replace('"', "'")[:240]
    try:
        subprocess.run(["osascript", "-e", f'display notification "{clean(message)}" '
                        f'with title "{clean(title)}" sound name "Basso"'],
                       capture_output=True, timeout=10)
    except Exception:
        pass


# --------------------------------------------------------------------------- locks

@contextmanager
def run_lock(name: str):
    """Yields True when this process holds the lock, False when another one does."""
    fh = open(_lock_dir() / f"{name}.lock", "a+")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} {_now().isoformat(timespec='seconds')}\n")
        fh.flush()
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def lock_is_held(name: str) -> bool:
    with run_lock(name) as got:
        return not got


# --------------------------------------------------------------------------- health

def read_health() -> dict:
    try:
        h = json.loads(_health_path().read_text())
        if isinstance(h, dict) and isinstance(h.get("jobs"), dict):
            return h
    except Exception:
        pass
    return {"version": 1, "jobs": {}}


@contextmanager
def _health_update():
    """Read-modify-write of health.json under its own lock, written atomically."""
    with open(_lock_dir() / "health.lock", "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            h = read_health()
            yield h
            h["updated"] = _now().isoformat(timespec="seconds")
            p = _health_path()
            tmp = p.with_name(p.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(h, indent=1))
            os.replace(tmp, p)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def record(name: str, status: str, *, error: str | None = None,
           duration_s: float | None = None) -> int:
    """status: 'running', 'ok' or 'failed'. Returns the failures in a row after this
    record. Never raises (health must not break a job)."""
    try:
        with _health_update() as h:
            j = h["jobs"].setdefault(name, {})
            ts = _now().isoformat(timespec="seconds")
            if status == "running":
                j["last_start"] = ts
            else:
                j["last_end"] = ts
                if duration_s is not None:
                    j["last_duration_s"] = round(duration_s, 1)
                if status == "ok":
                    j["last_success"] = ts
                    j["failures_in_a_row"] = 0
                else:
                    j["last_error"] = (error or "failed")[:1000]
                    j["last_error_at"] = ts
                    j["failures_in_a_row"] = int(j.get("failures_in_a_row") or 0) + 1
            j["last_status"] = status
            return int(j.get("failures_in_a_row") or 0)
    except Exception as exc:
        print(f"WARN health record for {name} not written: {exc.__class__.__name__}: {exc}")
        return 1


@contextmanager
def job(name: str, *, notify_failure: bool = True):
    """Records the run; on an exception records it as failed, notifies (the first
    failure and every 6th in a row after it - an hourly job must not nag every
    hour), re-raises."""
    t0 = time.time()
    record(name, "running")
    try:
        yield
    except BaseException as exc:
        msg = f"{exc.__class__.__name__}: {exc}"
        n = record(name, "failed", error=msg, duration_s=time.time() - t0)
        if notify_failure and not isinstance(exc, KeyboardInterrupt) and (n == 1 or n % 6 == 0):
            notify(f"Shaikh Capital - {name} FAILED", msg + (f" ({n} times in a row)" if n > 1 else ""))
        raise
    record(name, "ok", duration_s=time.time() - t0)


def succeeded_on(name: str, day: date, health: dict | None = None) -> bool:
    j = (health or read_health())["jobs"].get(name) or {}
    return str(j.get("last_success") or "")[:10] == day.isoformat()


# --------------------------------------------------------------------------- daily scan

def run_daily_scan(markets: list[str] | None = None, *, runners: dict | None = None,
                   progress=print) -> list[str]:
    """The scan command: backup, then every part isolated. Returns the failed parts
    (empty = all fine). runners replaces the real work in tests:
    {"market": f(market), "comeback": f(), "leverage_signal": f(), "backup": f(),
     "markets": f() -> default market list}."""
    if runners is None:
        from .comeback import scan_comeback
        from .leverage_signal import run_leverage_signal
        from .scan import load_config, run_scan

        runners = {"market": lambda m: run_scan([m]), "comeback": scan_comeback,
                   "leverage_signal": run_leverage_signal, "backup": backup_daily,
                   "markets": lambda: load_config()["markets"]}
    failed: list[str] = []
    t0 = time.time()
    with run_lock("scan") as got:
        if not got:
            progress("another scan is running - not starting a second one")
            return ["already running"]
        record("scan", "running")
        progress("=== scan start")
        steps: list[tuple[str, callable]] = [("backup", runners["backup"])]
        try:
            todo = markets or runners["markets"]()
        except Exception as exc:          # a broken config.json: say so, scan nothing
            progress(f"FAIL config: {exc.__class__.__name__}: {exc}")
            todo = []
            failed.append("config")
        steps += [(f"scan:{m}", (lambda m=m: runners["market"](m))) for m in todo]
        steps += [("comeback", runners["comeback"]), ("leverage_signal", runners["leverage_signal"])]
        for name, fn in steps:
            try:
                with job(name, notify_failure=False):
                    fn()
            except Exception as exc:
                failed.append(name)
                progress(f"FAIL {name}: {exc.__class__.__name__}: {exc}")
                progress(traceback.format_exc().rstrip())
        dur = time.time() - t0
        if failed:
            msg = "failed: " + ", ".join(failed)
            record("scan", "failed", error=msg, duration_s=dur)
            notify("Shaikh Capital - scan problem", msg + " (the other parts ran; see data/scan.log)")
        else:
            record("scan", "ok", duration_s=dur)
        progress(f"=== scan end: {'ok' if not failed else 'FAILED ' + ', '.join(failed)} "
                 f"({dur:.0f}s)")
    return failed


def self_heal(*, now: datetime | None = None, spawn=None, progress=print) -> str:
    """Start the daily scan when none succeeded today by HEAL_AFTER_HOUR on a weekday.
    Returns what it did ('not due', 'done today', 'running', 'gave up', 'started')."""
    now = now or _now()
    today = now.date()
    if now.weekday() >= 5 or now.hour < HEAL_AFTER_HOUR:
        return "not due"
    h = read_health()
    if succeeded_on("scan", today, h):
        return "done today"
    if lock_is_held("scan"):
        return "running"
    heal = h.get("self_heal") or {}
    tries = int(heal.get("attempts") or 0) if heal.get("day") == today.isoformat() else 0
    if tries >= HEAL_MAX_PER_DAY:
        return "gave up"
    with _health_update() as hh:
        hh["self_heal"] = {"day": today.isoformat(), "attempts": tries + 1,
                           "last_attempt": now.isoformat(timespec="seconds")}
    progress(f"self-heal: no successful scan today - starting one (attempt {tries + 1} "
             f"of {HEAL_MAX_PER_DAY})")
    if spawn is None:
        def spawn():
            log = open(DATA / "scan.log", "a")
            subprocess.Popen([sys.executable, "-m", "atlas2", "scan"], cwd=str(ROOT),
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    spawn()
    return "started"


# --------------------------------------------------------------------------- backups

def _integrity(path: Path, quick: bool = False) -> str:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return str(con.execute("PRAGMA quick_check" if quick else "PRAGMA integrity_check")
                   .fetchone()[0])
    finally:
        con.close()


def backup_daily(*, today: date | None = None, keep: int = BACKUP_KEEP_DAYS) -> dict:
    """Once a day; returns {'status': 'made' | 'exists', 'path': ...}. Raises when the
    journal fails its integrity check (and copies nothing then)."""
    day = (today or date.today()).isoformat()
    root = DATA / "backups"
    dest = root / day
    src = DATA / "journal.sqlite3"
    if (dest / "DONE").exists():          # an interrupted attempt has no DONE: redone
        return {"status": "exists", "path": str(dest)}
    if src.exists():
        ok = _integrity(src)
        if ok != "ok":
            raise RuntimeError(f"journal.sqlite3 failed its integrity check ({ok}) - not "
                               "backed up; the older copies in data/backups are kept")
    dest.mkdir(parents=True, exist_ok=True)
    if src.exists():
        tmp = dest / "journal.sqlite3.tmp"
        s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        d = sqlite3.connect(tmp)
        try:
            s.backup(d)
        finally:
            d.close()
            s.close()
        os.replace(tmp, dest / "journal.sqlite3")
    for name in BACKUP_FILES:
        if (DATA / name).exists():
            shutil.copy2(DATA / name, dest / name)
    (dest / "DONE").write_text(_now().isoformat(timespec="seconds") + "\n")
    days = sorted(p for p in root.iterdir() if p.is_dir() and DAY_RE.match(p.name))
    for old in days[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return {"status": "made", "path": str(dest)}


def latest_backup() -> str | None:
    root = DATA / "backups"
    if not root.exists():
        return None
    days = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and DAY_RE.match(p.name) and (p / "DONE").exists())
    return days[-1] if days else None


# --------------------------------------------------------------------------- logs

class _Stamped:
    """Prefixes every line written through it with the local time."""

    def __init__(self, stream):
        self._s = stream
        self._bol = True

    def write(self, text: str) -> int:
        out = []
        for part in str(text).splitlines(keepends=True):
            if self._bol:
                out.append(_now().strftime("[%Y-%m-%d %H:%M:%S] "))
            out.append(part)
            self._bol = part.endswith("\n")
        self._s.write("".join(out))
        return len(text)

    def flush(self) -> None:
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


def stamped_output() -> None:
    if not isinstance(sys.stdout, _Stamped):
        sys.stdout = _Stamped(sys.stdout)
    if not isinstance(sys.stderr, _Stamped):
        sys.stderr = _Stamped(sys.stderr)


def rotate_log(path: Path, max_bytes: int = LOG_MAX_BYTES, keep: int = LOG_KEEP) -> bool:
    """path -> path.1 -> ... -> path.<keep> when path is larger than max_bytes. A
    process already writing to it carries on in path.1 (same file); the next run
    starts a fresh one."""
    path = Path(path)
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return False
        for k in range(keep - 1, 0, -1):
            a = path.with_name(f"{path.name}.{k}")
            if a.exists():
                os.replace(a, path.with_name(f"{path.name}.{k + 1}"))
        os.replace(path, path.with_name(f"{path.name}.1"))
        return True
    except OSError:
        return False


def start_scheduled(log_name: str) -> None:
    """For the launchd commands: rotate their log, then time-stamp the output."""
    rotate_log(DATA / log_name)
    stamped_output()


# --------------------------------------------------------------------------- doctor

def _last_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _weekdays_between(a: date, b: date) -> int:
    """Weekdays strictly after a, up to and including b."""
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        n += d.weekday() < 5
    return n


def check_scans(markets: list[str], now: datetime) -> list[tuple[str, str, str]]:
    out = []
    for m in list(markets) + ["comeback"]:
        p = DATA / "scans" / f"{m}_latest.json"
        if not p.exists():
            out.append(("FAIL", f"scan {m}", "no scan file yet - run SCAN.command"))
            continue
        try:
            d = json.loads(p.read_text())
            gen = datetime.fromisoformat(str(d.get("generated"))[:19])
        except Exception as exc:
            out.append(("FAIL", f"scan {m}", f"unreadable ({exc.__class__.__name__})"))
            continue
        behind = _weekdays_between(gen.date(), now.date())
        due = now.hour >= 8 or now.weekday() >= 5     # before 08:00 today's scan may still run
        level = "OK" if behind == 0 or (behind == 1 and not due) else \
            "WARN" if behind <= 1 else "FAIL"
        detail = f"made {gen:%Y-%m-%d %H:%M}"
        stale = d.get("data_stale")
        if stale:
            level = "WARN" if level == "OK" else level
            detail += f"; prices stale: {stale.get('note', stale)}"
        out.append((level, f"scan {m}", detail))
    return out


def check_health(now: datetime, h: dict | None = None) -> list[tuple[str, str, str]]:
    h = h or read_health()
    out = []
    for name, j in sorted(h["jobs"].items()):
        if name.startswith("scan:") or name in ("scan", "monitor", "backup", "comeback",
                                                 "leverage_signal"):
            if j.get("last_status") == "failed":
                out.append(("FAIL", f"job {name}", f"last run failed at {j.get('last_error_at')}: "
                                                   f"{j.get('last_error')}"))
            elif j.get("last_status") == "running":
                started = str(j.get("last_start") or "")
                try:
                    age_h = (now - datetime.fromisoformat(started)).total_seconds() / 3600
                except ValueError:
                    age_h = 0
                out.append(("WARN" if age_h > 3 else "OK", f"job {name}",
                            f"running since {started}" + (" - stuck?" if age_h > 3 else "")))
    mon = h["jobs"].get("monitor") or {}
    if now.weekday() < 5 and 10 <= now.hour <= 22:
        last = str(mon.get("last_end") or "")
        try:
            age = (now - datetime.fromisoformat(last)).total_seconds() / 3600
        except ValueError:
            age = None
        if age is None:
            out.append(("WARN", "monitor", "no monitor run recorded yet"))
        elif age > 2.5:
            out.append(("WARN", "monitor", f"last ran {last} ({age:.1f} h ago; it runs hourly)"))
        else:
            out.append(("OK", "monitor", f"last ran {last}"))
    return out


def check_backup(now: datetime) -> list[tuple[str, str, str]]:
    last = latest_backup()
    if last is None:
        return [("WARN", "backup", "no backup yet (made by the next scan)")]
    days = (now.date() - date.fromisoformat(last)).days
    return [("OK" if days <= 1 else "WARN" if days <= 4 else "FAIL", "backup", f"latest {last}")]


def doctor(*, quick: bool = False, now: datetime | None = None,
           progress=print) -> list[tuple[str, str, str]]:
    """(level, check, detail) for every check; quick skips the slow ones (the price
    cache scan, Time Machine, launchd and the HTTP probe) - used by the dashboard."""
    now = now or _now()
    res: list[tuple[str, str, str]] = []
    markets = ["us", "de", "in"]
    try:
        from .scan import load_config

        cfg = load_config()
        markets = list(cfg["markets"])
        res.append(("OK", "config.json", "readable"))
    except Exception as exc:
        res.append(("FAIL", "config.json", f"cannot be read ({exc.__class__.__name__}) - the "
                                           "scan will not run until it is fixed"))
    res += check_scans(markets, now)
    res += check_health(now)
    res += check_backup(now)
    j = DATA / "journal.sqlite3"
    if j.exists():
        try:
            ok = _integrity(j)
            res.append(("OK" if ok == "ok" else "FAIL", "journal", "integrity ok" if ok == "ok"
                        else f"integrity check: {ok}"))
        except Exception as exc:
            res.append(("FAIL", "journal", f"cannot be opened ({exc.__class__.__name__})"))
    free = shutil.disk_usage(DATA).free / 1e9
    res.append(("OK" if free >= 5 else "WARN" if free >= 1 else "FAIL", "disk",
                f"{free:.0f} GB free"))
    if not quick:
        res += _deps_check()
        res += _launchd_check()
        res += _http_check()
        cache = DATA / "cache" / "prices.sqlite3"
        if cache.exists():
            try:
                ok = _integrity(cache, quick=True)
                res.append(("OK" if ok == "ok" else "FAIL", "price cache",
                            "quick check ok" if ok == "ok" else f"quick check: {ok}"))
            except Exception as exc:
                res.append(("FAIL", "price cache", f"cannot be opened ({exc.__class__.__name__})"))
        res += _time_machine_check()
    for level, name, detail in res:
        progress(f"{level:4s} {name:22s} {detail}")
    return res


def _deps_check() -> list[tuple[str, str, str]]:
    lock = ROOT / "requirements.lock"
    if not lock.exists():
        return [("WARN", "packages", "requirements.lock missing")]
    from importlib import metadata

    bad = []
    for line in lock.read_text().splitlines():
        name, sep, want = line.strip().partition("==")
        if not sep or name.startswith("#"):
            continue
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != want:
            bad.append(f"{name} {have or 'missing'} (lock {want})")
    return [("WARN", "packages", "differ from requirements.lock: " + ", ".join(bad[:6]))] if bad \
        else [("OK", "packages", "match requirements.lock")]


def _launchd_check() -> list[tuple[str, str, str]]:
    try:
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception as exc:
        return [("WARN", "launchd", f"cannot list jobs ({exc.__class__.__name__})")]
    rows = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            rows[parts[2]] = parts[0]
    res = []
    for label in LAUNCHD_LABELS:
        if label not in rows:
            res.append(("FAIL", f"launchd {label.split('.')[-1]}", "job not loaded"))
        elif label.endswith("serve") and rows[label] == "-":
            res.append(("FAIL", "launchd serve", "loaded but not running"))
        else:
            res.append(("OK", f"launchd {label.split('.')[-1]}", "loaded"))
    return res


def _http_check() -> list[tuple[str, str, str]]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(DASHBOARD_URL + "/login", timeout=5) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as exc:
        return [("FAIL", "dashboard", f"no answer on {DASHBOARD_URL} ({exc.__class__.__name__})")]
    return [("OK" if code == 200 else "FAIL", "dashboard", f"{DASHBOARD_URL} answers ({code})")]


def _time_machine_check() -> list[tuple[str, str, str]]:
    try:
        r = subprocess.run(["tmutil", "latestbackup"], capture_output=True, text=True, timeout=15)
        path = r.stdout.strip()
    except Exception:
        path = ""
    if not path:
        return [("WARN", "Time Machine", "no Time Machine backup found - data/backups is on the "
                                         "same disk and will not survive a dead disk")]
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    return [("OK", "Time Machine", f"latest {m.group(1) if m else path}")]


def summary(results: list[tuple[str, str, str]]) -> dict:
    levels = [r[0] for r in results]
    status = "fail" if "FAIL" in levels else "warn" if "WARN" in levels else "ok"
    return {"status": status, "checked": _now().isoformat(timespec="seconds"),
            "items": [{"level": a, "check": b, "detail": c} for a, b, c in results]}


def main_doctor(argv: list[str]) -> int:
    res = doctor(quick="--quick" in argv)
    s = summary(res)
    print(f"overall: {s['status'].upper()}")
    return 1 if s["status"] == "fail" else 0
