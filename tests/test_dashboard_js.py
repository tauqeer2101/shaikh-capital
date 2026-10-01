"""Regression tests for two dashboard functions in atlas2/static/index.html, run under
node against a small DOM stub (skipped when node is not installed). The functions are
extracted from the HTML verbatim, so the tests exercise the shipped code, not a copy."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "atlas2" / "static" / "index.html"
NODE = shutil.which("node")

# re-creates the child span on every innerHTML assignment, like a real element does
DOM_STUB = r"""
const ORIGINAL = 'at most <span id="help_cap">2</span> setups per sector make the active list each day';
function makeClause() {
  let html = "", span = null;
  const clause = { dataset: {} };
  const spanRe = /<span id="help_cap">([^<]*)<\/span>/;
  Object.defineProperty(clause, "innerHTML", {
    get() { return span ? html.replace(spanRe, `<span id="help_cap">${span.textContent}</span>`) : html; },
    set(v) { html = v; const m = v.match(spanRe); span = m ? { textContent: m[1] } : null; },
  });
  Object.defineProperty(clause, "textContent", {
    get() { return clause.innerHTML.replace(/<[^>]+>/g, ""); },
    set(v) { html = v; span = null; },
  });
  clause.innerHTML = ORIGINAL;
  return { clause, get span() { return span; } };
}
const detail = { innerHTML: "" };
const dom = makeClause();
const el = id => id === "detail" ? detail : id === "help_cap_clause" ? dom.clause
                 : id === "help_cap" ? dom.span : null;
const escH = s => String(s);
const fmt = x => String(x);
"""


def extract_function(name: str) -> str:
    """The full text of a top-level `function name(...) { ... }` in index.html."""
    src = INDEX.read_text(encoding="utf-8")
    m = re.search(rf"^function {name}\(.*?^\}}\n", src, re.S | re.M)
    if not m:
        raise AssertionError(f"function {name} not found in index.html")
    return m.group(0)


def run_js(body: str) -> list:
    """Run the stub + extracted functions + `body` under node; `body` prints one JSON line."""
    script = DOM_STUB + extract_function("renderAlerts") + extract_function("syncHelpCap") + body
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.js"
        path.write_text(script, encoding="utf-8")
        out = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


@unittest.skipIf(NODE is None, "node is not installed")
class AlertsStillOpenTest(unittest.TestCase):
    """'— still open' must follow the alert's trade_id, not its ticker: a closed position's
    historical stop alert is not 'still open' just because the same stock was re-entered."""

    def render(self, alerts: list[dict], open_rows: list[dict]) -> str:
        body = f"""
renderAlerts({json.dumps(alerts)}, {json.dumps(open_rows)});
console.log(JSON.stringify([detail.innerHTML]));
"""
        return run_js(body)[0]

    def stop_alert(self, trade_id, days: int = 1) -> dict:
        a = {"kind": "stop", "ticker": "SYF", "time": "2026-08-21T10:00:00", "message": "SYF: STOP HIT",
             "price": 71.55, "days_past_stop": days}
        if trade_id is not None:
            a["trade_id"] = trade_id
        return a

    def test_closed_positions_alert_is_not_still_open_after_the_ticker_is_re_entered(self):
        html = self.render([self.stop_alert(3)], [{"id": 7, "ticker": "SYF"}])
        self.assertIn("stop first hit 1 day earlier", html)
        self.assertNotIn("still open", html)

    def test_alert_for_the_open_trade_is_still_open(self):
        html = self.render([self.stop_alert(7, days=3)], [{"id": 7, "ticker": "SYF"}])
        self.assertIn("stop first hit 3 days earlier — still open", html)

    def test_legacy_alert_without_trade_id_falls_back_to_the_ticker(self):
        self.assertIn("still open", self.render([self.stop_alert(None)], [{"id": 7, "ticker": "SYF"}]))
        self.assertNotIn("still open", self.render([self.stop_alert(None)], [{"id": 8, "ticker": "JPM"}]))

    def test_no_open_positions_and_the_default_argument(self):
        self.assertNotIn("still open", self.render([self.stop_alert(7)], []))
        html = run_js(f"renderAlerts({json.dumps([self.stop_alert(7)])}); console.log(JSON.stringify([detail.innerHTML]));")[0]
        self.assertNotIn("still open", html)


@unittest.skipIf(NODE is None, "node is not installed")
class HelpCapWordingTest(unittest.TestCase):
    """The help modal's cap sentence follows the pattern scans; a tab whose scan file has
    no max_per_sector (Comeback, older files) must not reset it to the default 'at most 2'."""

    def wording_after(self, scans: list[dict]) -> list[str]:
        body = f"""
const out = [];
for (const s of {json.dumps(scans)}) {{ syncHelpCap(s); out.push(dom.clause.textContent); }}
console.log(JSON.stringify(out));
"""
        return run_js(body)

    def test_comeback_keeps_the_configured_cap(self):
        texts = self.wording_after([{"max_per_sector": 3}, {"setups": []}])
        self.assertTrue(texts[0].startswith("at most 3 setups per sector"), texts)
        self.assertTrue(texts[1].startswith("at most 3 setups per sector"), texts)

    def test_comeback_keeps_switched_off_and_the_cap_can_come_back(self):
        texts = self.wording_after([{"max_per_sector": 0}, {"setups": []}, {"max_per_sector": 3}, {"max_per_sector": 2}])
        self.assertIn("switched off", texts[0])
        self.assertIn("switched off", texts[1])
        self.assertTrue(texts[2].startswith("at most 3 setups"), texts)
        self.assertTrue(texts[3].startswith("at most 2 setups"), texts)

    def test_cold_start_on_a_tab_without_the_field_shows_the_default(self):
        texts = self.wording_after([{"setups": []}])
        self.assertTrue(texts[0].startswith("at most 2 setups per sector"), texts)

    def test_null_cap_counts_as_switched_off(self):
        self.assertIn("switched off", self.wording_after([{"max_per_sector": None}])[0])


if __name__ == "__main__":
    unittest.main()
