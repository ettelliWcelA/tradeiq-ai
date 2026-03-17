"""
tests/test_macro_calendar.py
=============================
Unit tests for data/macro_calendar.py — MacroCalendar and MacroCalendarResult.
Tests cover: keyword fallback logic, result object behavior, JSON parsing,
and the console_block output formatting.
No live API calls are made — the Anthropic client is mocked throughout.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import mock_deps  # noqa: F401

from data.macro_calendar import (
    MacroCalendar, MacroCalendarResult,
    SKIP_KEYWORDS, CAUTION_KEYWORDS,
)

WATCHLIST = ["NVDA", "AMD", "MSFT", "DELL", "META", "GOOGL", "AMZN",
             "ORCL", "PLTR", "AI", "BBAI", "SOUN", "IONQ", "HPE", "ANET"]


# ── TestMacroCalendarResult ───────────────────────────────────────────────────

class TestMacroCalendarResult(unittest.TestCase):

    def _clear(self, **kwargs):
        defaults = dict(should_trade=True, reason="", events=[],
                        cautions=[], earnings_conflicts=[], raw_summary="")
        defaults.update(kwargs)
        return MacroCalendarResult(**defaults)

    def _skip(self, **kwargs):
        defaults = dict(should_trade=False, reason="CPI today",
                        events=["CPI"], cautions=[], earnings_conflicts=[],
                        raw_summary="CPI report at 8:30 AM ET")
        defaults.update(kwargs)
        return MacroCalendarResult(**defaults)

    def test_bool_true_when_clear(self):
        r = self._clear()
        self.assertTrue(bool(r))

    def test_bool_false_when_skip(self):
        r = self._skip()
        self.assertFalse(bool(r))

    def test_console_block_clear_contains_clear_status(self):
        r = self._clear()
        block = r.console_block()
        self.assertIn("CLEAR TO TRADE", block)
        self.assertNotIn("DO NOT TRADE", block)

    def test_console_block_skip_contains_do_not_trade(self):
        r = self._skip()
        block = r.console_block()
        self.assertIn("DO NOT TRADE", block)
        self.assertIn("CPI today", block)

    def test_console_block_skip_shows_events(self):
        r = self._skip(events=["CPI", "FOMC"])
        block = r.console_block()
        self.assertIn("CPI", block)
        self.assertIn("FOMC", block)

    def test_console_block_skip_shows_earnings(self):
        r = self._skip(earnings_conflicts=["NVDA", "MSFT"])
        block = r.console_block()
        self.assertIn("NVDA", block)
        self.assertIn("MSFT", block)

    def test_console_block_clear_shows_cautions(self):
        r = self._clear(cautions=["trade balance", "consumer confidence"])
        block = r.console_block()
        self.assertIn("trade balance", block)

    def test_console_block_clear_no_cautions_is_clean(self):
        r = self._clear()
        block = r.console_block()
        self.assertNotIn("Caution", block)


# ── TestKeywordFallback ───────────────────────────────────────────────────────

class TestKeywordFallback(unittest.TestCase):
    """Tests the _keyword_fallback method which scans raw text for event keywords."""

    def setUp(self):
        self.cal = MacroCalendar(anthropic_key="")  # no key → uses fallback

    def test_cpi_triggers_skip(self):
        text = "Today the Bureau of Labor Statistics releases CPI data at 8:30 AM"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)
        self.assertIn("cpi", result.events)

    def test_fomc_triggers_skip(self):
        text = "The FOMC will announce its rate decision at 2:00 PM ET"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)

    def test_nfp_triggers_skip(self):
        text = "Non-farm payroll data drops at 8:30 AM this morning"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)

    def test_ppi_triggers_skip(self):
        text = "Producer Price Index PPI release scheduled for today"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)

    def test_gdp_triggers_skip(self):
        text = "GDP Q4 revised estimate published today"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)

    def test_pce_triggers_skip(self):
        text = "Personal consumption expenditure PCE report at 8:30"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)

    def test_federal_reserve_triggers_skip(self):
        text = "Federal Reserve interest rate decision today at 2 PM"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)

    def test_earnings_conflict_triggers_skip(self):
        # NVDA is in watchlist
        text = "NVDA reports quarterly earnings after market close today"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)
        self.assertIn("NVDA", result.earnings_conflicts)

    def test_earnings_non_watchlist_no_skip(self):
        # AAPL is NOT in our watchlist
        text = "AAPL earnings expected today after close"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertTrue(result.should_trade)

    def test_trade_balance_is_caution_not_skip(self):
        text = "US trade balance data released this morning, minor market impact expected"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertTrue(result.should_trade)  # trade balance doesn't skip
        self.assertIn("trade balance", result.cautions)

    def test_consumer_confidence_is_caution(self):
        text = "University of Michigan consumer confidence survey out today"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertTrue(result.should_trade)

    def test_clean_day_allows_trade(self):
        text = "No major economic events scheduled today. Markets open normally."
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertTrue(result.should_trade)

    def test_multiple_skip_events_detected(self):
        text = "CPI and PPI both release today along with jobless claims"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)
        self.assertGreaterEqual(len(result.events), 2)

    def test_case_insensitive_matching(self):
        text = "CONSUMER PRICE INDEX report scheduled for 8:30 AM today"
        result = self.cal._keyword_fallback(text, WATCHLIST)
        self.assertFalse(result.should_trade)


# ── TestMacroCalendarNoAPIKey ─────────────────────────────────────────────────

class TestMacroCalendarNoAPIKey(unittest.TestCase):
    """When no API key is present, check() should return should_trade=True with a warning."""

    def test_no_key_returns_safe_with_warning(self):
        cal = MacroCalendar(anthropic_key="")
        result = cal.check(WATCHLIST)
        # Without API key, system allows trading but warns
        self.assertTrue(result.should_trade)
        self.assertIn("API key", result.reason)

    def test_no_key_result_is_bool_true(self):
        cal = MacroCalendar(anthropic_key="")
        result = cal.check(WATCHLIST)
        self.assertTrue(bool(result))


# ── TestMacroCalendarJSONParsing ──────────────────────────────────────────────

class TestMacroCalendarJSONParsing(unittest.TestCase):
    """Tests the JSON parsing path using a mocked Anthropic client."""

    def _make_cal_with_response(self, json_str: str) -> MacroCalendar:
        """Create a MacroCalendar whose Anthropic client returns a fixed response."""
        import types

        class FakeBlock:
            def __init__(self, text): self.type = "text"; self.text = text

        class FakeResponse:
            def __init__(self, text): self.content = [FakeBlock(text)]

        class FakeMessages:
            def __init__(self, text): self._text = text
            def create(self, **kwargs): return FakeResponse(self._text)

        class FakeClient:
            def __init__(self, text): self.messages = FakeMessages(text)

        cal = MacroCalendar.__new__(MacroCalendar)
        cal.client = FakeClient(json_str)
        return cal

    def test_valid_skip_json(self):
        json_str = '''{
            "should_trade": false,
            "skip_reason": "CPI release at 8:30 AM ET",
            "high_impact_events": ["CPI", "Core CPI"],
            "caution_events": [],
            "earnings_conflicts": [],
            "summary": "CPI day — stay out."
        }'''
        cal = self._make_cal_with_response(json_str)
        result = cal._run_check("March 11, 2026", WATCHLIST)
        self.assertFalse(result.should_trade)
        self.assertIn("CPI", result.events)
        self.assertEqual(result.raw_summary, "CPI day — stay out.")

    def test_valid_clear_json(self):
        json_str = '''{
            "should_trade": true,
            "skip_reason": "",
            "high_impact_events": [],
            "caution_events": ["Trade Balance"],
            "earnings_conflicts": [],
            "summary": "Clean day, trade balance only."
        }'''
        cal = self._make_cal_with_response(json_str)
        result = cal._run_check("March 17, 2026", WATCHLIST)
        self.assertTrue(result.should_trade)
        self.assertEqual(result.events, [])
        self.assertIn("Trade Balance", result.cautions)

    def test_earnings_conflict_in_json(self):
        json_str = '''{
            "should_trade": false,
            "skip_reason": "NVDA earnings today",
            "high_impact_events": [],
            "caution_events": [],
            "earnings_conflicts": ["NVDA"],
            "summary": "NVDA reports after close."
        }'''
        cal = self._make_cal_with_response(json_str)
        result = cal._run_check("March 19, 2026", WATCHLIST)
        self.assertFalse(result.should_trade)
        self.assertIn("NVDA", result.earnings_conflicts)

    def test_json_with_markdown_fences_still_parses(self):
        json_str = '''```json
{
    "should_trade": true,
    "skip_reason": "",
    "high_impact_events": [],
    "caution_events": [],
    "earnings_conflicts": [],
    "summary": "Clean day."
}
```'''
        cal = self._make_cal_with_response(json_str)
        result = cal._run_check("March 16, 2026", WATCHLIST)
        self.assertTrue(result.should_trade)

    def test_malformed_json_falls_back_to_keyword_scan(self):
        # Malformed JSON but contains CPI keyword — fallback should catch it
        bad_json = "Error parsing response. Today has CPI release at 8:30 AM."
        cal = self._make_cal_with_response(bad_json)
        result = cal._run_check("March 11, 2026", WATCHLIST)
        # Keyword fallback should detect CPI
        self.assertFalse(result.should_trade)

    def test_fomc_day_json(self):
        json_str = '''{
            "should_trade": false,
            "skip_reason": "FOMC rate decision at 2 PM ET plus PPI in morning",
            "high_impact_events": ["FOMC Rate Decision", "PPI", "Core PPI"],
            "caution_events": [],
            "earnings_conflicts": [],
            "summary": "FOMC + PPI double whammy. Stay flat all day."
        }'''
        cal = self._make_cal_with_response(json_str)
        result = cal._run_check("March 18, 2026", WATCHLIST)
        self.assertFalse(result.should_trade)
        self.assertEqual(len(result.events), 3)


# ── TestSkipKeywordCoverage ───────────────────────────────────────────────────

class TestSkipKeywordCoverage(unittest.TestCase):
    """Verify all expected dangerous events are covered in SKIP_KEYWORDS."""

    def test_fomc_covered(self):
        self.assertTrue(any("fomc" in k for k in SKIP_KEYWORDS))

    def test_cpi_covered(self):
        self.assertTrue(any("cpi" in k for k in SKIP_KEYWORDS))

    def test_ppi_covered(self):
        self.assertTrue(any("ppi" in k for k in SKIP_KEYWORDS))

    def test_nfp_covered(self):
        self.assertTrue(any("nfp" in k or "payroll" in k for k in SKIP_KEYWORDS))

    def test_gdp_covered(self):
        self.assertTrue(any("gdp" in k for k in SKIP_KEYWORDS))

    def test_pce_covered(self):
        self.assertTrue(any("pce" in k for k in SKIP_KEYWORDS))

    def test_federal_reserve_covered(self):
        self.assertTrue(any("federal reserve" in k for k in SKIP_KEYWORDS))

    def test_at_least_7_skip_triggers(self):
        # Sanity check — we should have robust coverage
        self.assertGreaterEqual(len(SKIP_KEYWORDS), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
