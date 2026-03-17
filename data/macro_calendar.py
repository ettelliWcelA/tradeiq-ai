"""
TradeIQ AI — Macro Calendar Guard
===================================
Checks for high-impact macro events and earnings each morning before the
session begins. If a dangerous event is detected the engine will NOT trade
that day — it will print a clear warning and shut itself down gracefully.

Events that trigger a SKIP:
  - FOMC rate decision or minutes release
  - CPI (Consumer Price Index)
  - PPI (Producer Price Index)
  - NFP (Non-Farm Payrolls / Jobs Report)
  - GDP release
  - PCE / Personal Income & Spending
  - Earnings for any ticker in the watchlist

Uses the Anthropic web_search tool to fetch today's economic calendar so
the check is always current (no hard-coded dates that go stale).

Requires:  ANTHROPIC_API_KEY in .env
Fallback:  If Anthropic is unavailable, returns SAFE with a warning so the
           user can manually verify — the engine will proceed but print a
           prominent disclaimer.
"""

import json
from datetime import datetime
from typing import List, Optional
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

import pytz
EASTERN = pytz.timezone("America/New_York")

# ── High-impact event keywords ─────────────────────────────────────────────
# If any of these appear in today's calendar, the session is skipped.
SKIP_KEYWORDS = [
    "fomc", "federal reserve", "fed rate", "fed decision", "interest rate decision",
    "cpi", "consumer price index",
    "ppi", "producer price index",
    "nfp", "non-farm payroll", "nonfarm payroll", "jobs report",
    "gdp", "gross domestic product",
    "pce", "personal consumption expenditure", "personal income",
    "jobless claims",  # initial claims can spike vol too
]

# Caution-only keywords — these reduce quality but don't skip the session
CAUTION_KEYWORDS = [
    "trade balance",
    "university of michigan",
    "consumer confidence",
    "ism manufacturing",
    "retail sales",
    "housing starts",
    "building permits",
]


class MacroCalendarResult:
    """Result object returned by MacroCalendar.check()"""

    def __init__(self, should_trade: bool, reason: str,
                 events: List[str], cautions: List[str],
                 earnings_conflicts: List[str],
                 raw_summary: str = ""):
        self.should_trade       = should_trade
        self.reason             = reason
        self.events             = events          # SKIP-level events found
        self.cautions           = cautions        # Caution-level events found
        self.earnings_conflicts = earnings_conflicts  # watchlist tickers with earnings
        self.raw_summary        = raw_summary     # full text from AI for logging

    def __bool__(self):
        return self.should_trade

    def console_block(self) -> str:
        """Formatted console output for the engine banner."""
        today = datetime.now(tz=EASTERN).strftime("%A, %B %d, %Y")
        if self.should_trade:
            lines = [
                f"  MACRO CHECK — {today}",
                "  Status: ✅ CLEAR TO TRADE",
            ]
            if self.cautions:
                lines.append(f"  Caution events: {', '.join(self.cautions)}")
                lines.append("  (These are minor — system will run normally)")
        else:
            lines = [
                f"  MACRO CHECK — {today}",
                "  Status: ❌ DO NOT TRADE TODAY",
                f"  Reason: {self.reason}",
            ]
            if self.events:
                lines.append(f"  High-impact events: {', '.join(self.events)}")
            if self.earnings_conflicts:
                lines.append(f"  Earnings conflicts: {', '.join(self.earnings_conflicts)}")
            lines += [
                "",
                "  TradeIQ will NOT scan or place trades today.",
                "  Script will exit in 10 seconds.",
                "  Come back tomorrow for clean price action.",
            ]
        return "\n".join(lines)


class MacroCalendar:
    """
    Uses Claude's web_search tool to fetch today's economic calendar
    and determine whether it is safe to trade.
    """

    def __init__(self, anthropic_key: str = ""):
        self.client: Optional[object] = None
        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=anthropic_key)

    def check(self, watchlist: List[str], date_override: str = "") -> MacroCalendarResult:
        """
        Main entry point. Returns a MacroCalendarResult.
        date_override: 'YYYY-MM-DD' string, used for testing. Defaults to today.
        """
        today_str = date_override or datetime.now(tz=EASTERN).strftime("%B %d, %Y")
        logger.info(f"MacroCalendar: checking events for {today_str}...")

        if not self.client:
            logger.warning("MacroCalendar: No Anthropic API key — skipping macro check. Verify manually.")
            return MacroCalendarResult(
                should_trade=True,
                reason="API key not set — manual verification required",
                events=[], cautions=[], earnings_conflicts=[],
                raw_summary="API unavailable",
            )

        try:
            return self._run_check(today_str, watchlist)
        except Exception as e:
            logger.error(f"MacroCalendar check failed: {e}")
            # On error, allow trading but warn loudly
            return MacroCalendarResult(
                should_trade=True,
                reason=f"Check failed ({e}) — proceed with caution",
                events=[], cautions=[], earnings_conflicts=[],
                raw_summary=str(e),
            )

    def _run_check(self, today_str: str, watchlist: List[str]) -> MacroCalendarResult:
        """Call Claude with web_search to get today's macro events."""

        watchlist_str = ", ".join(watchlist[:20])

        prompt = f"""Today is {today_str}.

You are a macro calendar assistant for a day trading system. Search the web for today's US economic calendar and earnings calendar, then respond ONLY with a valid JSON object — no explanation, no markdown, no code fences.

Watchlist tickers: {watchlist_str}

Search for:
1. High-impact US economic events today (FOMC, CPI, PPI, NFP, GDP, PCE, jobless claims, etc.)
2. Earnings announcements today for any ticker in the watchlist above

Respond with ONLY this JSON structure:
{{
  "should_trade": true or false,
  "skip_reason": "short explanation if should_trade is false, else empty string",
  "high_impact_events": ["list of high-impact event names found today"],
  "caution_events": ["list of medium-impact events that don't require skipping"],
  "earnings_conflicts": ["list of watchlist tickers reporting earnings today"],
  "summary": "1-2 sentence plain English summary of today's macro landscape"
}}

Rules:
- should_trade = false if ANY of these exist today: FOMC rate decision, CPI, PPI, NFP/jobs report, GDP, PCE/personal income
- should_trade = false if ANY watchlist ticker has earnings today
- should_trade = true if only minor events exist (trade balance, consumer confidence, etc.)
- Be conservative — when in doubt, set should_trade = false"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the final text response (after tool use)
        raw_text = ""
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                raw_text += block.text

        logger.debug(f"MacroCalendar raw response: {raw_text[:300]}")

        # Parse JSON — strip any accidental markdown fences
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"MacroCalendar JSON parse error: {e} — raw: {raw_text[:200]}")
            # If we can't parse the response, check for keywords manually
            return self._keyword_fallback(raw_text, watchlist)

        should_trade       = bool(data.get("should_trade", True))
        skip_reason        = data.get("skip_reason", "")
        high_impact        = data.get("high_impact_events", [])
        cautions           = data.get("caution_events", [])
        earnings_conflicts = data.get("earnings_conflicts", [])
        summary            = data.get("summary", "")

        logger.info(f"MacroCalendar result: should_trade={should_trade} | events={high_impact} | earnings={earnings_conflicts}")

        return MacroCalendarResult(
            should_trade=should_trade,
            reason=skip_reason or (f"High-impact events: {', '.join(high_impact)}" if high_impact else ""),
            events=high_impact,
            cautions=cautions,
            earnings_conflicts=earnings_conflicts,
            raw_summary=summary,
        )

    def _keyword_fallback(self, text: str, watchlist: List[str]) -> MacroCalendarResult:
        """
        If JSON parsing fails, scan the raw text for skip keywords.
        Conservative: any hit → don't trade.
        """
        text_lower = text.lower()
        found_skip    = [kw for kw in SKIP_KEYWORDS    if kw in text_lower]
        found_caution = [kw for kw in CAUTION_KEYWORDS if kw in text_lower]
        found_earnings = [sym for sym in watchlist if sym.lower() in text_lower and "earn" in text_lower]

        should_trade = len(found_skip) == 0 and len(found_earnings) == 0
        reason = f"Keyword match: {', '.join(found_skip + found_earnings)}" if not should_trade else ""

        return MacroCalendarResult(
            should_trade=should_trade,
            reason=reason,
            events=found_skip,
            cautions=found_caution,
            earnings_conflicts=found_earnings,
            raw_summary=text[:200],
        )
