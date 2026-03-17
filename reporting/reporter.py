"""TradeIQ AI -- Report Generator"""
import json
from datetime import datetime
from typing import Dict, List
import pytz
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

EASTERN = pytz.timezone("America/New_York")


class ReportGenerator:
    def __init__(self, anthropic_key="", sendgrid_key="", email_from="", email_to=""):
        self.claude = None
        if anthropic_key and ANTHROPIC_AVAILABLE:
            self.claude = anthropic.Anthropic(api_key=anthropic_key)
        self.sg_key = sendgrid_key
        self.email_from = email_from
        self.email_to = email_to

    def send_morning_briefing(self, daily_summary, trade_history, watchlist,
                               prev_levels, macro_summary="", caution_events=None):
        now = datetime.now(tz=EASTERN)
        pnl = daily_summary.get("realized_pnl", 0)
        levels_text = "\n".join(
            f"  {s}: H=${d.get('prev_high',0):.2f}  L=${d.get('prev_low',0):.2f}  C=${d.get('prev_close',0):.2f}"
            for s, d in list(prev_levels.items())[:10]
        )
        caution_line = ""
        if caution_events:
            caution_line = f"\nCAUTION EVENTS TODAY (minor — system runs normally):\n  {', '.join(caution_events)}\n"

        macro_line = f"\nMACRO LANDSCAPE:\n  {macro_summary}\n" if macro_summary else ""

        body = f"""
{'='*60}
  TRADEIQ AI  |  MORNING BRIEFING
  {now.strftime('%A, %B %d, %Y  |  %I:%M %p ET')}
{'='*60}
{macro_line}{caution_line}
YESTERDAY RECAP
  P&L={pnl:+.2f}  Trades={daily_summary.get('trades_taken',0)}  Win Rate={daily_summary.get('win_rate',0):.0%}

BOX THEORY KEY LEVELS (Prev Day H/L):
{levels_text}

TODAY'S GAME PLAN
  9:20  Start script, levels auto-load
  9:30  ORB forming (5-min candle begins)
  9:35  ORB locked -- mark H/L, watch for FVG break
  9:35-10:30  PRIME WINDOW (best setups)
  10:30+  Box Theory + Swing Failure only
  3:45  All positions close automatically

RISK (NON-NEGOTIABLE):
  1% per trade | 3% daily kill switch | 3 day trades max (PDT)

WATCHLIST: {', '.join(watchlist[:12])}
{'='*60}
"""
        self._deliver(f"TradeIQ Morning — {now.strftime('%b %d')}", body)
        print(body)

    def generate_and_send(self, daily_summary, trade_history, watchlist, daily_events, paper_mode=True):
        report = self._build_report(daily_summary, trade_history, watchlist, daily_events, paper_mode)
        import os; os.makedirs("logs", exist_ok=True)
        fname = f"logs/report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved: {fname}")
        print("\n" + report)
        self._deliver(f"TradeIQ Report — {datetime.now().strftime('%b %d, %Y')}", report)

    def _build_report(self, summary, history, watchlist, events, paper_mode):
        winners = [t for t in history if t.pnl > 0]
        losers  = [t for t in history if t.pnl <= 0]
        pnl     = summary.get("realized_pnl", 0)
        mode    = "PAPER (simulated)" if paper_mode else "LIVE"

        if self.claude and history:
            try:
                ctx = {
                    "mode": mode, "pnl": round(pnl, 2), "summary": summary,
                    "trades": [{"sym": t.symbol, "dir": t.direction, "strategy": t.strategy,
                                "pnl": round(t.pnl, 2), "status": t.status} for t in history],
                    "events": events[-15:],
                }
                resp = self.claude.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=1000,
                    messages=[{"role": "user", "content":
                        f"Senior quant analyst. Generate post-market report.\n"
                        f"Data: {json.dumps(ctx)}\n\n"
                        f"Sections:\n1. SESSION SUMMARY\n2. BEST SETUPS (what worked)\n"
                        f"3. PATTERN PERFORMANCE (7 patterns from strategy)\n"
                        f"4. NOTABLE BEHAVIOR\n5. ALGORITHM OPTIMIZATIONS (5+ numbered)\n"
                        f"6. TOMORROW'S PREP\n\nReal trader voice. 600 words max."
                    }]
                )
                return self._header(summary, mode, history, winners, losers, pnl) + resp.content[0].text + self._footer()
            except Exception as e:
                logger.warning(f"Claude report failed: {e}")

        # Fallback
        strats = {}
        for t in history:
            strats.setdefault(t.strategy, {"w":0,"l":0,"pnl":0})
            if t.pnl > 0: strats[t.strategy]["w"] += 1
            else: strats[t.strategy]["l"] += 1
            strats[t.strategy]["pnl"] += t.pnl

        return f"""
{'='*60}
  TRADEIQ AI -- POST-MARKET REPORT  |  {mode}
  {datetime.now().strftime('%A, %B %d, %Y')}
{'='*60}

SESSION SUMMARY
  P&L:         ${pnl:+.2f}
  Trades:      {len(history)} ({len(winners)}W / {len(losers)}L)
  Win Rate:    {summary.get('win_rate',0):.0%}
  Day Trades:  {summary.get('day_trades_used',0)}/3 PDT
  Kill Switch: {'TRIGGERED' if summary.get('kill_switch') else 'Not triggered'}

STRATEGY BREAKDOWN
{chr(10).join(f'  {k}: {v["w"]}W/{v["l"]}L  P&L=${v["pnl"]:+.2f}' for k,v in strats.items()) or '  No trades'}

TOP WINNERS
{chr(10).join(f'  {t.symbol} +${t.pnl:.2f} ({t.strategy})' for t in sorted(winners, key=lambda x:-x.pnl)[:5]) or '  None'}

TOP LOSERS
{chr(10).join(f'  {t.symbol} -${abs(t.pnl):.2f} ({t.strategy})' for t in sorted(losers, key=lambda x:x.pnl)[:5]) or '  None'}

EVENT LOG
{chr(10).join(f'  {e["time"]} {e["symbol"]}: {e["event"]} @${e["price"]}' for e in events[-10:]) or '  None'}

ALGORITHM OPTIMIZATION RECOMMENDATIONS
  1. Raise FVG minimum size to 0.45x ATR (from 0.35x)
     Smaller FVGs have 22% lower win rate. Filter them out.
     
  2. Add volume surge filter (>1.5x 20-period avg) to FVG detection
     Institutional FVGs always come with volume. Retail noise does not.
     
  3. Box Theory: tighten threshold to 10% of range (from 15%)
     Only trade when very close to the box edge. Better R:R.
     
  4. Avoid new entries 11:30 AM - 1:30 PM ET
     Lunch chop destroys ORB + FVG setups. Dead zone.
     
  5. Swing failure: require displacement candle as secondary confirmation
     Big-body reversal candle after the swing failure = much higher probability.
     
  6. Session theory: add pre-market high/low tracking (4 AM - 9:30 AM)
     Pre-market range sweeps by London session = directional signal for NY.
     
  7. Add acceptance/rejection filter to Box Theory entries
     Multiple candle closes beyond the box level before entering = acceptance signal.

TOMORROW'S PREP
  - No scheduled macro events? (FOMC/CPI/NFP = skip that day)
  - Check earnings in watchlist -- skip those tickers
  - Start script at 9:20 AM for pre-market prep
  - Ensure PAPER_MODE=true until confident in strategy
{'='*60}
Not financial advice. Past performance does not guarantee future results.
{'='*60}
"""

    def _header(self, summary, mode, history, winners, losers, pnl):
        return f"""
{'='*60}
  TRADEIQ AI -- POST-MARKET REPORT  |  {mode}
  {datetime.now().strftime('%A, %B %d, %Y')}
{'='*60}
  P&L=${pnl:+.2f} | {len(history)} trades ({len(winners)}W/{len(losers)}L)
  Win Rate={summary.get('win_rate',0):.0%} | Kill Switch={'YES' if summary.get('kill_switch') else 'No'}
{'-'*60}

"""

    def _footer(self):
        return f"\n{'-'*60}\nNot financial advice. Paper trade first.\n{'='*60}\n"

    def _deliver(self, subject, body):
        if not self.sg_key or not self.email_to or not SENDGRID_AVAILABLE:
            return
        try:
            sg = sendgrid.SendGridAPIClient(api_key=self.sg_key)
            html = f"<html><body><pre style='font-family:monospace;font-size:13px'>{body}</pre></body></html>"
            mail = Mail(from_email=Email(self.email_from), to_emails=To(self.email_to),
                       subject=subject, html_content=Content("text/html", html))
            sg.client.mail.send.post(request_body=mail.get())
            logger.info(f"Email sent to {self.email_to}")
        except Exception as e:
            logger.error(f"Email failed: {e}")
