"""
TradeIQ AI -- Main Engine (Self-Learning Edition)
Loads learned params, runs session, records everything, optimizes nightly.
"""
import os, sys, time, signal, json
from datetime import datetime, date
from typing import Dict, List, Optional
import pytz, schedule
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv

from data.robinhood_data import RobinhoodDataManager, AI_STOCKS
from data.macro_calendar import MacroCalendar
from strategies.unified_strategy import UnifiedStrategy, SessionPhase, ORBLevels
from risk.risk_manager import RiskManager
from execution.executor import ExecutionEngine
from reporting.reporter import ReportGenerator
from learning.performance_db import PerformanceDB
from learning.algo_config import AlgoConfig
from learning.optimizer import NightlyOptimizer

load_dotenv()
EASTERN = pytz.timezone("America/New_York")
console = Console()


class TradeIQEngine:
    def __init__(self, watchlist=None, paper_mode=True):
        self.paper_mode = paper_mode
        self.watchlist  = watchlist or AI_STOCKS
        self.username    = os.getenv("ROBINHOOD_USERNAME", "")
        self.password    = os.getenv("ROBINHOOD_PASSWORD", "")
        self.totp_secret = os.getenv("ROBINHOOD_TOTP_SECRET", "")
        account_sz       = float(os.getenv("ACCOUNT_SIZE", "10000"))

        # Learning system first — everything reads from it
        self.db        = PerformanceDB()
        self.algo_cfg  = AlgoConfig()
        self.algo_cfg.save_defaults()
        self.optimizer = NightlyOptimizer(
            db=self.db, config=self.algo_cfg,
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        console.print(self.algo_cfg.describe())

        # Trading components using learned params
        self.data     = RobinhoodDataManager()
        self.macro    = MacroCalendar(anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""))
        self.risk     = RiskManager(
            max_risk_pct=self.algo_cfg.get("risk_per_trade_pct", 1.0),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0")),
            max_concurrent=int(os.getenv("MAX_CONCURRENT_POSITIONS", "3")),
            default_rr=self.algo_cfg.get("default_rr_ratio", 3.0),
            account_size=account_sz,
            paper_mode=paper_mode,
        )
        self.strategy = UnifiedStrategy({
            "min_rr":                           2.0,
            "default_rr":                       self.algo_cfg.get("default_rr_ratio", 3.0),
            "min_fvg_atr":                      self.algo_cfg.get("fvg_min_atr_mult", 0.45),
            "min_quality":                      self.algo_cfg.get("min_quality_score", 0.55),
            "engulf_ratio":                     self.algo_cfg.get("engulfing_min_ratio", 1.10),
            "disp_body_pct":                    self.algo_cfg.get("displacement_body_pct", 0.65),
            "box_threshold":                    self.algo_cfg.get("box_threshold_pct", 0.10),
            "box_acceptance_candles":           int(self.algo_cfg.get("box_acceptance_candles", 2)),
            "stop_atr_mult":                    self.algo_cfg.get("stop_atr_mult", 0.75),
            "vwap_std_mult":                    self.algo_cfg.get("vwap_std_mult", 1.5),
            "rvol_threshold":                   self.algo_cfg.get("rvol_threshold", 1.5),
            "rsi_period":                       int(self.algo_cfg.get("rsi_period", 14)),
            "fvg_rvol_min":                     self.algo_cfg.get("fvg_rvol_min", 1.5),
            "swing_failure_require_displacement": int(self.algo_cfg.get("swing_failure_require_displacement", 1)),
            "premarket_track_enabled":          int(self.algo_cfg.get("premarket_track_enabled", 1)),
            "premarket_sweep_quality_bonus":    self.algo_cfg.get("premarket_sweep_quality_bonus", 0.08),
        })
        self.executor = ExecutionEngine(self.risk, paper_mode=paper_mode)
        self.reporter = ReportGenerator(
            anthropic_key=os.getenv("ANTHROPIC_API_KEY", ""),
            sendgrid_key=os.getenv("SENDGRID_API_KEY", ""),
            email_from=os.getenv("EMAIL_FROM", ""),
            email_to=os.getenv("EMAIL_TO", ""),
        )

        self._orb_set:     Dict[str, bool] = {}
        self._no_setup:    Dict[str, bool] = {}
        self._running      = False
        self._prev_levels: Dict[str, dict] = {}
        self._daily_events: List[dict] = []
        self._session_atrs: Dict[str, float] = {}
        self._macro_clear: bool = True   # Gates trading — False = skip today
        self._macro_result = None        # MacroCalendarResult from morning check

    def start(self):
        self._print_banner()
        if not self._login():
            logger.error("Login failed")
            sys.exit(1)

        self._running = True
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self.db.record_parameters(str(date.today()), self.algo_cfg.get_all_values(), "session_start")
        schedule.every().day.at("06:30").do(self._morning_briefing)
        schedule.every().day.at("08:20").do(self._pre_market_prep)
        schedule.every().day.at("14:45").do(self._close_all)
        schedule.every().day.at("15:05").do(self._end_of_day)

        if self.data.is_market_open():
            self._pre_market_prep()

        logger.info("Engine running. Ctrl+C to stop.")
        while self._running:
            schedule.run_pending()
            if self.data.is_market_open():
                self._scan_cycle()
            time.sleep(30)

    def _login(self):
        if not self.username or not self.password:
            logger.error("Missing ROBINHOOD_USERNAME or ROBINHOOD_PASSWORD in .env")
            return False
        logger.info(f"Logging in as {self.username}...")
        return self.data.login(self.username, self.password, self.totp_secret)

    def _pre_market_prep(self):
        if not self._macro_clear:
            logger.warning("PRE-MARKET PREP skipped — macro event today.")
            return
        logger.info("=== PRE-MARKET PREP ===")
        equity = self.data.get_account_equity() or self.risk.account_size
        self.risk.initialize_day(equity)
        self.data.prefetch_all(self.watchlist)
        for sym in self.watchlist:
            levels = self.data.get_previous_day_levels(sym)
            if levels:
                self._prev_levels[sym] = levels
                self.strategy.initialize_symbol(sym, levels.get("prev_high",0), levels.get("prev_low",0))

            # Opt 6: Fetch pre-market high/low for directional bias
            pm_levels = self.data.get_premarket_levels(sym)
            if pm_levels:
                self.strategy.set_premarket_levels(
                    sym,
                    pm_levels.get("premarket_high", 0),
                    pm_levels.get("premarket_low", 0),
                )

            self._session_atrs[sym] = self.data.get_atr(sym)
            self._orb_set[sym]  = False
            self._no_setup[sym] = False
        logger.info(f"Ready | Equity=${equity:,.2f}")

    def _scan_cycle(self):
        if not self._macro_clear:
            return  # Macro event today — no scanning
        if self.risk.is_kill_switch():
            return
        now   = datetime.now(tz=EASTERN)
        phase = self.strategy.get_session_phase()
        if now.minute % 5 != 0 and now.minute % 5 != 1:
            return

        if now.hour == 9 and now.minute in (35, 36):
            for sym in self.watchlist:
                if not self._orb_set.get(sym):
                    self._set_orb(sym)

        for pos in self.risk.get_open_positions():
            price = self.data.get_latest_price(pos.symbol)
            if price:
                result = self.executor.check_targets(pos.symbol, price)
                if result:
                    self._on_position_closed(pos.symbol, price, result)

        market_minute = (now.hour - 9) * 60 + now.minute - 30
        avoid_s = int(self.algo_cfg.get("avoid_lunch_start", 120))
        avoid_e = int(self.algo_cfg.get("avoid_lunch_end", 210))
        in_lunch = avoid_s <= market_minute <= avoid_e

        if not in_lunch and phase not in (SessionPhase.PRE_MARKET, SessionPhase.ORB_FORMING, SessionPhase.CLOSED):
            for sym in self.watchlist:
                if self._no_setup.get(sym): continue
                if any(p.symbol == sym for p in self.risk.get_open_positions()): continue
                self._scan_symbol(sym, phase, market_minute)

        timeout = int(self.algo_cfg.get("orb_timeout_minutes", 60))
        if market_minute >= timeout:
            for sym in self.watchlist:
                if not self._no_setup.get(sym) and not any(p.symbol==sym for p in self.risk.get_open_positions()):
                    self._no_setup[sym] = True

    def _set_orb(self, symbol):
        candle = self.data.get_opening_candle(symbol)
        if candle is not None:
            orb = ORBLevels(
                high=float(candle["high"]), low=float(candle["low"]),
                open_price=float(candle["open"]), close_price=float(candle["close"]),
                range_size=float(candle["high"])-float(candle["low"]),
                formed_at=candle.name,
            )
            self.strategy.set_orb(symbol, orb)
            self._orb_set[symbol] = True
            self._log_event(symbol, "ORB_SET", float(candle["close"]))

    def _scan_symbol(self, symbol, phase, market_minute):
        try:
            df    = self.data.get_session_bars(symbol)
            if df.empty or len(df) < 4: return
            atr   = self._session_atrs.get(symbol) or self.data.get_atr(symbol)
            price = self.data.get_latest_price(symbol) or float(df.iloc[-1]["close"])
            setup = self.strategy.analyze(symbol, df, atr, price, phase)

            if setup:
                self.db.record_setup({
                    "session_date": str(date.today()), "symbol": symbol,
                    "strategy": setup.strategy, "direction": setup.direction,
                    "quality_score": setup.quality_score, "executed": 0,
                    "reject_reason": None, "rr_ratio": round(setup.rr_ratio, 2),
                    "signals": json.dumps(setup.signals), "market_minute": market_minute,
                })
                min_q = self.algo_cfg.get("min_quality_score", 0.55)
                if setup.quality_score < min_q:
                    logger.info(f"{symbol}: Quality {setup.quality_score:.2f} < {min_q:.2f} threshold")
                    return
                position = self.executor.execute_setup(setup)
                if position:
                    self._log_event(symbol, "TRADE_ENTERED", price, setup.strategy)
                    self._print_trade_alert(setup, position)
                    orb = self.strategy._orb.get(symbol)
                    position._meta = {
                        "orb_range": orb.range_size if orb else None,
                        "fvg_size":  setup.fvg.size if setup.fvg else None,
                        "atr": atr, "market_minute": market_minute, "signals": setup.signals,
                    }
        except Exception as e:
            logger.error(f"Scan error [{symbol}]: {e}")

    def _on_position_closed(self, symbol, exit_price, reason):
        if not self.risk._history: return
        pos = self.risk._history[-1]
        if pos.symbol != symbol: return
        meta = getattr(pos, "_meta", {}) or {}
        atr  = meta.get("atr") or self._session_atrs.get(symbol, 0)
        fvg  = meta.get("fvg_size")
        self.db.record_trade({
            "session_date": str(date.today()), "symbol": symbol,
            "direction": pos.direction, "strategy": pos.strategy,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "stop_loss": pos.stop_loss, "target": pos.target_2,
            "shares": pos.shares, "pnl": pos.pnl, "pnl_pct": pos.pnl_pct,
            "outcome": "win" if pos.pnl > 0 else "loss" if pos.pnl < 0 else "breakeven",
            "exit_reason": reason, "quality_score": pos.quality_score,
            "entry_hour": pos.entry_time.hour if pos.entry_time else None,
            "entry_minute": pos.entry_time.minute if pos.entry_time else None,
            "orb_range": meta.get("orb_range"), "fvg_size": fvg,
            "fvg_size_atr": (fvg/atr) if (fvg and atr) else None,
            "atr": atr, "market_minute": meta.get("market_minute"),
            "signals": json.dumps(meta.get("signals", [])),
        })
        logger.info(f"Recorded to DB: {symbol} {'WIN' if pos.pnl>0 else 'LOSS'} ${pos.pnl:+.2f}")

    def _close_all(self):
        logger.info("=== 3:45 PM CLOSE ALL ===")
        for pos in self.risk.get_open_positions():
            price = self.data.get_latest_price(pos.symbol) or pos.entry_price
            self._on_position_closed(pos.symbol, price, "eod_close")
        self.executor.close_all_positions("eod_close")

    def _end_of_day(self):
        logger.info("=== END OF DAY + OPTIMIZER ===")
        summary = self.risk.get_summary()
        history = self.risk.get_history()

        self.db.record_session({
            "session_date": str(date.today()),
            "total_trades": summary.get("trades_taken", 0),
            "wins": summary.get("trades_won", 0),
            "losses": summary.get("trades_taken",0) - summary.get("trades_won",0),
            "win_rate": summary.get("win_rate", 0),
            "total_pnl": summary.get("realized_pnl", 0),
            "pnl_pct": summary.get("pnl_pct", 0),
            "starting_equity": summary.get("starting_equity", 0),
            "ending_equity": summary.get("starting_equity",0) + summary.get("realized_pnl",0),
            "kill_switch": 1 if summary.get("kill_switch") else 0,
            "day_trades_used": summary.get("day_trades_used", 0),
            "paper_mode": 1 if self.paper_mode else 0,
        })

        self.reporter.generate_and_send(
            daily_summary=summary, trade_history=history,
            watchlist=self.watchlist, daily_events=self._daily_events,
            paper_mode=self.paper_mode,
        )

        # THE SELF-LEARNING STEP
        logger.info("Running nightly optimizer...")
        result = self.optimizer.run()

        if result["status"] == "optimized":
            changes = result["changes"]
            console.print(Panel(
                "\n".join([
                    f"[bold green]Optimizer updated {len(changes)} parameter(s) for tomorrow:[/]",
                    *[f"  [cyan]{c['param']}[/]: {c.get('old_value','?')} → [yellow]{c['new_value']}[/]\n  [dim]{c.get('reasoning','')}[/]"
                      for c in changes],
                ]),
                title="[bold]⬡ ALGORITHM SELF-OPTIMIZED[/bold]",
                border_style="green",
            ))
        elif result["status"] == "reverted_to_defaults":
            console.print(Panel("[bold red]3 losing days → reset to safe defaults[/]", border_style="red"))
        
        logger.info(f"Learning DB: {self.db.count_trades()} total trades across all history")

    def _morning_briefing(self):
        # ── Step 1: Macro calendar check ──────────────────────────────────
        logger.info("=== MORNING BRIEFING + MACRO CHECK ===")
        result = self.macro.check(watchlist=self.watchlist)
        self._macro_result = result
        self._macro_clear  = result.should_trade

        # Print the macro verdict prominently
        border = "=" * 62
        console.print(f"\n{border}")
        console.print(result.console_block())
        console.print(f"{border}\n")

        if not result.should_trade:
            # Log the skip to the DB as a session with 0 trades
            from datetime import date
            self.db.record_session({
                "session_date":    str(date.today()),
                "total_trades":    0, "wins": 0, "losses": 0,
                "win_rate":        0.0, "total_pnl": 0.0, "pnl_pct": 0.0,
                "starting_equity": self.risk.account_size,
                "ending_equity":   self.risk.account_size,
                "kill_switch":     0, "day_trades_used": 0,
                "paper_mode":      1 if self.paper_mode else 0,
            })
            logger.warning(f"SESSION SKIPPED — {result.reason}")
            # Give user 10 seconds to read the screen, then exit cleanly
            import time as _time
            _time.sleep(10)
            self._running = False
            return

        # ── Step 2: Normal morning briefing ───────────────────────────────
        now = datetime.now(tz=EASTERN)
        pnl = self.risk.get_summary().get("realized_pnl", 0)
        levels_text = "\n".join(
            f"  {s}: H=${d.get('prev_high',0):.2f}  L=${d.get('prev_low',0):.2f}  C=${d.get('prev_close',0):.2f}"
            for s, d in list(self._prev_levels.items())[:10]
        )
        caution_line = ""
        if result.cautions:
            caution_line = f"\nCAUTION EVENTS TODAY (minor — system runs normally):\n  {', '.join(result.cautions)}\n"

        self.reporter.send_morning_briefing(
            daily_summary=self.risk.get_summary(),
            trade_history=self.risk.get_history(),
            watchlist=self.watchlist,
            prev_levels=self._prev_levels,
            macro_summary=result.raw_summary,
            caution_events=result.cautions,
        )

    def _log_event(self, symbol, event, price, detail=""):
        self._daily_events.append({
            "time": datetime.now(tz=EASTERN).strftime("%H:%M:%S"),
            "symbol": symbol, "event": event,
            "price": round(price,2), "detail": detail,
        })

    def _print_trade_alert(self, setup, position):
        color = "green" if setup.direction == "long" else "red"
        mode  = "[PAPER]" if self.paper_mode else "[LIVE]"
        console.print(Panel(
            f"[bold {color}]{'▲ LONG' if setup.direction=='long' else '▼ SHORT'} {setup.symbol} {mode}[/]\n\n"
            f"Strategy: {setup.strategy} | Quality: {setup.quality_score:.0%}\n"
            f"Entry: ${setup.entry_price:.2f}  Stop: ${setup.stop_loss:.2f}  Target: ${setup.target_2:.2f}  R:R={setup.rr_ratio:.1f}\n"
            f"Signals: {' → '.join(setup.signals)}",
            title="TRADE EXECUTED", border_style=color,
        ))

    def _print_banner(self):
        db_stats  = self.db.get_stats_summary() or {}
        days      = int(db_stats.get("trading_days") or 0)
        trades    = int(db_stats.get("total_trades") or 0)
        wr        = float(db_stats.get("overall_win_rate") or 0.0)
        mode_col  = "green" if self.paper_mode else "red"
        mem_line  = f"{days} days | {trades} trades | {wr:.1f}% WR" if trades > 0 else "First session — learning starts today"
        console.print(Panel(
            Text.assemble(
                ("⬡ TRADEIQ AI — Self-Learning Edition\n\n", "bold blue"),
                ("Mode:       ", "dim"), (f"{'PAPER' if self.paper_mode else '⚠️  LIVE'}\n", f"bold {mode_col}"),
                ("Memory:     ", "dim"), (f"{mem_line}\n", "white"),
                ("Learns:     ", "dim"), ("Every night at 4:15 PM — gets better every session\n", "white"),
            ),
            border_style="blue",
        ))

    def _shutdown(self, sig, frame):
        self._close_all()
        self._running = False
        self.data.logout()
        sys.exit(0)
