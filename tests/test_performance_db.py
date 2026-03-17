"""
tests/test_performance_db.py
============================
Unit tests for learning/performance_db.py — PerformanceDB.
Uses in-memory / temp-dir SQLite so no files are left behind.
"""
import sys
import os
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import mock_deps  # noqa: F401

from learning.performance_db import PerformanceDB


def _temp_db():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "test_memory.db"
    db = PerformanceDB(db_path=db_path)
    return db, tmp


SAMPLE_TRADE = {
    "session_date": "2026-01-15",
    "symbol": "NVDA",
    "direction": "long",
    "strategy": "ORB_FVG_ENGULF",
    "entry_price": 100.0,
    "exit_price": 106.0,
    "stop_loss": 98.0,
    "target": 106.0,
    "shares": 10.0,
    "pnl": 60.0,
    "pnl_pct": 6.0,
    "outcome": "win",
    "exit_reason": "target_hit",
    "quality_score": 0.72,
    "entry_hour": 9,
    "entry_minute": 38,
    "orb_range": 2.5,
    "fvg_size": 1.2,
    "fvg_size_atr": 0.8,
    "atr": 1.5,
    "market_minute": 8,
    "signals": json.dumps(["ORB_BREAK", "FVG_1.20", "ENGULFING"]),
}

SAMPLE_SETUP = {
    "session_date": "2026-01-15",
    "symbol": "AMD",
    "strategy": "VWAP_PULLBACK",
    "direction": "long",
    "quality_score": 0.60,
    "executed": 1,
    "reject_reason": None,
    "rr_ratio": 3.0,
    "signals": json.dumps(["VWAP_PULLBACK", "EMA_TREND"]),
    "market_minute": 45,
}

SAMPLE_SESSION = {
    "session_date": "2026-01-15",
    "total_trades": 2,
    "wins": 1,
    "losses": 1,
    "win_rate": 0.50,
    "total_pnl": 30.0,
    "pnl_pct": 0.30,
    "starting_equity": 10_000.0,
    "ending_equity": 10_030.0,
    "kill_switch": 0,
    "day_trades_used": 2,
    "paper_mode": 1,
}


class TestDBInit(unittest.TestCase):
    def test_db_created_on_init(self):
        db, tmp = _temp_db()
        self.assertTrue(db.db_path.exists())
        tmp.cleanup()

    def test_empty_db_returns_zero_count(self):
        db, tmp = _temp_db()
        self.assertEqual(db.count_trades(), 0)
        tmp.cleanup()

    def test_schema_initialized(self):
        import sqlite3
        db, tmp = _temp_db()
        conn = sqlite3.connect(str(db.db_path))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        tmp.cleanup()
        for expected in ["trades", "setups", "sessions", "parameters", "optimizations"]:
            self.assertIn(expected, tables)


class TestRecordTrade(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_trade_increments_count(self):
        self.db.record_trade(SAMPLE_TRADE)
        self.assertEqual(self.db.count_trades(), 1)

    def test_record_multiple_trades(self):
        self.db.record_trade(SAMPLE_TRADE)
        t2 = SAMPLE_TRADE.copy()
        t2["symbol"] = "AMD"
        t2["pnl"] = -20.0
        t2["outcome"] = "loss"
        self.db.record_trade(t2)
        self.assertEqual(self.db.count_trades(), 2)

    def test_get_all_trades_returns_correct_data(self):
        self.db.record_trade(SAMPLE_TRADE)
        trades = self.db.get_all_trades()
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["symbol"], "NVDA")
        self.assertAlmostEqual(t["pnl"], 60.0)
        self.assertEqual(t["outcome"], "win")

    def test_get_all_trades_with_date_filter(self):
        self.db.record_trade(SAMPLE_TRADE)
        t2 = SAMPLE_TRADE.copy()
        t2["session_date"] = "2026-02-01"
        self.db.record_trade(t2)
        trades = self.db.get_all_trades(min_date="2026-02-01")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["session_date"], "2026-02-01")


class TestStrategyPerformance(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def _add_trades(self):
        for outcome, pnl in [("win", 60.0), ("win", 40.0), ("loss", -20.0)]:
            t = SAMPLE_TRADE.copy()
            t["outcome"] = outcome
            t["pnl"] = pnl
            self.db.record_trade(t)
        # Different strategy trade
        t2 = SAMPLE_TRADE.copy()
        t2["strategy"] = "VWAP_PULLBACK"
        t2["outcome"] = "loss"
        t2["pnl"] = -15.0
        self.db.record_trade(t2)

    def test_strategy_performance_correct_win_rate(self):
        self._add_trades()
        perf = self.db.get_strategy_performance()
        orb_perf = next(p for p in perf if p["strategy"] == "ORB_FVG_ENGULF")
        self.assertAlmostEqual(orb_perf["win_rate_pct"], 66.7, places=0)

    def test_strategy_performance_correct_total_pnl(self):
        self._add_trades()
        perf = self.db.get_strategy_performance()
        orb_perf = next(p for p in perf if p["strategy"] == "ORB_FVG_ENGULF")
        self.assertAlmostEqual(orb_perf["total_pnl"], 80.0)

    def test_strategy_performance_empty_db(self):
        perf = self.db.get_strategy_performance()
        self.assertEqual(perf, [])


class TestSetupRecording(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_setup(self):
        self.db.record_setup(SAMPLE_SETUP)
        # No exception = pass; verify via DB
        import sqlite3
        conn = sqlite3.connect(str(self.db.db_path))
        count = conn.execute("SELECT COUNT(*) FROM setups").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_record_rejected_setup(self):
        rejected = SAMPLE_SETUP.copy()
        rejected["executed"] = 0
        rejected["reject_reason"] = "R:R too low"
        self.db.record_setup(rejected)
        import sqlite3
        conn = sqlite3.connect(str(self.db.db_path))
        row = conn.execute("SELECT * FROM setups").fetchone()
        conn.close()
        self.assertEqual(row[6], 0)  # executed = 0


class TestSessionRecording(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_retrieve_session(self):
        self.db.record_session(SAMPLE_SESSION)
        sessions = self.db.get_recent_sessions()
        self.assertEqual(len(sessions), 1)
        s = sessions[0]
        self.assertEqual(s["session_date"], "2026-01-15")
        self.assertAlmostEqual(s["total_pnl"], 30.0)

    def test_session_upsert(self):
        # Two inserts with same date → second should replace
        self.db.record_session(SAMPLE_SESSION)
        updated = SAMPLE_SESSION.copy()
        updated["total_pnl"] = 50.0
        self.db.record_session(updated)
        sessions = self.db.get_recent_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertAlmostEqual(sessions[0]["total_pnl"], 50.0)


class TestParameterRecording(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_parameters(self):
        params = {"min_quality_score": 0.65, "stop_atr_mult": 0.75}
        self.db.record_parameters("2026-01-15", params)
        current = self.db.get_current_params()
        self.assertAlmostEqual(current["min_quality_score"], 0.65)
        self.assertAlmostEqual(current["stop_atr_mult"], 0.75)

    def test_get_current_params_most_recent(self):
        self.db.record_parameters("2026-01-15", {"min_quality_score": 0.60})
        self.db.record_parameters("2026-01-16", {"min_quality_score": 0.68})
        current = self.db.get_current_params()
        # Most recent value should be returned
        self.assertAlmostEqual(current["min_quality_score"], 0.68)


class TestOptimizationRecording(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_optimization(self):
        changes = [
            {"param": "min_quality_score", "old_value": 0.55, "new_value": 0.60,
             "reasoning": "Win rate higher above 0.60", "confidence": 0.75, "days_of_data": 5},
        ]
        self.db.record_optimization("2026-01-16", changes)
        history = self.db.get_optimization_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["param_name"], "min_quality_score")

    def test_record_multiple_optimizations(self):
        changes = [
            {"param": "fvg_min_atr_mult", "old_value": 0.35, "new_value": 0.45,
             "reasoning": "Bigger FVGs win more", "confidence": 0.8, "days_of_data": 7},
            {"param": "stop_atr_mult", "old_value": 0.75, "new_value": 0.80,
             "reasoning": "Less whipsaw", "confidence": 0.65, "days_of_data": 7},
        ]
        self.db.record_optimization("2026-01-16", changes)
        history = self.db.get_optimization_history()
        self.assertEqual(len(history), 2)


class TestStatsSummary(unittest.TestCase):
    def setUp(self):
        self.db, self.tmp = _temp_db()

    def tearDown(self):
        self.tmp.cleanup()

    def test_stats_summary_empty(self):
        summary = self.db.get_stats_summary()
        self.assertIn("total_trades", summary)
        self.assertEqual(summary["total_trades"], 0)

    def test_stats_summary_with_trades(self):
        self.db.record_trade(SAMPLE_TRADE)
        t2 = SAMPLE_TRADE.copy()
        t2["outcome"] = "loss"
        t2["pnl"] = -20.0
        self.db.record_trade(t2)
        summary = self.db.get_stats_summary()
        self.assertEqual(summary["total_trades"], 2)
        self.assertEqual(summary["total_wins"], 1)
        self.assertAlmostEqual(summary["overall_win_rate"], 50.0)
        self.assertAlmostEqual(summary["total_pnl"], 40.0)
        self.assertEqual(summary["trading_days"], 1)

    def test_quality_score_analysis(self):
        for qs in [0.55, 0.65, 0.55, 0.75]:
            t = SAMPLE_TRADE.copy()
            t["quality_score"] = qs
            self.db.record_trade(t)
        # With fewer than 2 trades in some buckets it may filter them out
        result = self.db.get_quality_score_analysis()
        self.assertIsInstance(result, list)

    def test_hour_analysis(self):
        for hour in [9, 10, 9]:
            t = SAMPLE_TRADE.copy()
            t["entry_hour"] = hour
            self.db.record_trade(t)
        result = self.db.get_hour_analysis()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_symbol_performance(self):
        for sym in ["NVDA", "NVDA", "AMD"]:
            t = SAMPLE_TRADE.copy()
            t["symbol"] = sym
            self.db.record_trade(t)
        # Only returns symbols with >= 2 trades
        result = self.db.get_symbol_performance()
        syms = [r["symbol"] for r in result]
        self.assertIn("NVDA", syms)


if __name__ == "__main__":
    unittest.main(verbosity=2)
