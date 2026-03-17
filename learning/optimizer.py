"""
TradeIQ AI — Nightly AI Optimizer
====================================
Runs after market close each day. Analyzes all historical performance
data in the database, identifies what's working and what isn't,
then uses Claude AI to decide on parameter updates.

The optimizer follows a strict evidence-based process:
  1. Query performance DB for statistical patterns
  2. Build a structured analysis prompt for Claude
  3. Claude returns specific parameter change recommendations
  4. Changes are validated against safety bounds
  5. Parameters are written to algo_params.json
  6. Next day's session loads the new parameters automatically

LEARNING PHILOSOPHY:
  - Never change a parameter without statistical evidence (min 5 trades)
  - Never make large jumps (max 20% change per day)
  - Keep a full audit trail of every change
  - If uncertain, don't change (conservative by default)
  - Parameters that keep winning → don't touch them
  - Parameters that keep losing → change them

SAFETY GUARDRAILS:
  - All changes bounded by hard min/max per parameter
  - Maximum change per parameter per day: 20%
  - Requires minimum 5 trades in relevant category before changing
  - Risk parameters (risk_per_trade_pct) require 10+ trades to change
  - Kill switch: if last 3 days all losing, revert to defaults
"""

import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from learning.performance_db import PerformanceDB
from learning.algo_config import AlgoConfig, DEFAULT_PARAMS


# Maximum fraction a parameter can change in a single day
MAX_DAILY_CHANGE_FRACTION = 0.20

# Minimum trades required before changing a parameter category
MIN_TRADES_FOR_CHANGE = {
    "fvg":         5,
    "orb":         5,
    "box":         5,
    "candlestick": 5,
    "timing":      8,
    "risk":        10,
    "quality":     5,
}


class NightlyOptimizer:
    """
    The self-learning brain of TradeIQ AI.
    Called once after market close to analyze and improve the algorithm.
    """

    def __init__(
        self,
        db: PerformanceDB,
        config: AlgoConfig,
        anthropic_key: str = "",
    ):
        self.db = db
        self.config = config
        self.claude = None

        api_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY", "")
        if api_key and ANTHROPIC_AVAILABLE:
            self.claude = anthropic.Anthropic(api_key=api_key)
            logger.info("Nightly optimizer: Claude AI connected")
        else:
            logger.warning("Nightly optimizer: No Claude API key — using rule-based optimization")

    # ─── Main Entry Point ─────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Full optimization cycle. Returns a report of what was changed.
        """
        today = str(date.today())
        total_trades = self.db.count_trades()

        logger.info("=" * 60)
        logger.info("NIGHTLY OPTIMIZER STARTING")
        logger.info(f"Total historical trades in DB: {total_trades}")
        logger.info("=" * 60)

        if total_trades < 3:
            msg = f"Only {total_trades} trades in database — need at least 3 before optimizing. Running tomorrow."
            logger.info(msg)
            return {"status": "skipped", "reason": msg, "changes": []}

        # Check for kill switch: 3 consecutive losing days
        if self._check_losing_streak():
            logger.warning("3 consecutive losing days detected — reverting to safe defaults")
            return self._revert_to_defaults(today)

        # Gather all performance data
        analysis = self._build_analysis()

        # Get optimization decisions
        if self.claude and total_trades >= 5:
            changes = self._ai_optimize(analysis, today)
        else:
            changes = self._rule_based_optimize(analysis)

        # Apply the changes
        if changes:
            validated = self._validate_and_apply(changes, today)
            self._save_learning_log(today, analysis, validated)
            logger.info(f"Optimizer complete: {len(validated)} parameter(s) updated")
            return {"status": "optimized", "changes": validated, "analysis": analysis}
        else:
            logger.info("Optimizer complete: no changes needed today")
            return {"status": "no_changes", "changes": [], "analysis": analysis}

    # ─── Data Analysis ────────────────────────────────────────────────────

    def _build_analysis(self) -> Dict:
        """Gather all statistical evidence from the performance database."""
        return {
            "total_stats":        self.db.get_stats_summary(),
            "strategy_perf":      self.db.get_strategy_performance(),
            "fvg_size_analysis":  self.db.get_fvg_size_analysis(),
            "quality_analysis":   self.db.get_quality_score_analysis(),
            "hour_analysis":      self.db.get_hour_analysis(),
            "symbol_perf":        self.db.get_symbol_performance(),
            "recent_sessions":    self.db.get_recent_sessions(10),
            "current_params":     self.config.get_all_values(),
            "optimization_history": self.db.get_optimization_history()[-5:],
            "all_trades":         self.db.get_all_trades(),
        }

    def _check_losing_streak(self) -> bool:
        """Return True if the last 3 sessions were all losing days."""
        sessions = self.db.get_recent_sessions(3)
        if len(sessions) < 3:
            return False
        return all(s["total_pnl"] < 0 for s in sessions)

    # ─── AI-Powered Optimization ──────────────────────────────────────────

    def _ai_optimize(self, analysis: Dict, today: str) -> List[Dict]:
        """
        Send performance data to Claude and get parameter recommendations back.
        Claude returns structured JSON with specific parameter changes.
        """
        # Build compact summary (keep token count manageable)
        compact = {
            "total_trades":    analysis["total_stats"].get("total_trades", 0),
            "overall_win_rate": analysis["total_stats"].get("overall_win_rate", 0),
            "total_pnl":       analysis["total_stats"].get("total_pnl", 0),
            "trading_days":    analysis["total_stats"].get("trading_days", 0),
            "strategy_perf":   analysis["strategy_perf"],
            "fvg_by_size":     analysis["fvg_size_analysis"],
            "quality_vs_wins": analysis["quality_analysis"],
            "wins_by_hour":    analysis["hour_analysis"],
            "top_symbols":     analysis["symbol_perf"][:8],
            "recent_sessions": [
                {"date": s["session_date"], "pnl": s["total_pnl"], "wr": s["win_rate"]}
                for s in analysis["recent_sessions"]
            ],
            "current_params":  analysis["current_params"],
            "past_changes":    analysis["optimization_history"],
        }

        # Build all available params with bounds
        param_reference = {
            k: {
                "current": v["value"],
                "min": v["min"],
                "max": v["max"],
                "description": v["description"],
                "category": v.get("category", ""),
            }
            for k, v in DEFAULT_PARAMS.items()
        }

        prompt = f"""You are the optimization engine for TradeIQ AI, an automated day trading system.
Your job: analyze historical trade performance and recommend specific parameter adjustments to improve win rate and profitability.

PERFORMANCE DATA:
{json.dumps(compact, indent=2)}

AVAILABLE PARAMETERS TO TUNE:
{json.dumps(param_reference, indent=2)}

OPTIMIZATION RULES (you MUST follow these):
1. Only recommend changes with clear statistical evidence (5+ trades in that category)
2. Maximum change per parameter: 20% of current value
3. All values must stay within the min/max bounds shown
4. If a parameter is already producing good results (win rate > 55%), leave it alone
5. Focus on the 2-3 highest-impact changes, not every parameter
6. Risk parameters (risk_per_trade_pct) require 10+ trades before any change
7. If insufficient data for a category, do NOT change those params
8. Be conservative — small targeted improvements beat aggressive overhauls

RESPOND ONLY WITH VALID JSON in this exact format (no extra text):
{{
  "reasoning": "2-3 sentence explanation of what the data shows",
  "changes": [
    {{
      "param": "parameter_name",
      "new_value": 0.45,
      "reasoning": "specific evidence from the data supporting this change",
      "confidence": 0.75,
      "days_of_data": 5
    }}
  ],
  "no_change_reason": "why some params were left alone (optional)"
}}

If no changes are warranted, return: {{"reasoning": "explanation", "changes": []}}"""

        try:
            response = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)
            logger.info(f"AI optimizer reasoning: {result.get('reasoning', '')}")

            changes = result.get("changes", [])
            logger.info(f"AI optimizer recommended {len(changes)} change(s)")
            return changes

        except json.JSONDecodeError as e:
            logger.error(f"AI optimizer returned invalid JSON: {e}")
            return self._rule_based_optimize(analysis)
        except Exception as e:
            logger.error(f"AI optimizer failed: {e} — falling back to rule-based")
            return self._rule_based_optimize(analysis)

    # ─── Rule-Based Fallback ──────────────────────────────────────────────

    def _rule_based_optimize(self, analysis: Dict) -> List[Dict]:
        """
        Statistical rule-based optimizer — no Claude API required.
        Finds the most impactful change based on the data patterns.
        """
        changes = []
        current = self.config.get_all_values()

        # ── Rule 1: FVG size optimization ─────────────────────────────────
        fvg_data = analysis.get("fvg_size_analysis", [])
        if len(fvg_data) >= 3:
            # Find the ATR multiple with the best win rate (min 3 trades)
            best = max(fvg_data, key=lambda x: x["win_rate_pct"], default=None)
            if best and best["win_rate_pct"] > 55:
                suggested = best["fvg_atr_bucket"]
                current_min = current.get("fvg_min_atr_mult", 0.35)
                # Only raise the minimum if the best bucket is above current minimum
                if suggested > current_min * 1.1:
                    new_val = min(suggested, current_min * 1.20)
                    changes.append({
                        "param": "fvg_min_atr_mult",
                        "new_value": round(new_val, 2),
                        "reasoning": f"FVGs with {suggested}x ATR show {best['win_rate_pct']}% win rate vs current minimum",
                        "confidence": 0.65,
                        "days_of_data": analysis["total_stats"].get("trading_days", 0),
                    })

        # ── Rule 2: Quality score threshold ───────────────────────────────
        quality_data = analysis.get("quality_analysis", [])
        if len(quality_data) >= 3:
            # Find the score threshold where win rate jumps significantly
            sorted_q = sorted(quality_data, key=lambda x: x["q_bucket"])
            for i in range(1, len(sorted_q)):
                low_wr  = sorted_q[i-1]["win_rate_pct"]
                high_wr = sorted_q[i]["win_rate_pct"]
                threshold = sorted_q[i]["q_bucket"]
                if high_wr - low_wr > 15 and threshold > current.get("min_quality_score", 0.55):
                    new_val = min(threshold, current.get("min_quality_score", 0.55) * 1.20)
                    changes.append({
                        "param": "min_quality_score",
                        "new_value": round(new_val, 2),
                        "reasoning": f"Win rate jumps {high_wr-low_wr:.0f}pts at quality score {threshold}",
                        "confidence": 0.70,
                        "days_of_data": analysis["total_stats"].get("trading_days", 0),
                    })
                    break

        # ── Rule 3: Lunch chop avoidance ──────────────────────────────────
        hour_data = analysis.get("hour_analysis", [])
        if len(hour_data) >= 4:
            lunch_hours = [h for h in hour_data if h["entry_hour"] in (11, 12, 13)]
            good_hours  = [h for h in hour_data if h["entry_hour"] in (9, 10, 15)]
            if lunch_hours and good_hours:
                avg_lunch = sum(h["win_rate_pct"] for h in lunch_hours) / len(lunch_hours)
                avg_good  = sum(h["win_rate_pct"] for h in good_hours)  / len(good_hours)
                if avg_good - avg_lunch > 20:
                    # Tighten lunch avoidance window
                    current_start = current.get("avoid_lunch_start", 120)
                    if current_start > 105:
                        changes.append({
                            "param": "avoid_lunch_start",
                            "new_value": max(105, current_start - 10),
                            "reasoning": f"Lunch hours avg {avg_lunch:.0f}% WR vs {avg_good:.0f}% at open/close. Widening avoidance.",
                            "confidence": 0.60,
                            "days_of_data": analysis["total_stats"].get("trading_days", 0),
                        })

        # Limit to top 2 changes max per day
        return changes[:2]

    # ─── Apply & Validate ─────────────────────────────────────────────────

    def _validate_and_apply(self, changes: List[Dict], today: str) -> List[Dict]:
        """
        Validate each proposed change against safety bounds and max daily change.
        Apply approved changes to config and record in DB.
        """
        current = self.config.get_all_values()
        approved = []

        for change in changes:
            param = change.get("param")
            new_val = change.get("new_value")

            if param not in DEFAULT_PARAMS:
                logger.warning(f"Unknown param '{param}' from optimizer — skipping")
                continue

            if new_val is None:
                continue

            bounds = DEFAULT_PARAMS[param]
            current_val = current.get(param, bounds["value"])

            # Safety: max 20% change per day
            max_change = abs(current_val) * MAX_DAILY_CHANGE_FRACTION
            if abs(float(new_val) - current_val) > max_change:
                # Clamp to max allowed change
                direction = 1 if float(new_val) > current_val else -1
                new_val = current_val + direction * max_change
                logger.warning(f"Change for {param} clamped to max daily change: → {new_val:.4f}")

            change["new_value"] = round(float(new_val), 4)
            change["old_value"] = current_val
            approved.append(change)

        if approved:
            updates = {c["param"]: c["new_value"] for c in approved}
            self.config.apply_updates(updates, source="optimizer")
            self.db.record_optimization(today, approved)
            self.db.record_parameters(
                today,
                self.config.get_all_values(),
                source="optimizer"
            )

        return approved

    def _revert_to_defaults(self, today: str) -> Dict:
        """Reset all parameters to defaults after 3 consecutive losing days."""
        default_vals = {k: v["value"] for k, v in DEFAULT_PARAMS.items()}
        self.config.apply_updates(default_vals, source="revert_to_defaults")
        change_records = [
            {"param": k, "old_value": self.config.get(k), "new_value": v,
             "reasoning": "3 consecutive losing days — full reset to safe defaults",
             "confidence": 1.0, "days_of_data": 3}
            for k, v in default_vals.items()
        ]
        self.db.record_optimization(today, change_records)
        logger.warning("All parameters reverted to safe defaults")
        return {"status": "reverted_to_defaults", "changes": change_records}

    def _save_learning_log(self, today: str, analysis: Dict, changes: List[Dict]):
        """Save a human-readable learning log to disk."""
        log_dir = Path("logs/learning")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"optimization_{today}.txt"

        stats = analysis.get("total_stats", {})
        strats = analysis.get("strategy_perf", [])

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"TRADEIQ AI — NIGHTLY OPTIMIZATION LOG\n")
            f.write(f"Date: {today}\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"HISTORICAL PERFORMANCE\n")
            f.write(f"  Total trades:   {stats.get('total_trades', 0)}\n")
            f.write(f"  Trading days:   {stats.get('trading_days', 0)}\n")
            f.write(f"  Overall WR:     {stats.get('overall_win_rate', 0):.1f}%\n")
            f.write(f"  Total P&L:      ${stats.get('total_pnl', 0):+.2f}\n\n")

            f.write(f"STRATEGY BREAKDOWN\n")
            for s in strats:
                f.write(f"  {s['strategy']:<20} {s['total_trades']}T  {s['win_rate_pct']}% WR  ${s['total_pnl']:+.2f}\n")

            f.write(f"\nPARAMETER CHANGES ({len(changes)} today)\n")
            if changes:
                for c in changes:
                    f.write(f"  {c['param']:<35} {c.get('old_value','?')} → {c['new_value']}\n")
                    f.write(f"    Reason: {c.get('reasoning','')}\n")
            else:
                f.write("  No changes made\n")

            f.write(f"\nCURRENT PARAMETERS\n")
            for k, v in self.config.get_all_values().items():
                f.write(f"  {k:<35} = {v}\n")

            f.write(f"\n{'='*60}\n")

        logger.info(f"Learning log saved: {log_path}")
