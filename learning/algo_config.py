"""
TradeIQ AI — Dynamic Parameter Config
=======================================
All tunable algorithm parameters live here.
The AI optimizer rewrites this file each night with improved values.

Parameters are loaded fresh at the start of each session,
so every day the algorithm runs with the latest optimized values.

PARAMETER SAFETY:
  Every parameter has hard min/max bounds.
  The optimizer CANNOT set values outside these bounds.
  This prevents runaway optimization destroying the strategy.
"""

import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from datetime import datetime

CONFIG_PATH = Path("learning/algo_params.json")

# ─── Default Parameters ───────────────────────────────────────────────────
# These are the starting values. The optimizer adjusts them over time.
# Format: param_name: { value, min, max, description }

DEFAULT_PARAMS = {

    # ── FVG Detection ──────────────────────────────────────────────────────
    "fvg_min_atr_mult": {
        "value": 0.45,
        "min": 0.20, "max": 0.80,
        "description": "Minimum FVG size as multiple of ATR. Raised to 0.45 (from 0.35) — smaller FVGs have 22% lower win rate.",
        "category": "fvg"
    },
    "fvg_body_ratio_min": {
        "value": 0.40,
        "min": 0.25, "max": 0.70,
        "description": "Middle candle must have body >= this % of candle range to qualify as FVG impulse.",
        "category": "fvg"
    },
    "fvg_retest_tolerance": {
        "value": 0.002,
        "min": 0.001, "max": 0.010,
        "description": "How close price must get to FVG zone to count as a retest (as % of price).",
        "category": "fvg"
    },
    "fvg_max_age_bars": {
        "value": 10,
        "min": 4, "max": 20,
        "description": "Maximum number of 5-min bars before an FVG expires.",
        "category": "fvg"
    },

    # ── ORB Settings ───────────────────────────────────────────────────────
    "orb_break_threshold": {
        "value": 0.001,
        "min": 0.0005, "max": 0.005,
        "description": "Price must close this % beyond ORB level to count as a break.",
        "category": "orb"
    },
    "orb_timeout_minutes": {
        "value": 60,
        "min": 30, "max": 90,
        "description": "Minutes after open before giving up on ORB setup.",
        "category": "orb"
    },

    # ── Box Theory ─────────────────────────────────────────────────────────
    "box_threshold_pct": {
        "value": 0.10,
        "min": 0.05, "max": 0.20,
        "description": "How close to box edge (as % of box range) to trigger Box Theory scan. Tightened to 0.10 for better R:R.",
        "category": "box"
    },
    "box_acceptance_candles": {
        "value": 2,
        "min": 1, "max": 4,
        "description": "Opt 7: Number of candle closes beyond box level required as acceptance confirmation before entry.",
        "category": "box"
    },

    # ── Candlestick Thresholds ─────────────────────────────────────────────
    "engulfing_min_ratio": {
        "value": 1.10,
        "min": 1.00, "max": 1.50,
        "description": "Engulfing candle body must be >= this multiple of prior candle body.",
        "category": "candlestick"
    },
    "displacement_body_pct": {
        "value": 0.65,
        "min": 0.50, "max": 0.85,
        "description": "Displacement candle body as % of candle range.",
        "category": "candlestick"
    },
    "momentum_candles_required": {
        "value": 3,
        "min": 2, "max": 5,
        "description": "Number of consecutive large-body candles required for momentum signal.",
        "category": "candlestick"
    },

    # ── Risk / Trade Management ────────────────────────────────────────────
    "risk_per_trade_pct": {
        "value": 1.00,
        "min": 0.50, "max": 2.00,
        "description": "Account % risked per trade.",
        "category": "risk"
    },
    "default_rr_ratio": {
        "value": 3.00,
        "min": 2.00, "max": 5.00,
        "description": "Default reward:risk ratio for target placement.",
        "category": "risk"
    },
    "min_quality_score": {
        "value": 0.55,
        "min": 0.40, "max": 0.75,
        "description": "Minimum setup quality score required to execute a trade.",
        "category": "risk"
    },
    "stop_atr_mult": {
        "value": 0.75,
        "min": 0.40, "max": 1.50,
        "description": "Stop loss distance as multiple of ATR.",
        "category": "risk"
    },

    # ── Session / Timing ───────────────────────────────────────────────────
    "prime_window_end_minute": {
        "value": 60,
        "min": 30, "max": 90,
        "description": "Minutes after open that the prime window ends (ORB+FVG priority).",
        "category": "timing"
    },
    "avoid_lunch_start": {
        "value": 120,
        "min": 90, "max": 150,
        "description": "Minutes after open to stop new entries (lunch chop avoidance).",
        "category": "timing"
    },
    "avoid_lunch_end": {
        "value": 210,
        "min": 150, "max": 240,
        "description": "Minutes after open to resume new entries after lunch.",
        "category": "timing"
    },

    # ── Quality Score Weights ──────────────────────────────────────────────
    "quality_momentum_bonus": {
        "value": 0.10,
        "min": 0.00, "max": 0.20,
        "description": "Quality score bonus when momentum aligns with setup direction.",
        "category": "quality"
    },
    "quality_structure_bonus": {
        "value": 0.10,
        "min": 0.00, "max": 0.20,
        "description": "Quality score bonus when structure break confirms direction.",
        "category": "quality"
    },

    # -- v3: VWAP Engine --------------------------------------------------
    "vwap_std_mult": {
        "value": 1.5, "min": 1.0, "max": 3.0,
        "description": "VWAP band standard deviation multiplier.",
        "category": "vwap"
    },
    "vwap_retest_tolerance": {
        "value": 0.003, "min": 0.001, "max": 0.010,
        "description": "How close price must get to VWAP to count as retest.",
        "category": "vwap"
    },

    # -- v3: RSI Engine ---------------------------------------------------
    "rsi_period": {
        "value": 14, "min": 7, "max": 21,
        "description": "RSI lookback period.",
        "category": "rsi"
    },
    "rsi_overbought": {
        "value": 70, "min": 60, "max": 80,
        "description": "RSI level considered overbought (avoid new longs above).",
        "category": "rsi"
    },
    "rsi_oversold": {
        "value": 30, "min": 20, "max": 40,
        "description": "RSI level considered oversold (avoid new shorts below).",
        "category": "rsi"
    },

    # -- v3: EMA Trend Filter ---------------------------------------------
    "ema_fast": {
        "value": 8, "min": 5, "max": 13,
        "description": "Fast EMA period.",
        "category": "ema"
    },
    "ema_slow": {
        "value": 21, "min": 13, "max": 34,
        "description": "Slow EMA period.",
        "category": "ema"
    },
    "ema_trend": {
        "value": 55, "min": 34, "max": 89,
        "description": "Macro trend EMA. Only trade in direction of this EMA.",
        "category": "ema"
    },

    # -- v3: Volume Analysis -----------------------------------------------
    "rvol_threshold": {
        "value": 1.5, "min": 1.0, "max": 3.0,
        "description": "Minimum relative volume for entry consideration.",
        "category": "volume"
    },
    "rvol_strong": {
        "value": 2.0, "min": 1.5, "max": 4.0,
        "description": "Relative volume considered strong institutional signal.",
        "category": "volume"
    },
    "stocks_in_play_mult": {
        "value": 2.0, "min": 1.5, "max": 3.5,
        "description": "Volume multiple vs 20-bar avg to classify stock as in-play.",
        "category": "volume"
    },

    # -- v4: Session Report Optimizations ------------------------------------
    # Opt 2: FVG volume surge filter
    "fvg_rvol_min": {
        "value": 1.5, "min": 1.0, "max": 3.0,
        "description": "Opt 2: Minimum relative volume on FVG impulse candle. Institutional FVGs come with volume.",
        "category": "fvg"
    },

    # Opt 5: Swing failure displacement confirmation
    "swing_failure_require_displacement": {
        "value": 1,  # 1 = required, 0 = optional
        "min": 0, "max": 1,
        "description": "Opt 5: Require a large-body displacement candle after swing failure for confirmation.",
        "category": "swing"
    },

    # Opt 6: Pre-market high/low tracking
    "premarket_track_enabled": {
        "value": 1,  # 1 = enabled, 0 = disabled
        "min": 0, "max": 1,
        "description": "Opt 6: Track pre-market (4 AM-9:30 AM) high/low as directional bias signal for NY session.",
        "category": "premarket"
    },
    "premarket_sweep_quality_bonus": {
        "value": 0.08, "min": 0.00, "max": 0.15,
        "description": "Opt 6: Quality score bonus when setup aligns with pre-market sweep direction.",
        "category": "premarket"
    },
}


class AlgoConfig:
    """
    Loads and manages algorithm parameters.
    Falls back to defaults if config file missing or corrupted.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self._params: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load params from JSON, falling back to defaults for any missing keys."""
        file_params = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, encoding="utf-8") as f:
                    file_params = json.load(f)
                logger.info(f"Loaded algo params from {self.config_path}")
            except Exception as e:
                logger.warning(f"Could not read algo_params.json: {e} — using defaults")

        # Merge: file values override defaults, but keep default bounds
        self._params = {}
        for key, default in DEFAULT_PARAMS.items():
            param = default.copy()
            if key in file_params:
                raw_val = file_params[key].get("value", default["value"])
                # Enforce bounds
                clamped = max(param["min"], min(param["max"], float(raw_val)))
                if clamped != raw_val:
                    logger.warning(f"Param {key}={raw_val} clamped to bounds [{param['min']}, {param['max']}] → {clamped}")
                param["value"] = clamped
            self._params[key] = param

    def get(self, key: str, fallback=None):
        """Get a parameter value."""
        if key in self._params:
            return self._params[key]["value"]
        if fallback is not None:
            return fallback
        raise KeyError(f"Unknown parameter: {key}")

    def get_all_values(self) -> Dict[str, float]:
        """Return flat dict of param_name → value."""
        return {k: v["value"] for k, v in self._params.items()}

    def get_full_params(self) -> Dict[str, Any]:
        """Return full param details including bounds and descriptions."""
        return self._params.copy()

    def apply_updates(self, updates: Dict[str, float], source: str = "optimizer") -> Dict[str, Any]:
        """
        Apply optimizer updates. Enforces bounds. Returns dict of changes made.
        Called by the optimizer after it decides on new values.
        """
        changes = []
        for key, new_val in updates.items():
            if key not in self._params:
                logger.warning(f"Unknown param: {key} — skipping")
                continue

            param = self._params[key]
            old_val = param["value"]

            # Clamp to safety bounds
            clamped = max(param["min"], min(param["max"], float(new_val)))

            if abs(clamped - old_val) < 1e-9:
                continue  # No meaningful change

            param["value"] = clamped
            changes.append({
                "param": key,
                "old_value": old_val,
                "new_value": clamped,
                "category": param.get("category", "unknown"),
            })
            logger.info(f"Param updated [{source}]: {key} = {old_val} → {clamped}")

        if changes:
            self._save()

        return changes

    def _save(self):
        """Write current params to JSON file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, param in self._params.items():
            data[key] = {
                "value": param["value"],
                "min": param["min"],
                "max": param["max"],
                "description": param["description"],
                "category": param.get("category", ""),
            }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Algo params saved to {self.config_path}")

    def save_defaults(self):
        """Write default params to file (first-run initialization)."""
        if not self.config_path.exists():
            self._save()
            logger.info(f"Default algo params written to {self.config_path}")

    def describe(self) -> str:
        """Human-readable summary of all current params."""
        lines = [f"{'─'*60}", "  CURRENT ALGORITHM PARAMETERS", f"{'─'*60}"]
        categories = {}
        for key, param in self._params.items():
            cat = param.get("category", "other")
            categories.setdefault(cat, []).append((key, param))

        for cat, params in sorted(categories.items()):
            lines.append(f"\n  [{cat.upper()}]")
            for key, param in params:
                lines.append(f"    {key:<35} = {param['value']}")
        lines.append(f"{'─'*60}")
        return "\n".join(lines)
