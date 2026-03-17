# TradeIQ AI

A self-learning algorithmic day trading system built in Python 3.9, 
trading AI-sector stocks through the Robinhood API.

## What It Does

TradeIQ runs a full intraday trading session autonomously — scanning 20 
AI-sector stocks every 5 minutes, detecting high-probability setups, 
managing risk, and optimizing its own parameters overnight using trade 
history stored in SQLite.

## Architecture
```
tradeiq_rh/
├── core/engine.py              # Main orchestrator + session scheduler
├── strategies/unified_strategy.py  # 6-strategy signal engine
├── data/
│   ├── robinhood_data.py       # Robinhood API wrapper
│   └── macro_calendar.py       # AI-powered macro event guard
├── risk/risk_manager.py        # Position sizing + kill switch + PDT
├── execution/executor.py       # Order placement
├── learning/
│   ├── performance_db.py       # SQLite trade memory
│   ├── algo_config.py          # Dynamic parameter system
│   └── optimizer.py            # Nightly self-optimization engine
├── reporting/reporter.py       # Post-market reports + email
└── tests/                      # 230-test unit suite
```

## Key Features

**6-Strategy Signal Engine**
- ORB + FVG + Engulfing (prime window, 9:35–10:30 AM ET)
- VWAP Pullback (trending regimes)
- Box Theory with acceptance filter (prev day H/L levels)
- Swing Failure Pattern with displacement confirmation
- RSI Divergence Reversal
- EMA Momentum Breakout (8/21 cross above 55 EMA)

**Quality Scoring System**
Every setup is scored 0–100% across 10+ factors: VWAP alignment, 
RSI, EMA trend, relative volume, market regime, pre-market sweep 
direction, and RSI divergence. Minimum 55% required to trade.

**Self-Learning Optimizer**
After each session, a nightly optimizer reads the SQLite trade 
history and tunes 35 algorithm parameters — FVG thresholds, stop 
distances, quality minimums, timing windows — using either 
rule-based logic or the Claude API.

**Risk Management**
- 1% account risk per trade (scaled by quality score)
- 3% daily kill switch
- Max 3 concurrent positions
- PDT compliance for accounts under $25k
- Force-close all positions at 3:45 PM ET

**Macro Calendar Guard**
Uses Claude's web search to check for FOMC, CPI, PPI, NFP, GDP, 
and earnings conflicts each morning. Automatically skips the session 
and exits cleanly on dangerous macro days.

**230-Test Unit Suite**
Full coverage across indicators, risk management, strategy logic, 
database operations, parameter config, and macro calendar.

## Tech Stack

- **Python 3.9** — core language
- **robin-stocks** — Robinhood API
- **pandas / numpy** — data processing and indicator math
- **SQLite** — persistent trade memory
- **Anthropic Claude API** — nightly optimizer + macro calendar
- **pytz / schedule** — market session timing

## Setup
```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/tradeiq-ai.git
cd tradeiq-ai
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Fill in your Robinhood credentials and Anthropic API key

# Run tests
python run_tests.py

# Start paper trading
python run.py
```

## Configuration

All 35 algorithm parameters are defined in `learning/algo_config.py` 
with hard min/max bounds. The optimizer cannot set values outside 
these bounds — a safety layer that prevents runaway optimization.

Key parameters:
| Parameter | Default | Description |
|---|---|---|
| `min_quality_score` | 0.55 | Minimum setup quality to trade |
| `fvg_min_atr_mult` | 0.45 | Minimum FVG size (× ATR) |
| `stop_atr_mult` | 0.75 | Stop loss distance (× ATR) |
| `default_rr_ratio` | 3.0 | Default reward:risk ratio |
| `rvol_threshold` | 1.5 | Minimum relative volume |
| `box_acceptance_candles` | 2 | Candle closes required for Box Theory |

## Paper Trading Results (ongoing)

| Date | P&L | Trades | Win Rate | Top Strategy |
|---|---|---|---|---|
| 2026-03-09 | $0.00 | 1 | — | BOX_THEORY |
| 2026-03-10 | +$41.20 | 4 | 75% | BOX_THEORY (3W/0L) |

*Paper trading only. Not financial advice.*

## License

MIT
```

---

## Step 5: Create a .env.example File

This shows employers the configuration structure without exposing real credentials:
```
ROBINHOOD_USERNAME=your_email@example.com
ROBINHOOD_PASSWORD=your_password
ROBINHOOD_TOTP_SECRET=
PAPER_MODE=true
ACCOUNT_SIZE=10000
ANTHROPIC_API_KEY=
SENDGRID_API_KEY=
EMAIL_FROM=
EMAIL_TO=
MAX_DAILY_LOSS_PCT=3.0
MAX_CONCURRENT_POSITIONS=3
