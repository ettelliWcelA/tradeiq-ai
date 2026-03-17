#!/usr/bin/env python3
"""
TradeIQ AI — Run Script
========================
Usage:
  python run.py              # Paper trading (default, safe)
  python run.py --live       # Live trading (real money — be careful!)
  python run.py --symbols NVDA MSFT GOOGL   # Custom watchlist
  python run.py --symbols NVDA --paper      # Single stock paper test
"""
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="TradeIQ AI - Robinhood Trading Engine")
    parser.add_argument("--live", action="store_true",
        help="Enable LIVE trading (default is paper/simulated)")
    parser.add_argument("--symbols", nargs="+", default=None,
        help="Custom watchlist (e.g. NVDA MSFT GOOGL)")
    parser.add_argument("--paper", action="store_true", default=True,
        help="Paper trading mode (default)")
    args = parser.parse_args()

    paper_mode = not args.live
    if args.live:
        env_paper = os.getenv("PAPER_MODE", "true").lower()
        if env_paper == "true":
            print("ERROR: .env has PAPER_MODE=true but --live flag passed.")
            print("To go live: set PAPER_MODE=false in .env AND use --live flag.")
            sys.exit(1)
        confirm = input(
            "\n⚠️  WARNING: LIVE TRADING MODE\n"
            "Real orders will be placed on your Robinhood account.\n"
            "Real money is at risk.\n"
            "Type 'I ACCEPT THE RISK' to continue: "
        )
        if confirm != "I ACCEPT THE RISK":
            print("Cancelled. Use --paper for safe paper trading.")
            sys.exit(0)
        paper_mode = False

    from core.engine import TradeIQEngine
    from data.robinhood_data import AI_STOCKS

    watchlist = args.symbols or AI_STOCKS

    print(f"\n{'='*50}")
    print(f"  TRADEIQ AI  |  {'PAPER TRADING' if paper_mode else '⚠️  LIVE TRADING'}")
    print(f"  Watchlist: {len(watchlist)} symbols")
    print(f"  Strategies: ORB+FVG | Box Theory | Swing Failure")
    print(f"{'='*50}\n")

    engine = TradeIQEngine(watchlist=watchlist, paper_mode=paper_mode)
    engine.start()

if __name__ == "__main__":
    main()
