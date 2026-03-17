#!/usr/bin/env python3
"""
run_tests.py — TradeIQ Unit Test Runner
========================================
Run from the tradeiq_rh/ directory:
    python run_tests.py

Or with verbose output:
    python run_tests.py -v
"""
import sys
import os
import unittest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

# Discover and run all tests in the tests/ folder
loader = unittest.TestLoader()
suite  = loader.discover(start_dir="tests", pattern="test_*.py")

verbosity = 2 if "-v" in sys.argv else 1

runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
result = runner.run(suite)

# Exit with non-zero code if any tests failed (useful for CI)
sys.exit(0 if result.wasSuccessful() else 1)
