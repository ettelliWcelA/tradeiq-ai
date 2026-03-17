"""
tests/test_algo_config.py
=========================
Unit tests for learning/algo_config.py — AlgoConfig class.
Tests cover: default loading, JSON file loading, bounds clamping,
apply_updates, get/get_all_values, and describe().
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

from learning.algo_config import AlgoConfig, DEFAULT_PARAMS


class TestAlgoConfigDefaults(unittest.TestCase):
    def setUp(self):
        # Use a non-existent temp path so defaults load
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "algo_params.json"
        self.cfg = AlgoConfig(config_path=self.config_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_default_params_present(self):
        values = self.cfg.get_all_values()
        for key in DEFAULT_PARAMS:
            self.assertIn(key, values, f"Missing default param: {key}")

    def test_default_values_correct(self):
        for key, meta in DEFAULT_PARAMS.items():
            self.assertAlmostEqual(self.cfg.get(key), meta["value"],
                                   msg=f"Wrong default for {key}")

    def test_get_unknown_with_fallback(self):
        self.assertEqual(self.cfg.get("nonexistent_key", fallback=42), 42)

    def test_get_unknown_no_fallback_raises(self):
        with self.assertRaises(KeyError):
            self.cfg.get("totally_unknown")

    def test_get_all_values_returns_flat_dict(self):
        values = self.cfg.get_all_values()
        self.assertIsInstance(values, dict)
        for k, v in values.items():
            self.assertIsInstance(v, (int, float))

    def test_get_full_params_has_bounds(self):
        full = self.cfg.get_full_params()
        for key, meta in full.items():
            self.assertIn("min", meta)
            self.assertIn("max", meta)
            self.assertIn("value", meta)

    def test_default_values_within_bounds(self):
        for key, meta in DEFAULT_PARAMS.items():
            val = meta["value"]
            self.assertGreaterEqual(val, meta["min"],
                                    f"{key} default below min")
            self.assertLessEqual(val, meta["max"],
                                 f"{key} default above max")


class TestAlgoConfigFileLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "algo_params.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write_params(self, overrides: dict):
        data = {}
        for key, meta in DEFAULT_PARAMS.items():
            data[key] = {
                "value": overrides.get(key, meta["value"]),
                "min": meta["min"], "max": meta["max"],
                "description": meta["description"],
                "category": meta.get("category", ""),
            }
        self.config_path.write_text(json.dumps(data))

    def test_loads_custom_value_from_file(self):
        self._write_params({"min_quality_score": 0.65})
        cfg = AlgoConfig(config_path=self.config_path)
        self.assertAlmostEqual(cfg.get("min_quality_score"), 0.65)

    def test_clamps_value_above_max(self):
        # min_quality_score max = 0.75; setting 0.90 should be clamped
        self._write_params({"min_quality_score": 0.90})
        cfg = AlgoConfig(config_path=self.config_path)
        self.assertAlmostEqual(cfg.get("min_quality_score"), 0.75)

    def test_clamps_value_below_min(self):
        # min_quality_score min = 0.40; setting 0.10 should be clamped
        self._write_params({"min_quality_score": 0.10})
        cfg = AlgoConfig(config_path=self.config_path)
        self.assertAlmostEqual(cfg.get("min_quality_score"), 0.40)

    def test_missing_key_falls_back_to_default(self):
        # Write file without fvg_min_atr_mult
        data = {"min_quality_score": {"value": 0.60, "min": 0.40, "max": 0.75,
                                       "description": "", "category": "risk"}}
        self.config_path.write_text(json.dumps(data))
        cfg = AlgoConfig(config_path=self.config_path)
        # fvg_min_atr_mult not in file → should use default 0.35
        self.assertAlmostEqual(cfg.get("fvg_min_atr_mult"),
                               DEFAULT_PARAMS["fvg_min_atr_mult"]["value"])

    def test_corrupt_file_falls_back_to_defaults(self):
        self.config_path.write_text("NOT_VALID_JSON{{}")
        cfg = AlgoConfig(config_path=self.config_path)
        # Should still load all defaults without error
        self.assertAlmostEqual(cfg.get("min_quality_score"),
                               DEFAULT_PARAMS["min_quality_score"]["value"])


class TestApplyUpdates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / "algo_params.json"
        self.cfg = AlgoConfig(config_path=self.config_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_update_applied(self):
        changes = self.cfg.apply_updates({"min_quality_score": 0.65})
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(self.cfg.get("min_quality_score"), 0.65)

    def test_update_clamped_to_max(self):
        changes = self.cfg.apply_updates({"min_quality_score": 0.99})
        self.assertAlmostEqual(self.cfg.get("min_quality_score"), 0.75)

    def test_update_clamped_to_min(self):
        changes = self.cfg.apply_updates({"min_quality_score": 0.01})
        self.assertAlmostEqual(self.cfg.get("min_quality_score"), 0.40)

    def test_no_change_for_same_value(self):
        default = DEFAULT_PARAMS["min_quality_score"]["value"]
        changes = self.cfg.apply_updates({"min_quality_score": default})
        self.assertEqual(len(changes), 0)

    def test_unknown_param_skipped(self):
        changes = self.cfg.apply_updates({"totally_made_up": 42.0})
        self.assertEqual(len(changes), 0)

    def test_multiple_updates(self):
        updates = {
            "min_quality_score": 0.65,
            "default_rr_ratio": 4.0,
            "stop_atr_mult": 1.0,
        }
        changes = self.cfg.apply_updates(updates)
        self.assertEqual(len(changes), 3)
        self.assertAlmostEqual(self.cfg.get("default_rr_ratio"), 4.0)

    def test_file_written_after_update(self):
        self.cfg.apply_updates({"min_quality_score": 0.68})
        self.assertTrue(self.config_path.exists())
        saved = json.loads(self.config_path.read_text())
        self.assertIn("min_quality_score", saved)
        self.assertAlmostEqual(saved["min_quality_score"]["value"], 0.68)

    def test_changes_dict_structure(self):
        changes = self.cfg.apply_updates({"min_quality_score": 0.65})
        change = changes[0]
        self.assertIn("param", change)
        self.assertIn("old_value", change)
        self.assertIn("new_value", change)
        self.assertAlmostEqual(change["new_value"], 0.65)


class TestDescribe(unittest.TestCase):
    def test_describe_returns_string(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = AlgoConfig(config_path=Path(tmp.name) / "p.json")
        desc = cfg.describe()
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 50)
        tmp.cleanup()

    def test_describe_contains_param_names(self):
        tmp = tempfile.TemporaryDirectory()
        cfg = AlgoConfig(config_path=Path(tmp.name) / "p.json")
        desc = cfg.describe()
        self.assertIn("min_quality_score", desc)
        self.assertIn("stop_atr_mult", desc)
        tmp.cleanup()


class TestSaveDefaults(unittest.TestCase):
    def test_save_defaults_creates_file(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "subdir" / "algo_params.json"
        cfg = AlgoConfig(config_path=path)
        cfg.save_defaults()
        self.assertTrue(path.exists())
        tmp.cleanup()

    def test_save_defaults_not_overwrite_existing(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "algo_params.json"
        # Write a custom value first
        data = {"min_quality_score": {"value": 0.70, "min": 0.40, "max": 0.75,
                                       "description": "", "category": "risk"}}
        path.write_text(json.dumps(data))
        cfg = AlgoConfig(config_path=path)
        cfg.save_defaults()  # Should NOT overwrite existing file
        saved = json.loads(path.read_text())
        # The file should still have the custom value (not reset)
        self.assertAlmostEqual(saved["min_quality_score"]["value"], 0.70)
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
