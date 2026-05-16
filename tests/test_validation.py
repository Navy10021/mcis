from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from mcis.validation import (
    assert_no_leakage,
    validate_claim_level,
    validate_data_validity_mode,
    validate_required_columns,
    validate_temporal_split,
    write_run_metadata,
    compute_file_hash,
)


class TestValidateDataValidityMode:
    def test_valid_modes(self):
        for mode in ["empirical", "synthetic", "mixed"]:
            config = {"project": {"data_validity_mode": mode}}
            validate_data_validity_mode(config)

    def test_missing_mode_raises(self):
        with pytest.raises(ValueError, match="required"):
            validate_data_validity_mode({})

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_data_validity_mode({"project": {"data_validity_mode": "invalid"}})


class TestValidateClaimLevel:
    def test_valid_levels_with_empirical(self):
        for level in ["engineering_demo", "descriptive_evidence", "inferential_evidence", "predictive_prototype"]:
            config = {"project": {"data_validity_mode": "empirical", "claim_level": level}}
            validate_claim_level(config)

    def test_inferential_with_synthetic_raises(self):
        config = {"project": {"data_validity_mode": "synthetic", "claim_level": "inferential_evidence"}}
        with pytest.raises(ValueError, match="Inferential claims"):
            validate_claim_level(config)

    def test_predictive_with_mixed_raises(self):
        config = {"project": {"data_validity_mode": "mixed", "claim_level": "predictive_prototype"}}
        with pytest.raises(ValueError, match="Inferential claims"):
            validate_claim_level(config)

    def test_descriptive_with_synthetic_ok(self):
        config = {"project": {"data_validity_mode": "synthetic", "claim_level": "descriptive_evidence"}}
        validate_claim_level(config)

    def test_missing_level_raises(self):
        with pytest.raises(ValueError, match="required"):
            validate_claim_level({"project": {}})


class TestAssertNoLeakage:
    def test_no_leakage_passes(self):
        config = {
            "validation": {
                "forbidden_features": ["days_to_t0", "post_conflict", "date"],
            }
        }
        assert_no_leakage(["mean_sog", "vessel_count"], config)

    def test_leakage_detected(self):
        config = {
            "validation": {
                "forbidden_features": ["days_to_t0", "post_conflict", "date"],
            }
        }
        with pytest.raises(ValueError, match="Leakage features"):
            assert_no_leakage(["mean_sog", "days_to_t0"], config)

    def test_multiple_leakage_detected(self):
        config = {
            "validation": {
                "forbidden_features": ["days_to_t0", "post_conflict", "date", "time_bucket"],
            }
        }
        with pytest.raises(ValueError, match="Leakage"):
            assert_no_leakage(["days_to_t0", "post_conflict", "mean_sog"], config)

    def test_empty_feature_cols_passes(self):
        config = {"validation": {"forbidden_features": ["days_to_t0"]}}
        assert_no_leakage([], config)

    def test_empty_forbidden_passes(self):
        config = {"validation": {"forbidden_features": []}}
        assert_no_leakage(["days_to_t0", "mean_sog"], config)


class TestValidateRequiredColumns:
    def test_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_required_columns(df, ["a", "b"])

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Required columns"):
            validate_required_columns(df, ["a", "b", "c"])

    def test_empty_required(self):
        df = pd.DataFrame({"a": [1]})
        validate_required_columns(df, [])


class TestValidateTemporalSplit:
    def test_basic_split_report(self, config):
        dates = pd.date_range("2021-08-24", "2022-08-24", freq="D")
        panel = pd.DataFrame({"date": dates, "value": range(len(dates))})
        result = validate_temporal_split(panel, config)
        assert "splits" in result
        assert "train" in result["splits"]
        assert "calibration" in result["splits"]
        assert "event_eval" in result["splits"]
        assert "post_event" in result["splits"]
        assert result["total_dates"] == 366

    def test_missing_date_column_raises(self, config):
        panel = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="date"):
            validate_temporal_split(panel, config)


class TestComputeFileHash:

    def test_compute_file_hash(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("abc", encoding="utf-8")
        h = compute_file_hash(p)
        assert len(h) == 64


class TestWriteRunMetadata:
    def test_writes_metadata_file(self, tmp_path, config):
        path = write_run_metadata(config, tmp_path, "test_stage")
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["stage"] == "test_stage"
        assert data["data_validity_mode"] == "mixed"
        assert data["claim_level"] == "descriptive_evidence"
        assert "conflict_t0" in data
        assert "timestamp" in data
        assert "git_commit_hash" in data
        assert "config_snapshot_hash" in data
        assert "environment" in data

    def test_extra_fields_included(self, tmp_path, config):
        import json
        extra = {"rows_loaded": 5000, "source_file": "test.csv"}
        path = write_run_metadata(config, tmp_path, "load", extra=extra)
        with open(path) as f:
            data = json.load(f)
        assert data["rows_loaded"] == 5000
        assert data["source_file"] == "test.csv"

    def test_creates_directory(self, tmp_path, config):
        nested = tmp_path / "a" / "b" / "c"
        path = write_run_metadata(config, nested, "test")
        assert path.exists()
