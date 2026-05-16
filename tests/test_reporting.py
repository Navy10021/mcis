from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.reporting import tables as tbl
from mcis.reporting import report_guardrails as rg


class TestDatasetStatisticsTable:
    def test_basic_loader_report(self):
        report = {
            "total_rows": 49000,
            "n_unique_mmsi": 1500,
            "n_unique_vessels": 1400,
            "n_vessel_types": 45,
            "n_flags": 15,
            "date_min": "2021-08-24",
            "date_max": "2022-08-24",
            "pos_src_distribution": {"TER": 30000, "SAT": 18000, "ROAM": 1000},
            "flag_distribution": {"TR": 5000, "RU": 3000, "UA": 2000},
        }
        result = tbl.dataset_statistics_table(report)
        assert isinstance(result, pd.DataFrame)
        assert "Statistic" in result.columns
        assert "Value" in result.columns
        assert result.loc[result["Statistic"] == "Total Rows", "Value"].iloc[0] == "49000"

    def test_with_cleaner_report(self):
        loader_rpt = {"total_rows": 50000, "date_min": "2021-01-01", "date_max": "2021-12-31"}
        cleaner_rpt = {"rows_before": 50000, "rows_after": 45000, "rows_dropped": 5000}
        result = tbl.dataset_statistics_table(loader_rpt, cleaner_rpt)
        assert "Cleaning: Rows Before" in result["Statistic"].values
        row = result.loc[result["Statistic"] == "Cleaning: Rows After", "Value"].iloc[0]
        assert row == "45000"

    def test_missing_fields(self):
        result = tbl.dataset_statistics_table({})
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestFeatureDescriptiveTable:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=60, freq="D")
        n = len(dates)
        return pd.DataFrame({
            "date": dates,
            "vessel_count": np.random.poisson(50, n),
            "mean_sog": np.random.uniform(5, 15, n),
            "rot_spike_count": np.random.poisson(2, n),
            "ais_silence_count": np.random.poisson(3, n),
            "cargo_fraction": np.random.beta(5, 5, n),
        })

    def test_basic_returns_dataframe(self, sample_df, sample_config):
        features = ["vessel_count", "mean_sog", "rot_spike_count", "ais_silence_count", "cargo_fraction"]
        result = tbl.feature_descriptive_table(sample_df, sample_config, feature_cols=features)
        assert isinstance(result, pd.DataFrame)
        assert "Feature" in result.columns
        assert "Pre_Mean" in result.columns
        assert "Post_Mean" in result.columns
        assert result["Feature"].tolist() == features

    def test_pre_post_split_correct(self, sample_df, sample_config):
        t0 = pd.Timestamp(sample_config["conflict"]["t0"])
        n_pre = (sample_df["date"] < t0).sum()
        n_post = (sample_df["date"] >= t0).sum()
        features = ["vessel_count"]
        result = tbl.feature_descriptive_table(sample_df, sample_config, feature_cols=features)
        assert result["N_Pre"].iloc[0] == n_pre
        assert result["N_Post"].iloc[0] == n_post

    def test_empty_feature_cols_falls_back(self, sample_df, sample_config):
        result = tbl.feature_descriptive_table(sample_df, sample_config, feature_cols=[])
        assert not result.empty


class TestEventStudyResultsTable:
    @pytest.fixture
    def es_result(self):
        return {
            "abnormal_values": pd.Series(
                {0: 1.5, 1: -0.5, 2: 2.0, 3: -1.0, 4: 0.3},
                name="abnormal",
            ),
            "t_stats": pd.Series({0: 2.5, 1: -0.8, 2: 3.1, 3: -1.5, 4: 0.5}),
            "p_values": pd.Series({0: 0.015, 1: 0.42, 2: 0.003, 3: 0.14, 4: 0.62}),
            "significant_dates": ["0", "2"],
        }

    def test_returns_dataframe(self, es_result, sample_config):
        result = tbl.event_study_results_table(
            es_result, "vessel_count", sample_config,
            windows=[(0, 2)],
        )
        assert isinstance(result, pd.DataFrame)
        assert "Event Day" in result.columns
        assert len(result) == 3

    def test_significant_dates_marked(self, es_result, sample_config):
        result = tbl.event_study_results_table(
            es_result, "vessel_count", sample_config,
            windows=[(0, 4)],
        )
        sig_rows = result.loc[result["Significant (p<0.05)"] == "Yes"]
        assert 0 in sig_rows["Event Day"].values
        assert 2 in sig_rows["Event Day"].values

    def test_empty_windows(self, es_result, sample_config):
        result = tbl.event_study_results_table(
            es_result, "vessel_count", sample_config,
            windows=[],
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestITSResultsTable:
    @pytest.fixture
    def its_result(self):
        return {
            "params": {
                "Intercept": 100.0,
                "time_idx": -0.5,
                "post_conflict": -15.0,
                "time_since_t0": 1.2,
            },
            "std_errors": {
                "Intercept": 5.0,
                "time_idx": 0.1,
                "post_conflict": 7.0,
                "time_since_t0": 0.3,
            },
            "pvalues": {
                "Intercept": 0.001,
                "time_idx": 0.002,
                "post_conflict": 0.04,
                "time_since_t0": 0.01,
            },
            "rsquared": 0.85,
            "rsquared_adj": 0.84,
        }

    def test_returns_dataframe(self, its_result):
        result = tbl.its_results_table(its_result, "vessel_count")
        assert isinstance(result, pd.DataFrame)
        assert "Coefficient" in result.columns
        assert "Estimate" in result.columns
        assert len(result) >= 4

    def test_r_squared_included(self, its_result):
        result = tbl.its_results_table(its_result, "vessel_count")
        r2_row = result.loc[result["Coefficient"] == "R²"]
        assert len(r2_row) == 1
        assert float(r2_row["Estimate"].iloc[0]) == 0.85

    def test_ci_calculated(self, its_result):
        result = tbl.its_results_table(its_result, "vessel_count")
        intercept = result.loc[result["Coefficient"] == "Intercept"]
        ci_lower = float(intercept["CI Lower (95%)"].iloc[0])
        ci_upper = float(intercept["CI Upper (95%)"].iloc[0])
        assert ci_lower < ci_upper
        assert np.isclose(ci_lower, 100 - 1.96 * 5)
        assert np.isclose(ci_upper, 100 + 1.96 * 5)

    def test_empty_std_errors(self):
        result = tbl.its_results_table(
            {"params": {"x": 10}, "std_errors": {}, "pvalues": {}},
            "metric",
        )
        assert len(result) == 1
        assert np.isnan(result["Std. Error"].iloc[0])


class TestGrangerResultsTable:
    @pytest.fixture
    def granger_result(self):
        return {
            "results": {
                "1": {
                    "ssr_ftest": (5.52, 0.021),
                },
                "2": {
                    "ssr_ftest": (3.14, 0.048),
                },
                "3": {
                    "ssr_ftest": (1.85, 0.142),
                },
            },
            "best_lag": 1,
            "n_obs": 300,
        }

    def test_returns_dataframe(self, granger_result):
        result = tbl.granger_results_table(granger_result, "mean_sog", "conflict_onset")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "Lag" in result.columns
        assert "Reject H0 (p<0.05)" in result.columns

    def test_best_lag_marked(self, granger_result):
        result = tbl.granger_results_table(granger_result, "mean_sog", "conflict_onset")
        best_row = result.loc[result["Best Lag"] == "★"]
        assert len(best_row) == 1
        assert best_row["Lag"].iloc[0] == 1

    def test_n_obs_in_attrs(self, granger_result):
        result = tbl.granger_results_table(granger_result, "mean_sog", "conflict_onset")
        assert result.attrs.get("n_obs") == 300

    def test_reject_flags(self, granger_result):
        result = tbl.granger_results_table(granger_result, "mean_sog", "conflict_onset")
        assert result.loc[result["Lag"] == 1, "Reject H0 (p<0.05)"].iloc[0] == "Yes"
        assert result.loc[result["Lag"] == 2, "Reject H0 (p<0.05)"].iloc[0] == "Yes"
        assert result.loc[result["Lag"] == 3, "Reject H0 (p<0.05)"].iloc[0] == "No"

    def test_no_best_lag(self):
        result_no_best = {
            "results": {"1": {"ssr_ftest": (0.5, 0.5)}},
            "best_lag": None,
            "n_obs": 100,
        }
        result = tbl.granger_results_table(result_no_best, "mean_sog", "conflict_onset")
        assert (result["Best Lag"] == "").all()


class TestSaveTable:
    def test_save_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        saved = tbl.save_table(df, str(path))
        assert Path(saved).exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    def test_save_latex(self, tmp_path):
        df = pd.DataFrame({"a": [1], "b": [2]})
        path = tmp_path / "test.tex"
        saved = tbl.save_table(df, str(path), caption="Test Table")
        assert Path(saved).exists()
        content = Path(saved).read_text()
        assert "Test Table" in content

    def test_save_markdown(self, tmp_path):
        df = pd.DataFrame({"a": [1], "b": [2]})
        path = tmp_path / "test.md"
        saved = tbl.save_table(df, str(path), caption="Test Table")
        assert Path(saved).exists()
        content = Path(saved).read_text()
        assert "Test Table" in content

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "sub" / "nested" / "test.csv"
        saved = tbl.save_table(df, str(path))
        assert Path(saved).exists()


class TestReportGuardrails:
    def test_assert_allowed_clean_text(self, sample_config):
        violations = rg.assert_allowed_claim_language(
            "The pipeline ingested 50K AIS records and computed 20 features.",
            sample_config,
        )
        assert len(violations) == 0

    def test_detect_forbidden_inference(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "synthetic"
        config["project"]["claim_level"] = "engineering_demo"
        violations = rg.assert_allowed_claim_language(
            "The model predicts conflict 7 days in advance with high accuracy.",
            config,
        )
        assert len(violations) >= 1
        assert any("predicts conflict" in v["matched"].lower() for v in violations)

    def test_allow_inference_in_empirical(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "empirical"
        config["project"]["claim_level"] = "inferential_evidence"
        violations = rg.assert_allowed_claim_language(
            "The model predicts conflict 7 days in advance with high accuracy.",
            config,
        )
        assert len(violations) == 0

    def test_detect_causal_language(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "synthetic"
        violations = rg.assert_allowed_claim_language(
            "This proves that AIS silence causes conflict onset.",
            config,
        )
        assert len(violations) >= 1
        matched_patterns = [v["matched"] for v in violations]
        assert any("proves" in m.lower() for m in matched_patterns)

    def test_strip_code_blocks(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "synthetic"
        text = "```\npredicts conflict\n```\nnormal text"
        violations = rg.assert_allowed_claim_language(text, config)
        code_violations = [v for v in violations if "predicts" in v["matched"]]
        assert len(code_violations) == 0

    def test_build_metadata_block(self, sample_config):
        block = rg.build_metadata_block(sample_config)
        assert "data_validity_mode:" in block
        assert "claim_level:" in block
        assert "conflict_t0:" in block
        assert "2022-02-24" in block
        assert "claim_description:" in block

    def test_metadata_block_with_extra(self, sample_config):
        extra = {"n_vessels": 1500, "pipeline_version": "1.0"}
        block = rg.build_metadata_block(sample_config, extra=extra)
        assert "n_vessels: 1500" in block
        assert "pipeline_version: 1.0" in block

    def test_validate_report_content_valid(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "empirical"
        config["project"]["claim_level"] = "descriptive_evidence"
        text = "data_validity_mode: empirical\nclaim_level: descriptive_evidence\nVessel count decreased after T0."
        result = rg.validate_report_content(text, config)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_report_content_invalid_claim(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "synthetic"
        config["project"]["claim_level"] = "inferential_evidence"
        text = "data_validity_mode: synthetic\nclaim_level: inferential_evidence"
        result = rg.validate_report_content(text, config)
        assert result["is_valid"] is False
        assert any("claim_level" in e for e in result["errors"])

    def test_validate_report_missing_metadata(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "empirical"
        config["project"]["claim_level"] = "engineering_demo"
        result = rg.validate_report_content("No metadata here", config)
        assert result["is_valid"] is True
        assert len(result["warnings"]) >= 1
        assert any("metadata" in w.lower() for w in result["warnings"])

    def test_validate_report_content_forbidden_phrases(self, sample_config):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = "synthetic"
        config["project"]["claim_level"] = "engineering_demo"
        text = "data_validity_mode: synthetic\nclaim_level: engineering_demo\nThe model predicts conflict reliably."
        result = rg.validate_report_content(text, config)
        assert len(result["violations"]) >= 1
        assert len(result["errors"]) == 0

    def test_build_metadata_block_all_claim_levels(self, sample_config):
        config = copy.deepcopy(sample_config)
        for level in ["engineering_demo", "descriptive_evidence", "inferential_evidence", "predictive_prototype"]:
            config["project"]["claim_level"] = level
            block = rg.build_metadata_block(config)
            assert f"claim_level: {level}" in block
            assert "claim_description:" in block

    @pytest.mark.parametrize("mode,level,is_valid", [
        ("synthetic", "engineering_demo", True),
        ("synthetic", "descriptive_evidence", True),
        ("synthetic", "inferential_evidence", False),
        ("synthetic", "predictive_prototype", False),
        ("empirical", "inferential_evidence", True),
        ("empirical", "predictive_prototype", True),
        ("mixed", "engineering_demo", True),
        ("mixed", "descriptive_evidence", True),
        ("mixed", "inferential_evidence", False),
    ])
    def test_mode_level_combinations(self, sample_config, mode, level, is_valid):
        config = copy.deepcopy(sample_config)
        config["project"]["data_validity_mode"] = mode
        config["project"]["claim_level"] = level
        text = f"data_validity_mode: {mode}\nclaim_level: {level}"
        result = rg.validate_report_content(text, config)
        assert result["is_valid"] == is_valid, f"mode={mode}, level={level}"
