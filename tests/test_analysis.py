from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mcis.analysis.did import run_did
from mcis.analysis.event_study import run_event_study
from mcis.analysis.granger import run_granger
from mcis.analysis.its import run_its

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


@pytest.fixture(scope="session")
def blacksea_panel():
    np.random.seed(42)
    dates = pd.date_range("2021-09-01", "2022-06-01", freq="D")
    n = len(dates)
    df = pd.DataFrame(index=dates)
    df["days_to_t0"] = (dates - pd.Timestamp("2022-02-24")).days
    df["post_conflict"] = (dates >= "2022-02-24").astype(int)

    df["vessel_count"] = np.random.poisson(80, n).astype(float)
    df["mean_sog"] = np.random.uniform(8, 12, n)
    df["std_sog"] = np.random.uniform(2, 5, n)
    df["max_abs_rot"] = np.random.uniform(0, 30, n)
    df["rot_spike_count"] = np.random.poisson(3, n).astype(float)
    df["ais_silence_count"] = np.random.poisson(5, n).astype(float)
    df["cargo_fraction"] = np.random.uniform(0.3, 0.6, n)
    df["tanker_fraction"] = np.random.uniform(0.1, 0.3, n)
    df["russian_flag_fraction"] = np.random.uniform(0.1, 0.3, n)
    df["route_entropy"] = np.random.uniform(2, 4, n)
    df["cog_variance"] = np.random.uniform(100, 500, n)

    return df


@pytest.fixture(scope="session")
def grid_panel():
    np.random.seed(42)
    dates = pd.date_range("2022-02-01", "2022-03-31", freq="D")
    grid_ids = [f"{i}_{j}" for i in range(3) for j in range(3)]

    rows = []
    for gid in grid_ids:
        for d in dates:
            rows.append({
                "grid_id": gid,
                "time_bucket": d,
                "vessel_count": np.random.poisson(10),
                "mean_sog": np.random.uniform(8, 12),
            })

    df = pd.DataFrame(rows)
    df["post_conflict"] = (df["time_bucket"] >= "2022-02-24").astype(int)
    df["days_to_t0"] = (df["time_bucket"] - pd.Timestamp("2022-02-24")).dt.days
    return df.set_index(["grid_id", "time_bucket"])


class TestEventStudy:
    def test_returns_expected_keys(self, blacksea_panel):
        result = run_event_study(blacksea_panel, "vessel_count")
        assert isinstance(result, dict)
        assert result["metric"] == "vessel_count"
        assert "abnormal_values" in result
        assert "cumulative_abnormal" in result
        assert "significant_dates" in result
        assert "baseline_mean" in result

    def test_baseline_mean_from_estimation_window(self, blacksea_panel):
        result = run_event_study(
            blacksea_panel, "vessel_count",
            estimation_window=(-90, -61), event_window=(-60, 60),
        )
        expected_mean = blacksea_panel.loc[
            (blacksea_panel["days_to_t0"] >= -90) &
            (blacksea_panel["days_to_t0"] <= -61),
            "vessel_count"
        ].mean()
        assert result["baseline_mean"] == pytest.approx(expected_mean)

    def test_abnormal_values_computed(self, blacksea_panel):
        result = run_event_study(blacksea_panel, "vessel_count")
        assert len(result["abnormal_values"]) > 0
        for k, v in result["abnormal_values"].items():
            assert isinstance(v, float)

    def test_cumulative_abnormal(self, blacksea_panel):
        result = run_event_study(blacksea_panel, "vessel_count")
        cum = result["cumulative_abnormal"]
        vals = list(cum.values())
        assert len(vals) > 1

    def test_missing_metric_raises(self, blacksea_panel):
        with pytest.raises(ValueError, match="not found"):
            run_event_study(blacksea_panel, "nonexistent")

    def test_missing_days_column_raises(self):
        panel = pd.DataFrame({"vessel_count": [1, 2, 3]})
        with pytest.raises(ValueError, match="days_to_t0"):
            run_event_study(panel, "vessel_count")

    def test_insufficient_estimation_data(self, blacksea_panel):
        short = blacksea_panel.iloc[:1].copy()
        short["days_to_t0"] = [0]
        result = run_event_study(short, "vessel_count")
        assert result["status"] == "insufficient_estimation_data"

    def test_status_ok_on_success(self, blacksea_panel):
        result = run_event_study(blacksea_panel, "vessel_count")
        assert result["status"] == "ok"

    def test_t_stats_type(self, blacksea_panel):
        result = run_event_study(blacksea_panel, "vessel_count")
        assert "t_stats" in result
        for v in result["t_stats"].values():
            assert isinstance(v, float)


class TestITS:
    def test_returns_expected_keys(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count")
        assert isinstance(result, dict)
        assert result["metric"] == "vessel_count"
        assert "params" in result
        assert "counterfactual" in result
        assert "r_squared" in result

    def test_has_level_and_slope_change(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count")
        assert "level_change" in result
        assert "slope_change" in result

    def test_counterfactual_length_matches_data(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count")
        cf = result["counterfactual"]
        assert len(cf["dates"]) == len(cf["predicted"])
        assert len(cf["dates"]) == len(cf["counterfactual"])

    def test_missing_metric_raises(self, blacksea_panel):
        with pytest.raises(ValueError, match="not found"):
            run_its(blacksea_panel, "nonexistent")

    def test_status_ok_on_success(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count")
        assert result["status"] == "ok"

    def test_insufficient_data(self, blacksea_panel):
        small = blacksea_panel.iloc[:3].copy()
        result = run_its(small, "vessel_count")
        assert result["status"] == "insufficient_data"

    def test_polynomial_degree_2(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count", polynomial_degree=2)
        params = result["params"]
        poly_keys = [k for k in params if k.startswith("T^")]
        assert len(poly_keys) == 1

    def test_r_squared_is_between_0_and_1(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count")
        assert 0 <= result["r_squared"] <= 1

    def test_level_change_coef_has_ci(self, blacksea_panel):
        result = run_its(blacksea_panel, "vessel_count")
        lc = result["level_change"]
        assert "coef" in lc
        assert "ci" in lc
        assert len(lc["ci"]) == 2


class TestGranger:
    def test_returns_expected_keys(self, blacksea_panel):
        result = run_granger(
            blacksea_panel, "mean_sog", "vessel_count",
            max_lag=3, alpha=0.05,
        )
        assert isinstance(result, dict)
        assert "lags" in result
        assert "significant_lags" in result
        assert "predictor_stationarity" in result

    def test_lag_structure(self, blacksea_panel):
        max_lag = 3
        result = run_granger(
            blacksea_panel, "mean_sog", "vessel_count",
            max_lag=max_lag, alpha=0.05,
        )
        for lag in range(1, max_lag + 1):
            assert str(lag) in result["lags"]

    def test_each_lag_has_required_fields(self, blacksea_panel):
        result = run_granger(
            blacksea_panel, "mean_sog", "vessel_count",
            max_lag=2, alpha=0.05,
        )
        for lag_key in result["lags"]:
            lag = result["lags"][lag_key]
            assert "ssr_f_stat" in lag
            assert "ssr_p_value" in lag
            assert "reject_h0" in lag

    def test_independent_series_not_significant(self, blacksea_panel):
        np.random.seed(99)
        panel = blacksea_panel.copy()
        panel["noise_a"] = np.random.normal(0, 1, len(panel))
        panel["noise_b"] = np.random.normal(0, 1, len(panel))
        result = run_granger(panel, "noise_a", "noise_b", max_lag=2, alpha=0.05)
        if result["status"] == "ok" and len(result["lags"]) > 0:
            for lag_key in result["lags"]:
                assert result["lags"][lag_key]["ssr_p_value"] >= 0.01

    def test_missing_predictor_raises(self, blacksea_panel):
        with pytest.raises(ValueError, match="not found"):
            run_granger(blacksea_panel, "nonexistent", "vessel_count")

    def test_missing_target_raises(self, blacksea_panel):
        with pytest.raises(ValueError, match="not found"):
            run_granger(blacksea_panel, "mean_sog", "nonexistent")

    def test_status_ok_on_success(self, blacksea_panel):
        result = run_granger(
            blacksea_panel, "mean_sog", "vessel_count",
            max_lag=2, alpha=0.05,
        )
        assert result["status"] == "ok"

    def test_stationarity_check_present(self, blacksea_panel):
        result = run_granger(
            blacksea_panel, "mean_sog", "vessel_count",
            max_lag=2, alpha=0.05,
        )
        assert "stationary" in result["predictor_stationarity"]
        assert "stationary" in result["target_stationarity"]


class TestDID:
    def test_returns_expected_keys(self, grid_panel):
        treated = ["0_0", "0_1"]
        control = ["1_0", "1_1"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert isinstance(result, dict)
        assert "delta" in result
        assert "did_coef" in result
        assert "parallel_trends" in result

    def test_delta_is_float(self, grid_panel):
        treated = ["0_0", "0_1"]
        control = ["1_0", "1_1"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert isinstance(result["delta"], float)

    def test_n_treated_pre_post_counts(self, grid_panel):
        treated = ["0_0"]
        control = ["1_0"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert result["n_treated_pre"] > 0
        assert result["n_treated_post"] > 0

    def test_missing_grid_level_raises(self):
        panel = pd.DataFrame({"vessel_count": [1, 2]})
        with pytest.raises(ValueError, match="MultiIndex"):
            run_did(panel, ["a"], ["b"])

    def test_missing_metric_in_panel_raises(self, grid_panel):
        with pytest.raises(ValueError, match="not found"):
            run_did(grid_panel, ["0_0"], ["1_0"], metric="nonexistent")

    def test_status_ok_on_success(self, grid_panel):
        treated = ["0_0"]
        control = ["1_0"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert result["status"] == "ok"

    def test_treated_and_control_group_sizes(self, grid_panel):
        treated = ["0_0", "0_1", "0_2"]
        control = ["1_0", "1_1"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert result["n_treated_grids"] == 3
        assert result["n_control_grids"] == 2

    def test_parallel_trends_has_testable_flag(self, grid_panel):
        treated = ["0_0"]
        control = ["1_0"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        parallel = result["parallel_trends"]
        assert "testable" in parallel

    def test_did_coef_is_float_or_none(self, grid_panel):
        treated = ["0_0"]
        control = ["1_0"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert result["did_coef"] is None or isinstance(result["did_coef"], float)

    def test_did_ci_length_2(self, grid_panel):
        treated = ["0_0"]
        control = ["1_0"]
        result = run_did(grid_panel, treated, control, metric="vessel_count")
        assert len(result["did_ci"]) == 2
