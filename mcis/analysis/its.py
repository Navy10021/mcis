from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_its(
    panel: pd.DataFrame,
    metric: str,
    event_date: str = "2022-02-24",
    polynomial_degree: int = 1,
    config: dict | None = None,
) -> dict[str, Any]:
    if metric not in panel.columns:
        raise ValueError(f"Metric '{metric}' not found in panel columns")

    if "days_to_t0" not in panel.columns:
        raise ValueError("Panel must contain 'days_to_t0' column")

    t0 = pd.Timestamp(event_date)
    panel = panel.sort_index()

    index = panel.index
    if hasattr(index, "tz") and index.tz is not None:
        t0 = t0.tz_localize(index.tz)

    T = np.arange(len(panel), dtype=float)
    D = (index >= t0).astype(float)
    T_rel = panel["days_to_t0"].values.astype(float)

    y = panel[metric].values.astype(float)

    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return {
            "metric": metric,
            "status": "insufficient_data",
            "n_observations": int(valid.sum()),
        }

    import statsmodels.api as sm

    T = T[valid]
    D = D[valid]
    T_rel = T_rel[valid]
    y = y[valid]

    X_cols: list[dict[str, Any]] = [{"name": "T", "data": T}]
    for p in range(2, polynomial_degree + 1):
        X_cols.append({"name": f"T^{p}", "data": T ** p})

    X_cols.append({"name": "D", "data": D})
    X_cols.append({"name": "DxT", "data": D * T_rel})

    X_list = [col["data"] for col in X_cols]
    X = np.column_stack(X_list)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})

    X_counterfactual = X.copy()
    for i, col in enumerate(X_cols):
        if col["name"] == "D":
            X_counterfactual[:, i + 1] = 0.0
        elif col["name"] == "DxT":
            X_counterfactual[:, i + 1] = 0.0

    counterfactual = X_counterfactual @ model.params

    result = {
        "metric": metric,
        "event_date": event_date,
        "polynomial_degree": polynomial_degree,
        "n_observations": int(len(y)),
        "status": "ok",
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "params": {},
        "conf_int": {},
        "p_values": {},
        "counterfactual": {
            "dates": [str(d.date()) for d in panel.index[valid]],
            "predicted": [float(v) for v in model.fittedvalues],
            "counterfactual": [float(v) for v in counterfactual],
            "actual": [float(v) for v in y],
        },
    }

    var_names = ["const"] + [col["name"] for col in X_cols]
    ci_array = model.conf_int()
    for i, name in enumerate(var_names):
        result["params"][name] = float(model.params[i])
        result["conf_int"][name] = [float(ci_array[i, 0]), float(ci_array[i, 1])]
        result["p_values"][name] = float(model.pvalues[i])

    for i, name in enumerate(var_names):
        if name == "D":
            result["level_change"] = {
                "coef": float(model.params[i]),
                "p_value": float(model.pvalues[i]),
                "ci": [float(ci_array[i, 0]), float(ci_array[i, 1])],
            }
        elif name == "DxT":
            result["slope_change"] = {
                "coef": float(model.params[i]),
                "p_value": float(model.pvalues[i]),
                "ci": [float(ci_array[i, 0]), float(ci_array[i, 1])],
            }

    return result
