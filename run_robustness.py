"""
Robustness analysis script for paper:
1. Placebo cut-dates test (100 random fake T0s) for top metrics
2. FDR + Holm correction for ITS p-values across 15 metrics
3. Save results to outputs/tables/robustness_*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mcis.analysis.robustness import (
    apply_multiple_testing,
    run_placebo_cutdates,
)

# Paths
PANEL_PATH = Path("data/aggregated/panel_blacksea.parquet")
OUT_DIR = Path("outputs/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Top metrics (paper의 메인 5개)
TOP_METRICS = [
    "vessel_count",
    "mean_sog",
    "std_sog",
    "ais_silence_count",
    "cog_variance",
]

# 모든 ITS-significant metrics (10개)
ALL_SIG_METRICS = [
    "vessel_count",
    "unique_mmsi",
    "mean_sog",
    "std_sog",
    "ais_silence_count",
    "mean_draught",
    "cargo_fraction",
    "tanker_fraction",
    "route_entropy",
    "cog_variance",
]


def main():
    print(f"Loading panel: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  Panel shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)[:5]}...")

    # ========================================================================
    # 1. Placebo cut-dates test (top 5 metrics)
    # ========================================================================
    print("\n=== Placebo Cut-Dates Test ===\n")

    # 100 random placebo offsets, excluding ±30 days around actual T0
    rng = np.random.default_rng(seed=42)
    candidate_offsets = sorted(
        rng.choice(
            [d for d in range(-150, 151) if abs(d) > 30],
            size=100,
            replace=False,
        ).tolist()
    )
    print(f"Generated {len(candidate_offsets)} placebo offsets")

    placebo_results = {}
    for metric in TOP_METRICS:
        print(f"\n  Metric: {metric}")
        try:
            result = run_placebo_cutdates(
                panel=panel,
                metric=metric,
                candidate_offsets=candidate_offsets,
                estimation_window=(-90, -31),
                event_window=(-30, 30),
            )
            placebo_results[metric] = result
            print(f"    Placebos run: {result['n_placebos']}")
            print(f"    Min p-value (placebos): {result['min_p_value']:.4f}")
            print(f"    Median p-value (placebos): {result['median_p_value']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            placebo_results[metric] = {"error": str(e)}

    # Save placebo results
    placebo_out = OUT_DIR / "robustness_placebo.json"
    with open(placebo_out, "w") as f:
        json.dump(placebo_results, f, indent=2, default=float)
    print(f"\nSaved: {placebo_out}")

    # ========================================================================
    # 2. FDR / Holm correction for ITS p-values
    # ========================================================================
    print("\n=== FDR + Holm Multiple Testing Correction ===\n")

    # Load all ITS results and extract level_change p-values
    its_pvalues = {}
    for metric in ALL_SIG_METRICS + [
        "max_abs_rot",
        "rot_spike_count",
        "russian_flag_fraction",
        "sat_src_fraction",
        "ukrainian_flag_fraction",
    ]:
        its_file = OUT_DIR / f"its_{metric}_2022-02-24.json"
        if not its_file.exists():
            print(f"  Missing: {its_file.name}")
            continue
        d = json.load(open(its_file))
        lc = d.get("level_change", {})
        p = lc.get("p_value")
        if isinstance(p, (int, float)) and np.isfinite(p):
            its_pvalues[metric] = float(p)

    print(f"Loaded {len(its_pvalues)} ITS level_change p-values")

    fdr_result = apply_multiple_testing(its_pvalues, alpha=0.05)

    # Pretty print
    print(f"\n{'Metric':<28} {'Raw p':<14} {'FDR-BH p':<14} {'BH sig':<8} {'Holm sig':<8}")
    print("-" * 75)
    for metric, vals in fdr_result["adjusted_p_values"].items():
        raw_p = f"{vals['raw_p_value']:.3e}"
        bh_p = f"{vals['fdr_bh_p_value']:.3e}"
        bh_sig = "*" if vals["fdr_bh_reject"] else " "
        holm_sig = "*" if vals["holm_reject"] else " "
        print(
            f"{metric:<28} {raw_p:<14} {bh_p:<14} {bh_sig:<8} {holm_sig:<8}"
        )

    n_bh = sum(v["fdr_bh_reject"] for v in fdr_result["adjusted_p_values"].values())
    n_holm = sum(v["holm_reject"] for v in fdr_result["adjusted_p_values"].values())
    total = len(fdr_result["adjusted_p_values"])
    print(f"\nFDR-BH significant: {n_bh} / {total}")
    print(f"Holm significant: {n_holm} / {total}")

    # Save FDR results
    fdr_out = OUT_DIR / "robustness_fdr.json"
    with open(fdr_out, "w") as f:
        json.dump(fdr_result, f, indent=2)
    print(f"\nSaved: {fdr_out}")

    print("\n=== Robustness Analysis Complete ===")


if __name__ == "__main__":
    main()