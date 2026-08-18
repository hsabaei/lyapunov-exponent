#!/usr/bin/env python3
"""
Final reporting/post-processing for Framework Experiment 4 (non-normal systems).

This script DOES NOT simulate trajectories. It reads the tables already produced by
run_framework_experiment4_nonnormal_subspace.py and creates a compact, report-ready
package.

Main report outputs
-------------------
Tables
  table1_experiment_design_report.csv/.md
  table2_primary_recovery_performance_report.csv/.md
  table3_primary_recovery_intervals_report.csv/.md
  table4_nonnormal_geometry_report.csv/.md

Figures
  figure1_ever_recovery_heatmap.png
  figure2a_recovery_intervals__strong_gap.png
  figure2b_recovery_intervals__moderate_gap.png
  figure2c_recovery_intervals__weak_gap.png
  figure3_first_recovery_iteration_heatmap.png
  figure4_subspace_vs_individual_eigenvector_diagnostic.png

Supplementary outputs
---------------------
  supplementary_individual_eigenvector_diagnostics.csv/.md
  supplementary_latest_primary_error_heatmap.png
  supplementary_distance_to_limit_first_recovery_heatmap.png

Important Experiment-4 convention
---------------------------------
Stage 1 target: q1, evaluated by sign-invariant angular error.
For k >= 2, the primary target is U_k = span(q1,...,qk), evaluated by the largest
principal angle between the estimated accumulated subspace and U_k.
Individual q2,...,q5 errors are diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SPECTRUM_ORDER = ["strong_gap", "moderate_gap", "weak_gap"]
GEOMETRY_ORDER = ["mild_nonnormal", "moderate_nonnormal", "strong_nonnormal"]
TARGET_ORDER = ["q1", "U2", "U3", "U4", "U5"]
TARGET_STAGE = {"q1": 1, "U2": 2, "U3": 3, "U4": 4, "U5": 5}
DISPLAY_GEOMETRY = {
    "mild_nonnormal": "mild non-normal",
    "moderate_nonnormal": "moderate non-normal",
    "strong_nonnormal": "strong non-normal",
}
DISPLAY_SPECTRUM = {
    "strong_gap": "strong gap",
    "moderate_gap": "moderate gap",
    "weak_gap": "weak gap",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create final report tables/figures for Framework Experiment 4")
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/framework_experiment4_nonnormal_subspace"),
        help="Directory containing Experiment-4 table*.csv files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <results-dir>/framework4_final_report",
    )
    return p.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required Experiment-4 result file not found: {path}")
    return path


def dataframe_to_markdown_no_tabulate(df: pd.DataFrame) -> str:
    """Write a simple Markdown table without pandas.to_markdown()/tabulate."""
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            if np.isclose(v, round(v)):
                return str(int(round(v)))
            return f"{float(v):.6g}"
        text = str(v)
        return text.replace("|", "\\|").replace("\n", " ")

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, outdir: Path, stem: str) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(dataframe_to_markdown_no_tabulate(df), encoding="utf-8")


def sort_cases(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_spectrum_order"] = x["spectrum_case"].map({v: i for i, v in enumerate(SPECTRUM_ORDER)})
    x["_geometry_order"] = x["geometry_case"].map({v: i for i, v in enumerate(GEOMETRY_ORDER)})
    if "stage" in x.columns:
        x["_stage_order"] = x["stage"]
        by = ["_spectrum_order", "_geometry_order", "_stage_order"]
    else:
        by = ["_spectrum_order", "_geometry_order"]
    x = x.sort_values(by).drop(columns=[c for c in ["_spectrum_order", "_geometry_order", "_stage_order"] if c in x])
    return x.reset_index(drop=True)


def load_source_tables(results_dir: Path):
    t1 = pd.read_csv(require_file(results_dir / "table1_experiment_design.csv"))
    t2 = pd.read_csv(require_file(results_dir / "table2_primary_stagewise_performance.csv"))
    t3 = pd.read_csv(require_file(results_dir / "table3_primary_recovery_intervals.csv"))
    t4 = pd.read_csv(require_file(results_dir / "table4_individual_eigenvector_diagnostics.csv"))
    t5 = pd.read_csv(require_file(results_dir / "table5_nonnormal_geometry_summary.csv"))
    return t1, t2, t3, t4, t5


def make_report_tables(t1, t2, t3, t4, t5):
    # Table 1: compact design table.
    table1 = t1[[
        "case_name", "spectrum_case", "leading_eigenvalues", "geometry_case", "shear_strength",
        "n_systems", "initial_states_per_system", "trajectories_per_case", "window_m", "steps",
        "stability_threshold_deg", "stability_patience", "pc1_energy_threshold",
        "external_correctness_tolerance_deg",
    ]].copy()
    table1 = sort_cases(table1)

    # Table 2: primary performance only; higher individual eigenvector errors are not correctness metrics.
    table2 = t2[[
        "spectrum_case", "geometry_case", "primary_target", "n_trajectories",
        "acceptance_rate", "ever_recovery_rate", "overall_recovery_latest_rate",
        "reliability_latest_given_accepted_rate", "false_acceptance_latest_rate",
        "median_latest_primary_error_deg_accepted",
    ]].copy()
    table2 = table2.rename(columns={
        "median_latest_primary_error_deg_accepted": "median_latest_primary_error_deg",
    })
    # recover stage to sort even after compacting
    table2["stage"] = table2["primary_target"].map(TARGET_STAGE)
    table2 = sort_cases(table2).drop(columns="stage")

    # Table 3: first/last recovery and closeness to the limit.
    table3 = t3[[
        "spectrum_case", "geometry_case", "primary_target", "n_trajectories", "n_ever_recovered",
        "ever_recovery_rate", "median_first_recovery_window_end", "median_last_recovery_window_end",
        "median_recovery_span_inclusive", "median_n_accepted_correct_windows",
        "median_recovery_continuity_fraction", "median_log10_relative_distance_at_first_recovery",
        "median_first_recovery_primary_error_deg",
    ]].copy()
    def interval_string(row):
        a, b = row["median_first_recovery_window_end"], row["median_last_recovery_window_end"]
        if pd.isna(a) or pd.isna(b):
            return "N/A"
        return f"[{int(round(a))}, {int(round(b))}]"
    table3.insert(
        table3.columns.get_loc("median_last_recovery_window_end") + 1,
        "median_recovery_interval",
        table3.apply(interval_string, axis=1),
    )
    table3["stage"] = table3["primary_target"].map(TARGET_STAGE)
    table3 = sort_cases(table3).drop(columns="stage")

    # Table 4: compact geometry evidence that non-normality/nonorthogonality actually increases.
    table4 = t5[[
        "geometry_case", "shear_strength", "median_eigenvector_condition_number",
        "median_leading_gram_nonorthogonality_fro", "median_relative_nonnormality",
        "median_min_pairwise_target_eigenvector_angle_deg",
        "median_median_pairwise_target_eigenvector_angle_deg",
    ]].copy()
    table4["_geometry_order"] = table4["geometry_case"].map({v: i for i, v in enumerate(GEOMETRY_ORDER)})
    table4 = table4.sort_values("_geometry_order").drop(columns="_geometry_order").reset_index(drop=True)

    supplementary_diag = sort_cases(t4.copy())
    return table1, table2, table3, table4, supplementary_diag


def heatmap_array(table: pd.DataFrame, value_col: str):
    case_order = [f"{s}__{g}" for s in SPECTRUM_ORDER for g in GEOMETRY_ORDER]
    arr = np.full((len(case_order), len(TARGET_ORDER)), np.nan, dtype=float)
    for i, case in enumerate(case_order):
        for j, target in enumerate(TARGET_ORDER):
            m = table[(table["case_name"] == case) & (table["primary_target"] == target)]
            if len(m) and pd.notna(m[value_col].iloc[0]):
                arr[i, j] = float(m[value_col].iloc[0])
    return case_order, arr


def draw_heatmap(table, value_col, output, title, cbar_label, fmt=".2f", vmin=None, vmax=None):
    case_order, arr = heatmap_array(table, value_col)
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(TARGET_ORDER)), labels=TARGET_ORDER)
    ax.set_yticks(range(len(case_order)), labels=case_order)
    ax.set_xlabel("Primary target")
    ax.set_ylabel("Spectrum / non-normality case")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    finite = arr[np.isfinite(arr)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                text_color = "white" if v < midpoint else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9, color=text_color)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def draw_recovery_intervals(table3: pd.DataFrame, spectrum: str, output: Path) -> None:
    """One recovery-range figure for a single spectral case.

    q1/U2/U3/U4/U5 are shown explicitly; no use of the ambiguous word 'stage'.
    """
    g = table3[table3["spectrum_case"] == spectrum].copy()
    rows = []
    for geometry in GEOMETRY_ORDER:
        for target in TARGET_ORDER:
            m = g[(g["geometry_case"] == geometry) & (g["primary_target"] == target)]
            if len(m):
                r = m.iloc[0]
                rows.append((geometry, target, r["median_first_recovery_window_end"], r["median_last_recovery_window_end"]))

    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(rows))[::-1]
    labels = []
    for yi, (geometry, target, first, last) in zip(y, rows):
        label = f"{DISPLAY_GEOMETRY[geometry]} / {target}"
        if pd.isna(first) or pd.isna(last):
            labels.append(label + " (N/A)")
            continue
        labels.append(label)
        ax.plot([first, last], [yi, yi], linewidth=2.2)
        ax.plot(first, yi, marker="o", markersize=6)
        ax.plot(last, yi, marker="s", markersize=6)

    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("Window-end iteration")
    ax.set_ylabel("Non-normality case / primary target")
    ax.set_title(
        f"Experiment 4: median recovery intervals — {DISPLAY_SPECTRUM[spectrum]}\n"
        "circle = first recovery, square = last recovery"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def draw_subspace_vs_individual_diagnostic(diag: pd.DataFrame, output: Path) -> None:
    """Conceptual non-normality diagnostic.

    Use the strong-gap condition so U2 and U3 are both successfully recovered for all
    three geometry levels. This isolates the geometry effect and avoids mixing in the
    weak-gap U3 failure.

    X-axis labels are U2 and U3, not 'stage 2'/'stage 3'.
    """
    g = diag[(diag["spectrum_case"] == "strong_gap") & (diag["stage"].isin([2, 3]))].copy()
    x_targets = ["U2", "U3"]
    x = np.arange(len(x_targets), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = {"mild_nonnormal": -0.045, "moderate_nonnormal": 0.0, "strong_nonnormal": 0.045}

    for geometry in GEOMETRY_ORDER:
        gg = g[g["geometry_case"] == geometry].set_index("primary_target")
        primary = [float(gg.loc[t, "median_primary_error_deg_latest_accepted"]) for t in x_targets]
        individual = [float(gg.loc[t, "median_individual_qi_error_deg_latest_accepted_diagnostic"]) for t in x_targets]
        xx = x + offsets[geometry]
        label_base = DISPLAY_GEOMETRY[geometry]
        ax.plot(xx, primary, marker="o", label=f"{label_base}: accumulated-subspace error")
        ax.plot(xx, individual, marker="x", linestyle="--", label=f"{label_base}: individual eigenvector diagnostic")

    ax.axhline(2.5, linestyle=":", linewidth=1.3, label="2.5° correctness tolerance")
    ax.set_xticks(x, labels=x_targets)
    ax.set_xlabel("Accumulated primary target")
    ax.set_ylabel("Median error at latest accepted window (degrees)")
    ax.set_title(
        "Experiment 4: accumulated-subspace error vs individual-eigenvector diagnostic\n"
        "strong spectral gap"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else results_dir / "framework4_final_report")
    output_dir.mkdir(parents=True, exist_ok=True)

    t1, t2, t3, t4, t5 = load_source_tables(results_dir)
    table1, table2, table3, table4, supplementary_diag = make_report_tables(t1, t2, t3, t4, t5)

    # Main report tables.
    save_table(table1, output_dir, "table1_experiment_design_report")
    save_table(table2, output_dir, "table2_primary_recovery_performance_report")
    save_table(table3, output_dir, "table3_primary_recovery_intervals_report")
    save_table(table4, output_dir, "table4_nonnormal_geometry_report")

    # Supplementary diagnostic table retained because it explains why individual q_i
    # should not be the primary target in non-normal systems.
    save_table(supplementary_diag, output_dir, "supplementary_individual_eigenvector_diagnostics")

    # Main Figure 1: what can ever be recovered?
    draw_heatmap(
        t2,
        "ever_recovery_rate",
        output_dir / "figure1_ever_recovery_heatmap.png",
        "Experiment 4: probability of ever recovering q1 / accumulated subspace U_k",
        "Ever-recovery rate",
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
    )

    # Main Figure 2a-c: explicit recovery intervals.
    for letter, spectrum in zip(["a", "b", "c"], SPECTRUM_ORDER):
        draw_recovery_intervals(
            t3,
            spectrum,
            output_dir / f"figure2{letter}_recovery_intervals__{spectrum}.png",
        )

    # Main Figure 3: how long until first successful recovery?
    draw_heatmap(
        t3,
        "median_first_recovery_window_end",
        output_dir / "figure3_first_recovery_iteration_heatmap.png",
        "Experiment 4: median first-recovery iteration",
        "Window-end iteration",
        fmt=".0f",
    )

    # Main Figure 4: conceptual reason for subspace target; renamed U2/U3 x-axis.
    draw_subspace_vs_individual_diagnostic(
        t4,
        output_dir / "figure4_subspace_vs_individual_eigenvector_diagnostic.png",
    )

    # Supplementary figures: useful but not necessary in the main Experiment-4 story.
    draw_heatmap(
        t2,
        "median_latest_primary_error_deg_accepted",
        output_dir / "supplementary_latest_primary_error_heatmap.png",
        "Experiment 4: latest accepted primary-target error",
        "Median primary error (degrees)",
        fmt=".2f",
    )
    draw_heatmap(
        t3,
        "median_log10_relative_distance_at_first_recovery",
        output_dir / "supplementary_distance_to_limit_first_recovery_heatmap.png",
        "Experiment 4: closeness to the limit at first recovery",
        "median log10(||x_t-L|| / ||x_0-L||)",
        fmt=".2f",
    )

    manifest = {
        "main_tables": [
            "table1_experiment_design_report.csv",
            "table2_primary_recovery_performance_report.csv",
            "table3_primary_recovery_intervals_report.csv",
            "table4_nonnormal_geometry_report.csv",
        ],
        "main_figures": [
            "figure1_ever_recovery_heatmap.png",
            "figure2a_recovery_intervals__strong_gap.png",
            "figure2b_recovery_intervals__moderate_gap.png",
            "figure2c_recovery_intervals__weak_gap.png",
            "figure3_first_recovery_iteration_heatmap.png",
            "figure4_subspace_vs_individual_eigenvector_diagnostic.png",
        ],
        "supplementary": [
            "supplementary_individual_eigenvector_diagnostics.csv",
            "supplementary_latest_primary_error_heatmap.png",
            "supplementary_distance_to_limit_first_recovery_heatmap.png",
        ],
        "notes": [
            "No trajectories are simulated by this script.",
            "q1 is the stage-1 primary target.",
            "For k>=2, U_k=span(q1,...,qk) is the primary target and is evaluated by largest principal angle.",
            "Individual higher-eigenvector angles are diagnostic only.",
            "The diagnostic figure uses U2/U3 labels instead of ambiguous stage-2/stage-3 labels.",
            "The diagnostic figure uses the strong-gap condition to isolate the non-normal geometry effect.",
        ],
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 64)
    print("Framework 4 final report generator")
    print("=" * 64)
    print(f"results dir: {results_dir}")
    print(f"output dir:  {output_dir}")
    print("No trajectories were simulated.")
    print("\nMain tables:")
    for x in manifest["main_tables"]:
        print(f"  {x}")
    print("\nMain figures:")
    for x in manifest["main_figures"]:
        print(f"  {x}")
    print("\nSupplementary:")
    for x in manifest["supplementary"]:
        print(f"  {x}")
    print("=" * 64)


if __name__ == "__main__":
    main()
