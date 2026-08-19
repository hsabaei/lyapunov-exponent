#!/usr/bin/env python3
"""
Final post-processing reporter for Framework Experiment 5.

This script merges the nonlinear Experiment 5 run with the beta=0 pair controls,
relabels stage-level outputs by the actual framework target, and writes the final
report tables and figures. It never simulates trajectories.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Tables can still be generated in minimal Python environments.
    plt = None


STRUCTURE_LABELS = {
    "normal_real": "Normal real",
    "nonnormal_real": "Non-normal real",
    "rotation_pair": "Rotation pair",
    "equal_magnitude_pair": "Equal-magnitude pair",
}

TARGET_LABELS = {
    "normal_real": {1: "q1", 2: "q2", 3: "q3", 4: "q4", 5: "q5"},
    "nonnormal_real": {1: "q1", 2: "U2", 3: "U3", 4: "U4", 5: "U5"},
    "rotation_pair": {1: "q1", 2: "q2", 3: "N/A", 4: "U4", 5: "q5"},
    "equal_magnitude_pair": {1: "q1", 2: "q2", 3: "N/A", 4: "U4", 5: "q5"},
}

REPRESENTATIVE_TARGET = {
    "normal_real": "q3",
    "nonnormal_real": "U3",
    "rotation_pair": "U4",
    "equal_magnitude_pair": "U4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final report tables and figures for Framework Experiment 5."
    )
    parser.add_argument(
        "--nonlinear-results-dir",
        "--results-dir",
        dest="nonlinear_results_dir",
        required=True,
        type=Path,
        help="Directory containing the beta=0.1,0.5,1 nonlinear Experiment 5 results.",
    )
    parser.add_argument(
        "--control-results-dir",
        required=True,
        type=Path,
        help="Directory containing the beta=0 pair linear-control results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Final report output directory. Default: <nonlinear-results-dir>/final-report.",
    )
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=2.5,
        help="External correctness tolerance used by the experiment.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Bootstrap replicates for simple rate confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def first_existing_file(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        p = root / name
        if p.exists():
            return p
    return None


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = False) -> str | None:
    exact = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    simplified = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in simplified:
            return simplified[key]
    if required:
        raise KeyError(
            f"Could not find one of {list(candidates)} in columns: {list(df.columns)}"
        )
    return None


def normalize_structure(value: object) -> str:
    s = str(value)
    s = s.replace("__linear_control", "")
    for key in STRUCTURE_LABELS:
        if key in s:
            return key
    lowered = s.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "normal": "normal_real",
        "normal_real": "normal_real",
        "non_normal_real": "nonnormal_real",
        "nonnormal": "nonnormal_real",
        "nonnormal_real": "nonnormal_real",
        "rotation": "rotation_pair",
        "rotation_pair": "rotation_pair",
        "equal_magnitude": "equal_magnitude_pair",
        "equal_magnitude_pair": "equal_magnitude_pair",
    }
    return aliases.get(lowered, lowered)


def parse_stage_value(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing stage/filter index.")
    text = str(value).strip().lower()
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse stage/filter index from {value!r}.")


def infer_beta(row: pd.Series, beta_col: str | None, case_col: str | None, default: float | None) -> float:
    if beta_col and pd.notna(row.get(beta_col)):
        return float(row[beta_col])
    if case_col:
        text = str(row.get(case_col, ""))
        match = re.search(r"beta[_=]?([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
    for value in row.values:
        text = str(value)
        match = re.search(r"beta[_=]?([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
    if default is not None:
        return float(default)
    return math.nan


def infer_structure_series(df: pd.DataFrame) -> pd.Series:
    structure_col = find_column(
        df,
        [
            "structure",
            "structure_name",
            "structure_label",
            "nonlinear_structure",
            "system_type",
            "case_structure",
            "case_type",
            "family",
            "scenario",
            "setting",
            "experiment_case",
            "experiment_condition",
        ],
    )
    case_col = find_column(
        df,
        [
            "case",
            "case_id",
            "case_name",
            "case_key",
            "condition",
            "condition_id",
            "configuration",
            "config",
        ],
    )

    if structure_col:
        return df[structure_col].map(normalize_structure)
    if case_col:
        return df[case_col].map(normalize_structure)

    # Last resort: scan text-valued columns for embedded case names such as
    # "rotation_pair__beta_0.5". This keeps the reporter usable across small
    # schema changes in the experiment runner.
    text_cols = [
        c
        for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    inferred = pd.Series([None] * len(df), index=df.index, dtype=object)
    known_tokens = tuple(STRUCTURE_LABELS)
    alias_tokens = {
        "non_normal_real": "nonnormal_real",
        "nonnormal": "nonnormal_real",
        "rotation": "rotation_pair",
        "equal_magnitude": "equal_magnitude_pair",
        "normal": "normal_real",
    }
    for col in text_cols:
        values = df[col].astype(str)
        for idx, value in values.items():
            if inferred.at[idx] is not None:
                continue
            lowered = value.lower().replace("-", "_").replace(" ", "_")
            for token in known_tokens:
                if token in lowered:
                    inferred.at[idx] = token
                    break
            if inferred.at[idx] is not None:
                continue
            for alias, canonical in alias_tokens.items():
                if alias in lowered:
                    inferred.at[idx] = canonical
                    break

    if inferred.notna().all():
        return inferred

    raise KeyError(
        "Could not infer structure column from events file. "
        f"Available columns are: {list(df.columns)}. "
        "Expected a structure/case column or text values containing one of "
        f"{list(STRUCTURE_LABELS)}."
    )


def normalize_events(df: pd.DataFrame, source: str, default_beta: float | None = None) -> pd.DataFrame:
    df = df.copy()
    case_col = find_column(
        df,
        [
            "case",
            "case_id",
            "case_name",
            "case_key",
            "condition",
            "condition_id",
            "configuration",
            "config",
            "experiment_case",
            "experiment_condition",
        ],
    )
    stage_col = find_column(
        df,
        ["stage", "stage_index", "filter", "filter_index", "k", "target_index", "direction", "direction_index"],
        required=True,
    )
    beta_col = find_column(df, ["beta", "nonlinear_beta"])
    traj_col = find_column(df, ["trajectory_id", "traj_id", "trajectory", "run_id"])
    system_col = find_column(df, ["system_id", "system", "system_index"])
    state_col = find_column(df, ["state_id", "initial_state_id", "state_index", "ic_id"])

    df["structure"] = infer_structure_series(df)
    df["beta"] = df.apply(lambda r: infer_beta(r, beta_col, case_col, default_beta), axis=1)
    df["stage"] = df[stage_col].map(parse_stage_value)
    df["target"] = [
        TARGET_LABELS.get(struct, {}).get(int(stage), f"stage{int(stage)}")
        for struct, stage in zip(df["structure"], df["stage"])
    ]
    df["valid_target"] = df["target"].ne("N/A")
    df["source_run"] = source

    if traj_col:
        df["trajectory_key"] = df[traj_col].astype(str)
    elif system_col and state_col:
        df["trajectory_key"] = df[system_col].astype(str) + ":" + df[state_col].astype(str)
    else:
        df["trajectory_key"] = np.arange(len(df)).astype(str)

    for canonical, candidates in {
        "accepted": ["accepted", "is_accepted", "A_k", "accepted_latest", "latest_accepted"],
        "correct": ["correct", "is_correct", "C_k", "primary_correct", "latest_correct"],
        "ever_recovered": ["ever_recovered", "ever_recover", "ever_correct", "ever_primary_correct"],
        "first_recovery": ["first_recovery", "first_recovery_iteration", "t_first", "first_t"],
        "last_recovery": ["last_recovery", "last_recovery_iteration", "t_last", "last_t"],
        "primary_error_deg": ["primary_error_deg", "primary_error", "error_deg", "angle_deg"],
        "distance_log10_first": [
            "log10_distance_to_limit_at_first_recovery",
            "log10_relative_distance_at_first_recovery",
            "distance_log10_first",
            "log10_distance_ratio_first",
        ],
        "jacobian_deviation_first": [
            "jacobian_deviation_at_first_recovery",
            "relative_jacobian_deviation_at_first_recovery",
            "jacobian_rel_error_first",
            "j_rel_first",
        ],
    }.items():
        col = find_column(df, candidates)
        if col and col != canonical:
            df[canonical] = df[col]

    if "accepted" not in df:
        df["accepted"] = np.isfinite(pd.to_numeric(df.get("primary_error_deg"), errors="coerce"))
    if "correct" not in df:
        err = pd.to_numeric(df.get("primary_error_deg"), errors="coerce")
        df["correct"] = err <= 2.5
    if "ever_recovered" not in df:
        if "first_recovery" in df:
            df["ever_recovered"] = pd.to_numeric(df["first_recovery"], errors="coerce").notna()
        else:
            df["ever_recovered"] = df["correct"]

    for col in ["accepted", "correct", "ever_recovered"]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in [
        "first_recovery",
        "last_recovery",
        "primary_error_deg",
        "distance_log10_first",
        "jacobian_deviation_first",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_events(nonlinear_dir: Path, control_dir: Path) -> pd.DataFrame:
    event_name = "trajectory_stage_recovery_events.csv"
    nonlinear = normalize_events(read_csv_required(nonlinear_dir / event_name), "nonlinear")
    control = normalize_events(read_csv_required(control_dir / event_name), "beta0_pair_control", 0.0)
    merged = pd.concat([nonlinear, control], ignore_index=True, sort=False)
    merged = merged[merged["valid_target"]].copy()
    order = {name: i for i, name in enumerate(STRUCTURE_LABELS)}
    merged["structure_order"] = merged["structure"].map(order).fillna(99)
    return merged.sort_values(["structure_order", "beta", "stage", "trajectory_key"])


def bootstrap_ci(values: pd.Series, rng: np.random.Generator, n_rep: int) -> tuple[float, float]:
    arr = values.astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan
    if len(arr) == 1 or n_rep <= 0:
        return float(arr[0]), float(arr[0])
    idx = rng.integers(0, len(arr), size=(n_rep, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def table1_design(events: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for structure, label in STRUCTURE_LABELS.items():
        sub = events[events["structure"].eq(structure)]
        betas = ", ".join(f"{b:g}" for b in sorted(sub["beta"].dropna().unique()))
        n_traj = sub.groupby(["beta", "trajectory_key"]).size().groupby("beta").size()
        rows.append(
            {
                "structure": label,
                "beta_values": betas,
                "filter_1_target": TARGET_LABELS[structure][1],
                "filter_2_target": TARGET_LABELS[structure][2],
                "filter_3_target": TARGET_LABELS[structure][3],
                "filter_4_target": TARGET_LABELS[structure][4],
                "filter_5_target": TARGET_LABELS[structure][5],
                "trajectories_per_beta": "; ".join(f"{b:g}: {int(n)}" for b, n in n_traj.items()),
                "correctness_tolerance_deg": tolerance,
                "simulation_status": "existing results only; no trajectories rerun",
            }
        )
    return pd.DataFrame(rows)


def table2_performance(events: pd.DataFrame, rng: np.random.Generator, n_rep: int) -> pd.DataFrame:
    rows = []
    for keys, g in events.groupby(["structure", "beta", "stage", "target"], sort=False):
        structure, beta, stage, target = keys
        n = len(g)
        accepted = g["accepted"]
        correct = g["correct"]
        ever = g["ever_recovered"]
        accepted_correct = accepted & correct
        false_accept = accepted & ~correct
        lo, hi = bootstrap_ci(ever, rng, n_rep)
        err = g.loc[g["accepted"], "primary_error_deg"] if "primary_error_deg" in g else pd.Series(dtype=float)
        rows.append(
            {
                "structure": STRUCTURE_LABELS.get(structure, structure),
                "beta": beta,
                "target": target,
                "filter_index": stage,
                "n_trajectories": n,
                "acceptance_rate": accepted.mean(),
                "accepted_and_correct_rate": accepted_correct.mean(),
                "reliability_given_accepted": accepted_correct.sum() / accepted.sum() if accepted.sum() else math.nan,
                "false_acceptance_rate": false_accept.mean(),
                "ever_recovery_rate": ever.mean(),
                "ever_recovery_ci95_low": lo,
                "ever_recovery_ci95_high": hi,
                "median_primary_error_deg_among_accepted": err.median() if len(err) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def table3_recovery(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in events.groupby(["structure", "beta", "stage", "target"], sort=False):
        structure, beta, stage, target = keys
        first = g.get("first_recovery", pd.Series(dtype=float)).dropna()
        last = g.get("last_recovery", pd.Series(dtype=float)).dropna()
        dist = g.get("distance_log10_first", pd.Series(dtype=float)).dropna()
        jac = g.get("jacobian_deviation_first", pd.Series(dtype=float)).dropna()
        rows.append(
            {
                "structure": STRUCTURE_LABELS.get(structure, structure),
                "beta": beta,
                "target": target,
                "filter_index": stage,
                "n_ever_recovered": int(g["ever_recovered"].sum()),
                "median_first_recovery": first.median() if len(first) else math.nan,
                "median_last_recovery": last.median() if len(last) else math.nan,
                "median_recovery_span": (last - first).median() + 1 if len(first) and len(last) else math.nan,
                "median_log10_distance_to_L_at_first_recovery": dist.median() if len(dist) else math.nan,
                "median_relative_jacobian_deviation_at_first_recovery": jac.median() if len(jac) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def table4_jacobian(nonlinear_dir: Path, control_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for root, default_beta, source in [
        (nonlinear_dir, None, "nonlinear"),
        (control_dir, 0.0, "beta0_pair_control"),
    ]:
        p = first_existing_file(
            root,
            [
                "all_trajectories.csv",
                "trajectory_jacobian_summary.csv",
                "table4_jacobian_variation_summary.csv",
            ],
        )
        if not p:
            continue
        df = pd.read_csv(p)
        case_col = find_column(
            df,
            [
                "case",
                "case_id",
                "case_name",
                "case_key",
                "condition",
                "condition_id",
                "configuration",
                "config",
                "experiment_case",
                "experiment_condition",
            ],
        )
        beta_col = find_column(df, ["beta", "nonlinear_beta"])
        df["structure"] = infer_structure_series(df)
        df["beta"] = df.apply(lambda r: infer_beta(r, beta_col, case_col, default_beta), axis=1)
        df["source_run"] = source
        frames.append(df)

    rows = []
    if frames:
        traj = pd.concat(frames, ignore_index=True, sort=False)
        initial_col = find_column(
            traj,
            ["initial_jacobian_deviation", "jacobian_deviation_initial", "j_rel_initial", "initial_relative_jacobian_deviation"],
        )
        max_col = find_column(
            traj,
            ["max_jacobian_deviation", "jacobian_deviation_max", "j_rel_max", "max_relative_jacobian_deviation"],
        )
        median_col = find_column(
            traj,
            ["median_jacobian_deviation", "jacobian_deviation_median", "j_rel_median", "median_relative_jacobian_deviation"],
        )
        final_col = find_column(
            traj,
            ["final_jacobian_deviation", "jacobian_deviation_final", "j_rel_final", "final_relative_jacobian_deviation"],
        )
        instant_col = find_column(
            traj,
            ["jacobian_deviation", "relative_jacobian_deviation", "j_rel", "jacobian_rel_error"],
        )
        distance_col = find_column(
            traj,
            ["log10_relative_distance", "log10_distance_ratio", "distance_log10", "log10_distance_to_limit"],
        )

        for keys, g in traj.groupby(["structure", "beta"], sort=False):
            structure, beta = keys
            row = {
                "structure": STRUCTURE_LABELS.get(structure, structure),
                "beta": beta,
                "n_rows": len(g),
            }
            for out, col in [
                ("median_initial_relative_jacobian_deviation", initial_col),
                ("median_max_relative_jacobian_deviation", max_col),
                ("median_trajectory_median_relative_jacobian_deviation", median_col),
                ("median_final_relative_jacobian_deviation", final_col),
            ]:
                row[out] = pd.to_numeric(g[col], errors="coerce").median() if col else math.nan
            if instant_col:
                vals = pd.to_numeric(g[instant_col], errors="coerce")
                row["pooled_median_relative_jacobian_deviation"] = vals.median()
                row["pooled_q95_relative_jacobian_deviation"] = vals.quantile(0.95)
            if distance_col:
                row["minimum_log10_relative_distance_observed"] = pd.to_numeric(
                    g[distance_col], errors="coerce"
                ).min()
            rows.append(row)

    if not rows:
        for keys, g in events.groupby(["structure", "beta"], sort=False):
            structure, beta = keys
            rows.append(
                {
                    "structure": STRUCTURE_LABELS.get(structure, structure),
                    "beta": beta,
                    "n_rows": len(g),
                    "median_relative_jacobian_deviation_at_first_recovery": g.get(
                        "jacobian_deviation_first", pd.Series(dtype=float)
                    ).median(),
                    "note": "No all_trajectories/jacobian summary file found; using event-level first-recovery values only.",
                }
            )
    return pd.DataFrame(rows)


def write_tables(events: pd.DataFrame, out: Path, tolerance: float, n_rep: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    table1_design(events, tolerance).to_csv(out / "table1_experiment_design_report.csv", index=False)
    perf = table2_performance(events, rng, n_rep)
    rec = table3_recovery(events)
    perf.to_csv(out / "table2_primary_recovery_performance_report.csv", index=False)
    rec.to_csv(out / "table3_recovery_intervals_and_locality_report.csv", index=False)
    return perf, rec


def plot_heatmap(table: pd.DataFrame, value_col: str, out: Path, title: str, cbar_label: str) -> None:
    if plt is None:
        print(f"Skipping {out.name}: matplotlib is not installed.")
        return
    structures = list(STRUCTURE_LABELS.values())
    betas = sorted(table["beta"].dropna().unique())
    nrows = len(structures)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 2.5 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    for ax, structure in zip(axes, structures):
        sub = table[table["structure"].eq(structure)]
        target_order = [
            TARGET_LABELS[k][stage]
            for k, v in STRUCTURE_LABELS.items()
            if v == structure
            for stage in sorted(TARGET_LABELS[k])
            if TARGET_LABELS[k][stage] != "N/A"
        ]
        mat = np.full((len(betas), len(target_order)), np.nan)
        for i, beta in enumerate(betas):
            for j, target in enumerate(target_order):
                row = sub[sub["beta"].eq(beta) & sub["target"].eq(target)]
                if not row.empty:
                    mat[i, j] = row[value_col].iloc[0]
        finite = mat[np.isfinite(mat)]
        if len(finite):
            vmin, vmax = np.nanmin(mat), np.nanmax(mat)
            if vmin == vmax:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(structure)
        ax.set_xticks(range(len(target_order)), target_order)
        ax.set_yticks(range(len(betas)), [f"{b:g}" for b in betas])
        ax.set_ylabel("beta")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", color="white", fontsize=8)
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label=cbar_label)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_recovery_intervals(rec: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        print("Skipping recovery-interval figures: matplotlib is not installed.")
        return
    for structure in STRUCTURE_LABELS.values():
        sub = rec[rec["structure"].eq(structure)].copy()
        sub = sub[np.isfinite(sub["median_first_recovery"]) & np.isfinite(sub["median_last_recovery"])]
        if sub.empty:
            continue
        sub = sub.sort_values(["target", "beta"])
        labels = [f"{r.target}, beta={r.beta:g}" for r in sub.itertuples()]
        y = np.arange(len(sub))
        left = sub["median_first_recovery"].to_numpy()
        width = sub["median_last_recovery"].to_numpy() - left + 1
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.35 * len(sub))))
        ax.barh(y, width, left=left, color="#3b82f6", edgecolor="#1f2937", alpha=0.85)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Iteration")
        ax.set_title(f"Recovery intervals: {structure}")
        ax.grid(axis="x", alpha=0.25)
        fname = "figure2_recovery_intervals__" + structure.lower().replace(" ", "_").replace("-", "_") + ".png"
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)


def copy_existing_representative_figures(nonlinear_dir: Path, out_dir: Path) -> None:
    figure_dir = out_dir / "supplementary_existing_source_figures"
    figure_dir.mkdir(exist_ok=True)
    copied = []
    for src in nonlinear_dir.rglob("*.png"):
        name = src.name.lower()
        if "primary_error_vs_distance" in name or "jacobian_variation_vs_distance" in name:
            dst = figure_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            copied.append(dst.name)
    manifest = figure_dir / "manifest.txt"
    manifest.write_text("\n".join(sorted(copied)) + ("\n" if copied else ""), encoding="utf-8")


def plot_representative_error_from_npz(nonlinear_dir: Path, out_dir: Path) -> None:
    if plt is None:
        print("Skipping NPZ-derived representative error figures: matplotlib is not installed.")
        return
    traces = list((nonlinear_dir / "time_traces").glob("*.npz"))
    if not traces:
        return
    for structure, target in REPRESENTATIVE_TARGET.items():
        paths = [p for p in traces if structure in p.stem]
        if not paths:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        made_line = False
        for p in sorted(paths):
            data = np.load(p)
            keys = set(data.files)
            beta_match = re.search(r"beta[_=]?([0-9]+(?:\.[0-9]+)?)", p.stem)
            beta_label = beta_match.group(1) if beta_match else p.stem
            stage = next((k for k, v in TARGET_LABELS[structure].items() if v == target), None)
            x_key = next((k for k in keys if "log10" in k.lower() and ("distance" in k.lower() or "dist" in k.lower())), None)
            y_key = None
            for candidate in [
                f"stage{stage}_primary_error_deg",
                f"primary_error_stage{stage}",
                f"stage{stage}_error_deg",
            ]:
                if candidate in keys:
                    y_key = candidate
                    break
            if not x_key or not y_key:
                continue
            x = np.asarray(data[x_key]).reshape(-1)
            y = np.asarray(data[y_key]).reshape(-1)
            n = min(len(x), len(y))
            if n:
                ax.plot(x[:n], y[:n], label=f"beta={beta_label}", alpha=0.9)
                made_line = True
        if made_line:
            ax.axhline(2.5, color="#dc2626", linestyle="--", linewidth=1, label="2.5 deg")
            ax.set_xlabel("log10(||x_t - L|| / ||x_0 - L||)")
            ax.set_ylabel(f"{target} primary error (deg)")
            ax.set_title(f"Primary error vs distance: {STRUCTURE_LABELS[structure]}, {target}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"figure3_primary_error_vs_distance__{structure}__{target}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    nonlinear_dir = args.nonlinear_results_dir.resolve()
    control_dir = args.control_results_dir.resolve()
    out = (args.output_dir or nonlinear_dir / "final-report").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Framework 5 final report generator")
    print("=" * 60)
    print(f"nonlinear results dir: {nonlinear_dir}")
    print(f"beta=0 control dir:   {control_dir}")
    print(f"output dir:           {out}")
    print("No trajectories will be simulated.")

    events = load_events(nonlinear_dir, control_dir)
    events.to_csv(out / "merged_trajectory_stage_recovery_events_report.csv", index=False)
    perf, rec = write_tables(events, out, args.tolerance_deg, args.bootstrap_replicates, args.seed)
    table4_jacobian(nonlinear_dir, control_dir, events).to_csv(
        out / "table4_jacobian_variation_summary_report.csv", index=False
    )

    plot_heatmap(
        perf,
        "ever_recovery_rate",
        out / "figure1_ever_recovery_by_actual_target_heatmap.png",
        "Ever recovery by actual target",
        "Ever recovery rate",
    )
    plot_recovery_intervals(rec, out)
    plot_heatmap(
        rec,
        "median_first_recovery",
        out / "figure3_median_first_recovery_iteration_heatmap.png",
        "Median first recovery iteration by actual target",
        "Iteration",
    )
    plot_heatmap(
        rec,
        "median_relative_jacobian_deviation_at_first_recovery",
        out / "figure4_jacobian_deviation_at_first_recovery_heatmap.png",
        "Jacobian deviation at first recovery",
        "Relative Jacobian deviation",
    )
    copy_existing_representative_figures(nonlinear_dir, out)
    plot_representative_error_from_npz(nonlinear_dir, out)

    print("\nGenerated final report outputs:")
    for path in sorted(out.iterdir()):
        print(f"  {path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
