#!/usr/bin/env python3
"""
Final post-processing reporter for Framework Experiment 5.

This script merges the nonlinear Experiment 5 run with the beta=0 pair
controls, preserves the framework's structure-specific primary targets, and
writes the final report tables and figures. It NEVER simulates trajectories.

Expected inputs
---------------
Nonlinear results directory (beta = 0.1, 0.5, 1.0):
    trajectory_stage_recovery_events.csv
    all_trajectories.csv
    time_traces/*.npz              (needed for error-vs-distance/Jacobian plots)

Pair-control results directory (beta = 0):
    trajectory_stage_recovery_events.csv
    all_trajectories.csv
    time_traces/*.npz              (optional but used when available)

Main outputs
------------
Table 1  Experimental design and structure-specific reference targets
Table 2  Primary acceptance/recovery performance
Table 3  Recovery intervals + locality + Jacobian deviation at first recovery
Table 4  Jacobian-variation summary
Figure 1 Ever recovery by actual target
Figure 2 Recovery intervals (one figure per structural case)
Figure 3 Primary error vs distance to L (representative later target, 4 panels)
Figure 4 Jacobian variation vs distance to L (4 panels)

Supplementary outputs include first-recovery and Jacobian-at-first-recovery
heatmaps. No simulation is run by this reporter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # tables still work without matplotlib
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

EXPECTED_BETAS = {
    "normal_real": (0.1, 0.5, 1.0),
    "nonnormal_real": (0.1, 0.5, 1.0),
    "rotation_pair": (0.0, 0.1, 0.5, 1.0),
    "equal_magnitude_pair": (0.0, 0.1, 0.5, 1.0),
}

REPRESENTATIVE_TARGET = {
    "normal_real": "q3",
    "nonnormal_real": "U3",
    "rotation_pair": "U4",
    "equal_magnitude_pair": "U4",
}

REPORTER_VERSION = "2026-08-19-fix3"


NONLINEARITY_TO_BETA = {
    "mild_nonlinearity": 0.1,
    "moderate_nonlinearity": 0.5,
    "strong_nonlinearity": 1.0,
    "linear_control": 0.0,
}


# ---------------------------------------------------------------------------
# CLI / file helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final report tables/figures for Framework Experiment 5."
    )
    parser.add_argument(
        "--nonlinear-results-dir",
        "--results-dir",
        dest="nonlinear_results_dir",
        required=True,
        type=Path,
        help="Directory containing beta=0.1,0.5,1 nonlinear Experiment 5 results.",
    )
    parser.add_argument(
        "--control-results-dir",
        required=True,
        type=Path,
        help="Directory containing beta=0 pair linear-control results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <nonlinear-results-dir>/final-report.",
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
        help="Hierarchical bootstrap replicates for rate CIs.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--expected-trajectories-per-case",
        type=int,
        default=200,
        help="Strict integrity check. Use 0 to disable.",
    )
    parser.add_argument(
        "--expected-systems-per-case",
        type=int,
        default=10,
        help="Strict integrity check. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-strict-validation",
        action="store_true",
        help="Do not fail on missing/unexpected structure-beta cells.",
    )
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = False,
) -> str | None:
    exact = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]

    simplified = {
        re.sub(r"[^a-z0-9]", "", c.lower()): c
        for c in df.columns
    }
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in simplified:
            return simplified[key]

    if required:
        raise KeyError(
            f"Could not find one of {list(candidates)} in columns: {list(df.columns)}"
        )
    return None


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Dependency-free markdown writer (no optional `tabulate` package)."""
    cols = list(df.columns)

    def clean(x: object) -> str:
        try:
            if pd.isna(x):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(x, float):
            return f"{x:.8g}"
        return str(x).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, out: Path, stem: str) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(dataframe_to_markdown(df), encoding="utf-8")


# ---------------------------------------------------------------------------
# Normalization / integrity
# ---------------------------------------------------------------------------


def normalize_structure(value: object) -> str:
    """
    Convert a structure/case string to a canonical structure key.

    IMPORTANT: `normal_real` is a substring of `nonnormal_real`.  Therefore
    non-normal must be checked FIRST.  The previous reporter checked
    `normal_real` first, which incorrectly merged normal and non-normal cases.
    """
    s = str(value).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)

    # Exact aliases first.
    exact_aliases = {
        "normal": "normal_real",
        "normal_real": "normal_real",
        "nonnormal": "nonnormal_real",
        "non_normal": "nonnormal_real",
        "nonnormal_real": "nonnormal_real",
        "non_normal_real": "nonnormal_real",
        "rotation": "rotation_pair",
        "rotation_pair": "rotation_pair",
        "equal_magnitude": "equal_magnitude_pair",
        "equal_magnitude_pair": "equal_magnitude_pair",
    }
    if s in exact_aliases:
        return exact_aliases[s]

    # Embedded case names, most specific first.
    embedded = [
        ("equal_magnitude_pair", "equal_magnitude_pair"),
        ("non_normal_real", "nonnormal_real"),
        ("nonnormal_real", "nonnormal_real"),
        ("rotation_pair", "rotation_pair"),
        ("normal_real", "normal_real"),
    ]
    for token, canonical in embedded:
        if token in s:
            return canonical

    return s


def infer_structure_series(df: pd.DataFrame) -> pd.Series:
    # Prefer the runner's explicit column.  This is the critical fix.
    structure_col = find_column(
        df,
        [
            "structure_case",
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
    if structure_col:
        return df[structure_col].map(normalize_structure)

    case_col = find_column(
        df,
        [
            "case_name",
            "case",
            "case_id",
            "case_key",
            "condition",
            "condition_id",
            "configuration",
            "config",
        ],
    )
    if case_col:
        return df[case_col].map(normalize_structure)

    raise KeyError(
        "Could not infer structure. Expected `structure_case` or a case-name column. "
        f"Columns: {list(df.columns)}"
    )


def parse_stage_value(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing stage/filter index.")
    match = re.search(r"(\d+)", str(value).strip().lower())
    if not match:
        raise ValueError(f"Could not parse stage/filter index from {value!r}.")
    return int(match.group(1))


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin(["true", "t", "yes", "y", "1"])


def infer_beta(
    row: pd.Series,
    beta_col: str | None,
    case_col: str | None,
    default: float | None,
) -> float:
    if beta_col and pd.notna(row.get(beta_col)):
        return float(row[beta_col])
    if case_col:
        text = str(row.get(case_col, ""))
        match = re.search(r"beta[_=]?([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
        for name, beta in NONLINEARITY_TO_BETA.items():
            if name in text:
                return float(beta)
    if default is not None:
        return float(default)
    return math.nan


def normalize_events(
    df: pd.DataFrame,
    source: str,
    tolerance_deg: float,
    default_beta: float | None = None,
) -> pd.DataFrame:
    df = df.copy()

    case_col = find_column(
        df,
        ["case_name", "case", "case_id", "case_key", "condition", "configuration"],
    )
    stage_col = find_column(
        df,
        ["stage", "stage_index", "filter", "filter_index", "k", "target_index"],
        required=True,
    )
    beta_col = find_column(df, ["beta", "nonlinear_beta"])
    traj_col = find_column(
        df,
        ["trajectory_uid", "trajectory_id", "traj_id", "trajectory", "run_id"],
    )
    system_col = find_column(
        df,
        ["system_uid", "system_id", "system", "system_index", "system_replicate"],
    )
    state_col = find_column(
        df,
        ["initial_state_within_system", "state_id", "initial_state_id", "state_index", "ic_id"],
    )

    df["structure"] = infer_structure_series(df)
    df["beta"] = df.apply(
        lambda r: infer_beta(r, beta_col, case_col, default_beta), axis=1
    )
    df["stage"] = df[stage_col].map(parse_stage_value)
    df["target"] = [
        TARGET_LABELS.get(structure, {}).get(int(stage), f"stage{int(stage)}")
        for structure, stage in zip(df["structure"], df["stage"])
    ]

    # Keep the runner-provided target for an integrity check.  This prevents a
    # reporting bug from silently turning non-normal U2/U3/... into q2/q3/....
    source_target_col = find_column(df, ["primary_target", "target_name", "reference_target"])
    if source_target_col is not None:
        df["source_primary_target"] = df[source_target_col].astype(str)
    else:
        df["source_primary_target"] = ""

    df["valid_target"] = df["target"].ne("N/A")
    df["source_run"] = source

    if traj_col:
        df["trajectory_key"] = df[traj_col].astype(str)
    elif system_col and state_col:
        df["trajectory_key"] = (
            df[system_col].astype(str) + ":" + df[state_col].astype(str)
        )
    else:
        raise KeyError("Could not construct a trajectory key from the events file.")

    if system_col:
        df["system_key"] = df[system_col].astype(str)
    elif traj_col:
        # Only a fallback; actual Experiment 5 files contain system_uid.
        df["system_key"] = df[traj_col].astype(str).str.replace(
            r"__state\d+$", "", regex=True
        )
    else:
        df["system_key"] = "system_unknown"

    # Canonical raw fields.
    field_candidates = {
        "ever_accepted": ["ever_accepted", "accepted", "is_accepted", "A_k"],
        "ever_recovered": [
            "ever_recovered_primary_target",
            "ever_recovered",
            "ever_recover",
            "ever_correct",
        ],
        "first_recovery": [
            "first_recovery_window_end",
            "first_recovery_iteration",
            "first_recovery",
            "t_first",
        ],
        "last_recovery": [
            "last_recovery_window_end",
            "last_recovery_iteration",
            "last_recovery",
            "t_last",
        ],
        "latest_primary_error_deg": [
            "latest_accepted_primary_error_deg",
            "primary_error_deg",
            "primary_error",
            "error_deg",
        ],
        "first_recovery_primary_error_deg": [
            "first_recovery_primary_error_deg",
            "first_recovery_error_deg",
        ],
        "distance_log10_first": [
            "first_recovery_log10_relative_distance",
            "log10_relative_distance_at_first_recovery",
            "log10_distance_to_limit_at_first_recovery",
        ],
        "jacobian_deviation_first": [
            "first_recovery_jacobian_relative_difference",
            "relative_jacobian_deviation_at_first_recovery",
            "jacobian_deviation_at_first_recovery",
        ],
        "correctness_evaluated": ["correctness_evaluated"],
    }

    for canonical, candidates in field_candidates.items():
        col = find_column(df, candidates)
        if col is not None:
            df[canonical] = df[col]

    if "ever_accepted" not in df:
        df["ever_accepted"] = pd.to_numeric(
            df.get("latest_primary_error_deg"), errors="coerce"
        ).notna()
    else:
        df["ever_accepted"] = parse_bool_series(df["ever_accepted"])

    if "ever_recovered" not in df:
        if "first_recovery" in df:
            df["ever_recovered"] = pd.to_numeric(
                df["first_recovery"], errors="coerce"
            ).notna()
        else:
            raise KeyError("Could not determine ever-recovery status.")
    else:
        df["ever_recovered"] = parse_bool_series(df["ever_recovered"])

    if "correctness_evaluated" in df:
        df["correctness_evaluated"] = parse_bool_series(df["correctness_evaluated"])
    else:
        df["correctness_evaluated"] = df["valid_target"]

    numeric_cols = [
        "first_recovery",
        "last_recovery",
        "latest_primary_error_deg",
        "first_recovery_primary_error_deg",
        "distance_log10_first",
        "jacobian_deviation_first",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # Framework A/C definitions at the selected (latest accepted) output.
    # Rejection => correctness is not evaluated, not set to incorrect.
    selected_exists = df["latest_primary_error_deg"].notna()
    df["accepted"] = df["ever_accepted"] | selected_exists
    df["accepted_and_correct"] = (
        selected_exists
        & df["correctness_evaluated"]
        & (df["latest_primary_error_deg"] <= tolerance_deg)
    )
    df["false_acceptance"] = (
        selected_exists
        & df["correctness_evaluated"]
        & (df["latest_primary_error_deg"] > tolerance_deg)
    )

    return df


def load_events(
    nonlinear_dir: Path,
    control_dir: Path,
    tolerance_deg: float,
) -> pd.DataFrame:
    event_name = "trajectory_stage_recovery_events.csv"
    nonlinear = normalize_events(
        read_csv_required(nonlinear_dir / event_name),
        source="nonlinear",
        tolerance_deg=tolerance_deg,
    )
    control = normalize_events(
        read_csv_required(control_dir / event_name),
        source="beta0_pair_control",
        tolerance_deg=tolerance_deg,
        default_beta=0.0,
    )
    merged = pd.concat([nonlinear, control], ignore_index=True, sort=False)

    # Invalid partial pair stage is deliberately not a correctness target.
    merged = merged[merged["valid_target"]].copy()

    order = {name: i for i, name in enumerate(STRUCTURE_LABELS)}
    merged["structure_order"] = merged["structure"].map(order).fillna(99)
    return merged.sort_values(
        ["structure_order", "beta", "stage", "trajectory_key"]
    ).reset_index(drop=True)


def validate_events(
    events: pd.DataFrame,
    expected_trajectories_per_case: int,
    expected_systems_per_case: int,
    strict: bool,
) -> None:
    errors: list[str] = []

    unknown = sorted(set(events["structure"]) - set(STRUCTURE_LABELS))
    if unknown:
        errors.append(f"Unknown structures after normalization: {unknown}")

    # Cross-check the report target against the target written by the
    # experiment runner.  Pair stage 3 is removed before this validation.
    if "source_primary_target" in events.columns:
        source_target = events["source_primary_target"].fillna("").astype(str).str.strip()
        comparable = source_target.ne("") & source_target.ne("nan")
        mismatch = comparable & source_target.ne(events["target"].astype(str))
        if mismatch.any():
            sample = events.loc[
                mismatch,
                ["structure", "beta", "stage", "target", "source_primary_target"],
            ].drop_duplicates().head(10)
            errors.append(
                "Reporter target labels disagree with runner primary_target values:\n"
                + sample.to_string(index=False)
            )

    # One event row per trajectory and valid target/stage.
    dup = events.duplicated(
        ["source_run", "structure", "beta", "stage", "trajectory_key"]
    )
    if dup.any():
        errors.append(f"Found {int(dup.sum())} duplicate trajectory-stage rows.")

    for structure, expected_betas in EXPECTED_BETAS.items():
        sub = events[events["structure"].eq(structure)]
        actual_betas = tuple(sorted(float(x) for x in sub["beta"].dropna().unique()))
        expected = tuple(sorted(expected_betas))
        if actual_betas != expected:
            errors.append(
                f"{structure}: beta cells {actual_betas}, expected {expected}."
            )

        for beta in expected:
            cell = sub[np.isclose(sub["beta"].astype(float), beta)]
            n_traj = int(cell["trajectory_key"].nunique())
            n_sys = int(cell["system_key"].nunique())
            if expected_trajectories_per_case and n_traj != expected_trajectories_per_case:
                errors.append(
                    f"{structure}, beta={beta:g}: {n_traj} unique trajectories; "
                    f"expected {expected_trajectories_per_case}."
                )
            if expected_systems_per_case and n_sys != expected_systems_per_case:
                errors.append(
                    f"{structure}, beta={beta:g}: {n_sys} systems; "
                    f"expected {expected_systems_per_case}."
                )

    if errors:
        message = "\n".join("  - " + e for e in errors)
        if strict:
            raise RuntimeError("Experiment-5 report integrity check failed:\n" + message)
        print("WARNING: integrity checks found:\n" + message)


# ---------------------------------------------------------------------------
# Bootstrap and tables
# ---------------------------------------------------------------------------


def hierarchical_bootstrap_ci(
    g: pd.DataFrame,
    value_col: str,
    rng: np.random.Generator,
    n_rep: int,
) -> tuple[float, float]:
    """Bootstrap systems first, then initial states within sampled systems."""
    clean = g[["system_key", value_col]].copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean[np.isfinite(clean[value_col])]
    if clean.empty:
        return math.nan, math.nan

    systems = list(clean["system_key"].unique())
    arrays = {
        s: clean.loc[clean["system_key"].eq(s), value_col].to_numpy(dtype=float)
        for s in systems
    }
    if n_rep <= 0:
        estimate = float(clean[value_col].mean())
        return estimate, estimate

    boot = np.empty(n_rep, dtype=float)
    n_systems = len(systems)
    for r in range(n_rep):
        sampled_system_indices = rng.integers(0, n_systems, size=n_systems)
        sampled_values: list[np.ndarray] = []
        for idx in sampled_system_indices:
            arr = arrays[systems[int(idx)]]
            state_idx = rng.integers(0, len(arr), size=len(arr))
            sampled_values.append(arr[state_idx])
        boot[r] = float(np.concatenate(sampled_values).mean())

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def table1_design(events: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for structure, label in STRUCTURE_LABELS.items():
        sub = events[events["structure"].eq(structure)]
        betas = sorted(float(b) for b in sub["beta"].dropna().unique())
        counts = {
            beta: int(
                sub[np.isclose(sub["beta"].astype(float), beta)]["trajectory_key"].nunique()
            )
            for beta in betas
        }
        systems = {
            beta: int(
                sub[np.isclose(sub["beta"].astype(float), beta)]["system_key"].nunique()
            )
            for beta in betas
        }
        rows.append(
            {
                "structure": label,
                "beta_values": ", ".join(f"{b:g}" for b in betas),
                "filter_1_target": TARGET_LABELS[structure][1],
                "filter_2_target": TARGET_LABELS[structure][2],
                "filter_3_target": (
                    "Not evaluated (partial pair)"
                    if TARGET_LABELS[structure][3] == "N/A"
                    else TARGET_LABELS[structure][3]
                ),
                "filter_4_target": TARGET_LABELS[structure][4],
                "filter_5_target": TARGET_LABELS[structure][5],
                "systems_per_beta": "; ".join(
                    f"{b:g}: {systems[b]}" for b in betas
                ),
                "trajectories_per_beta": "; ".join(
                    f"{b:g}: {counts[b]}" for b in betas
                ),
                "correctness_tolerance_deg": tolerance,
                "simulation_status": "existing results only; no trajectories rerun",
            }
        )
    return pd.DataFrame(rows)


def table2_performance(
    events: pd.DataFrame,
    rng: np.random.Generator,
    n_rep: int,
) -> pd.DataFrame:
    rows = []
    group_cols = ["structure", "beta", "stage", "target"]
    for (structure, beta, stage, target), g in events.groupby(group_cols, sort=False):
        g = g.copy()
        n = int(g["trajectory_key"].nunique())
        accepted = g["accepted"].astype(bool)
        ac = g["accepted_and_correct"].astype(bool)
        false = g["false_acceptance"].astype(bool)
        ever = g["ever_recovered"].astype(bool)

        g["_accepted"] = accepted.astype(float)
        g["_ac"] = ac.astype(float)
        g["_false"] = false.astype(float)
        g["_ever"] = ever.astype(float)

        acc_lo, acc_hi = hierarchical_bootstrap_ci(g, "_accepted", rng, n_rep)
        ac_lo, ac_hi = hierarchical_bootstrap_ci(g, "_ac", rng, n_rep)
        ever_lo, ever_hi = hierarchical_bootstrap_ci(g, "_ever", rng, n_rep)

        selected = g[g["latest_primary_error_deg"].notna()]
        reliability = (
            float(ac.sum() / accepted.sum())
            if accepted.sum()
            else math.nan
        )

        rows.append(
            {
                "structure": STRUCTURE_LABELS[structure],
                "beta": float(beta),
                "target": target,
                "filter_index": int(stage),
                "n_systems": int(g["system_key"].nunique()),
                "n_trajectories": n,
                "acceptance_rate": float(accepted.mean()),
                "acceptance_ci95_low": acc_lo,
                "acceptance_ci95_high": acc_hi,
                "accepted_and_correct_rate": float(ac.mean()),
                "accepted_and_correct_ci95_low": ac_lo,
                "accepted_and_correct_ci95_high": ac_hi,
                "reliability_given_accepted": reliability,
                "false_acceptance_rate": float(false.mean()),
                "ever_recovery_rate": float(ever.mean()),
                "ever_recovery_ci95_low": ever_lo,
                "ever_recovery_ci95_high": ever_hi,
                "median_primary_error_deg_among_accepted": (
                    float(selected["latest_primary_error_deg"].median())
                    if len(selected)
                    else math.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    structure_order = {label: i for i, label in enumerate(STRUCTURE_LABELS.values())}
    result["_structure_order"] = result["structure"].map(structure_order)
    result = result.sort_values(["_structure_order", "beta", "filter_index"])
    return result.drop(columns="_structure_order").reset_index(drop=True)


def table3_recovery(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (structure, beta, stage, target), g in events.groupby(
        ["structure", "beta", "stage", "target"], sort=False
    ):
        recovered = g[g["ever_recovered"]].copy()
        first = recovered["first_recovery"].dropna()
        last = recovered["last_recovery"].dropna()
        dist = recovered["distance_log10_first"].dropna()
        jac = recovered["jacobian_deviation_first"].dropna()
        first_err = recovered["first_recovery_primary_error_deg"].dropna()

        spans = (
            recovered["last_recovery"] - recovered["first_recovery"] + 1
        ).dropna()

        rows.append(
            {
                "structure": STRUCTURE_LABELS[structure],
                "beta": float(beta),
                "target": target,
                "filter_index": int(stage),
                "n_trajectories": int(g["trajectory_key"].nunique()),
                "n_ever_recovered": int(g["ever_recovered"].sum()),
                "ever_recovery_rate": float(g["ever_recovered"].mean()),
                "median_first_recovery": first.median() if len(first) else math.nan,
                "q25_first_recovery": first.quantile(0.25) if len(first) else math.nan,
                "q75_first_recovery": first.quantile(0.75) if len(first) else math.nan,
                "median_last_recovery": last.median() if len(last) else math.nan,
                "q25_last_recovery": last.quantile(0.25) if len(last) else math.nan,
                "q75_last_recovery": last.quantile(0.75) if len(last) else math.nan,
                "median_recovery_span_inclusive": spans.median() if len(spans) else math.nan,
                "median_log10_distance_to_L_at_first_recovery": (
                    dist.median() if len(dist) else math.nan
                ),
                "median_relative_jacobian_deviation_at_first_recovery": (
                    jac.median() if len(jac) else math.nan
                ),
                "median_first_recovery_primary_error_deg": (
                    first_err.median() if len(first_err) else math.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    structure_order = {label: i for i, label in enumerate(STRUCTURE_LABELS.values())}
    result["_structure_order"] = result["structure"].map(structure_order)
    result = result.sort_values(["_structure_order", "beta", "filter_index"])
    return result.drop(columns="_structure_order").reset_index(drop=True)


def normalize_trajectory_summary(
    df: pd.DataFrame,
    default_beta: float | None,
) -> pd.DataFrame:
    df = df.copy()
    case_col = find_column(df, ["case_name", "case", "condition"])
    beta_col = find_column(df, ["beta", "nonlinear_beta"])
    traj_col = find_column(df, ["trajectory_uid", "trajectory_id", "trajectory"])
    system_col = find_column(df, ["system_uid", "system_id", "system"])

    df["structure"] = infer_structure_series(df)
    df["beta"] = df.apply(
        lambda r: infer_beta(r, beta_col, case_col, default_beta), axis=1
    )
    if traj_col:
        df["trajectory_key"] = df[traj_col].astype(str)
    else:
        df["trajectory_key"] = np.arange(len(df)).astype(str)
    if system_col:
        df["system_key"] = df[system_col].astype(str)
    else:
        df["system_key"] = "system_unknown"
    return df


def table4_jacobian(nonlinear_dir: Path, control_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root, default_beta in [(nonlinear_dir, None), (control_dir, 0.0)]:
        p = root / "all_trajectories.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Required for final Jacobian summary: {p}"
            )
        frames.append(normalize_trajectory_summary(pd.read_csv(p), default_beta))

    traj = pd.concat(frames, ignore_index=True, sort=False)

    # Exact Experiment-5 runner names first, then tolerant aliases.
    initial_col = find_column(
        traj,
        [
            "initial_jacobian_relative_difference",
            "initial_jacobian_relative_deviation",
            "initial_jacobian_deviation",
            "jacobian_deviation_initial",
        ],
        required=True,
    )
    max_col = find_column(
        traj,
        [
            "max_jacobian_relative_difference",
            "max_jacobian_relative_deviation",
            "max_jacobian_deviation",
            "jacobian_deviation_max",
        ],
        required=True,
    )
    median_col = find_column(
        traj,
        [
            "median_jacobian_relative_difference",
            "median_jacobian_relative_deviation",
            "median_jacobian_deviation",
            "jacobian_deviation_median",
        ],
        required=True,
    )
    final_col = find_column(
        traj,
        [
            "final_jacobian_relative_difference",
            "final_jacobian_relative_deviation",
            "final_jacobian_deviation",
            "jacobian_deviation_final",
        ],
        required=True,
    )
    final_distance_col = find_column(
        traj,
        ["final_relative_distance", "relative_distance_final"],
    )

    rows = []
    for (structure, beta), g in traj.groupby(["structure", "beta"], sort=False):
        initial = pd.to_numeric(g[initial_col], errors="coerce")
        maxv = pd.to_numeric(g[max_col], errors="coerce")
        medv = pd.to_numeric(g[median_col], errors="coerce")
        final = pd.to_numeric(g[final_col], errors="coerce")
        final_dist = (
            pd.to_numeric(g[final_distance_col], errors="coerce")
            if final_distance_col
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "structure": STRUCTURE_LABELS[structure],
                "beta": float(beta),
                "n_systems": int(g["system_key"].nunique()),
                "n_trajectories": int(g["trajectory_key"].nunique()),
                "median_initial_relative_jacobian_deviation": float(initial.median()),
                "q25_initial_relative_jacobian_deviation": float(initial.quantile(0.25)),
                "q75_initial_relative_jacobian_deviation": float(initial.quantile(0.75)),
                "median_max_relative_jacobian_deviation": float(maxv.median()),
                "q25_max_relative_jacobian_deviation": float(maxv.quantile(0.25)),
                "q75_max_relative_jacobian_deviation": float(maxv.quantile(0.75)),
                "median_trajectory_median_relative_jacobian_deviation": float(medv.median()),
                "median_final_relative_jacobian_deviation": float(final.median()),
                "median_final_relative_distance": (
                    float(final_dist.median()) if len(final_dist) else math.nan
                ),
            }
        )

    result = pd.DataFrame(rows)
    structure_order = {label: i for i, label in enumerate(STRUCTURE_LABELS.values())}
    result["_structure_order"] = result["structure"].map(structure_order)
    result = result.sort_values(["_structure_order", "beta"])
    return result.drop(columns="_structure_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _structure_key_from_label(label: str) -> str:
    for key, value in STRUCTURE_LABELS.items():
        if value == label:
            return key
    raise KeyError(label)


def plot_heatmap(
    table: pd.DataFrame,
    value_col: str,
    out: Path,
    title: str,
    cbar_label: str,
    fixed_range: tuple[float, float] | None = None,
    value_format: str = ".2g",
) -> None:
    if plt is None:
        print(f"Skipping {out.name}: matplotlib is not installed.")
        return

    structures = list(STRUCTURE_LABELS.values())
    fig, axes = plt.subplots(
        len(structures), 1, figsize=(9.5, 2.55 * len(structures)), constrained_layout=True
    )
    if len(structures) == 1:
        axes = [axes]

    # A single shared colorbar is used for all panels, therefore every panel
    # must use the same numerical range.  The previous reporter could scale
    # each panel separately while showing only the last panel's colorbar.
    if fixed_range is not None:
        global_vmin, global_vmax = fixed_range
    else:
        all_values = pd.to_numeric(table[value_col], errors="coerce").to_numpy(dtype=float)
        finite_all = all_values[np.isfinite(all_values)]
        if len(finite_all):
            global_vmin = float(finite_all.min())
            global_vmax = float(finite_all.max())
            if np.isclose(global_vmin, global_vmax):
                global_vmax = global_vmin + 1.0
        else:
            global_vmin, global_vmax = 0.0, 1.0

    image_for_colorbar = None
    for ax, structure in zip(axes, structures):
        sub = table[table["structure"].eq(structure)].copy()
        key = _structure_key_from_label(structure)
        betas = sorted(float(b) for b in sub["beta"].dropna().unique())
        target_order = [
            TARGET_LABELS[key][stage]
            for stage in sorted(TARGET_LABELS[key])
            if TARGET_LABELS[key][stage] != "N/A"
        ]

        mat = np.full((len(betas), len(target_order)), np.nan, dtype=float)
        for i, beta in enumerate(betas):
            for j, target in enumerate(target_order):
                row = sub[
                    np.isclose(sub["beta"].astype(float), beta)
                    & sub["target"].eq(target)
                ]
                if not row.empty:
                    mat[i, j] = float(row[value_col].iloc[0])

        image_for_colorbar = ax.imshow(
            mat,
            aspect="auto",
            vmin=global_vmin,
            vmax=global_vmax,
            cmap="viridis",
        )
        ax.set_title(structure)
        ax.set_xticks(range(len(target_order)), target_order)
        ax.set_yticks(range(len(betas)), [f"{b:g}" for b in betas])
        ax.set_ylabel(r"$\beta$")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        format(mat[i, j], value_format),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=8)

    fig.suptitle(title)
    if image_for_colorbar is not None:
        fig.colorbar(image_for_colorbar, ax=axes, shrink=0.8, label=cbar_label)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_recovery_intervals(rec: pd.DataFrame, out_dir: Path) -> list[str]:
    if plt is None:
        print("Skipping recovery-interval figures: matplotlib is not installed.")
        return []

    created: list[str] = []
    for structure_key, structure_label in STRUCTURE_LABELS.items():
        sub = rec[rec["structure"].eq(structure_label)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["filter_index", "beta"])

        labels = [f"{r.target}, beta={r.beta:g}" for r in sub.itertuples()]
        y = np.arange(len(sub))
        first = sub["median_first_recovery"].to_numpy(dtype=float)
        last = sub["median_last_recovery"].to_numpy(dtype=float)
        finite = np.isfinite(first) & np.isfinite(last)

        if finite.any():
            xmin = max(0.0, float(np.nanmin(first[finite])) - 15.0)
            xmax = max(500.0, float(np.nanmax(last[finite])) + 10.0)
        else:
            xmin, xmax = 0.0, 500.0

        fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.40 * len(sub))))
        for i in range(len(sub)):
            if finite[i]:
                ax.barh(
                    y[i],
                    last[i] - first[i] + 1.0,
                    left=first[i],
                    alpha=0.8,
                )
                ax.plot(first[i], y[i], marker="o", linestyle="none", markersize=4)
                ax.plot(last[i], y[i], marker="s", linestyle="none", markersize=4)
            else:
                ax.text(
                    xmin + 3.0,
                    y[i],
                    "no recovery",
                    va="center",
                    fontsize=8,
                )

        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Window-end iteration")
        ax.set_title(
            f"Recovery intervals: {structure_label}\n"
            "circle = first recovery, square = last recovery"
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        fname = f"figure2_recovery_intervals__{structure_key}.png"
        fig.savefig(out_dir / fname, dpi=220)
        plt.close(fig)
        created.append(fname)

    return created


def trace_files_by_beta(
    nonlinear_dir: Path,
    control_dir: Path,
    structure: str,
) -> dict[float, list[Path]]:
    result: dict[float, list[Path]] = {}

    nonlinear_trace_dir = nonlinear_dir / "time_traces"
    if nonlinear_trace_dir.exists():
        for name, beta in NONLINEARITY_TO_BETA.items():
            if beta == 0.0:
                continue
            paths = sorted(
                nonlinear_trace_dir.glob(f"{structure}__{name}__sys*_state*.npz")
            )
            if paths:
                result[float(beta)] = paths

    if structure in {"rotation_pair", "equal_magnitude_pair"}:
        control_trace_dir = control_dir / "time_traces"
        if control_trace_dir.exists():
            paths = sorted(
                control_trace_dir.glob(
                    f"{structure}__linear_control__sys*_state*.npz"
                )
            )
            if paths:
                result[0.0] = paths

    return dict(sorted(result.items()))


def binned_trace_median(
    paths: list[Path],
    stage: int | None,
    metric: str,
    require_accepted: bool,
    n_bins: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for path in paths:
        with np.load(path) as z:
            x = np.asarray(z["log10_relative_distance"], dtype=float)
            if metric == "primary_error":
                assert stage is not None
                y = np.asarray(z["primary_errors_deg"], dtype=float)[stage - 1]
                if require_accepted:
                    accepted = np.asarray(z["accepted"], dtype=bool)[stage - 1]
                else:
                    accepted = np.ones_like(y, dtype=bool)
            elif metric == "jacobian":
                y = np.asarray(z["jacobian_relative_difference"], dtype=float)
                accepted = np.ones_like(y, dtype=bool)
            else:
                raise ValueError(metric)

            mask = np.isfinite(x) & np.isfinite(y) & accepted
            xs.append(x[mask])
            ys.append(y[mask])

    if not xs:
        return np.array([]), np.array([])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if len(x) < 10:
        return np.array([]), np.array([])

    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if np.isclose(lo, hi):
        return np.array([lo]), np.array([float(np.nanmedian(y))])

    edges = np.linspace(lo, hi, n_bins)
    centers: list[float] = []
    medians: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (x >= left) & (x < right)
        if int(mask.sum()) >= 10:
            centers.append(0.5 * (left + right))
            medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def plot_representative_error_vs_distance(
    nonlinear_dir: Path,
    control_dir: Path,
    out: Path,
    tolerance_deg: float,
) -> bool:
    if plt is None:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    made_any = False

    for ax, structure in zip(axes.flat, STRUCTURE_LABELS):
        target = REPRESENTATIVE_TARGET[structure]
        stage = next(
            s for s, label in TARGET_LABELS[structure].items() if label == target
        )
        groups = trace_files_by_beta(nonlinear_dir, control_dir, structure)
        made_panel = False
        for beta, paths in groups.items():
            x, y = binned_trace_median(
                paths,
                stage=stage,
                metric="primary_error",
                require_accepted=True,
            )
            if len(x):
                ax.plot(x, y, marker="o", markersize=3, label=rf"$\beta={beta:g}$")
                made_panel = True
                made_any = True

        ax.axhline(
            tolerance_deg,
            linestyle="--",
            linewidth=1.0,
            label=f"{tolerance_deg:g}° tolerance",
        )
        ax.set_title(f"{STRUCTURE_LABELS[structure]} — {target}")
        ax.set_xlabel(r"$\log_{10}(\|x_t-L\|/\|x_0-L\|)$")
        ax.set_ylabel("Median primary error (degrees)")
        ax.grid(alpha=0.25)
        if made_panel:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "trace files not found", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Primary error versus distance to the fixed point")
    if made_any:
        fig.savefig(out, dpi=220)
    plt.close(fig)
    return made_any


def plot_jacobian_vs_distance(
    nonlinear_dir: Path,
    control_dir: Path,
    out: Path,
) -> bool:
    if plt is None:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    made_any = False

    for ax, structure in zip(axes.flat, STRUCTURE_LABELS):
        groups = trace_files_by_beta(nonlinear_dir, control_dir, structure)
        made_panel = False
        for beta, paths in groups.items():
            x, y = binned_trace_median(
                paths,
                stage=None,
                metric="jacobian",
                require_accepted=False,
            )
            if len(x):
                ax.plot(x, y, marker="o", markersize=3, label=rf"$\beta={beta:g}$")
                made_panel = True
                made_any = True

        ax.set_title(STRUCTURE_LABELS[structure])
        ax.set_xlabel(r"$\log_{10}(\|x_t-L\|/\|x_0-L\|)$")
        ax.set_ylabel(r"Median $\|J_{x_t}-J_L\|_F/\|J_L\|_F$")
        ax.grid(alpha=0.25)
        if made_panel:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "trace files not found", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle("Jacobian variation as the orbit approaches the fixed point")
    if made_any:
        fig.savefig(out, dpi=220)
    plt.close(fig)
    return made_any


def integrity_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Compact audit table proving that structure/beta cells were not merged."""
    rows = []
    for (structure, beta), g in events.groupby(["structure", "beta"], sort=False):
        rows.append(
            {
                "structure_key": structure,
                "structure": STRUCTURE_LABELS.get(structure, structure),
                "beta": float(beta),
                "n_systems": int(g["system_key"].nunique()),
                "n_trajectories": int(g["trajectory_key"].nunique()),
                "targets": ", ".join(
                    g[["stage", "target"]]
                    .drop_duplicates()
                    .sort_values("stage")["target"]
                    .astype(str)
                    .tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    nonlinear_dir = args.nonlinear_results_dir.resolve()
    control_dir = args.control_results_dir.resolve()
    out = (args.output_dir or nonlinear_dir / "final-report").resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Framework 5 final report generator — {REPORTER_VERSION}")
    print("=" * 72)
    print(f"nonlinear results dir: {nonlinear_dir}")
    print(f"beta=0 control dir:   {control_dir}")
    print(f"output dir:           {out}")
    print("No trajectories will be simulated.")

    events = load_events(nonlinear_dir, control_dir, args.tolerance_deg)
    validate_events(
        events,
        expected_trajectories_per_case=args.expected_trajectories_per_case,
        expected_systems_per_case=args.expected_systems_per_case,
        strict=not args.no_strict_validation,
    )

    # Keep merged events and a compact integrity audit for traceability.
    events.to_csv(out / "merged_trajectory_stage_recovery_events_report.csv", index=False)
    integrity_summary(events).to_csv(out / "integrity_structure_beta_counts.csv", index=False)

    rng = np.random.default_rng(args.seed)
    t1 = table1_design(events, args.tolerance_deg)
    t2 = table2_performance(events, rng, args.bootstrap_replicates)
    t3 = table3_recovery(events)
    t4 = table4_jacobian(nonlinear_dir, control_dir)

    save_table(t1, out, "table1_experiment_design_report")
    save_table(t2, out, "table2_primary_recovery_performance_report")
    save_table(t3, out, "table3_recovery_intervals_and_locality_report")
    save_table(t4, out, "table4_jacobian_variation_summary_report")

    # Main Figure 1.
    plot_heatmap(
        t2,
        "ever_recovery_rate",
        out / "figure1_ever_recovery_by_actual_target_heatmap.png",
        "Ever recovery by actual framework target",
        "Ever-recovery rate",
        fixed_range=(0.0, 1.0),
        value_format=".2f",
    )

    # Main Figure 2 group.
    interval_files = plot_recovery_intervals(t3, out)

    # Main Figure 3.
    made_error = plot_representative_error_vs_distance(
        nonlinear_dir,
        control_dir,
        out / "figure3_primary_error_vs_distance.png",
        args.tolerance_deg,
    )

    # Main Figure 4.
    made_jacobian = plot_jacobian_vs_distance(
        nonlinear_dir,
        control_dir,
        out / "figure4_jacobian_variation_vs_distance.png",
    )

    # Supplementary diagnostic heatmaps.
    plot_heatmap(
        t3,
        "median_first_recovery",
        out / "supplementary_first_recovery_iteration_heatmap.png",
        "Median first-recovery iteration by actual target",
        "Window-end iteration",
        value_format=".0f",
    )
    plot_heatmap(
        t3,
        "median_relative_jacobian_deviation_at_first_recovery",
        out / "supplementary_jacobian_deviation_at_first_recovery_heatmap.png",
        "Jacobian deviation at first recovery",
        "Relative Jacobian deviation",
        value_format=".2g",
    )

    manifest = {
        "reporter_version": REPORTER_VERSION,
        "integrity_audit": "integrity_structure_beta_counts.csv",
        "main_tables": [
            "table1_experiment_design_report.csv",
            "table2_primary_recovery_performance_report.csv",
            "table3_recovery_intervals_and_locality_report.csv",
            "table4_jacobian_variation_summary_report.csv",
        ],
        "main_figures": [
            "figure1_ever_recovery_by_actual_target_heatmap.png",
            *interval_files,
            *( ["figure3_primary_error_vs_distance.png"] if made_error else [] ),
            *( ["figure4_jacobian_variation_vs_distance.png"] if made_jacobian else [] ),
        ],
        "supplementary": [
            "supplementary_first_recovery_iteration_heatmap.png",
            "supplementary_jacobian_deviation_at_first_recovery_heatmap.png",
        ],
        "notes": [
            "No trajectories are simulated by this script.",
            "normal_real and nonnormal_real are kept distinct.",
            "Non-normal targets are q1, U2, U3, U4, U5.",
            "Pair stage 3 is N/A; U4 is the complete pair-containing subspace target.",
            "Pair beta=0 controls are integrated as ordinary control conditions.",
            "Rate confidence intervals use a hierarchical bootstrap: systems first, then states.",
        ],
    }
    (out / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nIntegrity check passed.")
    print("Generated final report outputs:")
    for path in sorted(out.iterdir()):
        print(f"  {path.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
