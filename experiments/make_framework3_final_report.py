from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def find_first_glob(directory: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def safe_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0.0)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def q(values: pd.Series, p: float) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.quantile(p)) if len(x) else np.nan


def med(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def rate_ci_hierarchical(
    frame: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int,
    seed: int,
    system_col: str = "system_uid",
) -> Dict[str, float]:
    """
    Hierarchical bootstrap: systems first, then trajectories within system.
    value_col should contain 0/1 values (NaN rows are omitted).
    """
    data = frame[[system_col, value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if data.empty:
        return {"rate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    system_ids = data[system_col].drop_duplicates().to_numpy()
    grouped = {
        sid: data.loc[data[system_col] == sid, value_col].to_numpy(dtype=float)
        for sid in system_ids
    }
    estimate = float(data[value_col].mean())

    if n_boot <= 0:
        return {"rate": estimate, "ci95_low": np.nan, "ci95_high": np.nan}

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_systems = rng.choice(system_ids, size=len(system_ids), replace=True)
        chunks: List[np.ndarray] = []
        for sid in sampled_systems:
            vals = grouped[sid]
            chunks.append(rng.choice(vals, size=len(vals), replace=True))
        boot[b] = float(np.mean(np.concatenate(chunks)))

    return {
        "rate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def dataframe_to_markdown_no_tabulate(df: pd.DataFrame) -> str:
    """Return a simple GitHub-style Markdown table without optional dependencies."""
    def fmt_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " " )

    headers = [str(c).replace("|", "\\|") for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt_value(v) for v in row) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown_no_tabulate(df))
        f.write("\n")


def annotate_heatmap(ax, arr: np.ndarray, fmt: str) -> None:
    finite = arr[np.isfinite(arr)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            # Let matplotlib choose plot colors; text contrast is only black/white.
            txt_color = "white" if v < midpoint else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", color=txt_color, fontsize=8)


def plot_matrix(
    table: pd.DataFrame,
    value_col: str,
    *,
    output_path: Path,
    title: str,
    cbar_label: str,
    fmt: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    pivot = table.pivot(index="case_name", columns="target", values=value_col)

    spectrum_order = ["strong_gap", "moderate_gap", "weak_gap"]
    excitation_order = [
        "balanced",
        "mild_decreasing",
        "strong_decreasing",
        "increasing",
        "later_modes_strong",
    ]
    desired_rows = [f"{s}__{a}" for s in spectrum_order for a in excitation_order]
    desired_rows = [r for r in desired_rows if r in pivot.index]
    remaining = [r for r in pivot.index if r not in desired_rows]
    pivot = pivot.reindex(desired_rows + sorted(remaining))

    desired_cols = [f"q{i}" for i in range(1, 6)]
    cols = [c for c in desired_cols if c in pivot.columns] + [
        c for c in pivot.columns if c not in desired_cols
    ]
    pivot = pivot[cols]

    arr = pivot.to_numpy(dtype=float)
    fig_h = max(6.0, 0.48 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    image = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Direction")
    ax.set_ylabel("Spectrum / excitation case")
    ax.set_title(title)
    annotate_heatmap(ax, arr, fmt)
    cb = fig.colorbar(image, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Table 1: experimental design
# -----------------------------------------------------------------------------


def load_design(results_dir: Path, config: Dict[str, object]) -> pd.DataFrame:
    # Prefer reconstructing from the matched config because a directory may contain
    # design CSVs from older Experiment-3 runs.
    spectra = config.get("spectrum_cases", {})
    excitations = config.get("excitation_profiles", {})
    if not spectra or not excitations:
        design_path = first_existing(results_dir, ["table1_experiment_design.csv"])
        if design_path is None:
            candidates = sorted(results_dir.glob("table1_experiment_design*.csv"))
            # Prefer a 15-row spectrum x excitation design if multiple files exist.
            for candidate in candidates:
                try:
                    d = pd.read_csv(candidate)
                    if len(d) == 15 and {"spectrum_case", "excitation_case"}.issubset(d.columns):
                        return d
                except Exception:
                    pass
            design_path = candidates[-1] if candidates else None
        if design_path is not None:
            return pd.read_csv(design_path)

    rows: List[Dict[str, object]] = []
    for sname, eigs in spectra.items():
        for aname, amps in excitations.items():
            rows.append({
                "case_name": f"{sname}__{aname}",
                "spectrum_case": sname,
                "leading_eigenvalues": json.dumps(eigs),
                "excitation_case": aname,
                "target_excitation_magnitudes": json.dumps(amps),
                "n_systems": config.get("system_replicates", np.nan),
                "initial_states_per_system": config.get("initial_states_per_system", np.nan),
                "trajectories_per_case": config.get("trajectories_per_case", np.nan),
                "window_m": config.get("window", np.nan),
                "steps": config.get("steps", np.nan),
                "stability_threshold_deg": config.get("stability_threshold_deg", np.nan),
                "stability_patience": config.get("stability_patience", np.nan),
                "pc1_energy_threshold": config.get("min_stage_pc1_energy_fraction", np.nan),
                "external_correctness_tolerance_deg": config.get("recovery_tolerance_deg", np.nan),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Table 2: framework stage-wise performance at the latest accepted window
# plus ever-recovery as a time-resolved supplement.
# -----------------------------------------------------------------------------


def build_stagewise_performance(
    events: pd.DataFrame,
    tolerance_deg: float,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    work = events.copy()
    work["ever_accepted"] = safe_bool_series(work["ever_accepted"])
    work["ever_accepted_and_correct"] = safe_bool_series(work["ever_accepted_and_correct"])

    latest_error = pd.to_numeric(work["latest_accepted_error_deg"], errors="coerce")
    work["latest_correct"] = np.where(
        work["ever_accepted"],
        (latest_error <= tolerance_deg).astype(float),
        np.nan,
    )
    work["latest_success"] = (
        work["ever_accepted"] & (latest_error <= tolerance_deg)
    ).astype(float)
    work["latest_false_accept"] = (
        work["ever_accepted"] & (latest_error > tolerance_deg)
    ).astype(float)
    work["acceptance_numeric"] = work["ever_accepted"].astype(float)
    work["ever_recovery_numeric"] = work["ever_accepted_and_correct"].astype(float)

    rows: List[Dict[str, object]] = []
    grouped = work.groupby(
        ["spectrum_case", "excitation_case", "case_name", "stage", "target"],
        sort=True,
        dropna=False,
    )

    for idx, (keys, g) in enumerate(grouped):
        spectrum_case, excitation_case, case_name, stage, target = keys

        acc = rate_ci_hierarchical(g, "acceptance_numeric", n_boot=n_boot, seed=seed + idx * 101 + 1)
        succ = rate_ci_hierarchical(g, "latest_success", n_boot=n_boot, seed=seed + idx * 101 + 2)
        false = rate_ci_hierarchical(g, "latest_false_accept", n_boot=n_boot, seed=seed + idx * 101 + 3)
        ever = rate_ci_hierarchical(g, "ever_recovery_numeric", n_boot=n_boot, seed=seed + idx * 101 + 4)
        accepted = g.loc[g["ever_accepted"]].copy()
        rel = rate_ci_hierarchical(
            accepted,
            "latest_correct",
            n_boot=n_boot,
            seed=seed + idx * 101 + 5,
        )

        rows.append({
            "spectrum_case": spectrum_case,
            "excitation_case": excitation_case,
            "case_name": case_name,
            "stage": int(stage),
            "target": target,
            "n_systems": int(g["system_uid"].nunique()),
            "n_trajectories": int(len(g)),
            "acceptance_rate": acc["rate"],
            "acceptance_ci95_low": acc["ci95_low"],
            "acceptance_ci95_high": acc["ci95_high"],
            "overall_recovery_latest_rate": succ["rate"],
            "overall_recovery_latest_ci95_low": succ["ci95_low"],
            "overall_recovery_latest_ci95_high": succ["ci95_high"],
            "reliability_latest_given_accepted_rate": rel["rate"],
            "reliability_latest_ci95_low": rel["ci95_low"],
            "reliability_latest_ci95_high": rel["ci95_high"],
            "false_acceptance_latest_rate": false["rate"],
            "false_acceptance_latest_ci95_low": false["ci95_low"],
            "false_acceptance_latest_ci95_high": false["ci95_high"],
            "ever_recovery_rate": ever["rate"],
            "ever_recovery_ci95_low": ever["ci95_low"],
            "ever_recovery_ci95_high": ever["ci95_high"],
            "median_latest_accepted_error_deg": med(accepted["latest_accepted_error_deg"]),
            "q25_latest_accepted_error_deg": q(accepted["latest_accepted_error_deg"], 0.25),
            "q75_latest_accepted_error_deg": q(accepted["latest_accepted_error_deg"], 0.75),
        })

    return pd.DataFrame(rows).sort_values(["spectrum_case", "excitation_case", "stage"])


# -----------------------------------------------------------------------------
# Table 3: recovery range and distance to fixed point
# -----------------------------------------------------------------------------


def build_recovery_intervals(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["ever_accepted_and_correct"] = safe_bool_series(work["ever_accepted_and_correct"])

    first = pd.to_numeric(work["first_recovery_window_end"], errors="coerce")
    last = pd.to_numeric(work["last_recovery_window_end"], errors="coerce")
    ncorrect = pd.to_numeric(work["n_accepted_correct_windows"], errors="coerce")
    work["recovery_span_inclusive"] = np.where(
        first.notna() & last.notna(), last - first + 1.0, np.nan
    )
    work["recovery_continuity_fraction"] = np.where(
        work["recovery_span_inclusive"] > 0,
        ncorrect / work["recovery_span_inclusive"],
        np.nan,
    )

    rows: List[Dict[str, object]] = []
    grouped = work.groupby(
        ["spectrum_case", "excitation_case", "case_name", "stage", "target"],
        sort=True,
        dropna=False,
    )
    for keys, g in grouped:
        spectrum_case, excitation_case, case_name, stage, target = keys
        recovered = g.loc[g["ever_accepted_and_correct"]].copy()
        rows.append({
            "spectrum_case": spectrum_case,
            "excitation_case": excitation_case,
            "case_name": case_name,
            "stage": int(stage),
            "target": target,
            "n_trajectories": int(len(g)),
            "n_ever_recovered": int(len(recovered)),
            "ever_recovery_rate": float(len(recovered) / len(g)) if len(g) else np.nan,
            "median_first_recovery_window_end": med(recovered["first_recovery_window_end"]),
            "q25_first_recovery_window_end": q(recovered["first_recovery_window_end"], 0.25),
            "q75_first_recovery_window_end": q(recovered["first_recovery_window_end"], 0.75),
            "median_last_recovery_window_end": med(recovered["last_recovery_window_end"]),
            "q25_last_recovery_window_end": q(recovered["last_recovery_window_end"], 0.25),
            "q75_last_recovery_window_end": q(recovered["last_recovery_window_end"], 0.75),
            "median_recovery_span_inclusive": med(recovered["recovery_span_inclusive"]),
            "median_n_accepted_correct_windows": med(recovered["n_accepted_correct_windows"]),
            "median_recovery_continuity_fraction": med(recovered["recovery_continuity_fraction"]),
            "median_log10_relative_distance_at_first_recovery": med(recovered["first_recovery_log10_relative_distance"]),
            "q25_log10_relative_distance_at_first_recovery": q(recovered["first_recovery_log10_relative_distance"], 0.25),
            "q75_log10_relative_distance_at_first_recovery": q(recovered["first_recovery_log10_relative_distance"], 0.75),
            "median_first_recovery_error_deg": med(recovered["first_recovery_error_deg"]),
        })

    return pd.DataFrame(rows).sort_values(["spectrum_case", "excitation_case", "stage"])


# -----------------------------------------------------------------------------
# Table 4: same-window reference vs estimated deflation and conditional
# propagation at each trajectory's latest accepted stage-i window.
# -----------------------------------------------------------------------------


def latest_condition_rows_from_npz(
    results_dir: Path,
    events: pd.DataFrame,
    tolerance_deg: float,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    case_names = sorted(events["case_name"].dropna().unique())
    uid_to_system = events.drop_duplicates("trajectory_uid").set_index("trajectory_uid")["system_uid"].to_dict()

    for case_name in case_names:
        # Experiment 3 stores per-case traces under results_dir/time_traces/.
        # Keep a fallback to the results root for compatibility with older runs.
        candidate_paths = [
            results_dir / "time_traces" / f"{case_name}.npz",
            results_dir / f"{case_name}.npz",
        ]
        npz_path = next((p for p in candidate_paths if p.exists()), None)

        # Final fallback: search recursively inside this experiment's result folder.
        if npz_path is None:
            matches = list(results_dir.rglob(f"{case_name}.npz"))
            if len(matches) == 1:
                npz_path = matches[0]

        if npz_path is None:
            missing.append(case_name)
            continue
        z = np.load(npz_path, allow_pickle=True)
        accepted = np.asarray(z["accepted"], dtype=bool)           # [traj, stage, window]
        errors = np.asarray(z["errors_deg"], dtype=float)
        accepted_correct = np.asarray(z["accepted_correct"], dtype=bool)
        trajectory_uid = np.asarray(z["trajectory_uid"]).astype(str)
        window_end = np.asarray(z["window_end"], dtype=int)

        n_traj, n_stage, _ = accepted.shape
        for ti in range(n_traj):
            uid = str(trajectory_uid[ti])
            system_uid = uid_to_system.get(uid, "unknown")
            for sidx in range(1, n_stage):  # q2...qk only
                A = accepted[ti, sidx]
                idxs = np.flatnonzero(A)
                if len(idxs) == 0:
                    continue
                wi = int(idxs[-1])
                prior_correct = accepted_correct[ti, :sidx, wi]
                all_prior_correct = bool(np.all(prior_correct))
                current_correct = bool(accepted_correct[ti, sidx, wi])
                rows.append({
                    "case_name": case_name,
                    "trajectory_uid": uid,
                    "system_uid": system_uid,
                    "stage": sidx + 1,
                    "target": f"q{sidx + 1}",
                    "latest_accepted_window_end": int(window_end[wi]),
                    "current_error_deg_same_window": float(errors[ti, sidx, wi]),
                    "current_correct_same_window": float(current_correct),
                    "all_prior_correct_same_window": all_prior_correct,
                    "any_prior_incorrect_same_window": not all_prior_correct,
                })

    return pd.DataFrame(rows), missing


def build_propagation_table(
    events: pd.DataFrame,
    results_dir: Path,
    tolerance_deg: float,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, List[str]]:
    cond, missing_npz = latest_condition_rows_from_npz(results_dir, events, tolerance_deg)

    event_cols = [
        "case_name", "spectrum_case", "excitation_case", "trajectory_uid", "system_uid",
        "stage", "target", "latest_accepted_error_deg",
        "reference_error_deg_at_latest_accept",
        "estimated_minus_reference_penalty_deg_at_latest_accept",
    ]
    base = events[event_cols].copy()
    base = base.loc[base["stage"] >= 2]

    if len(cond):
        merged = base.merge(
            cond[[
                "case_name", "trajectory_uid", "stage",
                "all_prior_correct_same_window", "any_prior_incorrect_same_window",
                "current_correct_same_window",
            ]],
            on=["case_name", "trajectory_uid", "stage"],
            how="left",
        )
    else:
        merged = base.copy()
        merged["all_prior_correct_same_window"] = np.nan
        merged["any_prior_incorrect_same_window"] = np.nan
        merged["current_correct_same_window"] = np.nan

    rows: List[Dict[str, object]] = []
    grouped = merged.groupby(
        ["spectrum_case", "excitation_case", "case_name", "stage", "target"],
        sort=True,
        dropna=False,
    )
    for idx, (keys, g) in enumerate(grouped):
        spectrum_case, excitation_case, case_name, stage, target = keys
        accepted = g.loc[pd.to_numeric(g["latest_accepted_error_deg"], errors="coerce").notna()].copy()

        est_err = pd.to_numeric(accepted["latest_accepted_error_deg"], errors="coerce")
        ref_err = pd.to_numeric(accepted["reference_error_deg_at_latest_accept"], errors="coerce")
        penalty = pd.to_numeric(accepted["estimated_minus_reference_penalty_deg_at_latest_accept"], errors="coerce")

        ref_evaluable = accepted.loc[ref_err.notna()].copy()
        if len(ref_evaluable):
            ref_evaluable["reference_correct"] = (
                pd.to_numeric(ref_evaluable["reference_error_deg_at_latest_accept"], errors="coerce")
                <= tolerance_deg
            ).astype(float)
            ref_correct = rate_ci_hierarchical(
                ref_evaluable, "reference_correct", n_boot=n_boot,
                seed=seed + idx * 313 + 1,
            )
        else:
            ref_correct = {"rate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

        prior_good = accepted.loc[accepted["all_prior_correct_same_window"] == True].copy()
        prior_bad = accepted.loc[accepted["any_prior_incorrect_same_window"] == True].copy()

        if len(prior_good):
            good_rate = rate_ci_hierarchical(
                prior_good, "current_correct_same_window", n_boot=n_boot,
                seed=seed + idx * 313 + 2,
            )
        else:
            good_rate = {"rate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

        if len(prior_bad):
            bad_rate = rate_ci_hierarchical(
                prior_bad, "current_correct_same_window", n_boot=n_boot,
                seed=seed + idx * 313 + 3,
            )
        else:
            bad_rate = {"rate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

        rows.append({
            "spectrum_case": spectrum_case,
            "excitation_case": excitation_case,
            "case_name": case_name,
            "stage": int(stage),
            "target": target,
            "n_latest_accepted": int(len(accepted)),
            "median_estimated_error_deg_latest_accept": float(est_err.median()) if len(est_err.dropna()) else np.nan,
            "median_reference_error_deg_same_window": float(ref_err.median()) if len(ref_err.dropna()) else np.nan,
            "median_estimated_minus_reference_penalty_deg": float(penalty.median()) if len(penalty.dropna()) else np.nan,
            "reference_correct_rate_same_window": ref_correct["rate"],
            "reference_correct_ci95_low": ref_correct["ci95_low"],
            "reference_correct_ci95_high": ref_correct["ci95_high"],
            "n_all_prior_correct": int(len(prior_good)),
            "current_correct_rate_all_prior_correct": good_rate["rate"],
            "current_correct_all_prior_correct_ci95_low": good_rate["ci95_low"],
            "current_correct_all_prior_correct_ci95_high": good_rate["ci95_high"],
            "n_any_prior_incorrect": int(len(prior_bad)),
            "current_correct_rate_any_prior_incorrect": bad_rate["rate"],
            "current_correct_any_prior_incorrect_ci95_low": bad_rate["ci95_low"],
            "current_correct_any_prior_incorrect_ci95_high": bad_rate["ci95_high"],
        })

    return pd.DataFrame(rows).sort_values(["spectrum_case", "excitation_case", "stage"]), missing_npz


# -----------------------------------------------------------------------------
# Figure 2: recovery interval plots (one panel/file per spectrum)
# -----------------------------------------------------------------------------


def plot_recovery_intervals(intervals: pd.DataFrame, output_dir: Path) -> List[Path]:
    produced: List[Path] = []
    excitation_order = [
        "balanced", "mild_decreasing", "strong_decreasing",
        "increasing", "later_modes_strong",
    ]

    for spectrum_case, spec in intervals.groupby("spectrum_case", sort=False):
        fig, ax = plt.subplots(figsize=(11, 8))
        y_positions: List[float] = []
        y_labels: List[str] = []
        y = 0.0

        for excitation in excitation_order:
            gcase = spec.loc[spec["excitation_case"] == excitation].sort_values("stage")
            if gcase.empty:
                continue
            for _, row in gcase.iterrows():
                first = row["median_first_recovery_window_end"]
                last = row["median_last_recovery_window_end"]
                y_positions.append(y)
                recovered = np.isfinite(first) and np.isfinite(last)
                y_labels.append(
                    f"{excitation} / {row['target']}" if recovered
                    else f"{excitation} / {row['target']} (N/A)"
                )
                if recovered:
                    line = ax.plot([first, last], [y, y], linewidth=3)[0]
                    c = line.get_color()
                    ax.plot(first, y, marker="o", color=c, linestyle="None")
                    ax.plot(last, y, marker="s", color=c, linestyle="None")
                y += 1.0
            y += 0.5

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Window-end iteration")
        ax.set_ylabel("Excitation case / direction")
        ax.set_title(
            f"Experiment 3: median recovery ranges — {spectrum_case}\n"
            "circle = first recovery, square = last recovery"
        )
        ax.grid(True, axis="x", alpha=0.25)
        ax.invert_yaxis()
        fig.tight_layout()
        path = output_dir / f"figure2_recovery_intervals__{spectrum_case}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        produced.append(path)

    return produced


# -----------------------------------------------------------------------------
# Config selection
# -----------------------------------------------------------------------------


def load_matching_config(results_dir: Path, n_trajectories: int) -> Tuple[Dict[str, object], Optional[Path]]:
    standard = results_dir / "experiment_config.json"
    candidates = [standard] if standard.exists() else []
    candidates += [p for p in sorted(results_dir.glob("experiment_config*.json")) if p != standard]

    parsed: List[Tuple[Path, Dict[str, object]]] = []
    for p in candidates:
        try:
            with p.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            parsed.append((p, cfg))
        except Exception:
            continue

    # Prefer an exact total-trajectory match and the spectrum/excitation design fields.
    for p, cfg in parsed:
        total = cfg.get("total_trajectories")
        if total is not None and int(total) == int(n_trajectories) and cfg.get("spectrum_cases") and cfg.get("excitation_profiles"):
            return cfg, p

    # Secondary match from n_cases * trajectories_per_case.
    for p, cfg in parsed:
        n_cases = cfg.get("n_cases")
        per_case = cfg.get("trajectories_per_case")
        if n_cases is not None and per_case is not None:
            if int(n_cases) * int(per_case) == int(n_trajectories):
                return cfg, p

    if parsed:
        return parsed[0][1], parsed[0][0]
    return {}, None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the final Framework-3 reporting package from an already-completed "
            "spectrum x excitation x time-resolved Experiment 3 run. This script does "
            "NOT simulate trajectories."
        )
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("results/framework_experiment3_multifilter_propagation_normal"),
        help="Directory containing trajectory_stage_recovery_events.csv and case NPZ traces.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Default: <results-dir>/framework3_final_report",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=None,
        help="Default: use value in experiment_config.json, otherwise 2000.",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "framework3_final_report"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = first_existing(results_dir, ["trajectory_stage_recovery_events.csv"])
    if events_path is None:
        events_path = find_first_glob(results_dir, ["trajectory_stage_recovery_events*.csv"])
    if events_path is None:
        raise FileNotFoundError(
            "Could not find trajectory_stage_recovery_events.csv in the results directory."
        )

    events = pd.read_csv(events_path)
    n_unique_trajectories = int(events["trajectory_uid"].nunique())
    config, config_path = load_matching_config(results_dir, n_unique_trajectories)

    tolerance_deg = float(config.get("recovery_tolerance_deg", 2.5))
    n_boot = int(
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else config.get("bootstrap_replicates", 2000)
    )
    required = {
        "case_name", "spectrum_case", "excitation_case", "system_uid",
        "trajectory_uid", "stage", "target", "ever_accepted",
        "ever_accepted_and_correct", "first_recovery_window_end",
        "last_recovery_window_end", "latest_accepted_error_deg",
        "first_recovery_log10_relative_distance",
        "reference_error_deg_at_latest_accept",
        "estimated_minus_reference_penalty_deg_at_latest_accept",
    }
    missing_cols = sorted(required - set(events.columns))
    if missing_cols:
        raise ValueError(f"Event file is missing required columns: {missing_cols}")

    print("============================================================")
    print("Framework 3 final report generator")
    print("============================================================")
    print("results dir:", results_dir)
    print("events file:", events_path.name)
    print("trajectories represented:", events["trajectory_uid"].nunique())
    print("case-direction rows:", len(events))
    print("external correctness tolerance:", tolerance_deg, "degrees")
    print("bootstrap replicates:", n_boot)
    print("No trajectories will be simulated.")
    print("============================================================")

    # Table 1
    table1 = load_design(results_dir, config)
    save_table(table1, output_dir, "table1_experiment_design")

    # Table 2
    table2 = build_stagewise_performance(
        events, tolerance_deg, n_boot=n_boot, seed=args.seed + 10000
    )
    save_table(table2, output_dir, "table2_stagewise_performance")

    # Table 3
    table3 = build_recovery_intervals(events)
    save_table(table3, output_dir, "table3_recovery_intervals_and_limit_distance")

    # Table 4
    table4, missing_npz = build_propagation_table(
        events, results_dir, tolerance_deg,
        n_boot=n_boot, seed=args.seed + 20000,
    )
    save_table(table4, output_dir, "table4_reference_and_propagation")

    # Figure 1: ever recovery heatmap
    plot_matrix(
        table3,
        "ever_recovery_rate",
        output_path=output_dir / "figure1_ever_recovery_heatmap.png",
        title="Experiment 3: probability of ever recovering each direction",
        cbar_label=r"P(there exists t with A_i(t)=1 and C_i(t)=1)",
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
    )

    # Figure 2a-c: recovery intervals
    interval_figs = plot_recovery_intervals(table3, output_dir)

    # Figure 3: first recovery iteration
    plot_matrix(
        table3,
        "median_first_recovery_window_end",
        output_path=output_dir / "figure3_median_first_recovery_iteration_heatmap.png",
        title="Experiment 3: median first-recovery iteration",
        cbar_label="median window-end iteration",
        fmt=".0f",
    )

    # Figure 4: distance to limit at first recovery
    plot_matrix(
        table3,
        "median_log10_relative_distance_at_first_recovery",
        output_path=output_dir / "figure4_distance_to_limit_at_first_recovery_heatmap.png",
        title="Experiment 3: closeness to the limit at first recovery",
        cbar_label=r"median log10(||x_t-L|| / ||x_0-L||)",
        fmt=".2f",
    )

    manifest = {
        "source_results_dir": str(results_dir),
        "source_event_file": str(events_path),
        "source_config_file": str(config_path) if config_path is not None else None,
        "external_correctness_tolerance_deg": tolerance_deg,
        "bootstrap_replicates": n_boot,
        "tables": [
            "table1_experiment_design.csv",
            "table2_stagewise_performance.csv",
            "table3_recovery_intervals_and_limit_distance.csv",
            "table4_reference_and_propagation.csv",
        ],
        "figures": [
            "figure1_ever_recovery_heatmap.png",
            *[p.name for p in interval_figs],
            "figure3_median_first_recovery_iteration_heatmap.png",
            "figure4_distance_to_limit_at_first_recovery_heatmap.png",
        ],
        "missing_npz_cases_for_conditional_propagation": missing_npz,
        "notes": [
            "Table 2 framework rates use each trajectory's latest accepted window.",
            "Ever-recovery is included separately as a time-resolved supplement.",
            "Recovery interval [first,last] is a span and may contain gaps; see continuity fraction.",
            "Reference errors use the same observation-only selected event window.",
            "No simulation is performed by this reporting script.",
        ],
    }
    with (output_dir / "report_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nGenerated Tables 1-4 and Figures 1-4 in:")
    print(output_dir)
    if missing_npz:
        print("\nWARNING: NPZ traces were missing for these cases:")
        for c in missing_npz:
            print("  -", c)
        print("Table 4 reference-error columns are still available from the event CSV, but")
        print("conditional prior-correct/prior-incorrect rates are blank for those cases.")
    else:
        print("All case NPZ traces found; conditional propagation table is complete.")

    print("\nMain files:")
    for name in manifest["tables"]:
        print("  ", name)
    for name in manifest["figures"]:
        print("  ", name)


if __name__ == "__main__":
    main()
