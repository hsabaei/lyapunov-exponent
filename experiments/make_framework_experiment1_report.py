from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


UNIQUE_CASES = ("strong_gap", "weak_gap")
CASES = ("strong_gap", "weak_gap", "equal_magnitude", "rotation")

CASE_LABELS = {
    "strong_gap": "Strong gap",
    "weak_gap": "Weak gap",
    "equal_magnitude": "Equal magnitude",
    "rotation": "Contracting rotation",
}

TARGET_LABELS = {
    "strong_gap": "Unique q1",
    "weak_gap": "Unique q1",
    "equal_magnitude": "No unique 1-D target",
    "rotation": "No unique 1-D target",
}


def load_config(results_dir: Path) -> dict:
    path = results_dir / "experiment_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_window_length(all_trials: pd.DataFrame) -> float:
    good = all_trials[["selected_window_start", "selected_window_end"]].dropna()
    if good.empty:
        return np.nan
    lengths = good["selected_window_end"] - good["selected_window_start"] + 1
    return float(lengths.median())


def build_design_table(
    all_trials: pd.DataFrame,
    systems: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    rows = []
    window_m = cfg.get("window", infer_window_length(all_trials))
    for case in CASES:
        sg = systems.loc[systems["case"] == case]
        tg = all_trials.loc[all_trials["case"] == case]
        if sg.empty or tg.empty:
            continue

        if case == "rotation":
            lam1 = cfg.get("lambda1", float(sg["lambda1_abs"].iloc[0]))
            angle = cfg.get("rotation_angle_deg", np.nan)
            second = (
                f"{lam1:.2f} R({angle:.0f} deg)"
                if np.isfinite(angle)
                else "contracting rotation"
            )
        elif case == "equal_magnitude":
            lam1 = cfg.get("lambda1", float(sg["lambda1_abs"].iloc[0]))
            second = f"{-abs(float(lam1)):.2f}"
        else:
            second = f"{float(sg['lambda2_abs'].iloc[0]):.2f}"

        counts = tg.groupby("system_replicate").size()
        initial_states_per_system = (
            int(counts.iloc[0]) if len(counts) and counts.nunique() == 1 else np.nan
        )

        rows.append(
            {
                "case": CASE_LABELS[case],
                "target": TARGET_LABELS[case],
                "lambda1_abs": float(sg["lambda1_abs"].iloc[0]),
                "second_leading_structure": second,
                "abs_lambda2_over_lambda1": float(
                    sg["spectral_ratio_abs_lambda2_over_lambda1"].iloc[0]
                ),
                "n_systems": int(sg["system_replicate"].nunique()),
                "initial_states_per_system": initial_states_per_system,
                "total_trajectories": int(len(tg)),
                "window_m": window_m,
                "stability_threshold_deg": cfg.get("stability_threshold_deg", np.nan),
                "stability_patience": cfg.get("stability_patience", np.nan),
                "pc1_energy_threshold": cfg.get(
                    "min_stage_pc1_energy_fraction", np.nan
                ),
                "external_tolerance_deg": cfg.get(
                    "recovery_tolerance_deg",
                    float(tg["recovery_tolerance_deg"].dropna().iloc[0])
                    if tg["recovery_tolerance_deg"].notna().any()
                    else np.nan,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_results_table(
    all_trials: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for case in CASES:
        sg = summary.loc[summary["case"] == case]
        tg = all_trials.loc[all_trials["case"] == case]
        if sg.empty or tg.empty:
            continue
        row = sg.iloc[0]

        n_total = len(tg)
        n_accepted = int(tg["accepted_stage_1"].astype(bool).sum())
        accepted_text = f"{n_accepted}/{n_total} ({100*n_accepted/n_total:.1f}%)"

        if case in UNIQUE_CASES:
            n_success = int(tg["accepted_and_correct_stage_1"].astype(bool).sum())
            n_false = int(tg["false_acceptance_stage_1"].astype(bool).sum())
            successful = f"{n_success}/{n_total} ({100*n_success/n_total:.1f}%)"
            reliability = (
                f"{100*n_success/n_accepted:.1f}%" if n_accepted else "N/A"
            )
            false_accept = f"{n_false}/{n_total} ({100*n_false/n_total:.1f}%)"
            errors = tg.loc[
                tg["accepted_stage_1"], "qhat1_vs_reference_q1_deg"
            ].dropna()
            if len(errors):
                error_text = (
                    f"{errors.median():.6f} "
                    f"[{errors.quantile(0.25):.6f}, {errors.quantile(0.75):.6f}]"
                )
            else:
                error_text = "N/A"
        else:
            successful = "N/A"
            reliability = "N/A"
            n_false_unique = int(
                tg["false_unique_direction_acceptance"].astype(bool).sum()
            )
            false_accept = (
                f"{n_false_unique}/{n_total} "
                f"({100*n_false_unique/n_total:.1f}%)"
            )
            error_text = "N/A"

        if "first_acceptable_window_end" in tg.columns:
            first_vals = tg.loc[
                tg["accepted_stage_1"], "first_acceptable_window_end"
            ].dropna()
            first_text = (
                f"{first_vals.median():.1f} "
                f"[{first_vals.quantile(0.25):.1f}, {first_vals.quantile(0.75):.1f}]"
                if len(first_vals)
                else "N/A"
            )
        else:
            first_text = "not saved in original run"

        rows.append(
            {
                "case": CASE_LABELS[case],
                "accepted": accepted_text,
                "overall_successful_recovery": successful,
                "reliability_given_accepted": reliability,
                "false_acceptance_or_false_unique_acceptance": false_accept,
                "median_q1_error_deg_IQR": error_text,
                "first_acceptable_window_end_median_IQR": first_text,
            }
        )
    return pd.DataFrame(rows)


def save_markdown_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    def cell(v) -> str:
        if pd.isna(v):
            return "N/A"
        return str(v).replace("|", "\\|")

    headers = [cell(c) for c in frame.columns]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_main_outcomes(all_trials: pd.DataFrame, output_path: Path) -> None:
    labels = []
    values = []
    for case in CASES:
        tg = all_trials.loc[all_trials["case"] == case]
        if tg.empty:
            continue
        if case in UNIQUE_CASES:
            value = float(tg["accepted_and_correct_stage_1"].astype(bool).mean())
            suffix = "successful recovery"
        else:
            value = float(
                tg["false_unique_direction_acceptance"].astype(bool).mean()
            )
            suffix = "false unique acceptance"
        labels.append(f"{CASE_LABELS[case]}\n{suffix}")
        values.append(100.0 * value)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Percentage of trajectories")
    ax.set_title("Experiment 1: recovery and non-uniqueness controls")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_q1_errors(all_trials: pd.DataFrame, output_path: Path) -> None:
    data = []
    labels = []
    for case in UNIQUE_CASES:
        values = all_trials.loc[
            (all_trials["case"] == case) & all_trials["accepted_stage_1"],
            "qhat1_vs_reference_q1_deg",
        ].dropna().to_numpy(dtype=float)
        if len(values):
            data.append(values)
            labels.append(CASE_LABELS[case])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(1.0, linestyle="--", label="1 degree tolerance")
    ax.set_ylabel("Sign-invariant q1 angle error (degrees)")
    ax.set_title("Experiment 1: q1 error among accepted estimates")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_first_acceptable(all_trials: pd.DataFrame, output_path: Path) -> bool:
    if "first_acceptable_window_end" not in all_trials.columns:
        return False

    data = []
    labels = []
    for case in UNIQUE_CASES:
        values = all_trials.loc[
            (all_trials["case"] == case) & all_trials["accepted_stage_1"],
            "first_acceptable_window_end",
        ].dropna().to_numpy(dtype=float)
        if len(values):
            data.append(values)
            labels.append(CASE_LABELS[case])

    if not data:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("First acceptable window end")
    ax.set_title("Experiment 1: time to first internally acceptable estimate")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the final Experiment 1 report from an existing result directory."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/framework_experiment1_single_filter"),
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    all_trials_path = results_dir / "all_trials.csv"
    summary_path = results_dir / "summary.csv"
    systems_path = results_dir / "systems.csv"

    for path in (all_trials_path, summary_path, systems_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    all_trials = pd.read_csv(all_trials_path)
    summary = pd.read_csv(summary_path)
    systems = pd.read_csv(systems_path)
    cfg = load_config(results_dir)

    table1 = build_design_table(all_trials, systems, cfg)
    table2 = build_results_table(all_trials, summary)

    table1.to_csv(results_dir / "table1_experiment_design.csv", index=False)
    table2.to_csv(results_dir / "table2_main_results.csv", index=False)
    save_markdown_table(
        table1,
        results_dir / "table1_experiment_design.md",
        "Experiment 1 - Experimental design",
    )
    save_markdown_table(
        table2,
        results_dir / "table2_main_results.md",
        "Experiment 1 - Main results",
    )

    plot_main_outcomes(all_trials, results_dir / "01_single_filter_outcomes_report.png")
    plot_q1_errors(all_trials, results_dir / "02_q1_error_distribution_report.png")
    made_first = plot_first_acceptable(
        all_trials, results_dir / "03_first_acceptable_window.png"
    )

    print("\n=== Table 1: Experimental design ===")
    print(table1.to_string(index=False))
    print("\n=== Table 2: Main results ===")
    print(table2.to_string(index=False))
    if made_first:
        print("\nCreated 03_first_acceptable_window.png")
    else:
        print(
            "\nNOTE: 03_first_acceptable_window.png was not created because "
            "the original run did not save first_acceptable_window_end. "
            "Use the report-ready Experiment 1 script for one rerun."
        )
    print(f"\nReport written to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
