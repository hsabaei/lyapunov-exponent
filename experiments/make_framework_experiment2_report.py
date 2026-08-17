from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def hierarchical_bootstrap_rate(df, value_col, n_boot=2000, seed=42):
    work = df[["system_uid", value_col]].dropna().copy()
    if work.empty:
        return np.nan, np.nan, np.nan

    estimate = float(work[value_col].astype(float).mean())
    systems = work["system_uid"].drop_duplicates().to_numpy()
    grouped = {
        s: work.loc[work["system_uid"] == s, value_col].astype(float).to_numpy()
        for s in systems
    }

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_systems = rng.choice(systems, size=len(systems), replace=True)
        pieces = []
        for s in sampled_systems:
            x = grouped[s]
            pieces.append(rng.choice(x, size=len(x), replace=True))
        vals[b] = np.concatenate(pieces).mean()

    return estimate, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def build_reanalysis(df, threshold):
    out = df.copy()
    A2 = out["A2_accepted"].astype(bool)
    alpha0 = np.isclose(out["excitation_ratio_abs_a2_over_a1"].to_numpy(float), 0.0)

    # q2 is not identifiable at alpha=0. Any accepted stage-2 direction is
    # therefore a target-specific false acceptance for fixed-q2 validation.
    c2 = np.full(len(out), np.nan)
    accepted_idx = np.where(A2.to_numpy())[0]
    for i in accepted_idx:
        if alpha0[i]:
            c2[i] = 0.0
        else:
            err = float(out.iloc[i]["q2_error_deg_estimated_sequence"])
            c2[i] = float(np.isfinite(err) and err <= threshold)

    c1_stage2 = np.full(len(out), np.nan)
    for i in accepted_idx:
        err = float(out.iloc[i]["q1_error_deg_at_stage2_window"])
        c1_stage2[i] = float(np.isfinite(err) and err <= threshold)

    out["C2_correct_reanalysis"] = c2
    out["C1_at_stage2_reanalysis"] = c1_stage2
    out["A2_and_C2_reanalysis"] = A2.to_numpy() & (np.nan_to_num(c2, nan=0.0) == 1.0)
    out["A2_and_not_C2_reanalysis"] = A2.to_numpy() & (np.nan_to_num(c2, nan=0.0) == 0.0)
    return out


def cell_summary(df, threshold, n_boot, seed):
    rows = []
    group_cols = [
        "lambda2",
        "spectral_ratio_g_abs_lambda2_over_lambda1",
        "excitation_ratio_abs_a2_over_a1",
    ]

    for k, (keys, gdf) in enumerate(df.groupby(group_cols, sort=True)):
        l2, gr, alpha = keys
        temp = gdf.copy()
        temp["acceptance_value"] = temp["A2_accepted"].astype(float)
        temp["success_value"] = temp["A2_and_C2_reanalysis"].astype(float)
        temp["false_accept_value"] = temp["A2_and_not_C2_reanalysis"].astype(float)

        accepted = temp[temp["A2_accepted"].astype(bool)].copy()
        accepted["reliability_value"] = pd.to_numeric(
            accepted["C2_correct_reanalysis"], errors="coerce"
        )

        acc = hierarchical_bootstrap_rate(temp, "acceptance_value", n_boot, seed + 1000*k)
        suc = hierarchical_bootstrap_rate(temp, "success_value", n_boot, seed + 1000*k + 1)
        fal = hierarchical_bootstrap_rate(temp, "false_accept_value", n_boot, seed + 1000*k + 2)
        rel = hierarchical_bootstrap_rate(accepted, "reliability_value", n_boot, seed + 1000*k + 3)

        rows.append({
            "lambda2": l2,
            "g_abs_lambda2_over_lambda1": gr,
            "alpha_abs_a2_over_a1": alpha,
            "n_systems": int(gdf["system_uid"].nunique()),
            "n_trajectories": int(len(gdf)),
            "acceptance_rate": acc[0],
            "acceptance_ci95_low": acc[1],
            "acceptance_ci95_high": acc[2],
            "successful_recovery_rate": suc[0],
            "successful_recovery_ci95_low": suc[1],
            "successful_recovery_ci95_high": suc[2],
            "reliability_given_accepted": rel[0],
            "reliability_ci95_low": rel[1],
            "reliability_ci95_high": rel[2],
            "false_acceptance_rate": fal[0],
            "false_acceptance_ci95_low": fal[1],
            "false_acceptance_ci95_high": fal[2],
            "median_q2_error_deg_accepted": (
                float(accepted["q2_error_deg_estimated_sequence"].median())
                if len(accepted) else np.nan
            ),
            "q25_q2_error_deg_accepted": (
                float(accepted["q2_error_deg_estimated_sequence"].quantile(.25))
                if len(accepted) else np.nan
            ),
            "q75_q2_error_deg_accepted": (
                float(accepted["q2_error_deg_estimated_sequence"].quantile(.75))
                if len(accepted) else np.nan
            ),
            "median_deflation_penalty_deg": (
                float(accepted["q2_error_penalty_estimated_minus_oracle_deg"].median())
                if len(accepted) else np.nan
            ),
        })
    return pd.DataFrame(rows).sort_values(["g_abs_lambda2_over_lambda1", "alpha_abs_a2_over_a1"])


def overall_table(df, threshold, n_boot, seed):
    work = df.copy()
    work["acceptance_value"] = work["A2_accepted"].astype(float)
    work["success_value"] = work["A2_and_C2_reanalysis"].astype(float)
    work["false_accept_value"] = work["A2_and_not_C2_reanalysis"].astype(float)
    accepted = work[work["A2_accepted"].astype(bool)].copy()
    accepted["reliability_value"] = pd.to_numeric(
        accepted["C2_correct_reanalysis"], errors="coerce"
    )

    acc = hierarchical_bootstrap_rate(work, "acceptance_value", n_boot, seed+10)
    suc = hierarchical_bootstrap_rate(work, "success_value", n_boot, seed+20)
    rel = hierarchical_bootstrap_rate(accepted, "reliability_value", n_boot, seed+30)
    fal = hierarchical_bootstrap_rate(work, "false_accept_value", n_boot, seed+40)

    zero = work[np.isclose(work["excitation_ratio_abs_a2_over_a1"], 0.0)].copy()
    zero["zero_false_accept"] = zero["A2_accepted"].astype(float)
    zfa = hierarchical_bootstrap_rate(zero, "zero_false_accept", n_boot, seed+50)

    rows = [
        ("Stage-2 acceptance", acc),
        (f"Successful q2 recovery (<= {threshold:g} deg)", suc),
        (f"Reliability given acceptance (<= {threshold:g} deg)", rel),
        ("Target-specific false acceptance", fal),
        ("alpha=0 target-specific false acceptance", zfa),
    ]

    result = pd.DataFrame([
        {
            "quantity": name,
            "estimate": vals[0],
            "ci95_low": vals[1],
            "ci95_high": vals[2],
        }
        for name, vals in rows
    ])
    return result


def propagation_table(df, threshold, n_boot, seed):
    # alpha=0 excluded: q2 target is absent there.
    pos = df[
        (df["A2_accepted"].astype(bool))
        & (df["excitation_ratio_abs_a2_over_a1"] > 0)
    ].copy()
    pos["stage2_correct_value"] = pd.to_numeric(
        pos["C2_correct_reanalysis"], errors="coerce"
    )

    rows = []
    for label, val in [("preceding_q1_correct", 1.0), ("preceding_q1_incorrect", 0.0)]:
        sub = pos[pos["C1_at_stage2_reanalysis"] == val].copy()
        if len(sub) == 0:
            rows.append({
                "preceding_stage_condition": label,
                "n_trajectories": 0,
                "n_systems": 0,
                "stage2_correct_rate": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
            })
        else:
            est = hierarchical_bootstrap_rate(
                sub, "stage2_correct_value", n_boot, seed + (100 if val == 1.0 else 200)
            )
            rows.append({
                "preceding_stage_condition": label,
                "n_trajectories": int(len(sub)),
                "n_systems": int(sub["system_uid"].nunique()),
                "stage2_correct_rate": est[0],
                "ci95_low": est[1],
                "ci95_high": est[2],
            })
    return pd.DataFrame(rows)


def reference_deflation_table(df, threshold):
    pos = df[
        df["A2_accepted"].astype(bool)
        & (df["excitation_ratio_abs_a2_over_a1"] > 0)
    ].copy()

    if pos.empty:
        return pd.DataFrame()

    cell = (
        pos.groupby(
            [
                "spectral_ratio_g_abs_lambda2_over_lambda1",
                "excitation_ratio_abs_a2_over_a1",
            ],
            as_index=False,
        )
        .agg(
            median_estimated_q2_error_deg=("q2_error_deg_estimated_sequence", "median"),
            median_reference_q2_error_deg=("oracle_q2_error_deg_same_window", "median"),
            median_added_error_deg=("q2_error_penalty_estimated_minus_oracle_deg", "median"),
        )
    )
    worst = cell.loc[cell["median_added_error_deg"].idxmax()]

    return pd.DataFrame([{
        "n_accepted_positive_alpha": int(len(pos)),
        "estimated_q2_correct_rate": float(
            (pos["q2_error_deg_estimated_sequence"] <= threshold).mean()
        ),
        "reference_deflation_q2_correct_rate": float(
            (pos["oracle_q2_error_deg_same_window"] <= threshold).mean()
        ),
        "median_estimated_q2_error_deg": float(
            pos["q2_error_deg_estimated_sequence"].median()
        ),
        "q25_estimated_q2_error_deg": float(
            pos["q2_error_deg_estimated_sequence"].quantile(0.25)
        ),
        "q75_estimated_q2_error_deg": float(
            pos["q2_error_deg_estimated_sequence"].quantile(0.75)
        ),
        "median_reference_q2_error_deg": float(
            pos["oracle_q2_error_deg_same_window"].median()
        ),
        "median_added_error_from_estimated_q1_deflation_deg": float(
            pos["q2_error_penalty_estimated_minus_oracle_deg"].median()
        ),
        "worst_cell_g": float(
            worst["spectral_ratio_g_abs_lambda2_over_lambda1"]
        ),
        "worst_cell_alpha": float(
            worst["excitation_ratio_abs_a2_over_a1"]
        ),
        "worst_cell_median_added_error_deg": float(
            worst["median_added_error_deg"]
        ),
        "external_reporting_tolerance_deg": float(threshold),
    }])



def plot_lines(summary, ycol, ylabel, title, output, include_alpha0=True, ylim=None):
    data = summary.copy()
    if not include_alpha0:
        data = data[data["alpha_abs_a2_over_a1"] > 0]

    alphas = sorted(data["alpha_abs_a2_over_a1"].unique())
    x = np.arange(len(alphas))

    fig, ax = plt.subplots(figsize=(9, 6))
    for gr, group in data.groupby("g_abs_lambda2_over_lambda1", sort=True):
        group = group.set_index("alpha_abs_a2_over_a1").reindex(alphas)
        ax.plot(x, group[ycol], marker="o", label=f"g={gr:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"Excitation $\alpha=|a_2/a_1|$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_q2_error(summary, threshold, output):
    data = summary[summary["alpha_abs_a2_over_a1"] > 0].copy()
    alphas = sorted(data["alpha_abs_a2_over_a1"].unique())
    x = np.arange(len(alphas))

    fig, ax = plt.subplots(figsize=(9, 6))
    for gr, group in data.groupby("g_abs_lambda2_over_lambda1", sort=True):
        group = group.set_index("alpha_abs_a2_over_a1").reindex(alphas)
        y = np.maximum(group["median_q2_error_deg_accepted"].to_numpy(float), 1e-8)
        ax.plot(x, y, marker="o", label=f"g={gr:.3f}")

    ax.axhline(threshold, linestyle="--", label=f"{threshold:g} deg threshold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"Excitation $\alpha=|a_2/a_1|$")
    ax.set_ylabel(r"Median accepted $q_2$ angular error (degrees)")
    ax.set_title("Experiment 2: continuous q2 error versus excitation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_deflation_penalty(summary, output):
    data = summary[summary["alpha_abs_a2_over_a1"] > 0].copy()
    alphas = sorted(data["alpha_abs_a2_over_a1"].unique())
    x = np.arange(len(alphas))

    fig, ax = plt.subplots(figsize=(9, 6))
    for gr, group in data.groupby("g_abs_lambda2_over_lambda1", sort=True):
        group = group.set_index("alpha_abs_a2_over_a1").reindex(alphas)
        ax.plot(
            x,
            group["median_deflation_penalty_deg"],
            marker="o",
            label=f"g={gr:.3f}",
        )

    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"Excitation $\alpha=|a_2/a_1|$")
    ax.set_ylabel(
        r"Median $\theta_{2,\mathrm{est}}-\theta_{2,\mathrm{ref}}$ (degrees)"
    )
    ax.set_title("Experiment 2: error added by estimated q1 deflation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_alpha0_negative_control(summary, output):
    data = summary[np.isclose(summary["alpha_abs_a2_over_a1"], 0.0)].sort_values(
        "g_abs_lambda2_over_lambda1"
    )
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, data["acceptance_rate"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g:.3f}" for g in data["g_abs_lambda2_over_lambda1"]])
    ax.set_xlabel(r"Spectral ratio $g=|\lambda_2|/|\lambda_1|$")
    ax.set_ylabel(r"$P(A_2=1\mid \alpha=0)$")
    ax.set_ylim(0, 1)
    ax.set_title("Experiment 2: unexcited-q2 negative control")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_markdown(df, path, title):
    """Write a simple Markdown table without requiring pandas/tabulate."""
    display = df.copy()

    def format_value(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    columns = [str(c) for c in display.columns]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")

        for _, row in display.iterrows():
            values = []
            for col in display.columns:
                value = format_value(row[col])
                value = value.replace("|", "\\|").replace("\n", " ")
                values.append(value)
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Framework Experiment 2 at a chosen external angle threshold."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/framework_experiment2_two_filter_normal"),
    )
    parser.add_argument("--threshold-deg", type=float, default=2.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = args.results_dir
    input_csv = results_dir / "all_trials.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing {input_csv}")

    outdir = results_dir / f"report_{args.threshold_deg:g}deg"
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)

    config = {}
    config_path = results_dir / "experiment_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

    data = build_reanalysis(raw, args.threshold_deg)
    summary = cell_summary(
        data, args.threshold_deg, args.bootstrap_replicates, args.seed
    )
    overall = overall_table(
        data, args.threshold_deg, args.bootstrap_replicates, args.seed
    )
    propagation = propagation_table(
        data, args.threshold_deg, args.bootstrap_replicates, args.seed
    )
    reference = reference_deflation_table(data, args.threshold_deg)

    design = pd.DataFrame([{
        "dimension": int(config.get("dim", 20)),
        "lambda1": float(raw["lambda1"].iloc[0]),
        "lambda2_values": ", ".join(f"{x:g}" for x in sorted(raw["lambda2"].unique(), reverse=True)),
        "g_values": ", ".join(f"{x:.3f}" for x in sorted(raw["spectral_ratio_g_abs_lambda2_over_lambda1"].unique(), reverse=True)),
        "alpha_values": ", ".join(f"{x:g}" for x in sorted(raw["excitation_ratio_abs_a2_over_a1"].unique())),
        "systems_per_g": int(
            raw.groupby("lambda2")["system_uid"].nunique().median()
        ),
        "n_systems_total": int(raw["system_uid"].nunique()),
        "initial_states_per_system_per_alpha": int(
            raw.groupby(["system_uid", "excitation_ratio_abs_a2_over_a1"]).size().median()
        ),
        "total_trajectories": int(len(raw)),
        "external_reporting_tolerance_deg": args.threshold_deg,
        "note": "2.5 deg is post-processing only; observation-only acceptance is unchanged.",
    }])

    data.to_csv(outdir / "all_trials_reanalysed.csv", index=False)
    summary.to_csv(outdir / "summary_by_g_and_alpha_2p5deg.csv", index=False)
    overall.to_csv(outdir / "table2_main_results.csv", index=False)
    reference.to_csv(outdir / "table3_reference_vs_estimated_deflation.csv", index=False)
    propagation.to_csv(outdir / "supplementary_thresholded_propagation.csv", index=False)
    design.to_csv(outdir / "table1_experimental_design.csv", index=False)

    write_markdown(design, outdir / "table1_experimental_design.md", "Experiment 2 - Experimental design")
    write_markdown(overall, outdir / "table2_main_results.md", "Experiment 2 - Main results")
    write_markdown(
        reference,
        outdir / "table3_reference_vs_estimated_deflation.md",
        "Experiment 2 - Reference versus estimated q1 deflation",
    )
    write_markdown(
        propagation,
        outdir / "supplementary_thresholded_propagation.md",
        "Experiment 2 - Thresholded propagation diagnostic",
    )

    plot_lines(
        summary,
        "successful_recovery_rate",
        rf"$P(A_2=1,C_2=1)$ at {args.threshold_deg:g} deg",
        f"Experiment 2: successful q2 recovery ({args.threshold_deg:g} deg criterion)",
        outdir / "01_successful_q2_recovery_vs_excitation.png",
        include_alpha0=True,
        ylim=(-0.02, 1.02),
    )
    plot_lines(
        summary,
        "acceptance_rate",
        r"$P(A_2=1)$",
        "Experiment 2: stage-2 acceptance versus excitation",
        outdir / "02_stage2_acceptance_vs_excitation.png",
        include_alpha0=True,
        ylim=(-0.02, 1.02),
    )
    plot_q2_error(
        summary,
        args.threshold_deg,
        outdir / "03_q2_angular_error_vs_excitation.png",
    )
    plot_deflation_penalty(
        summary,
        outdir / "04_estimated_q1_deflation_penalty_vs_excitation.png",
    )
    plot_alpha0_negative_control(
        summary,
        outdir / "05_alpha0_negative_control.png",
    )

    print("\n=== Main results at %.2f degrees ===" % args.threshold_deg)
    print(overall.to_string(index=False))
    print("\n=== Reference versus estimated q1 deflation ===")
    print(reference.to_string(index=False))
    print("\n=== Thresholded propagation diagnostic ===")
    print(propagation.to_string(index=False))
    print("\nReport written to:", outdir.resolve())


if __name__ == "__main__":
    main()
