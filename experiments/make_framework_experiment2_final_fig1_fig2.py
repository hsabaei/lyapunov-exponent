from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _asymmetric_yerr(group: pd.DataFrame, estimate: str, low: str, high: str):
    y = group[estimate].to_numpy(dtype=float)
    lo = group[low].to_numpy(dtype=float)
    hi = group[high].to_numpy(dtype=float)
    return np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])


def _prepare(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "g_abs_lambda2_over_lambda1",
        "alpha_abs_a2_over_a1",
        "successful_recovery_rate",
        "successful_recovery_ci95_low",
        "successful_recovery_ci95_high",
        "acceptance_rate",
        "acceptance_ci95_low",
        "acceptance_ci95_high",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    # alpha=0 is an unexcited-target negative control and is reported separately
    # in Figure 5. Main recovery/acceptance trends use identifiable q2 cases only.
    out = summary.loc[summary["alpha_abs_a2_over_a1"] > 0].copy()
    return out.sort_values(
        ["g_abs_lambda2_over_lambda1", "alpha_abs_a2_over_a1"]
    )


def plot_final(
    data: pd.DataFrame,
    *,
    estimate_col: str,
    low_col: str,
    high_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    alphas = sorted(data["alpha_abs_a2_over_a1"].unique())
    x = np.arange(len(alphas), dtype=float)

    # These three curves are exactly identical for every positive-alpha point
    # in the current Experiment-2 results, including their bootstrap CIs.
    combined_g = (0.9583333333333334, 0.9791666666666666, 0.9895833333333334)
    single_g = (0.9166666666666667, 0.9375)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    for g in single_g:
        group = data.loc[
            np.isclose(data["g_abs_lambda2_over_lambda1"], g)
        ].set_index("alpha_abs_a2_over_a1").reindex(alphas).reset_index()

        y = group[estimate_col].to_numpy(dtype=float)
        yerr = _asymmetric_yerr(group, estimate_col, low_col, high_col)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2,
            label=f"g={g:.3f}",
        )

    # Verify that the three advertised curves really coincide before combining.
    members = []
    for g in combined_g:
        member = data.loc[
            np.isclose(data["g_abs_lambda2_over_lambda1"], g)
        ].set_index("alpha_abs_a2_over_a1").reindex(alphas).reset_index()
        members.append(member)

    for col in (estimate_col, low_col, high_col):
        reference = members[0][col].to_numpy(dtype=float)
        for member in members[1:]:
            if not np.allclose(
                reference,
                member[col].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise RuntimeError(
                    "The curves g=0.958, 0.979, 0.990 are no longer identical "
                    f"for {col}; refusing to combine them."
                )

    group = members[0]
    y = group[estimate_col].to_numpy(dtype=float)
    yerr = _asymmetric_yerr(group, estimate_col, low_col, high_col)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="s",
        capsize=4,
        linewidth=2.4,
        label="g=0.958, 0.979, 0.990 (identical)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r"Excitation $\alpha=|a_2/a_1|$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()

    # A concise note prevents the combined curve from being mistaken for omitted data.
    ax.text(
        0.99,
        0.03,
        r"$\alpha=0$ negative control reported separately",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create final Experiment-2 Figures 1 and 2 with hierarchical "
            "95% CIs and combined exactly coincident curves."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/framework_experiment2_two_filter_normal/report_2.5deg"),
    )
    args = parser.parse_args()

    summary_path = args.results_dir / "summary_by_g_and_alpha_2p5deg.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    summary = pd.read_csv(summary_path)
    data = _prepare(summary)

    plot_final(
        data,
        estimate_col="successful_recovery_rate",
        low_col="successful_recovery_ci95_low",
        high_col="successful_recovery_ci95_high",
        ylabel=r"$P(A_2=1,C_2=1)$",
        title=r"Experiment 2: successful $q_2$ recovery ($2.5^\circ$ criterion)",
        output_path=args.results_dir / "01_successful_q2_recovery_vs_excitation_FINAL.png",
    )

    plot_final(
        data,
        estimate_col="acceptance_rate",
        low_col="acceptance_ci95_low",
        high_col="acceptance_ci95_high",
        ylabel=r"$P(A_2=1)$",
        title="Experiment 2: stage-2 acceptance versus excitation",
        output_path=args.results_dir / "02_stage2_acceptance_vs_excitation_FINAL.png",
    )

    print("Created:")
    print(args.results_dir / "01_successful_q2_recovery_vs_excitation_FINAL.png")
    print(args.results_dir / "02_stage2_acceptance_vs_excitation_FINAL.png")
    print("\nCurves g=0.958, 0.979, and 0.990 were combined only after exact numerical verification.")
    print("alpha=0 is intentionally omitted here because it is reported separately as the unexcited-q2 negative control.")


if __name__ == "__main__":
    main()
