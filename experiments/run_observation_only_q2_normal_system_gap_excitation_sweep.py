from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Search both the script directory and the repository root. This supports:
#   repo/experiments/this_script.py
#   repo/run_observation_only_same_window_deflation_normal.py
# as well as both files being in experiments/.
for candidate_dir in (SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "experiments"):
    candidate_text = str(candidate_dir)
    if candidate_text not in sys.path:
        sys.path.insert(0, candidate_text)

try:
    from run_observation_only_same_window_deflation_normal import (
        Config as EstimatorConfig,
        angle_deg,
        estimate_from_observations_only,
        max_principal_angle_deg,
        simulate_trajectory,
    )
except ImportError as exc:
    searched = [
        SCRIPT_DIR
        / "run_observation_only_same_window_deflation_normal.py",
        REPO_ROOT
        / "run_observation_only_same_window_deflation_normal.py",
        REPO_ROOT
        / "experiments"
        / "run_observation_only_same_window_deflation_normal.py",
    ]
    searched_text = "\n".join(f"  - {path}" for path in searched)
    raise ImportError(
        "Could not import "
        "'run_observation_only_same_window_deflation_normal'.\n"
        "Place that file at one of these locations:\n"
        f"{searched_text}"
    ) from exc


@dataclass(frozen=True)
class SweepConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20

    lambda1: float = 0.96
    lambda2_values: tuple[float, ...] = (
        0.95,
        0.94,
        0.92,
        -0.92,
        0.90,
        0.88,
        -0.88,
        0.86,
    )

    # Controlled magnitude |a2/a1|.
    # ratio=0 is a negative control: q2 is absent from the trajectory.
    excitation_ratios: tuple[float, ...] = (
        0.0,
        0.01,
        0.03,
        0.10,
        0.30,
        1.00,
        3.00,
    )

    system_replicates: int = 5
    trials_per_system: int = 20
    seed: int = 42

    # Coefficients of q3 and lower modes.
    other_mode_scale: float = 0.25

    # Tail eigenvalues are random for every system replicate, while remaining
    # strictly below |lambda2|.
    tail_max: float = 0.84
    tail_min: float = 0.20
    tail_gap_below_lambda2: float = 0.02

    # Observation-only estimator settings.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # Ground truth is used only after estimation to define empirical recovery.
    recovery_angle_tolerance_deg: float = 1.0


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip())
        for item in text.split(",")
        if item.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected at least one comma-separated number."
        )
    return values


def validate_config(cfg: SweepConfig) -> None:
    if cfg.dim < 3:
        raise ValueError("dim must be at least 3.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if cfg.window < 2 or cfg.window > cfg.steps + 1:
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if cfg.system_replicates < 1:
        raise ValueError("system_replicates must be positive.")
    if cfg.trials_per_system < 1:
        raise ValueError("trials_per_system must be positive.")
    if not (0.0 < cfg.lambda1 < 1.0):
        raise ValueError("lambda1 must be in (0, 1).")
    if cfg.other_mode_scale < 0.0:
        raise ValueError("other_mode_scale must be nonnegative.")
    if cfg.tail_min <= 0.0:
        raise ValueError("tail_min must be positive.")
    if cfg.tail_max <= cfg.tail_min:
        raise ValueError("tail_max must exceed tail_min.")
    if cfg.tail_gap_below_lambda2 <= 0.0:
        raise ValueError("tail_gap_below_lambda2 must be positive.")
    if cfg.recovery_angle_tolerance_deg <= 0.0:
        raise ValueError("recovery_angle_tolerance_deg must be positive.")

    for lambda2 in cfg.lambda2_values:
        if lambda2 == 0.0:
            raise ValueError("lambda2 must be nonzero.")

        if abs(lambda2) >= abs(cfg.lambda1):
            raise ValueError(
                "Every lambda2 must satisfy "
                "abs(lambda2) < abs(lambda1). "
                f"Received lambda2={lambda2}."
            )
        available_tail_max = min(
            cfg.tail_max,
            abs(lambda2) - cfg.tail_gap_below_lambda2,
        )
        if available_tail_max <= cfg.tail_min:
            raise ValueError(
                "The tail spectrum has no valid interval for "
                f"lambda2={lambda2}. Reduce tail_min or the tail gap."
            )

    for ratio in cfg.excitation_ratios:
        if ratio < 0.0:
            raise ValueError(
                "Excitation ratios represent |a2/a1| and must be nonnegative."
            )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            p * (1.0 - p) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def theoretical_equal_amplitude_time(
    lambda1: float,
    lambda2: float,
    excitation_ratio: float,
) -> float:
    """
    Solve

        |a1| lambda1^t = |a2| lambda2^t,

    with |a1|=1 and |a2|=excitation_ratio.

    A negative value means q1 is already larger at t=0.
    For excitation_ratio=0, equality never occurs.
    """
    if excitation_ratio <= 0.0:
        return np.nan

    return float(
        math.log(excitation_ratio)
        / math.log(abs(lambda1) / abs(lambda2))
    )


def build_random_normal_system(
    cfg: SweepConfig,
    lambda2: float,
    system_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Construct a genuinely different real normal linear system for every
    replicate:

        A = Q diag(lambda_1, lambda_2, lambda_3, ...) Q^T.

    The first two eigenvalues are controlled by the sweep. The remaining
    eigenvalues and the orthogonal eigenbasis are independently randomized.
    """
    rng = np.random.default_rng(system_seed)

    gaussian = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(gaussian)

    available_tail_max = min(
        cfg.tail_max,
        abs(lambda2) - cfg.tail_gap_below_lambda2,
    )

    tail_magnitudes = rng.uniform(
        cfg.tail_min,
        available_tail_max,
        size=cfg.dim - 2,
    )
    tail_magnitudes = np.sort(tail_magnitudes)[::-1]

    tail_signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=cfg.dim - 2,
    )
    tail_eigenvalues = tail_signs * tail_magnitudes

    eigenvalues = np.concatenate(
        (
            np.array([cfg.lambda1, lambda2], dtype=float),
            tail_eigenvalues,
        )
    )

    A = Q @ np.diag(eigenvalues) @ Q.T

    normality_error = float(
        np.linalg.norm(
            A.T @ A - A @ A.T,
            ord="fro",
        )
    )

    if normality_error > 1e-10:
        raise RuntimeError(
            "Constructed system is not numerically normal: "
            f"{normality_error:.3e}"
        )

    return A, Q, eigenvalues, normality_error


def construct_controlled_initial_state(
    true_basis: np.ndarray,
    excitation_ratio: float,
    other_mode_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct x0 in the true eigenbasis.

    We fix |a1|=1 and |a2/a1| to the requested value. Independent random signs
    avoid making the experiment depend on one sign convention.

    Coefficients for q3 and lower modes are random with configurable scale.
    """
    dim = true_basis.shape[0]

    coefficients = rng.normal(
        loc=0.0,
        scale=other_mode_scale,
        size=dim,
    )

    sign1 = float(rng.choice(np.array([-1.0, 1.0])))
    sign2 = float(rng.choice(np.array([-1.0, 1.0])))

    coefficients[0] = sign1
    coefficients[1] = sign2 * excitation_ratio

    x0 = true_basis @ coefficients

    return x0, coefficients


def make_estimator_config(cfg: SweepConfig) -> EstimatorConfig:
    return EstimatorConfig(
        dim=cfg.dim,
        steps=cfg.steps,
        trials=1,
        window=cfg.window,
        n_directions=2,
        seed=cfg.seed,
        stability_threshold_deg=cfg.stability_threshold_deg,
        stability_patience=cfg.stability_patience,
        relative_window_norm_floor=cfg.relative_window_norm_floor,
        min_residual_energy_fraction=(
            cfg.min_residual_energy_fraction
        ),
        numeric_relative_residual_floor=(
            cfg.numeric_relative_residual_floor
        ),
        min_stage_pc1_energy_fraction=(
            cfg.min_stage_pc1_energy_fraction
        ),
    )


def analyse_one_trial(
    *,
    cfg: SweepConfig,
    estimator_cfg: EstimatorConfig,
    A: np.ndarray,
    true_basis: np.ndarray,
    eigenvalues: np.ndarray,
    lambda2: float,
    system_replicate: int,
    system_seed: int,
    excitation_ratio: float,
    trial_within_system: int,
    trial_seed: int,
) -> dict:
    rng = np.random.default_rng(trial_seed)

    x0, coefficients = construct_controlled_initial_state(
        true_basis=true_basis,
        excitation_ratio=excitation_ratio,
        other_mode_scale=cfg.other_mode_scale,
        rng=rng,
    )

    X = simulate_trajectory(
        A=A,
        x0=x0,
        steps=cfg.steps,
    )

    L = np.zeros(cfg.dim, dtype=float)

    directions, info, _diagnostics = (
        estimate_from_observations_only(
            X=X,
            L=L,
            cfg=estimator_cfg,
        )
    )

    accepted = bool(
        info.get("success", False)
        and len(directions) == 2
    )

    q1_error = np.nan
    q2_error = np.nan
    leading_2_subspace_error = np.nan
    orthogonality_error = np.nan

    if accepted:
        estimated_basis = np.column_stack(directions)
        true_leading_basis = true_basis[:, :2]

        q1_error = angle_deg(
            directions[0],
            true_basis[:, 0],
        )
        q2_error = angle_deg(
            directions[1],
            true_basis[:, 1],
        )
        leading_2_subspace_error = max_principal_angle_deg(
            estimated_basis,
            true_leading_basis,
        )
        orthogonality_error = float(
            np.linalg.norm(
                estimated_basis.T @ estimated_basis
                - np.eye(2),
                ord="fro",
            )
        )

    q1_recovered = bool(
        accepted
        and q1_error <= cfg.recovery_angle_tolerance_deg
    )
    q2_recovered = bool(
        accepted
        and q2_error <= cfg.recovery_angle_tolerance_deg
    )
    joint_recovered = bool(
        q1_recovered and q2_recovered
    )

    row = {
        "lambda1": cfg.lambda1,
        "lambda2": lambda2,
        "lambda2_sign": int(np.sign(lambda2)),
        "spectral_gap_abs": (
            abs(cfg.lambda1) - abs(lambda2)
        ),
        "lambda2_abs_over_lambda1_abs": (
            abs(lambda2) / abs(cfg.lambda1)
        ),
        "excitation_ratio_abs_a2_over_a1": excitation_ratio,
        "theoretical_equal_amplitude_time": (
            theoretical_equal_amplitude_time(
                cfg.lambda1,
                lambda2,
                excitation_ratio,
            )
        ),
        "system_replicate": system_replicate,
        "system_seed": system_seed,
        "trial_within_system": trial_within_system,
        "trial_seed": trial_seed,
        "accepted_by_observation_criteria": accepted,
        "q1_recovered_within_tolerance": q1_recovered,
        "q2_recovered_within_tolerance": q2_recovered,
        "joint_q1_q2_recovered_within_tolerance": (
            joint_recovered
        ),
        "recovery_angle_tolerance_deg": (
            cfg.recovery_angle_tolerance_deg
        ),
        "qhat1_vs_q1_deg": q1_error,
        "qhat2_vs_q2_deg": q2_error,
        "leading_2_subspace_error_deg": (
            leading_2_subspace_error
        ),
        "estimated_orthogonality_fro_error": (
            orthogonality_error
        ),
        "selected_window_start": info.get(
            "window_start",
            np.nan,
        ),
        "selected_window_end": info.get(
            "window_end",
            np.nan,
        ),
        "relative_window_norm": info.get(
            "relative_window_norm",
            np.nan,
        ),
        "n_common_stable_candidates": info.get(
            "n_common_stable_candidates",
            0,
        ),
        "a1": coefficients[0],
        "a2": coefficients[1],
        "abs_a1": abs(coefficients[0]),
        "abs_a2": abs(coefficients[1]),
        "realized_abs_a2_over_a1": (
            abs(coefficients[1])
            / abs(coefficients[0])
        ),
        "tail_coefficient_l2_norm": float(
            np.linalg.norm(coefficients[2:])
        ),
        "x0_norm": float(np.linalg.norm(x0)),
        "third_eigenvalue_abs": float(
            abs(eigenvalues[2])
        ),
    }

    for stage in (1, 2):
        for name in (
            "direction_change_deg",
            "stage_pc1_energy_fraction",
            "singular_value_ratio_1_to_2",
            "residual_energy_before_fraction",
            "residual_energy_after_fraction",
            "extracted_energy_fraction_original",
        ):
            row[f"stage_{stage}_{name}"] = info.get(
                f"stage_{stage}_{name}",
                np.nan,
            )

    return row


def safe_quantile(
    values: pd.Series,
    quantile: float,
) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(quantile))


def summarise_group(group: pd.DataFrame) -> pd.Series:
    total = len(group)

    accepted = group[
        group["accepted_by_observation_criteria"]
    ]
    q2_recovered_count = int(
        group[
            "q2_recovered_within_tolerance"
        ].sum()
    )
    joint_recovered_count = int(
        group[
            "joint_q1_q2_recovered_within_tolerance"
        ].sum()
    )

    accepted_count = len(accepted)

    acceptance_low, acceptance_high = wilson_interval(
        accepted_count,
        total,
    )
    q2_low, q2_high = wilson_interval(
        q2_recovered_count,
        total,
    )

    conditional_q2_accuracy = (
        q2_recovered_count / accepted_count
        if accepted_count > 0
        else np.nan
    )

    return pd.Series(
        {
            "n_trials": total,
            "n_accepted": accepted_count,
            "acceptance_rate": accepted_count / total,
            "acceptance_rate_wilson95_low": acceptance_low,
            "acceptance_rate_wilson95_high": acceptance_high,
            "n_q2_recovered": q2_recovered_count,
            "q2_recovery_rate": q2_recovered_count / total,
            "q2_recovery_rate_wilson95_low": q2_low,
            "q2_recovery_rate_wilson95_high": q2_high,
            "conditional_q2_accuracy_given_accepted": (
                conditional_q2_accuracy
            ),
            "n_joint_q1_q2_recovered": joint_recovered_count,
            "joint_q1_q2_recovery_rate": (
                joint_recovered_count / total
            ),
            "median_qhat1_vs_q1_deg_accepted": (
                float(
                    accepted[
                        "qhat1_vs_q1_deg"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
            "median_qhat2_vs_q2_deg_accepted": (
                float(
                    accepted[
                        "qhat2_vs_q2_deg"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
            "q25_qhat2_vs_q2_deg_accepted": safe_quantile(
                accepted["qhat2_vs_q2_deg"],
                0.25,
            ),
            "q75_qhat2_vs_q2_deg_accepted": safe_quantile(
                accepted["qhat2_vs_q2_deg"],
                0.75,
            ),
            "median_leading_2_subspace_error_deg_accepted": (
                float(
                    accepted[
                        "leading_2_subspace_error_deg"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
            "median_selected_window_end_accepted": (
                float(
                    accepted[
                        "selected_window_end"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
            "median_stage_2_residual_energy_before_fraction": (
                float(
                    accepted[
                        "stage_2_residual_energy_before_fraction"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
        }
    )


def build_summaries(
    all_trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_cell = (
        all_trials.groupby(
            [
                "lambda2",
                "lambda2_sign",
                "spectral_gap_abs",
                "lambda2_abs_over_lambda1_abs",
                "excitation_ratio_abs_a2_over_a1",
            ],
            dropna=False,
        )
        .apply(summarise_group)
        .reset_index()
        .sort_values(
            [
                "spectral_gap_abs",
                "excitation_ratio_abs_a2_over_a1",
            ]
        )
    )

    by_gap = (
        all_trials.groupby(
            [
                "lambda2",
                "lambda2_sign",
                "spectral_gap_abs",
                "lambda2_abs_over_lambda1_abs",
            ],
            dropna=False,
        )
        .apply(summarise_group)
        .reset_index()
        .sort_values("spectral_gap_abs")
    )

    by_excitation = (
        all_trials.groupby(
            ["excitation_ratio_abs_a2_over_a1"],
            dropna=False,
        )
        .apply(summarise_group)
        .reset_index()
        .sort_values(
            "excitation_ratio_abs_a2_over_a1"
        )
    )

    return by_cell, by_gap, by_excitation


def format_ratio_label(value: float) -> str:
    return f"{value:g}"


def format_gap_label(value: float) -> str:
    return f"{value:.3f}"


def plot_heatmap(
    *,
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    colourbar_label: str,
    output_path: Path,
    value_format: str,
) -> None:
    pivot = summary.pivot(
        index="lambda2",
        columns="excitation_ratio_abs_a2_over_a1",
        values=value_column,
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        origin="lower",
    )

    ax.set_xticks(
        np.arange(len(pivot.columns))
    )
    ax.set_xticklabels(
        [
            format_ratio_label(float(value))
            for value in pivot.columns
        ]
    )

    ax.set_yticks(
        np.arange(len(pivot.index))
    )
    ax.set_yticklabels(
        [
            (
                f"{float(value):+.2f} "
                f"(gap={float(summary.loc[summary['lambda2'] == value, 'spectral_gap_abs'].iloc[0]):.2f})"
            )
            for value in pivot.index
        ]
    )

    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(
        r"Signed $\lambda_2$ (magnitude gap in parentheses)"
    )
    ax.set_title(title)

    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[
                row_index,
                column_index,
            ]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    format(value, value_format),
                    ha="center",
                    va="center",
                )

    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label(colourbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recovery_vs_excitation(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for lambda2, group in summary.groupby(
        "lambda2",
        sort=True,
    ):
        ordered = group.sort_values(
            "excitation_ratio_abs_a2_over_a1"
        )

        gap = float(ordered["spectral_gap_abs"].iloc[0])

        ax.plot(
            np.arange(len(ordered)),
            ordered["q2_recovery_rate"],
            marker="o",
            label=f"lambda2={lambda2:+.2f}, gap={gap:.2f}",
        )

    ratios = sorted(
        summary[
            "excitation_ratio_abs_a2_over_a1"
        ].unique()
    )

    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels(
        [format_ratio_label(value) for value in ratios]
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(
        "Empirical q2 recovery rate"
    )
    ax.set_title(
        "Second-direction recovery versus trajectory excitation"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recovery_vs_gap(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for (ratio, sign), group in summary.groupby(
        [
            "excitation_ratio_abs_a2_over_a1",
            "lambda2_sign",
        ],
        sort=True,
    ):
        ordered = group.sort_values(
            "spectral_gap_abs"
        )

        sign_label = "positive" if sign > 0 else "negative"

        ax.plot(
            ordered["spectral_gap_abs"],
            ordered["q2_recovery_rate"],
            marker="o",
            label=(
                f"|a2/a1|={ratio:g}, "
                f"lambda2 {sign_label}"
            ),
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        r"Absolute spectral gap $|\lambda_1|-|\lambda_2|$"
    )
    ax.set_ylabel(
        "Empirical q2 recovery rate"
    )
    ax.set_title(
        "Second-direction recovery versus spectral gap"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_vs_excitation(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for lambda2, group in summary.groupby(
        "lambda2",
        sort=True,
    ):
        ordered = group.sort_values(
            "excitation_ratio_abs_a2_over_a1"
        )

        gap = float(ordered["spectral_gap_abs"].iloc[0])

        ax.plot(
            np.arange(len(ordered)),
            ordered[
                "median_qhat2_vs_q2_deg_accepted"
            ],
            marker="o",
            label=f"lambda2={lambda2:+.2f}, gap={gap:.2f}",
        )

    ratios = sorted(
        summary[
            "excitation_ratio_abs_a2_over_a1"
        ].unique()
    )

    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels(
        [format_ratio_label(value) for value in ratios]
    )
    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(
        "Median q2 angle error among accepted trials (degrees)"
    )
    ax.set_title(
        "Conditional second-direction error versus excitation"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def add_bar_annotations(
    ax: plt.Axes,
    bars: Iterable,
    percentages: Iterable[float],
    counts: Iterable[int] | None = None,
) -> None:
    percentages = list(percentages)
    counts_list = list(counts) if counts is not None else None

    for index, (bar, percentage) in enumerate(
        zip(bars, percentages)
    ):
        label = f"{percentage:.2f}%"

        if counts_list is not None:
            label += f"\n(n={counts_list[index]:,})"

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            label,
            ha="center",
            va="bottom",
        )


def plot_overall_trial_outcomes(
    all_trials: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Mutually exclusive outcomes across all trajectories.

    This plot makes the difference between observation-based acceptance and
    validated q2 recovery explicit.
    """
    total = int(len(all_trials))
    accepted = int(
        all_trials[
            "accepted_by_observation_criteria"
        ].sum()
    )
    recovered = int(
        all_trials[
            "q2_recovered_within_tolerance"
        ].sum()
    )

    accepted_not_q2 = accepted - recovered
    rejected = total - accepted

    labels = [
        r"Recovered $q_2$",
        "Accepted,\nbut not $q_2$",
        "No accepted\npair",
    ]
    counts = [
        recovered,
        accepted_not_q2,
        rejected,
    ]
    percentages = [
        100.0 * count / total
        for count in counts
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, percentages)

    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Percentage of all trajectories")
    ax.set_title(
        "Overall outcomes of the two-direction estimator"
    )
    ax.grid(True, axis="y", alpha=0.25)

    add_bar_annotations(
        ax=ax,
        bars=bars,
        percentages=percentages,
        counts=counts,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_zero_excitation_control(
    all_trials: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Show that an accepted stable residual direction is not necessarily q2
    when a2=0 and q2 is absent from the observed trajectory.
    """
    zero = all_trials[
        np.isclose(
            all_trials[
                "excitation_ratio_abs_a2_over_a1"
            ].to_numpy(dtype=float),
            0.0,
        )
    ]

    total = int(len(zero))
    accepted = int(
        zero[
            "accepted_by_observation_criteria"
        ].sum()
    )
    recovered = int(
        zero[
            "q2_recovered_within_tolerance"
        ].sum()
    )

    labels = [
        "Accepted stable\nsecond direction",
        r"Recovered $q_2$",
    ]
    counts = [accepted, recovered]
    percentages = [
        100.0 * count / total
        if total > 0
        else np.nan
        for count in counts
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, percentages)

    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel(
        r"Percentage of trajectories with $a_2=0$"
    )
    ax.set_title(
        r"Negative control: $q_2$ is absent when $a_2=0$"
    )
    ax.grid(True, axis="y", alpha=0.25)

    add_bar_annotations(
        ax=ax,
        bars=bars,
        percentages=percentages,
        counts=counts,
    )

    ax.text(
        0.5,
        -0.20,
        (
            "Acceptance can correspond to another lower "
            "residual direction."
        ),
        transform=ax.transAxes,
        ha="center",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overall_rates_vs_excitation(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Aggregate over all tested signed lambda2 values and system replicates.
    """
    ordered = summary.sort_values(
        "excitation_ratio_abs_a2_over_a1"
    )

    x = np.arange(len(ordered))
    ratios = ordered[
        "excitation_ratio_abs_a2_over_a1"
    ].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        x,
        100.0 * ordered["acceptance_rate"],
        marker="o",
        label="Accepted pair",
    )
    ax.plot(
        x,
        100.0 * ordered["q2_recovery_rate"],
        marker="o",
        label=r"Recovered $q_2$",
    )

    ax.set_xticks(
        x,
        [
            format_ratio_label(value)
            for value in ratios
        ],
    )
    ax.set_ylim(0.0, 105.0)
    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel("Percentage of trajectories")
    ax.set_title(
        r"Effect of the initial $q_2$ component"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_positive_lambda2_gap_heatmap(
    summary: pd.DataFrame,
    cfg: SweepConfig,
    output_path: Path,
) -> None:
    """
    Isolate positive lambda2 values so that the magnitude-gap effect is not
    mixed with the alternating-sign effect.
    """
    positive = summary[
        summary["lambda2"] > 0.0
    ].copy()

    row_order = (
        positive[
            ["lambda2", "spectral_gap_abs"]
        ]
        .drop_duplicates()
        .sort_values("spectral_gap_abs")
    )

    lambda2_order = row_order[
        "lambda2"
    ].to_list()

    excitation_order = list(
        cfg.excitation_ratios
    )

    pivot = (
        positive.pivot(
            index="lambda2",
            columns="excitation_ratio_abs_a2_over_a1",
            values="q2_recovery_rate",
        )
        .reindex(
            index=lambda2_order,
            columns=excitation_order,
        )
    )

    values = 100.0 * pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(
        values,
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=100.0,
    )

    ax.set_xticks(
        np.arange(len(excitation_order)),
        [
            format_ratio_label(value)
            for value in excitation_order
        ],
    )

    y_labels = []
    for lambda2 in lambda2_order:
        gap = abs(cfg.lambda1) - abs(lambda2)
        y_labels.append(
            rf"$\lambda_2={lambda2:.2f}$  "
            rf"(gap={gap:.2f})"
        )

    ax.set_yticks(
        np.arange(len(lambda2_order)),
        y_labels,
    )

    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(
        r"Positive $\lambda_2$ and magnitude gap"
    )
    ax.set_title(
        "Recovery across excitation and eigenvalue-magnitude gap"
    )

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]

            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                )

    colourbar = fig.colorbar(image, ax=ax)
    colourbar.set_label(r"$q_2$ recovery rate (%)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_sign_effect_for_magnitude(
    summary: pd.DataFrame,
    magnitude: float,
    output_path: Path,
) -> None:
    """
    Directly compare +|lambda2| and -|lambda2| at matched excitation levels.
    """
    selected = summary[
        np.isclose(
            np.abs(
                summary["lambda2"].to_numpy(dtype=float)
            ),
            magnitude,
        )
    ].copy()

    if selected.empty:
        return

    ratios = sorted(
        selected[
            "excitation_ratio_abs_a2_over_a1"
        ].unique()
    )
    x = np.arange(len(ratios))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for lambda2 in (magnitude, -magnitude):
        group = selected[
            np.isclose(
                selected["lambda2"].to_numpy(dtype=float),
                lambda2,
            )
        ].sort_values(
            "excitation_ratio_abs_a2_over_a1"
        )

        if group.empty:
            continue

        ax.plot(
            x,
            100.0 * group["q2_recovery_rate"],
            marker="o",
            label=rf"$\lambda_2={lambda2:+.2f}$",
        )

    ax.set_xticks(
        x,
        [
            format_ratio_label(value)
            for value in ratios
        ],
    )
    ax.set_ylim(0.0, 105.0)
    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(r"$q_2$ recovery rate (%)")
    ax.set_title(
        rf"Effect of the sign of $\lambda_2$ "
        rf"when $|\lambda_2|={magnitude:.2f}$"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_accepted_q2_errors_by_excitation(
    all_trials: pd.DataFrame,
    cfg: SweepConfig,
    output_path: Path,
) -> None:
    """
    Show the distribution of validation errors among estimates that passed
    the observation-only acceptance criteria.
    """
    accepted = all_trials[
        all_trials[
            "accepted_by_observation_criteria"
        ]
    ]

    labels: list[str] = []
    data: list[np.ndarray] = []

    for ratio in cfg.excitation_ratios:
        values = accepted.loc[
            np.isclose(
                accepted[
                    "excitation_ratio_abs_a2_over_a1"
                ].to_numpy(dtype=float),
                ratio,
            ),
            "qhat2_vs_q2_deg",
        ].dropna().to_numpy(dtype=float)

        if len(values) == 0:
            continue

        labels.append(format_ratio_label(ratio))
        data.append(
            np.maximum(values, 1e-8)
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if data:
        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
        )
        ax.set_yscale("log")

    ax.axhline(
        cfg.recovery_angle_tolerance_deg,
        linestyle="--",
        label=(
            f"{cfg.recovery_angle_tolerance_deg:g}° "
            "recovery threshold"
        ),
    )

    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(
        r"Angle between $\widehat u_2$ and $q_2$ "
        "(degrees, log scale)"
    )
    ax.set_title(
        "Accuracy among observation-based accepted estimates"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_presentation_plot_summary(
    all_trials: pd.DataFrame,
    output_path: Path,
) -> None:
    total = int(len(all_trials))
    accepted = int(
        all_trials[
            "accepted_by_observation_criteria"
        ].sum()
    )
    recovered = int(
        all_trials[
            "q2_recovered_within_tolerance"
        ].sum()
    )

    zero = all_trials[
        np.isclose(
            all_trials[
                "excitation_ratio_abs_a2_over_a1"
            ].to_numpy(dtype=float),
            0.0,
        )
    ]
    zero_total = int(len(zero))
    zero_accepted = int(
        zero[
            "accepted_by_observation_criteria"
        ].sum()
    )
    zero_recovered = int(
        zero[
            "q2_recovered_within_tolerance"
        ].sum()
    )

    rows = [
        {
            "result": "all_accepted",
            "count": accepted,
            "denominator": total,
            "rate": accepted / total,
        },
        {
            "result": "all_q2_recovered",
            "count": recovered,
            "denominator": total,
            "rate": recovered / total,
        },
        {
            "result": "accepted_but_not_q2",
            "count": accepted - recovered,
            "denominator": total,
            "rate": (accepted - recovered) / total,
        },
        {
            "result": "zero_excitation_accepted",
            "count": zero_accepted,
            "denominator": zero_total,
            "rate": (
                zero_accepted / zero_total
                if zero_total > 0
                else np.nan
            ),
        },
        {
            "result": "zero_excitation_q2_recovered",
            "count": zero_recovered,
            "denominator": zero_total,
            "rate": (
                zero_recovered / zero_total
                if zero_total > 0
                else np.nan
            ),
        },
    ]

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate observation-only recovery of the second direction "
            "across multiple normal linear systems, spectral gaps, and "
            "controlled initial q2 excitation levels."
        )
    )

    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--lambda1", type=float, default=0.96)
    parser.add_argument(
        "--lambda2-values",
        type=parse_float_list,
        default=parse_float_list(
            "0.95,0.94,0.92,-0.92,0.90,0.88,-0.88,0.86"
        ),
    )
    parser.add_argument(
        "--excitation-ratios",
        type=parse_float_list,
        default=parse_float_list(
            "0,0.01,0.03,0.1,0.3,1,3"
        ),
    )
    parser.add_argument(
        "--system-replicates",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--trials-per-system",
        type=int,
        default=20,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--other-mode-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--tail-max",
        type=float,
        default=0.84,
    )
    parser.add_argument(
        "--tail-min",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--tail-gap-below-lambda2",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--stability-threshold-deg",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--stability-patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--relative-window-norm-floor",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--min-residual-energy-fraction",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--numeric-relative-residual-floor",
        type=float,
        default=1e-15,
    )
    parser.add_argument(
        "--min-stage-pc1-energy-fraction",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--recovery-angle-tolerance-deg",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "observation_only_q2_"
            "normal_system_gap_excitation_sweep"
        ),
    )

    args = parser.parse_args()

    cfg = SweepConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        lambda1=args.lambda1,
        lambda2_values=tuple(args.lambda2_values),
        excitation_ratios=tuple(
            args.excitation_ratios
        ),
        system_replicates=args.system_replicates,
        trials_per_system=args.trials_per_system,
        seed=args.seed,
        other_mode_scale=args.other_mode_scale,
        tail_max=args.tail_max,
        tail_min=args.tail_min,
        tail_gap_below_lambda2=(
            args.tail_gap_below_lambda2
        ),
        stability_threshold_deg=(
            args.stability_threshold_deg
        ),
        stability_patience=args.stability_patience,
        relative_window_norm_floor=(
            args.relative_window_norm_floor
        ),
        min_residual_energy_fraction=(
            args.min_residual_energy_fraction
        ),
        numeric_relative_residual_floor=(
            args.numeric_relative_residual_floor
        ),
        min_stage_pc1_energy_fraction=(
            args.min_stage_pc1_energy_fraction
        ),
        recovery_angle_tolerance_deg=(
            args.recovery_angle_tolerance_deg
        ),
    )

    validate_config(cfg)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    estimator_cfg = make_estimator_config(cfg)

    total_trials = (
        len(cfg.lambda2_values)
        * len(cfg.excitation_ratios)
        * cfg.system_replicates
        * cfg.trials_per_system
    )

    print(
        "============================================================"
    )
    print(
        "Observation-only q2 recovery sweep: normal linear systems"
    )
    print(
        "============================================================"
    )
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"window: {cfg.window}")
    print(f"lambda1: {cfg.lambda1}")
    print(f"signed lambda2 values: {cfg.lambda2_values}")
    print(
        "All spectral ordering uses magnitudes: "
        "|lambda1| > |lambda2| > remaining modes"
    )
    print(
        "controlled |a2/a1| values: "
        f"{cfg.excitation_ratios}"
    )
    print(
        "system replicates per spectral gap: "
        f"{cfg.system_replicates}"
    )
    print(
        "trials per system and excitation setting: "
        f"{cfg.trials_per_system}"
    )
    print(f"total trajectories: {total_trials}")
    print(
        "q2 recovery tolerance: "
        f"{cfg.recovery_angle_tolerance_deg} degrees"
    )
    print(
        "The estimator receives only X and L. "
        "True directions are used only for validation."
    )
    print(
        "============================================================"
    )

    config_json = asdict(cfg)
    config_json["lambda2_values"] = list(
        cfg.lambda2_values
    )
    config_json["excitation_ratios"] = list(
        cfg.excitation_ratios
    )

    with (
        output / "experiment_config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            config_json,
            file,
            indent=2,
        )

    trial_rows: list[dict] = []
    system_rows: list[dict] = []

    completed = 0

    for lambda2_index, lambda2 in enumerate(
        cfg.lambda2_values
    ):
        gap = abs(cfg.lambda1) - abs(lambda2)

        for system_replicate in range(
            cfg.system_replicates
        ):
            system_seed = (
                cfg.seed
                + 1_000_000 * lambda2_index
                + 10_000 * system_replicate
            )

            (
                A,
                true_basis,
                eigenvalues,
                normality_error,
            ) = build_random_normal_system(
                cfg=cfg,
                lambda2=lambda2,
                system_seed=system_seed,
            )

            system_rows.append(
                {
                    "lambda1": cfg.lambda1,
                    "lambda2": lambda2,
                    "spectral_gap_abs": gap,
                    "lambda2_sign": int(np.sign(lambda2)),
                    "lambda2_abs_over_lambda1_abs": (
                        abs(lambda2) / abs(cfg.lambda1)
                    ),
                    "system_replicate": (
                        system_replicate
                    ),
                    "system_seed": system_seed,
                    "normality_error": (
                        normality_error
                    ),
                    "tail_max_abs_eigenvalue": float(
                        np.max(
                            np.abs(
                                eigenvalues[2:]
                            )
                        )
                    ),
                    "tail_eigenvalues_json": json.dumps(
                        eigenvalues[2:].tolist()
                    ),
                }
            )

            for ratio_index, ratio in enumerate(
                cfg.excitation_ratios
            ):
                for trial_within_system in range(
                    cfg.trials_per_system
                ):
                    trial_seed = (
                        cfg.seed
                        + 100_000_000 * lambda2_index
                        + 1_000_000 * system_replicate
                        + 10_000 * ratio_index
                        + trial_within_system
                    )

                    row = analyse_one_trial(
                        cfg=cfg,
                        estimator_cfg=estimator_cfg,
                        A=A,
                        true_basis=true_basis,
                        eigenvalues=eigenvalues,
                        lambda2=lambda2,
                        system_replicate=(
                            system_replicate
                        ),
                        system_seed=system_seed,
                        excitation_ratio=ratio,
                        trial_within_system=(
                            trial_within_system
                        ),
                        trial_seed=trial_seed,
                    )

                    trial_rows.append(row)
                    completed += 1

            print(
                f"completed lambda2={lambda2:+.3f}, "
                f"magnitude gap={gap:.3f}, "
                f"system {system_replicate + 1}/"
                f"{cfg.system_replicates}; "
                f"{completed}/{total_trials} trajectories"
            )

    all_trials = pd.DataFrame(trial_rows)

    all_trials.to_csv(
        output / "all_trial_metrics.csv",
        index=False,
    )

    pd.DataFrame(system_rows).to_csv(
        output / "system_definitions.csv",
        index=False,
    )

    (
        summary_by_cell,
        summary_by_gap,
        summary_by_excitation,
    ) = build_summaries(all_trials)

    summary_by_cell.to_csv(
        output
        / "summary_by_spectral_gap_and_excitation.csv",
        index=False,
    )
    summary_by_gap.to_csv(
        output / "summary_by_spectral_gap.csv",
        index=False,
    )
    summary_by_excitation.to_csv(
        output / "summary_by_excitation_ratio.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Presentation-ready plots
    # ------------------------------------------------------------
    plot_overall_trial_outcomes(
        all_trials=all_trials,
        output_path=(
            output / "01_overall_trial_outcomes.png"
        ),
    )

    plot_zero_excitation_control(
        all_trials=all_trials,
        output_path=(
            output / "02_zero_excitation_negative_control.png"
        ),
    )

    plot_overall_rates_vs_excitation(
        summary=summary_by_excitation,
        output_path=(
            output / "03_acceptance_and_recovery_vs_excitation.png"
        ),
    )

    plot_positive_lambda2_gap_heatmap(
        summary=summary_by_cell,
        cfg=cfg,
        output_path=(
            output / "04_positive_lambda2_gap_effect.png"
        ),
    )

    plot_sign_effect_for_magnitude(
        summary=summary_by_cell,
        magnitude=0.88,
        output_path=(
            output / "05_sign_effect_abs_lambda2_088.png"
        ),
    )

    plot_sign_effect_for_magnitude(
        summary=summary_by_cell,
        magnitude=0.92,
        output_path=(
            output / "06_sign_effect_abs_lambda2_092.png"
        ),
    )

    plot_accepted_q2_errors_by_excitation(
        all_trials=all_trials,
        cfg=cfg,
        output_path=(
            output / "07_accepted_q2_error_by_excitation.png"
        ),
    )

    save_presentation_plot_summary(
        all_trials=all_trials,
        output_path=(
            output / "presentation_plot_summary.csv"
        ),
    )

    # ------------------------------------------------------------
    # Existing detailed diagnostic plots retained as appendix plots
    # ------------------------------------------------------------
    plot_heatmap(
        summary=summary_by_cell,
        value_column="q2_recovery_rate",
        title=(
            "Empirical recovery rate of the second direction"
        ),
        colourbar_label="q2 recovery rate",
        output_path=(
            output / "appendix_01_q2_recovery_rate_heatmap.png"
        ),
        value_format=".2f",
    )

    plot_heatmap(
        summary=summary_by_cell,
        value_column="acceptance_rate",
        title=(
            "Observation-based common-window acceptance rate"
        ),
        colourbar_label="acceptance rate",
        output_path=(
            output / "appendix_02_acceptance_rate_heatmap.png"
        ),
        value_format=".2f",
    )

    plot_heatmap(
        summary=summary_by_cell,
        value_column=(
            "median_qhat2_vs_q2_deg_accepted"
        ),
        title=(
            "Median second-direction angle error "
            "among accepted trials"
        ),
        colourbar_label="median q2 error (degrees)",
        output_path=(
            output / "appendix_03_median_q2_error_heatmap.png"
        ),
        value_format=".3f",
    )

    plot_recovery_vs_excitation(
        summary=summary_by_cell,
        output_path=(
            output
            / "appendix_04_q2_recovery_rate_vs_excitation.png"
        ),
    )

    plot_recovery_vs_gap(
        summary=summary_by_cell,
        output_path=(
            output
            / "appendix_05_q2_recovery_rate_vs_spectral_gap.png"
        ),
    )

    plot_error_vs_excitation(
        summary=summary_by_cell,
        output_path=(
            output
            / "appendix_06_median_q2_error_vs_excitation.png"
        ),
    )

    overall = summarise_group(all_trials)
    overall_df = overall.to_frame().T

    overall_df.insert(0, "lambda1", cfg.lambda1)
    overall_df.insert(
        1,
        "n_lambda2_values",
        len(cfg.lambda2_values),
    )
    overall_df.insert(
        2,
        "n_excitation_ratios",
        len(cfg.excitation_ratios),
    )
    overall_df.insert(
        3,
        "system_replicates",
        cfg.system_replicates,
    )
    overall_df.insert(
        4,
        "trials_per_system",
        cfg.trials_per_system,
    )
    overall_df.insert(
        5,
        "recovery_angle_tolerance_deg",
        cfg.recovery_angle_tolerance_deg,
    )

    overall_df.to_csv(
        output / "overall_summary.csv",
        index=False,
    )

    print("\nOverall results across the complete sweep:")
    print(
        "acceptance rate: "
        f"{overall['acceptance_rate']:.4f}"
    )
    print(
        "q2 recovery rate: "
        f"{overall['q2_recovery_rate']:.4f}"
    )
    print(
        "conditional q2 accuracy given acceptance: "
        f"{overall['conditional_q2_accuracy_given_accepted']:.4f}"
    )
    print(
        "median accepted q2 angle error: "
        f"{overall['median_qhat2_vs_q2_deg_accepted']:.6f} degrees"
    )
    print(
        "\nImportant distinction:"
        "\n- acceptance_rate uses only observation-based criteria;"
        "\n- q2_recovery_rate additionally requires the synthetic validation "
        "angle to be within the specified tolerance;"
        "\n- the validation angle is never used by the estimator."
    )

    print(f"\nResults written to: {output.resolve()}")


if __name__ == "__main__":
    main()
