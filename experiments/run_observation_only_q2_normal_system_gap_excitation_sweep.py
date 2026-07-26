from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from run_observation_only_same_window_deflation_normalized import (
        Config as EstimatorConfig,
        angle_deg,
        estimate_from_observations_only,
        max_principal_angle_deg,
        simulate_trajectory,
    )
except ImportError as exc:
    raise ImportError(
        "Place this script in the same directory as "
        "'run_observation_only_same_window_deflation_normalized.py'."
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
        0.90,
        0.88,
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
        lambda2 - cfg.tail_gap_below_lambda2,
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
                "spectral_gap_abs",
                "lambda2_over_lambda1",
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
                "spectral_gap_abs",
                "lambda2_over_lambda1",
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
        index="spectral_gap_abs",
        columns="excitation_ratio_abs_a2_over_a1",
        values=value_column,
    ).sort_index()

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
            format_gap_label(float(value))
            for value in pivot.index
        ]
    )

    ax.set_xlabel(
        r"Controlled excitation magnitude $|a_2/a_1|$"
    )
    ax.set_ylabel(
        r"Absolute spectral gap $|\lambda_1|-|\lambda_2|$"
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

    for gap, group in summary.groupby(
        "spectral_gap_abs",
        sort=True,
    ):
        ordered = group.sort_values(
            "excitation_ratio_abs_a2_over_a1"
        )

        ax.plot(
            np.arange(len(ordered)),
            ordered["q2_recovery_rate"],
            marker="o",
            label=f"gap={gap:.3f}",
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

    for ratio, group in summary.groupby(
        "excitation_ratio_abs_a2_over_a1",
        sort=True,
    ):
        ordered = group.sort_values(
            "spectral_gap_abs"
        )

        ax.plot(
            ordered["spectral_gap_abs"],
            ordered["q2_recovery_rate"],
            marker="o",
            label=f"|a2/a1|={ratio:g}",
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

    for gap, group in summary.groupby(
        "spectral_gap_abs",
        sort=True,
    ):
        ordered = group.sort_values(
            "excitation_ratio_abs_a2_over_a1"
        )

        ax.plot(
            np.arange(len(ordered)),
            ordered[
                "median_qhat2_vs_q2_deg_accepted"
            ],
            marker="o",
            label=f"gap={gap:.3f}",
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
            "0.95,0.94,0.92,0.90,0.88"
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
    print(f"lambda2 values: {cfg.lambda2_values}")
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
        gap = cfg.lambda1 - lambda2

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
                    "lambda2_over_lambda1": (
                        lambda2 / cfg.lambda1
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
                f"completed lambda2={lambda2:.3f}, "
                f"gap={gap:.3f}, "
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

    plot_heatmap(
        summary=summary_by_cell,
        value_column="q2_recovery_rate",
        title=(
            "Empirical recovery rate of the second direction"
        ),
        colourbar_label="q2 recovery rate",
        output_path=(
            output / "01_q2_recovery_rate_heatmap.png"
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
            output / "02_acceptance_rate_heatmap.png"
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
            output / "03_median_q2_error_heatmap.png"
        ),
        value_format=".3f",
    )

    plot_recovery_vs_excitation(
        summary=summary_by_cell,
        output_path=(
            output
            / "04_q2_recovery_rate_vs_excitation.png"
        ),
    )

    plot_recovery_vs_gap(
        summary=summary_by_cell,
        output_path=(
            output
            / "05_q2_recovery_rate_vs_spectral_gap.png"
        ),
    )

    plot_error_vs_excitation(
        summary=summary_by_cell,
        output_path=(
            output
            / "06_median_q2_error_vs_excitation.png"
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
