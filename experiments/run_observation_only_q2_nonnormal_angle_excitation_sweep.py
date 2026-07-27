from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Support both:
#   repo/experiments/this_script.py
#   repo/experiments/run_observation_only_same_window_deflation_normal.py
#
# and:
#   repo/experiments/this_script.py
#   repo/run_observation_only_same_window_deflation_normal.py
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
        "The code looked at:\n"
        f"{searched_text}"
    ) from exc


@dataclass(frozen=True)
class SweepConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20

    lambda1: float = 0.96

    # A targeted subset of the normal-system sweep. Positive/negative pairs
    # isolate the effect of sign while two magnitude gaps are represented.
    lambda2_values: tuple[float, ...] = (
        0.92,
        -0.92,
        0.88,
        -0.88,
    )

    # Angle between the first two true right eigenvectors.
    # 90 degrees is the normal control. Smaller angles are increasingly
    # non-normal.
    eigenvector_angles_deg: tuple[float, ...] = (
        90.0,
        75.0,
        60.0,
        45.0,
        30.0,
    )

    # Controlled modal excitation |a2/a1|.
    excitation_ratios: tuple[float, ...] = (
        0.0,
        0.03,
        0.10,
        0.30,
        1.00,
        3.00,
    )

    system_replicates: int = 5
    trials_per_system: int = 20
    seed: int = 42

    # Modal coefficients for q3 and lower modes.
    other_mode_scale: float = 0.25

    # Remaining eigenvalue magnitudes are random but strictly below |lambda2|.
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

    # True synthetic quantities are used only after estimation.
    recovery_angle_tolerance_deg: float = 1.0
    subspace_angle_tolerance_deg: float = 1.0


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
        raise ValueError(
            "window must satisfy 2 <= window <= steps + 1."
        )

    if cfg.system_replicates < 1:
        raise ValueError(
            "system_replicates must be positive."
        )

    if cfg.trials_per_system < 1:
        raise ValueError(
            "trials_per_system must be positive."
        )

    if not (0.0 < abs(cfg.lambda1) < 1.0):
        raise ValueError(
            "lambda1 must be nonzero and stable: abs(lambda1) < 1."
        )

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
                "No valid tail spectrum remains for "
                f"lambda2={lambda2}."
            )

    for angle in cfg.eigenvector_angles_deg:
        if not (0.0 < angle <= 90.0):
            raise ValueError(
                "Every eigenvector angle must satisfy "
                "0 < angle <= 90 degrees."
            )

    for ratio in cfg.excitation_ratios:
        if ratio < 0.0:
            raise ValueError(
                "Excitation ratios represent |a2/a1| and "
                "must be nonnegative."
            )

    if cfg.other_mode_scale < 0.0:
        raise ValueError(
            "other_mode_scale must be nonnegative."
        )

    if cfg.tail_min <= 0.0:
        raise ValueError("tail_min must be positive.")

    if cfg.tail_max <= cfg.tail_min:
        raise ValueError(
            "tail_max must exceed tail_min."
        )

    if cfg.tail_gap_below_lambda2 <= 0.0:
        raise ValueError(
            "tail_gap_below_lambda2 must be positive."
        )

    if cfg.recovery_angle_tolerance_deg <= 0.0:
        raise ValueError(
            "recovery_angle_tolerance_deg must be positive."
        )

    if cfg.subspace_angle_tolerance_deg <= 0.0:
        raise ValueError(
            "subspace_angle_tolerance_deg must be positive."
        )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError(
            "Cannot normalize a zero or non-finite vector."
        )

    return vector / norm


def signed_direction_angle_deg(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """
    Sign-invariant angle in [0, 90] degrees.
    """
    vector_a = normalize_vector(vector_a)
    vector_b = normalize_vector(vector_b)

    cosine = np.clip(
        abs(float(vector_a @ vector_b)),
        0.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan

    probability = successes / total
    z_squared = z * z

    denominator = 1.0 + z_squared / total

    centre = (
        probability
        + z_squared / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            probability
            * (1.0 - probability)
            / total
            + z_squared
            / (4.0 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def build_controlled_nonnormal_system(
    *,
    cfg: SweepConfig,
    lambda2: float,
    eigenvector_angle_deg: float,
    system_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    """
    Construct

        A = V diag(lambda_1, lambda_2, ...) V^{-1}

    with a controlled angle between the first two right eigenvectors.

    In canonical coordinates:

        q1 = e1,
        q2 = cos(theta)e1 + sin(theta)e2.

    The other eigenvectors are e3,...,ed. A random orthogonal rotation is
    applied to the complete basis so the experiment is not tied to coordinate
    axes.

    theta=90 degrees gives an orthogonal eigenbasis and therefore a normal
    control. Smaller theta produces a genuinely non-normal matrix.
    """
    rng = np.random.default_rng(system_seed)

    theta = np.deg2rad(eigenvector_angle_deg)

    canonical_basis = np.eye(cfg.dim, dtype=float)

    canonical_basis[:, 0] = 0.0
    canonical_basis[0, 0] = 1.0

    canonical_basis[:, 1] = 0.0
    canonical_basis[0, 1] = np.cos(theta)
    canonical_basis[1, 1] = np.sin(theta)

    gaussian = rng.normal(
        size=(cfg.dim, cfg.dim)
    )
    rotation, _ = np.linalg.qr(gaussian)

    true_right_eigenvectors = (
        rotation @ canonical_basis
    )

    # Numerical normalization does not change A because diagonal column
    # scaling commutes with the diagonal eigenvalue matrix.
    true_right_eigenvectors = (
        true_right_eigenvectors
        / np.linalg.norm(
            true_right_eigenvectors,
            axis=0,
            keepdims=True,
        )
    )

    available_tail_max = min(
        cfg.tail_max,
        abs(lambda2) - cfg.tail_gap_below_lambda2,
    )

    tail_magnitudes = rng.uniform(
        cfg.tail_min,
        available_tail_max,
        size=cfg.dim - 2,
    )
    tail_magnitudes = np.sort(
        tail_magnitudes
    )[::-1]

    tail_signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=cfg.dim - 2,
    )

    tail_eigenvalues = (
        tail_signs * tail_magnitudes
    )

    eigenvalues = np.concatenate(
        (
            np.array(
                [cfg.lambda1, lambda2],
                dtype=float,
            ),
            tail_eigenvalues,
        )
    )

    inverse_eigenvectors = np.linalg.inv(
        true_right_eigenvectors
    )

    matrix_a = (
        true_right_eigenvectors
        @ np.diag(eigenvalues)
        @ inverse_eigenvectors
    )

    q1 = normalize_vector(
        true_right_eigenvectors[:, 0]
    )
    q2 = normalize_vector(
        true_right_eigenvectors[:, 1]
    )

    measured_angle = signed_direction_angle_deg(
        q1,
        q2,
    )

    q2_qr_compatible = q2 - float(q1 @ q2) * q1
    q2_qr_compatible = normalize_vector(
        q2_qr_compatible
    )

    eigenvector_condition_number = float(
        np.linalg.cond(
            true_right_eigenvectors
        )
    )

    gram_nonorthogonality = float(
        np.linalg.norm(
            true_right_eigenvectors.T
            @ true_right_eigenvectors
            - np.eye(cfg.dim),
            ord="fro",
        )
    )

    nonnormality_fro = float(
        np.linalg.norm(
            matrix_a.T @ matrix_a
            - matrix_a @ matrix_a.T,
            ord="fro",
        )
    )

    matrix_norm_squared = float(
        np.linalg.norm(
            matrix_a,
            ord="fro",
        )
        ** 2
    )

    relative_nonnormality = (
        nonnormality_fro / matrix_norm_squared
        if matrix_norm_squared > 0.0
        else np.nan
    )

    metrics = {
        "requested_q1_q2_angle_deg": (
            eigenvector_angle_deg
        ),
        "measured_q1_q2_angle_deg": (
            measured_angle
        ),
        "nonorthogonality_defect_deg": (
            90.0 - measured_angle
        ),
        # Any orthonormal pair approximating q1 and q2 has a minimax
        # individual-direction error at least this large.
        "theoretical_orthogonal_minimax_floor_deg": (
            (90.0 - measured_angle) / 2.0
        ),
        # If qhat1 is fixed exactly at q1, an orthogonal qhat2 cannot be
        # closer to q2 than this.
        "q1_fixed_q2_error_floor_deg": (
            90.0 - measured_angle
        ),
        "eigenvector_condition_number": (
            eigenvector_condition_number
        ),
        "gram_nonorthogonality_fro": (
            gram_nonorthogonality
        ),
        "nonnormality_fro": (
            nonnormality_fro
        ),
        "relative_nonnormality": (
            relative_nonnormality
        ),
    }

    if eigenvector_condition_number > 1e8:
        raise RuntimeError(
            "Generated eigenvector basis is too "
            "ill-conditioned: "
            f"{eigenvector_condition_number:.3e}"
        )

    return (
        matrix_a,
        true_right_eigenvectors,
        eigenvalues,
        metrics,
    )


def construct_controlled_initial_state(
    *,
    true_right_eigenvectors: np.ndarray,
    excitation_ratio: float,
    other_mode_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct x0 using controlled MODAL coefficients:

        x0 = V a,

    with |a1|=1 and |a2/a1| equal to excitation_ratio.

    In a non-normal system, these modal coefficients are not the same as
    Euclidean projections x0^T qj.
    """
    dimension = true_right_eigenvectors.shape[0]

    coefficients = rng.normal(
        loc=0.0,
        scale=other_mode_scale,
        size=dimension,
    )

    sign1 = float(
        rng.choice(
            np.array([-1.0, 1.0])
        )
    )
    sign2 = float(
        rng.choice(
            np.array([-1.0, 1.0])
        )
    )

    coefficients[0] = sign1
    coefficients[1] = (
        sign2 * excitation_ratio
    )

    initial_state = (
        true_right_eigenvectors
        @ coefficients
    )

    return initial_state, coefficients


def make_estimator_config(
    cfg: SweepConfig,
) -> EstimatorConfig:
    """
    The repository contains two closely related estimator versions. One has
    numeric_relative_residual_floor and one does not. Pass only parameters
    supported by the installed Config class.
    """
    requested = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": 1,
        "window": cfg.window,
        "n_directions": 2,
        "seed": cfg.seed,
        "stability_threshold_deg": (
            cfg.stability_threshold_deg
        ),
        "stability_patience": (
            cfg.stability_patience
        ),
        "relative_window_norm_floor": (
            cfg.relative_window_norm_floor
        ),
        "min_residual_energy_fraction": (
            cfg.min_residual_energy_fraction
        ),
        "numeric_relative_residual_floor": (
            cfg.numeric_relative_residual_floor
        ),
        "min_stage_pc1_energy_fraction": (
            cfg.min_stage_pc1_energy_fraction
        ),
    }

    supported_parameters = set(
        inspect.signature(
            EstimatorConfig
        ).parameters
    )

    supported = {
        name: value
        for name, value in requested.items()
        if name in supported_parameters
    }

    return EstimatorConfig(**supported)


def analyse_one_trial(
    *,
    cfg: SweepConfig,
    estimator_cfg: EstimatorConfig,
    matrix_a: np.ndarray,
    true_right_eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    system_metrics: dict[str, float],
    lambda2: float,
    eigenvector_angle_deg: float,
    system_replicate: int,
    system_seed: int,
    excitation_ratio: float,
    trial_within_system: int,
    trial_seed: int,
) -> dict:
    rng = np.random.default_rng(
        trial_seed
    )

    initial_state, modal_coefficients = (
        construct_controlled_initial_state(
            true_right_eigenvectors=(
                true_right_eigenvectors
            ),
            excitation_ratio=(
                excitation_ratio
            ),
            other_mode_scale=(
                cfg.other_mode_scale
            ),
            rng=rng,
        )
    )

    trajectory = simulate_trajectory(
        A=matrix_a,
        x0=initial_state,
        steps=cfg.steps,
    )

    limit = np.zeros(
        cfg.dim,
        dtype=float,
    )

    directions, info, _diagnostics = (
        estimate_from_observations_only(
            X=trajectory,
            L=limit,
            cfg=estimator_cfg,
        )
    )

    accepted = bool(
        info.get("success", False)
        and len(directions) == 2
    )

    q1 = normalize_vector(
        true_right_eigenvectors[:, 0]
    )
    q2 = normalize_vector(
        true_right_eigenvectors[:, 1]
    )

    q2_qr_compatible = q2 - float(q1 @ q2) * q1
    q2_qr_compatible = normalize_vector(
        q2_qr_compatible
    )

    q1_error = np.nan
    q2_true_error = np.nan
    q2_qr_error = np.nan
    leading_2_subspace_error = np.nan
    estimated_orthogonality_error = np.nan

    if accepted:
        estimated_basis = np.column_stack(
            directions
        )

        true_leading_basis = (
            true_right_eigenvectors[:, :2]
        )

        q1_error = angle_deg(
            directions[0],
            q1,
        )

        q2_true_error = angle_deg(
            directions[1],
            q2,
        )

        q2_qr_error = angle_deg(
            directions[1],
            q2_qr_compatible,
        )

        leading_2_subspace_error = (
            max_principal_angle_deg(
                estimated_basis,
                true_leading_basis,
            )
        )

        estimated_orthogonality_error = float(
            np.linalg.norm(
                estimated_basis.T
                @ estimated_basis
                - np.eye(2),
                ord="fro",
            )
        )

    # excitation_ratio=0 is a negative control. Even accidental alignment does
    # not constitute identifiable recovery because q2 is absent from the orbit.
    q2_is_present = excitation_ratio > 0.0

    q1_recovered = bool(
        accepted
        and q1_error
        <= cfg.recovery_angle_tolerance_deg
    )

    q2_true_recovered = bool(
        q2_is_present
        and accepted
        and q2_true_error
        <= cfg.recovery_angle_tolerance_deg
    )

    q2_qr_recovered = bool(
        q2_is_present
        and accepted
        and q2_qr_error
        <= cfg.recovery_angle_tolerance_deg
    )

    leading_2_subspace_recovered = bool(
        q2_is_present
        and accepted
        and leading_2_subspace_error
        <= cfg.subspace_angle_tolerance_deg
    )

    row: dict = {
        "lambda1": cfg.lambda1,
        "lambda2": lambda2,
        "lambda2_sign": int(
            np.sign(lambda2)
        ),
        "spectral_gap_abs": (
            abs(cfg.lambda1)
            - abs(lambda2)
        ),
        "lambda2_abs_over_lambda1_abs": (
            abs(lambda2)
            / abs(cfg.lambda1)
        ),
        "eigenvector_angle_deg": (
            eigenvector_angle_deg
        ),
        "excitation_ratio_abs_a2_over_a1": (
            excitation_ratio
        ),
        "q2_present_in_trajectory": (
            q2_is_present
        ),
        "system_replicate": (
            system_replicate
        ),
        "system_seed": system_seed,
        "trial_within_system": (
            trial_within_system
        ),
        "trial_seed": trial_seed,
        "accepted_by_observation_criteria": (
            accepted
        ),
        "q1_recovered_within_tolerance": (
            q1_recovered
        ),
        "q2_true_recovered_within_tolerance": (
            q2_true_recovered
        ),
        "q2_qr_compatible_recovered_within_tolerance": (
            q2_qr_recovered
        ),
        "leading_2_subspace_recovered_within_tolerance": (
            leading_2_subspace_recovered
        ),
        "recovery_angle_tolerance_deg": (
            cfg.recovery_angle_tolerance_deg
        ),
        "subspace_angle_tolerance_deg": (
            cfg.subspace_angle_tolerance_deg
        ),
        "qhat1_vs_true_q1_deg": (
            q1_error
        ),
        "qhat2_vs_true_q2_deg": (
            q2_true_error
        ),
        "qhat2_vs_qr_compatible_q2_deg": (
            q2_qr_error
        ),
        "leading_2_subspace_error_deg": (
            leading_2_subspace_error
        ),
        "estimated_orthogonality_fro_error": (
            estimated_orthogonality_error
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
        "a1_modal": modal_coefficients[0],
        "a2_modal": modal_coefficients[1],
        "abs_a1_modal": abs(
            modal_coefficients[0]
        ),
        "abs_a2_modal": abs(
            modal_coefficients[1]
        ),
        "realized_abs_a2_over_a1_modal": (
            abs(modal_coefficients[1])
            / abs(modal_coefficients[0])
        ),
        "tail_modal_coefficient_l2_norm": float(
            np.linalg.norm(
                modal_coefficients[2:]
            )
        ),
        "x0_norm": float(
            np.linalg.norm(initial_state)
        ),
        "third_eigenvalue_abs": float(
            abs(eigenvalues[2])
        ),
    }

    row.update(system_metrics)

    for stage in (1, 2):
        for metric_name in (
            "direction_change_deg",
            "stage_pc1_energy_fraction",
            "singular_value_ratio_1_to_2",
            "residual_energy_before_fraction",
            "residual_energy_after_fraction",
            "extracted_energy_fraction_original",
        ):
            row[
                f"stage_{stage}_{metric_name}"
            ] = info.get(
                f"stage_{stage}_{metric_name}",
                np.nan,
            )

    return row


def safe_quantile(
    values: pd.Series,
    quantile: float,
) -> float:
    finite = values.dropna()

    if finite.empty:
        return np.nan

    return float(
        finite.quantile(quantile)
    )


def summarise_group(
    group: pd.DataFrame,
) -> pd.Series:
    total = len(group)

    accepted = group[
        group[
            "accepted_by_observation_criteria"
        ]
    ]

    q2_present = group[
        group[
            "q2_present_in_trajectory"
        ]
    ]

    accepted_count = len(accepted)
    present_count = len(q2_present)

    true_q2_count = int(
        group[
            "q2_true_recovered_within_tolerance"
        ].sum()
    )

    qr_q2_count = int(
        group[
            "q2_qr_compatible_recovered_within_tolerance"
        ].sum()
    )

    subspace_count = int(
        group[
            "leading_2_subspace_recovered_within_tolerance"
        ].sum()
    )

    acceptance_low, acceptance_high = (
        wilson_interval(
            accepted_count,
            total,
        )
    )

    true_low, true_high = wilson_interval(
        true_q2_count,
        present_count,
    )

    qr_low, qr_high = wilson_interval(
        qr_q2_count,
        present_count,
    )

    subspace_low, subspace_high = (
        wilson_interval(
            subspace_count,
            present_count,
        )
    )

    accepted_present = accepted[
        accepted[
            "q2_present_in_trajectory"
        ]
    ]

    conditional_true_accuracy = (
        true_q2_count
        / len(accepted_present)
        if len(accepted_present) > 0
        else np.nan
    )

    conditional_qr_accuracy = (
        qr_q2_count
        / len(accepted_present)
        if len(accepted_present) > 0
        else np.nan
    )

    conditional_subspace_accuracy = (
        subspace_count
        / len(accepted_present)
        if len(accepted_present) > 0
        else np.nan
    )

    return pd.Series(
        {
            "n_trials": total,
            "n_q2_present_trials": (
                present_count
            ),
            "n_accepted": accepted_count,
            "acceptance_rate": (
                accepted_count / total
            ),
            "acceptance_rate_wilson95_low": (
                acceptance_low
            ),
            "acceptance_rate_wilson95_high": (
                acceptance_high
            ),
            "n_true_q2_recovered": (
                true_q2_count
            ),
            "true_q2_recovery_rate_given_q2_present": (
                true_q2_count / present_count
                if present_count > 0
                else np.nan
            ),
            "true_q2_recovery_wilson95_low": (
                true_low
            ),
            "true_q2_recovery_wilson95_high": (
                true_high
            ),
            "n_qr_compatible_q2_recovered": (
                qr_q2_count
            ),
            "qr_compatible_q2_recovery_rate_given_q2_present": (
                qr_q2_count / present_count
                if present_count > 0
                else np.nan
            ),
            "qr_q2_recovery_wilson95_low": (
                qr_low
            ),
            "qr_q2_recovery_wilson95_high": (
                qr_high
            ),
            "n_leading_2_subspace_recovered": (
                subspace_count
            ),
            "leading_2_subspace_recovery_rate_given_q2_present": (
                subspace_count / present_count
                if present_count > 0
                else np.nan
            ),
            "subspace_recovery_wilson95_low": (
                subspace_low
            ),
            "subspace_recovery_wilson95_high": (
                subspace_high
            ),
            "conditional_true_q2_accuracy_given_accepted_and_present": (
                conditional_true_accuracy
            ),
            "conditional_qr_q2_accuracy_given_accepted_and_present": (
                conditional_qr_accuracy
            ),
            "conditional_subspace_accuracy_given_accepted_and_present": (
                conditional_subspace_accuracy
            ),
            "median_qhat1_vs_true_q1_deg_accepted": (
                float(
                    accepted[
                        "qhat1_vs_true_q1_deg"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
            "median_qhat2_vs_true_q2_deg_accepted": (
                float(
                    accepted[
                        "qhat2_vs_true_q2_deg"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
            ),
            "q25_qhat2_vs_true_q2_deg_accepted": (
                safe_quantile(
                    accepted[
                        "qhat2_vs_true_q2_deg"
                    ],
                    0.25,
                )
            ),
            "q75_qhat2_vs_true_q2_deg_accepted": (
                safe_quantile(
                    accepted[
                        "qhat2_vs_true_q2_deg"
                    ],
                    0.75,
                )
            ),
            "median_qhat2_vs_qr_compatible_q2_deg_accepted": (
                float(
                    accepted[
                        "qhat2_vs_qr_compatible_q2_deg"
                    ].median()
                )
                if accepted_count > 0
                else np.nan
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
            "median_eigenvector_condition_number": (
                float(
                    group[
                        "eigenvector_condition_number"
                    ].median()
                )
            ),
            "median_relative_nonnormality": (
                float(
                    group[
                        "relative_nonnormality"
                    ].median()
                )
            ),
            "theoretical_orthogonal_minimax_floor_deg": (
                float(
                    group[
                        "theoretical_orthogonal_minimax_floor_deg"
                    ].median()
                )
            ),
            "q1_fixed_q2_error_floor_deg": (
                float(
                    group[
                        "q1_fixed_q2_error_floor_deg"
                    ].median()
                )
            ),
        }
    )


def build_summaries(
    all_trials: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    cell_keys = [
        "lambda2",
        "lambda2_sign",
        "spectral_gap_abs",
        "lambda2_abs_over_lambda1_abs",
        "eigenvector_angle_deg",
        "excitation_ratio_abs_a2_over_a1",
    ]

    by_cell = (
        all_trials.groupby(
            cell_keys,
            dropna=False,
        )
        .apply(summarise_group)
        .reset_index()
        .sort_values(
            [
                "lambda2",
                "eigenvector_angle_deg",
                "excitation_ratio_abs_a2_over_a1",
            ]
        )
    )

    by_angle = (
        all_trials.groupby(
            ["eigenvector_angle_deg"],
            dropna=False,
        )
        .apply(summarise_group)
        .reset_index()
        .sort_values(
            "eigenvector_angle_deg",
            ascending=False,
        )
    )

    by_lambda_angle = (
        all_trials.groupby(
            [
                "lambda2",
                "lambda2_sign",
                "spectral_gap_abs",
                "eigenvector_angle_deg",
            ],
            dropna=False,
        )
        .apply(summarise_group)
        .reset_index()
        .sort_values(
            [
                "lambda2",
                "eigenvector_angle_deg",
            ]
        )
    )

    return (
        by_cell,
        by_angle,
        by_lambda_angle,
    )


def create_row_label(
    lambda2: float,
    angle: float,
    gap: float,
) -> str:
    return (
        f"lambda2={lambda2:+.2f}, "
        f"gap={gap:.2f}, "
        f"angle={angle:.0f}"
    )


def plot_heatmap(
    *,
    summary_by_cell: pd.DataFrame,
    value_column: str,
    title: str,
    colourbar_label: str,
    output_path: Path,
    value_format: str,
) -> None:
    plotting = (
        summary_by_cell.copy()
    )

    plotting["row_label"] = [
        create_row_label(
            float(lambda2),
            float(angle),
            float(gap),
        )
        for lambda2, angle, gap in zip(
            plotting["lambda2"],
            plotting["eigenvector_angle_deg"],
            plotting["spectral_gap_abs"],
        )
    ]

    row_order = (
        plotting[
            [
                "lambda2",
                "eigenvector_angle_deg",
                "row_label",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "lambda2",
                "eigenvector_angle_deg",
            ],
            ascending=[False, False],
        )
        ["row_label"]
        .tolist()
    )

    pivot = plotting.pivot(
        index="row_label",
        columns=(
            "excitation_ratio_abs_a2_over_a1"
        ),
        values=value_column,
    )

    pivot = pivot.reindex(
        row_order
    )

    figure_height = max(
        7.0,
        0.38 * len(pivot.index) + 2.0,
    )

    fig, ax = plt.subplots(
        figsize=(11, figure_height)
    )

    image = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        origin="upper",
    )

    ax.set_xticks(
        np.arange(len(pivot.columns))
    )
    ax.set_xticklabels(
        [
            f"{float(value):g}"
            for value in pivot.columns
        ]
    )

    ax.set_yticks(
        np.arange(len(pivot.index))
    )
    ax.set_yticklabels(
        pivot.index
    )

    ax.set_xlabel(
        r"Controlled modal excitation $|a_2/a_1|$"
    )
    ax.set_ylabel(
        "Signed lambda2, magnitude gap, and "
        "angle(q1,q2) in degrees"
    )
    ax.set_title(title)

    for row_index in range(
        len(pivot.index)
    ):
        for column_index in range(
            len(pivot.columns)
        ):
            value = pivot.iloc[
                row_index,
                column_index,
            ]

            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    format(
                        value,
                        value_format,
                    ),
                    ha="center",
                    va="center",
                )

    colourbar = fig.colorbar(
        image,
        ax=ax,
    )
    colourbar.set_label(
        colourbar_label
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def plot_aggregate_recovery_by_angle(
    *,
    all_trials: pd.DataFrame,
    output_path: Path,
) -> None:
    # Exclude the negative control because q2 is absent there.
    present = all_trials[
        all_trials[
            "q2_present_in_trajectory"
        ]
    ]

    rows: list[dict] = []

    for angle, group in present.groupby(
        "eigenvector_angle_deg",
        sort=True,
    ):
        total = len(group)

        rows.append(
            {
                "angle": angle,
                "true_q2": (
                    group[
                        "q2_true_recovered_within_tolerance"
                    ].sum()
                    / total
                ),
                "qr_q2": (
                    group[
                        "q2_qr_compatible_recovered_within_tolerance"
                    ].sum()
                    / total
                ),
                "subspace": (
                    group[
                        "leading_2_subspace_recovered_within_tolerance"
                    ].sum()
                    / total
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    ).sort_values("angle")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        frame["angle"],
        frame["true_q2"],
        marker="o",
        label="true q2",
    )

    ax.plot(
        frame["angle"],
        frame["qr_q2"],
        marker="o",
        label="QR-compatible q2 residual",
    )

    ax.plot(
        frame["angle"],
        frame["subspace"],
        marker="o",
        label="leading 2D subspace",
    )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(
        r"Angle between true right eigenvectors "
        r"$q_1$ and $q_2$ (degrees)"
    )
    ax.set_ylabel(
        "Empirical recovery rate"
    )
    ax.set_title(
        "What the orthogonal SVD estimator recovers "
        "as non-normality increases"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def plot_aggregate_error_by_angle(
    *,
    all_trials: pd.DataFrame,
    output_path: Path,
) -> None:
    accepted_present = all_trials[
        all_trials[
            "accepted_by_observation_criteria"
        ]
        & all_trials[
            "q2_present_in_trajectory"
        ]
    ]

    rows: list[dict] = []

    for angle, group in accepted_present.groupby(
        "eigenvector_angle_deg",
        sort=True,
    ):
        rows.append(
            {
                "angle": angle,
                "true_q2_error": float(
                    group[
                        "qhat2_vs_true_q2_deg"
                    ].median()
                ),
                "qr_q2_error": float(
                    group[
                        "qhat2_vs_qr_compatible_q2_deg"
                    ].median()
                ),
                "subspace_error": float(
                    group[
                        "leading_2_subspace_error_deg"
                    ].median()
                ),
                "orthogonal_minimax_floor": float(
                    group[
                        "theoretical_orthogonal_minimax_floor_deg"
                    ].median()
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    ).sort_values("angle")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        frame["angle"],
        frame["true_q2_error"],
        marker="o",
        label="qhat2 versus true q2",
    )

    ax.plot(
        frame["angle"],
        frame["qr_q2_error"],
        marker="o",
        label="qhat2 versus QR-compatible q2",
    )

    ax.plot(
        frame["angle"],
        frame["subspace_error"],
        marker="o",
        label="leading 2D subspace error",
    )

    ax.plot(
        frame["angle"],
        frame["orthogonal_minimax_floor"],
        linestyle="--",
        label="orthogonal-basis minimax floor",
    )

    ax.set_xlabel(
        r"Angle between true right eigenvectors "
        r"$q_1$ and $q_2$ (degrees)"
    )
    ax.set_ylabel(
        "Median angular error among accepted trials "
        "(degrees)"
    )
    ax.set_title(
        "Individual directions versus invariant-subspace recovery"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)




def _format_percent(rate: float) -> str:
    if rate is None or not np.isfinite(rate):
        return "—"
    return f"{100.0 * rate:.1f}%"


def _ordered_angles(frame: pd.DataFrame) -> list[float]:
    return sorted(
        [float(value) for value in frame["eigenvector_angle_deg"].unique()],
        reverse=True,
    )


def _ordered_excitations(frame: pd.DataFrame) -> list[float]:
    return sorted(
        [float(value) for value in frame["excitation_ratio_abs_a2_over_a1"].unique()]
    )


def plot_overall_summary_bars(
    *,
    overall: pd.Series,
    output_path: Path,
) -> None:
    labels = [
        "Accepted\n(all trials)",
        "Recovered\ntrue q2",
        "Recovered\nq2 perpendicular part",
        "Recovered\n(q1,q2) plane",
    ]

    rates = [
        float(overall["acceptance_rate"]),
        float(overall["true_q2_recovery_rate_given_q2_present"]),
        float(overall["qr_compatible_q2_recovery_rate_given_q2_present"]),
        float(overall["leading_2_subspace_recovery_rate_given_q2_present"]),
    ]

    counts = [
        (int(overall["n_accepted"]), int(overall["n_trials"])),
        (int(overall["n_true_q2_recovered"]), int(overall["n_q2_present_trials"])),
        (int(overall["n_qr_compatible_q2_recovered"]), int(overall["n_q2_present_trials"])),
        (int(overall["n_leading_2_subspace_recovered"]), int(overall["n_q2_present_trials"])),
    ]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(x, [100.0 * value for value in rates], width=0.65)

    ax.set_ylim(0.0, 105.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Overall summary of accepted and recovered trials")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, rate, (numerator, denominator) in zip(bars, rates, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.0,
            f"{100.0 * rate:.2f}%\n({numerator}/{denominator})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_results_by_angle(
    *,
    summary_by_angle: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = summary_by_angle.sort_values(
        "eigenvector_angle_deg", ascending=False
    ).copy()

    x = frame["eigenvector_angle_deg"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        x,
        100.0 * frame["acceptance_rate"].to_numpy(dtype=float),
        marker="o",
        label="Accepted by observation criteria",
    )
    ax.plot(
        x,
        100.0 * frame[
            "true_q2_recovery_rate_given_q2_present"
        ].to_numpy(dtype=float),
        marker="o",
        label="Recovered true q2",
    )
    ax.plot(
        x,
        100.0 * frame[
            "qr_compatible_q2_recovery_rate_given_q2_present"
        ].to_numpy(dtype=float),
        marker="o",
        label="Recovered q2 perpendicular part",
    )
    ax.plot(
        x,
        100.0 * frame[
            "leading_2_subspace_recovery_rate_given_q2_present"
        ].to_numpy(dtype=float),
        marker="o",
        label="Recovered (q1,q2) plane",
    )

    ax.set_xticks(x)
    ax.set_ylim(0.0, 105.0)
    ax.set_xlabel(r"Angle between $q_1$ and $q_2$ (degrees)")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Results by angle between the first two true eigendirections")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_true_q2_error_relationship(
    *,
    summary_by_angle: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = summary_by_angle.sort_values(
        "eigenvector_angle_deg", ascending=False
    ).copy()

    angle = frame["eigenvector_angle_deg"].to_numpy(dtype=float)
    observed_true_error = frame[
        "median_qhat2_vs_true_q2_deg_accepted"
    ].to_numpy(dtype=float)
    observed_qr_error = frame[
        "median_qhat2_vs_qr_compatible_q2_deg_accepted"
    ].to_numpy(dtype=float)
    expected_true_error = 90.0 - angle

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        angle,
        observed_true_error,
        marker="o",
        label=r"Median $\angle(\hat u_2,q_2)$",
    )
    ax.plot(
        angle,
        expected_true_error,
        linestyle="--",
        marker="o",
        label=r"Expected $90^\circ-\theta$",
    )
    ax.plot(
        angle,
        observed_qr_error,
        marker="o",
        label=r"Median $\angle(\hat u_2,q_{2,\perp})$",
    )

    ax.set_xticks(angle)
    ax.set_xlabel(r"Angle between $q_1$ and $q_2$ (degrees)")
    ax.set_ylabel("Median angular error (degrees)")
    ax.set_title("The estimated second direction follows the perpendicular part of q2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_simple_heatmap(
    *,
    matrix: pd.DataFrame,
    title: str,
    colourbar_label: str,
    output_path: Path,
    value_format: str = ".2f",
    as_percent: bool = True,
) -> None:
    plot_values = matrix.to_numpy(dtype=float)
    show_values = 100.0 * plot_values if as_percent else plot_values

    fig_height = max(4.5, 0.75 * len(matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    image = ax.imshow(show_values, aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([f"{float(value):g}" for value in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([f"{float(value):.0f}" for value in matrix.index])
    ax.set_xlabel(r"Controlled excitation $|a_2/a_1|$")
    ax.set_ylabel(r"Angle between $q_1$ and $q_2$ (degrees)")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = show_values[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    fontsize=10,
                )
            else:
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colourbar_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_angle_excitation_recovery_heatmap(
    *,
    all_trials: pd.DataFrame,
    metric_column: str,
    title: str,
    colourbar_label: str,
    output_path: Path,
    require_q2_present: bool,
) -> None:
    if require_q2_present:
        frame = all_trials[all_trials["q2_present_in_trajectory"]].copy()
    else:
        frame = all_trials.copy()

    grouped = (
        frame.groupby(
            ["eigenvector_angle_deg", "excitation_ratio_abs_a2_over_a1"],
            dropna=False,
        )[metric_column]
        .mean()
        .reset_index()
    )

    angle_order = _ordered_angles(frame)
    excitation_order = _ordered_excitations(frame)

    pivot = grouped.pivot(
        index="eigenvector_angle_deg",
        columns="excitation_ratio_abs_a2_over_a1",
        values=metric_column,
    )
    pivot = pivot.reindex(index=angle_order, columns=excitation_order)

    if require_q2_present and 0.0 in pivot.columns:
        pivot.loc[:, 0.0] = np.nan

    plot_simple_heatmap(
        matrix=pivot,
        title=title,
        colourbar_label=colourbar_label,
        output_path=output_path,
        value_format=".0f",
        as_percent=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Observation-only q2 validation across controlled "
            "non-normal linear systems. The angle between q1 and q2 "
            "controls eigenvector non-orthogonality."
        )
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.96,
    )
    parser.add_argument(
        "--lambda2-values",
        type=parse_float_list,
        default=parse_float_list(
            "0.92,-0.92,0.88,-0.88"
        ),
    )
    parser.add_argument(
        "--eigenvector-angles-deg",
        type=parse_float_list,
        default=parse_float_list(
            "90,75,60,45,30"
        ),
    )
    parser.add_argument(
        "--excitation-ratios",
        type=parse_float_list,
        default=parse_float_list(
            "0,0.03,0.1,0.3,1,3"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
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
        "--subspace-angle-tolerance-deg",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "observation_only_q2_"
            "nonnormal_angle_excitation_sweep"
        ),
    )

    args = parser.parse_args()

    cfg = SweepConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        lambda1=args.lambda1,
        lambda2_values=tuple(
            args.lambda2_values
        ),
        eigenvector_angles_deg=tuple(
            args.eigenvector_angles_deg
        ),
        excitation_ratios=tuple(
            args.excitation_ratios
        ),
        system_replicates=(
            args.system_replicates
        ),
        trials_per_system=(
            args.trials_per_system
        ),
        seed=args.seed,
        other_mode_scale=(
            args.other_mode_scale
        ),
        tail_max=args.tail_max,
        tail_min=args.tail_min,
        tail_gap_below_lambda2=(
            args.tail_gap_below_lambda2
        ),
        stability_threshold_deg=(
            args.stability_threshold_deg
        ),
        stability_patience=(
            args.stability_patience
        ),
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
        subspace_angle_tolerance_deg=(
            args.subspace_angle_tolerance_deg
        ),
    )

    validate_config(cfg)

    output = args.output
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    estimator_cfg = (
        make_estimator_config(cfg)
    )

    total_trajectories = (
        len(cfg.lambda2_values)
        * len(cfg.eigenvector_angles_deg)
        * len(cfg.excitation_ratios)
        * cfg.system_replicates
        * cfg.trials_per_system
    )

    print(
        "============================================================"
    )
    print(
        "Observation-only q2 sweep: controlled non-normal systems"
    )
    print(
        "============================================================"
    )
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"window: {cfg.window}")
    print(f"lambda1: {cfg.lambda1}")
    print(
        f"signed lambda2 values: "
        f"{cfg.lambda2_values}"
    )
    print(
        "q1-q2 eigenvector angles: "
        f"{cfg.eigenvector_angles_deg}"
    )
    print(
        "controlled modal |a2/a1| values: "
        f"{cfg.excitation_ratios}"
    )
    print(
        "system replicates per cell: "
        f"{cfg.system_replicates}"
    )
    print(
        "trials per system and excitation: "
        f"{cfg.trials_per_system}"
    )
    print(
        f"total trajectories: "
        f"{total_trajectories}"
    )
    print(
        "The estimator receives only X and L. "
        "Eigenvectors and subspaces are used only afterward "
        "for synthetic validation."
    )
    print(
        "============================================================"
    )

    config_json = asdict(cfg)
    config_json[
        "lambda2_values"
    ] = list(cfg.lambda2_values)
    config_json[
        "eigenvector_angles_deg"
    ] = list(
        cfg.eigenvector_angles_deg
    )
    config_json[
        "excitation_ratios"
    ] = list(
        cfg.excitation_ratios
    )

    with (
        output / "experiment_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
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
        magnitude_gap = (
            abs(cfg.lambda1)
            - abs(lambda2)
        )

        for angle_index, eigenvector_angle in enumerate(
            cfg.eigenvector_angles_deg
        ):
            for system_replicate in range(
                cfg.system_replicates
            ):
                system_seed = (
                    cfg.seed
                    + 100_000_000
                    * lambda2_index
                    + 1_000_000
                    * angle_index
                    + 10_000
                    * system_replicate
                )

                (
                    matrix_a,
                    true_right_eigenvectors,
                    eigenvalues,
                    system_metrics,
                ) = build_controlled_nonnormal_system(
                    cfg=cfg,
                    lambda2=lambda2,
                    eigenvector_angle_deg=(
                        eigenvector_angle
                    ),
                    system_seed=system_seed,
                )

                system_row = {
                    "lambda1": cfg.lambda1,
                    "lambda2": lambda2,
                    "lambda2_sign": int(
                        np.sign(lambda2)
                    ),
                    "spectral_gap_abs": (
                        magnitude_gap
                    ),
                    "eigenvector_angle_deg": (
                        eigenvector_angle
                    ),
                    "system_replicate": (
                        system_replicate
                    ),
                    "system_seed": (
                        system_seed
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

                system_row.update(
                    system_metrics
                )
                system_rows.append(
                    system_row
                )

                for ratio_index, ratio in enumerate(
                    cfg.excitation_ratios
                ):
                    for trial_within_system in range(
                        cfg.trials_per_system
                    ):
                        trial_seed = (
                            cfg.seed
                            + 1_000_000_000
                            * lambda2_index
                            + 10_000_000
                            * angle_index
                            + 100_000
                            * system_replicate
                            + 1_000
                            * ratio_index
                            + trial_within_system
                        )

                        trial_rows.append(
                            analyse_one_trial(
                                cfg=cfg,
                                estimator_cfg=(
                                    estimator_cfg
                                ),
                                matrix_a=matrix_a,
                                true_right_eigenvectors=(
                                    true_right_eigenvectors
                                ),
                                eigenvalues=(
                                    eigenvalues
                                ),
                                system_metrics=(
                                    system_metrics
                                ),
                                lambda2=lambda2,
                                eigenvector_angle_deg=(
                                    eigenvector_angle
                                ),
                                system_replicate=(
                                    system_replicate
                                ),
                                system_seed=(
                                    system_seed
                                ),
                                excitation_ratio=(
                                    ratio
                                ),
                                trial_within_system=(
                                    trial_within_system
                                ),
                                trial_seed=(
                                    trial_seed
                                ),
                            )
                        )

                        completed += 1

                print(
                    f"completed lambda2={lambda2:+.3f}, "
                    f"gap={magnitude_gap:.3f}, "
                    f"angle={eigenvector_angle:.1f}, "
                    f"system "
                    f"{system_replicate + 1}/"
                    f"{cfg.system_replicates}; "
                    f"{completed}/"
                    f"{total_trajectories} trajectories"
                )

    all_trials = pd.DataFrame(
        trial_rows
    )

    all_trials["acceptance_rate_placeholder"] = all_trials[
        "accepted_by_observation_criteria"
    ].astype(float)

    all_trials.to_csv(
        output / "all_trial_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        system_rows
    ).to_csv(
        output / "system_definitions.csv",
        index=False,
    )

    (
        summary_by_cell,
        summary_by_angle,
        summary_by_lambda_angle,
    ) = build_summaries(
        all_trials
    )

    summary_by_cell.to_csv(
        output
        / "summary_by_lambda_angle_excitation.csv",
        index=False,
    )

    summary_by_angle.to_csv(
        output
        / "summary_by_eigenvector_angle.csv",
        index=False,
    )

    summary_by_lambda_angle.to_csv(
        output
        / "summary_by_lambda_and_angle.csv",
        index=False,
    )
    plot_overall_summary_bars(
        overall=summarise_group(all_trials),
        output_path=(
            output / "01_overall_summary_rates.png"
        ),
    )

    plot_results_by_angle(
        summary_by_angle=summary_by_angle,
        output_path=(
            output / "02_results_by_angle.png"
        ),
    )

    plot_true_q2_error_relationship(
        summary_by_angle=summary_by_angle,
        output_path=(
            output / "03_true_q2_error_vs_angle.png"
        ),
    )

    plot_angle_excitation_recovery_heatmap(
        all_trials=all_trials,
        metric_column="acceptance_rate_placeholder",
        title="Observation-only acceptance rate by angle and excitation",
        colourbar_label="acceptance rate (%)",
        output_path=(output / "04_acceptance_by_angle_and_excitation.png"),
        require_q2_present=False,
    )

    plot_angle_excitation_recovery_heatmap(
        all_trials=all_trials,
        metric_column="q2_true_recovered_within_tolerance",
        title="True q2 recovery by angle and excitation",
        colourbar_label="true q2 recovery rate (%)",
        output_path=(output / "05_true_q2_recovery_by_angle_and_excitation.png"),
        require_q2_present=True,
    )

    plot_angle_excitation_recovery_heatmap(
        all_trials=all_trials,
        metric_column="q2_qr_compatible_recovered_within_tolerance",
        title="Recovery of the perpendicular part of q2 by angle and excitation",
        colourbar_label="q2 perpendicular-part recovery rate (%)",
        output_path=(output / "06_q2_perpendicular_recovery_by_angle_and_excitation.png"),
        require_q2_present=True,
    )

    overall = summarise_group(

        all_trials
    )
    overall_frame = (
        overall.to_frame().T
    )

    overall_frame.insert(
        0,
        "lambda1",
        cfg.lambda1,
    )
    overall_frame.insert(
        1,
        "n_lambda2_values",
        len(cfg.lambda2_values),
    )
    overall_frame.insert(
        2,
        "n_eigenvector_angles",
        len(
            cfg.eigenvector_angles_deg
        ),
    )
    overall_frame.insert(
        3,
        "n_excitation_ratios",
        len(cfg.excitation_ratios),
    )
    overall_frame.insert(
        4,
        "system_replicates",
        cfg.system_replicates,
    )
    overall_frame.insert(
        5,
        "trials_per_system",
        cfg.trials_per_system,
    )

    overall_frame.to_csv(
        output / "overall_summary.csv",
        index=False,
    )

    print(
        "\nOverall results across the complete non-normal sweep:"
    )
    print(
        "acceptance rate: "
        f"{overall['acceptance_rate']:.4f}"
    )
    print(
        "true q2 recovery rate given q2 is present: "
        f"{overall['true_q2_recovery_rate_given_q2_present']:.4f}"
    )
    print(
        "QR-compatible q2 recovery rate given q2 is present: "
        f"{overall['qr_compatible_q2_recovery_rate_given_q2_present']:.4f}"
    )
    print(
        "leading 2D subspace recovery rate given q2 is present: "
        f"{overall['leading_2_subspace_recovery_rate_given_q2_present']:.4f}"
    )
    print(
        "median accepted true-q2 error: "
        f"{overall['median_qhat2_vs_true_q2_deg_accepted']:.6f} degrees"
    )
    print(
        "median accepted QR-compatible-q2 error: "
        f"{overall['median_qhat2_vs_qr_compatible_q2_deg_accepted']:.6f} degrees"
    )
    print(
        "median accepted leading-subspace error: "
        f"{overall['median_leading_2_subspace_error_deg_accepted']:.6f} degrees"
    )

    print(
        "\nInterpretation:"
        "\n- true-q2 recovery asks whether the orthogonal SVD direction equals "
        "the non-orthogonal right eigenvector q2;"
        "\n- QR-compatible recovery compares qhat2 with q2 after removing its "
        "q1 component;"
        "\n- subspace recovery asks whether span(qhat1,qhat2) equals "
        "span(q1,q2)."
    )

    print(
        f"\nResults written to: "
        f"{output.resolve()}"
    )


if __name__ == "__main__":
    main()
