from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EPS = 1e-14


def _load_estimator_module():
    """Load the existing same-window observation-only estimator."""
    canonical_name = "run_observation_only_same_window_deflation_normal"
    try:
        return __import__(canonical_name)
    except ImportError:
        pass

    candidates = [
        SCRIPT_DIR / "run_observation_only_same_window_deflation_normal.py",
        REPO_ROOT / "run_observation_only_same_window_deflation_normal.py",
        REPO_ROOT / "experiments" / "run_observation_only_same_window_deflation_normal.py",
        SCRIPT_DIR / "run_observation_only_same_window_deflation_normal(2).py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(canonical_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[canonical_name] = module
        spec.loader.exec_module(module)
        return module

    searched = "\n".join("  - %s" % p for p in candidates)
    raise ImportError(
        "Could not locate run_observation_only_same_window_deflation_normal.py.\n"
        "Searched:\n%s" % searched
    )


ESTIMATOR = _load_estimator_module()
EstimatorConfig = ESTIMATOR.Config
estimate_from_observations_only = ESTIMATOR.estimate_from_observations_only
simulate_trajectory = ESTIMATOR.simulate_trajectory
angle_deg = ESTIMATOR.angle_deg
select_common_window = ESTIMATOR.select_common_window


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20

    lambda1: float = 0.96
    lambda2_values: Tuple[float, ...] = (0.95, 0.94, 0.92, 0.90, 0.88)
    excitation_ratios: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.10, 0.30, 1.0, 3.0)

    system_replicates: int = 10
    initial_states_per_system: int = 20
    seed: int = 42

    # q3 and lower modes.
    other_mode_scale: float = 0.25
    tail_max: float = 0.84
    tail_min: float = 0.20
    tail_gap_below_lambda2: float = 0.02

    # Observation-only acceptance criteria. These are frozen before evaluation.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External validation tolerance only.
    recovery_tolerance_deg: float = 1.0

    # Hierarchical bootstrap.
    bootstrap_replicates: int = 2000


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number.")
    return values


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.dim < 3:
        raise ValueError("dim must be at least 3.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if cfg.window < 2 or cfg.window > cfg.steps + 1:
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if not (0.0 < cfg.lambda1 < 1.0):
        raise ValueError("lambda1 must lie in (0,1).")
    if cfg.system_replicates < 1 or cfg.initial_states_per_system < 1:
        raise ValueError("replicate counts must be positive.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be at least 100.")
    if cfg.recovery_tolerance_deg <= 0.0:
        raise ValueError("recovery_tolerance_deg must be positive.")
    if cfg.tail_min <= 0.0 or cfg.tail_max <= cfg.tail_min:
        raise ValueError("Require 0 < tail_min < tail_max.")
    if cfg.tail_gap_below_lambda2 <= 0.0:
        raise ValueError("tail_gap_below_lambda2 must be positive.")

    abs_l2 = [abs(v) for v in cfg.lambda2_values]
    if len(set(round(v, 14) for v in abs_l2)) != len(abs_l2):
        raise ValueError(
            "Experiment 2 uses spectral magnitude g as the controlled factor; "
            "lambda2_values must therefore have distinct absolute magnitudes."
        )

    for l2 in cfg.lambda2_values:
        if not (0.0 < abs(l2) < cfg.lambda1):
            raise ValueError("Each lambda2 must satisfy 0 < |lambda2| < lambda1.")
        available_tail_max = min(cfg.tail_max, abs(l2) - cfg.tail_gap_below_lambda2)
        if available_tail_max <= cfg.tail_min:
            raise ValueError(
                "No valid tail interval for lambda2=%g. Reduce tail_min or tail gap." % l2
            )

    for ratio in cfg.excitation_ratios:
        if ratio < 0.0:
            raise ValueError("Excitation ratios |a2/a1| must be nonnegative.")


def _estimator_config_kwargs(cfg: ExperimentConfig, n_directions: int) -> Dict[str, object]:
    """Build only Config fields supported by the estimator version in the repo."""
    desired = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": 1,
        "window": cfg.window,
        "n_directions": n_directions,
        "seed": cfg.seed,
        "stability_threshold_deg": cfg.stability_threshold_deg,
        "stability_patience": cfg.stability_patience,
        "relative_window_norm_floor": cfg.relative_window_norm_floor,
        "min_residual_energy_fraction": cfg.min_residual_energy_fraction,
        "numeric_relative_residual_floor": cfg.numeric_relative_residual_floor,
        "min_stage_pc1_energy_fraction": cfg.min_stage_pc1_energy_fraction,
        "recovery_tolerance_deg": cfg.recovery_tolerance_deg,
    }
    supported = {f.name for f in fields(EstimatorConfig)}
    return {k: v for k, v in desired.items() if k in supported}


def make_estimator_config(cfg: ExperimentConfig, n_directions: int) -> EstimatorConfig:
    return EstimatorConfig(**_estimator_config_kwargs(cfg, n_directions))


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        raise ValueError("Cannot normalize a non-finite or near-zero vector.")
    return v / n


def top_right_singular_direction(matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return PC1 direction, PC1 energy fraction, and sigma1/sigma2."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if float(np.linalg.norm(matrix, ord="fro")) <= EPS:
        return np.full(matrix.shape[1], np.nan), np.nan, np.nan

    _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    direction = normalize(vt[0])
    energy = float(np.sum(s * s))
    pc1 = float((s[0] * s[0]) / energy) if energy > EPS else np.nan
    ratio = float(s[0] / s[1]) if len(s) > 1 and s[1] > EPS else np.inf
    return direction, pc1, ratio


def deflate_window(error_window: np.ndarray, direction: np.ndarray) -> np.ndarray:
    q = normalize(direction)
    return error_window @ (np.eye(error_window.shape[1]) - np.outer(q, q))


def fro_energy(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord="fro") ** 2)


def theoretical_equal_amplitude_time(
    lambda1: float, lambda2: float, excitation_ratio: float
) -> float:
    if excitation_ratio <= 0.0:
        return np.nan
    return float(math.log(excitation_ratio) / math.log(lambda1 / abs(lambda2)))


def build_random_normal_system(
    cfg: ExperimentConfig, lambda2: float, system_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(system_seed)
    gaussian = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(gaussian)

    available_tail_max = min(cfg.tail_max, abs(lambda2) - cfg.tail_gap_below_lambda2)
    tail_magnitudes = rng.uniform(cfg.tail_min, available_tail_max, size=cfg.dim - 2)
    tail_magnitudes = np.sort(tail_magnitudes)[::-1]
    tail_signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.dim - 2)
    tail = tail_signs * tail_magnitudes

    eigenvalues = np.concatenate((np.array([cfg.lambda1, lambda2]), tail))
    A = Q @ np.diag(eigenvalues) @ Q.T
    normality_error = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    if normality_error > 1e-10:
        raise RuntimeError("Constructed system is not numerically normal.")
    return A, Q, eigenvalues, normality_error


def construct_controlled_initial_state(
    true_basis: np.ndarray,
    excitation_ratio: float,
    other_mode_scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = true_basis.shape[0]
    coefficients = rng.normal(0.0, other_mode_scale, size=dim)
    coefficients[0] = float(rng.choice(np.array([-1.0, 1.0])))
    coefficients[1] = float(rng.choice(np.array([-1.0, 1.0]))) * excitation_ratio
    x0 = true_basis @ coefficients
    return x0, coefficients


def correctness_value(accepted: bool, error_deg: float, tolerance: float) -> Optional[bool]:
    """Correctness is not evaluated when the stage is rejected."""
    if not accepted:
        return None
    if not np.isfinite(error_deg):
        return False
    return bool(error_deg <= tolerance)


def target_q2_correctness(
    accepted: bool,
    excitation_ratio: float,
    q2_error_deg: float,
    tolerance: float,
) -> Optional[bool]:
    """At alpha=0, any accepted stage-2 direction is a target-specific false accept."""
    if not accepted:
        return None
    if np.isclose(excitation_ratio, 0.0):
        return False
    if not np.isfinite(q2_error_deg):
        return False
    return bool(q2_error_deg <= tolerance)


def bool_or_nan(value: Optional[bool]):
    if value is None:
        return np.nan
    return bool(value)


def analyse_one_trial(
    *,
    cfg: ExperimentConfig,
    cfg_stage1: EstimatorConfig,
    cfg_stage2: EstimatorConfig,
    A: np.ndarray,
    true_basis: np.ndarray,
    eigenvalues: np.ndarray,
    lambda2: float,
    lambda2_index: int,
    system_replicate: int,
    system_seed: int,
    excitation_ratio: float,
    excitation_index: int,
    trial_within_system: int,
    trial_seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(trial_seed)
    x0, coefficients = construct_controlled_initial_state(
        true_basis, excitation_ratio, cfg.other_mode_scale, rng
    )
    X = simulate_trajectory(A=A, x0=x0, steps=cfg.steps)
    L = np.zeros(cfg.dim, dtype=float)

    # One rolling pass with two directions. Stage-1 acceptance is recovered by
    # applying the one-direction acceptance rule to the same diagnostics.
    directions2, info2, diagnostics = estimate_from_observations_only(
        X=X, L=L, cfg=cfg_stage2
    )

    selected1, n_candidates1 = select_common_window(diagnostics, cfg_stage1)
    A1 = selected1 is not None and selected1.get("direction_1") is not None
    q1_stage1_error = np.nan
    C1 = None
    stage1_window_start = np.nan
    stage1_window_end = np.nan
    if A1:
        u1_stage1 = normalize(np.asarray(selected1["direction_1"], dtype=float))
        q1_stage1_error = angle_deg(u1_stage1, true_basis[:, 0])
        C1 = correctness_value(True, q1_stage1_error, cfg.recovery_tolerance_deg)
        stage1_window_start = int(selected1["window_start"])
        stage1_window_end = int(selected1["window_end"])

    A2 = bool(info2.get("success", False) and len(directions2) == 2)

    q1_at_stage2_error = np.nan
    q2_estimated_error = np.nan
    C1_at_stage2 = None
    C2 = None
    selected_window_start = np.nan
    selected_window_end = np.nan

    oracle_q2_error = np.nan
    oracle_q2_correct = None
    oracle_pc1_fraction = np.nan
    oracle_sv_ratio = np.nan
    estimated_rebuilt_q2_error = np.nan
    estimated_residual_energy_fraction = np.nan
    oracle_residual_energy_fraction = np.nan
    residual_difference_relative_to_oracle = np.nan
    estimated_q2_energy_fraction = np.nan
    oracle_q2_energy_fraction = np.nan
    q2_error_penalty_estimated_minus_oracle = np.nan

    if A2:
        u1_stage2 = normalize(np.asarray(directions2[0], dtype=float))
        u2_stage2 = normalize(np.asarray(directions2[1], dtype=float))
        q1_at_stage2_error = angle_deg(u1_stage2, true_basis[:, 0])
        q2_estimated_error = angle_deg(u2_stage2, true_basis[:, 1])
        C1_at_stage2 = correctness_value(
            True, q1_at_stage2_error, cfg.recovery_tolerance_deg
        )
        C2 = target_q2_correctness(
            True, excitation_ratio, q2_estimated_error, cfg.recovery_tolerance_deg
        )

        selected_window_start = int(info2["window_start"])
        selected_window_end = int(info2["window_end"])
        R = X[selected_window_start : selected_window_end + 1] - L

        # Framework comparison: same original selected window, exact q1 removal
        # versus observation-only estimated q1 removal.
        R_est = deflate_window(R, u1_stage2)
        R_ref = deflate_window(R, true_basis[:, 0])
        e0 = fro_energy(R)
        e_est = fro_energy(R_est)
        e_ref = fro_energy(R_ref)
        estimated_residual_energy_fraction = e_est / e0 if e0 > EPS else np.nan
        oracle_residual_energy_fraction = e_ref / e0 if e0 > EPS else np.nan
        residual_difference_relative_to_oracle = (
            float(np.linalg.norm(R_est - R_ref, ord="fro")) /
            float(np.linalg.norm(R_ref, ord="fro"))
            if float(np.linalg.norm(R_ref, ord="fro")) > EPS else np.nan
        )

        oracle_u2, oracle_pc1_fraction, oracle_sv_ratio = top_right_singular_direction(R_ref)
        rebuilt_u2, _rebuilt_pc1, _rebuilt_ratio = top_right_singular_direction(R_est)
        if np.all(np.isfinite(oracle_u2)):
            oracle_q2_error = angle_deg(oracle_u2, true_basis[:, 1])
            oracle_q2_correct = target_q2_correctness(
                True, excitation_ratio, oracle_q2_error, cfg.recovery_tolerance_deg
            )
        if np.all(np.isfinite(rebuilt_u2)):
            estimated_rebuilt_q2_error = angle_deg(rebuilt_u2, true_basis[:, 1])

        # Ground-truth diagnostics only: amount of q2 remaining after each deflation.
        q2 = true_basis[:, 1]
        if e_est > EPS:
            estimated_q2_energy_fraction = float(np.linalg.norm(R_est @ q2) ** 2 / e_est)
        if e_ref > EPS:
            oracle_q2_energy_fraction = float(np.linalg.norm(R_ref @ q2) ** 2 / e_ref)

        if np.isfinite(q2_estimated_error) and np.isfinite(oracle_q2_error):
            q2_error_penalty_estimated_minus_oracle = q2_estimated_error - oracle_q2_error

    accepted_and_correct1 = bool(A1 and C1 is True)
    false_accept1 = bool(A1 and C1 is False)
    accepted_and_correct2 = bool(A2 and C2 is True)
    false_accept2 = bool(A2 and C2 is False)

    system_uid = "l2_%02d_rep_%03d" % (lambda2_index, system_replicate)

    row: Dict[str, object] = {
        "lambda1": cfg.lambda1,
        "lambda2": lambda2,
        "lambda2_abs": abs(lambda2),
        "spectral_ratio_g_abs_lambda2_over_lambda1": abs(lambda2) / cfg.lambda1,
        "spectral_separation_1_minus_g": 1.0 - abs(lambda2) / cfg.lambda1,
        "spectral_gap_magnitude": cfg.lambda1 - abs(lambda2),
        "excitation_ratio_abs_a2_over_a1": excitation_ratio,
        "theoretical_equal_amplitude_time": theoretical_equal_amplitude_time(
            cfg.lambda1, lambda2, excitation_ratio
        ),
        "lambda2_index": lambda2_index,
        "system_replicate": system_replicate,
        "system_uid": system_uid,
        "system_seed": system_seed,
        "excitation_index": excitation_index,
        "initial_state_within_system": trial_within_system,
        "trial_seed": trial_seed,
        "a1": coefficients[0],
        "a2": coefficients[1],
        "abs_a1": abs(coefficients[0]),
        "abs_a2": abs(coefficients[1]),
        "realized_abs_a2_over_a1": abs(coefficients[1]) / abs(coefficients[0]),
        "tail_coefficient_l2_norm": float(np.linalg.norm(coefficients[2:])),
        "x0_norm": float(np.linalg.norm(x0)),
        "third_eigenvalue_abs": float(abs(eigenvalues[2])),

        # Stage 1: A1 and C1 are distinct; C1 is blank when A1=0.
        "A1_accepted": bool(A1),
        "C1_correct": bool_or_nan(C1),
        "A1_and_C1": accepted_and_correct1,
        "A1_and_not_C1": false_accept1,
        "q1_error_deg_stage1_selected": q1_stage1_error,
        "stage1_selected_window_start": stage1_window_start,
        "stage1_selected_window_end": stage1_window_end,
        "stage1_n_acceptable_windows": int(n_candidates1),

        # Stage 2: A2 means a joint same-window stage-1/stage-2 estimate was selected.
        "A2_accepted": bool(A2),
        "C2_correct": bool_or_nan(C2),
        "A2_and_C2": accepted_and_correct2,
        "A2_and_not_C2": false_accept2,
        "C1_at_stage2_window": bool_or_nan(C1_at_stage2),
        "q1_error_deg_at_stage2_window": q1_at_stage2_error,
        "q2_error_deg_estimated_sequence": q2_estimated_error,
        "stage2_selected_window_start": selected_window_start,
        "stage2_selected_window_end": selected_window_end,
        "stage2_n_acceptable_windows": int(info2.get("n_common_stable_candidates", 0)),

        # Same-selected-window reference-filter comparison.
        "oracle_q2_error_deg_same_window": oracle_q2_error,
        "oracle_q2_correct_same_window": bool_or_nan(oracle_q2_correct),
        "q2_error_penalty_estimated_minus_oracle_deg": q2_error_penalty_estimated_minus_oracle,
        "oracle_stage2_pc1_energy_fraction_same_window": oracle_pc1_fraction,
        "oracle_stage2_singular_value_ratio_1_to_2_same_window": oracle_sv_ratio,
        "rebuilt_estimated_q2_error_deg_same_window": estimated_rebuilt_q2_error,
        "estimated_q1_deflation_residual_energy_fraction": estimated_residual_energy_fraction,
        "reference_q1_deflation_residual_energy_fraction": oracle_residual_energy_fraction,
        "estimated_vs_reference_residual_relative_fro_difference": residual_difference_relative_to_oracle,
        "q2_energy_fraction_after_estimated_q1_deflation": estimated_q2_energy_fraction,
        "q2_energy_fraction_after_reference_q1_deflation": oracle_q2_energy_fraction,
        "recovery_tolerance_deg": cfg.recovery_tolerance_deg,
    }

    # Keep the estimator's own selected-window diagnostics.
    for stage in (1, 2):
        for name in (
            "direction_change_deg",
            "stage_pc1_energy_fraction",
            "singular_value_ratio_1_to_2",
            "residual_energy_before_fraction",
            "residual_energy_after_fraction",
            "extracted_energy_fraction_original",
        ):
            row["stage_%d_%s" % (stage, name)] = info2.get(
                "stage_%d_%s" % (stage, name), np.nan
            )

    return row


def hierarchical_bootstrap_mean(
    frame: pd.DataFrame,
    value_col: str,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    data = frame[["system_uid", value_col]].dropna().copy()
    if data.empty:
        return {"estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    system_ids = data["system_uid"].drop_duplicates().to_numpy()
    grouped = {
        sid: data.loc[data["system_uid"] == sid, value_col].astype(float).to_numpy()
        for sid in system_ids
    }
    estimate = float(data[value_col].astype(float).mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sampled_systems = rng.choice(system_ids, size=len(system_ids), replace=True)
        chunks: List[np.ndarray] = []
        for sid in sampled_systems:
            values = grouped[sid]
            values = rng.choice(values, size=len(values), replace=True)
            chunks.append(np.asarray(values, dtype=float))
        boot[b] = float(np.mean(np.concatenate(chunks)))

    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def _rate_columns(group: pd.DataFrame, cfg: ExperimentConfig, seed_offset: int) -> Dict[str, float]:
    work = group.copy()
    work["stage2_reliability_value"] = np.where(
        work["A2_accepted"].astype(bool),
        pd.to_numeric(work["C2_correct"], errors="coerce"),
        np.nan,
    )

    metrics = {
        "acceptance": "A2_accepted",
        "overall_recovery": "A2_and_C2",
        "reliability_given_accepted": "stage2_reliability_value",
        "false_acceptance": "A2_and_not_C2",
    }
    result: Dict[str, float] = {}
    for j, (name, col) in enumerate(metrics.items()):
        boot = hierarchical_bootstrap_mean(
            work, col, cfg.bootstrap_replicates, cfg.seed + seed_offset + j * 997
        )
        result[name + "_rate"] = boot["estimate"]
        result[name + "_ci95_low"] = boot["ci95_low"]
        result[name + "_ci95_high"] = boot["ci95_high"]
    return result


def summarize_cell(group: pd.DataFrame, cfg: ExperimentConfig, seed_offset: int) -> pd.Series:
    rates = _rate_columns(group, cfg, seed_offset)
    accepted = group.loc[group["A2_accepted"].astype(bool)].copy()
    positive_excitation = not np.isclose(
        float(group["excitation_ratio_abs_a2_over_a1"].iloc[0]), 0.0
    )

    oracle_evaluable = accepted.loc[
        accepted["oracle_q2_error_deg_same_window"].notna()
    ].copy()
    if positive_excitation and len(oracle_evaluable):
        oracle_rate = float(
            (oracle_evaluable["oracle_q2_error_deg_same_window"] <= cfg.recovery_tolerance_deg).mean()
        )
    else:
        oracle_rate = np.nan

    return pd.Series({
        "n_systems": int(group["system_uid"].nunique()),
        "n_initial_states_per_system": int(cfg.initial_states_per_system),
        "n_trajectories": int(len(group)),
        "n_stage2_accepted": int(group["A2_accepted"].astype(bool).sum()),
        **rates,
        "median_q1_error_deg_at_stage2_window": float(accepted["q1_error_deg_at_stage2_window"].median()) if len(accepted) else np.nan,
        "median_q2_error_deg_accepted": float(accepted["q2_error_deg_estimated_sequence"].median()) if len(accepted) else np.nan,
        "q25_q2_error_deg_accepted": float(accepted["q2_error_deg_estimated_sequence"].quantile(0.25)) if len(accepted) else np.nan,
        "q75_q2_error_deg_accepted": float(accepted["q2_error_deg_estimated_sequence"].quantile(0.75)) if len(accepted) else np.nan,
        "median_selected_window_end": float(accepted["stage2_selected_window_end"].median()) if len(accepted) else np.nan,
        "median_reference_q1_deflation_q2_error_deg_same_window": float(oracle_evaluable["oracle_q2_error_deg_same_window"].median()) if len(oracle_evaluable) else np.nan,
        "median_estimated_minus_reference_q2_error_deg": float(oracle_evaluable["q2_error_penalty_estimated_minus_oracle_deg"].median()) if len(oracle_evaluable) else np.nan,
        "reference_q1_deflation_q2_correct_rate_same_window": oracle_rate,
        "median_estimated_vs_reference_residual_relative_fro_difference": float(accepted["estimated_vs_reference_residual_relative_fro_difference"].median()) if len(accepted) else np.nan,
        "median_stage2_residual_energy_before_fraction": float(accepted["stage_2_residual_energy_before_fraction"].median()) if len(accepted) else np.nan,
    })


def build_cell_summary(all_trials: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[pd.Series] = []
    grouped = all_trials.groupby(
        ["lambda2", "lambda2_abs", "spectral_ratio_g_abs_lambda2_over_lambda1", "excitation_ratio_abs_a2_over_a1"],
        sort=True,
        dropna=False,
    )
    for idx, (keys, group) in enumerate(grouped):
        l2, l2abs, g, alpha = keys
        s = summarize_cell(group, cfg, seed_offset=10000 * idx)
        s["lambda2"] = l2
        s["lambda2_abs"] = l2abs
        s["spectral_ratio_g_abs_lambda2_over_lambda1"] = g
        s["excitation_ratio_abs_a2_over_a1"] = alpha
        rows.append(s)
    out = pd.DataFrame(rows)
    columns_front = [
        "lambda2", "lambda2_abs", "spectral_ratio_g_abs_lambda2_over_lambda1",
        "excitation_ratio_abs_a2_over_a1"
    ]
    return out[columns_front + [c for c in out.columns if c not in columns_front]].sort_values(
        ["spectral_ratio_g_abs_lambda2_over_lambda1", "excitation_ratio_abs_a2_over_a1"]
    )


def build_propagation_summary(all_trials: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Stage-2 correctness conditional on whether accepted stage-1 output was correct.

    alpha=0 is excluded because there is no identifiable q2 target there; including
    it would mix target absence with propagation of a wrong q1 deflation.
    """
    data = all_trials.loc[
        all_trials["A2_accepted"].astype(bool)
        & (all_trials["excitation_ratio_abs_a2_over_a1"] > 0)
    ].copy()

    rows: List[Dict[str, object]] = []
    for label, mask in (
        ("preceding_q1_correct", data["C1_at_stage2_window"] == True),
        ("preceding_q1_incorrect", data["C1_at_stage2_window"] == False),
    ):
        group = data.loc[mask].copy()
        if group.empty:
            rows.append({
                "preceding_stage_condition": label,
                "n_trajectories": 0,
                "n_systems": 0,
                "stage2_correct_rate": np.nan,
                "stage2_correct_ci95_low": np.nan,
                "stage2_correct_ci95_high": np.nan,
            })
            continue
        group["C2_numeric"] = pd.to_numeric(group["C2_correct"], errors="coerce")
        boot = hierarchical_bootstrap_mean(
            group, "C2_numeric", cfg.bootstrap_replicates,
            cfg.seed + (880001 if label == "preceding_q1_correct" else 880002),
        )
        rows.append({
            "preceding_stage_condition": label,
            "n_trajectories": int(len(group)),
            "n_systems": int(group["system_uid"].nunique()),
            "stage2_correct_rate": boot["estimate"],
            "stage2_correct_ci95_low": boot["ci95_low"],
            "stage2_correct_ci95_high": boot["ci95_high"],
        })
    return pd.DataFrame(rows)


def build_design_table(cfg: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for l2 in cfg.lambda2_values:
        rows.append({
            "lambda1": cfg.lambda1,
            "lambda2": l2,
            "g_abs_lambda2_over_lambda1": abs(l2) / cfg.lambda1,
            "n_systems": cfg.system_replicates,
            "initial_states_per_system_per_alpha": cfg.initial_states_per_system,
            "alpha_values": ",".join("%g" % a for a in cfg.excitation_ratios),
            "window_m": cfg.window,
            "stability_threshold_deg": cfg.stability_threshold_deg,
            "stability_patience": cfg.stability_patience,
            "pc1_energy_threshold": cfg.min_stage_pc1_energy_fraction,
            "external_tolerance_deg": cfg.recovery_tolerance_deg,
        })
    return pd.DataFrame(rows)


def plot_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    output_path: Path,
    fmt: str = ".2f",
) -> None:
    pivot = summary.pivot(
        index="spectral_ratio_g_abs_lambda2_over_lambda1",
        columns="excitation_ratio_abs_a2_over_a1",
        values=value_col,
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    arr = pivot.to_numpy(dtype=float)
    image = ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(["%g" % float(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(["%.3f" % float(v) for v in pivot.index])
    ax.set_xlabel(r"Excitation $\alpha=|a_2/a_1|$")
    ax.set_ylabel(r"Spectral ratio $g=|\lambda_2|/|\lambda_1|$")
    ax.set_title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center")

    cb = fig.colorbar(image, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_recovery_vs_excitation(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    alphas = sorted(summary["excitation_ratio_abs_a2_over_a1"].unique())
    x = np.arange(len(alphas))
    for g, group in summary.groupby("spectral_ratio_g_abs_lambda2_over_lambda1", sort=True):
        ordered = group.sort_values("excitation_ratio_abs_a2_over_a1")
        ax.plot(x, ordered["overall_recovery_rate"], marker="o", label="g=%.3f" % g)
    ax.set_xticks(x)
    ax.set_xticklabels(["%g" % a for a in alphas])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Excitation $\alpha=|a_2/a_1|$")
    ax.set_ylabel(r"$P(A_2=1,C_2=1)$")
    ax.set_title("Experiment 2: successful q2 recovery versus excitation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_oracle_penalty(summary: pd.DataFrame, output_path: Path) -> None:
    positive = summary.loc[summary["excitation_ratio_abs_a2_over_a1"] > 0].copy()
    plot_heatmap(
        positive,
        "median_estimated_minus_reference_q2_error_deg",
        "Experiment 2: q2 error added by estimated q1 deflation",
        "median estimated error - reference-deflation error (degrees)",
        output_path,
        fmt=".3f",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Framework Experiment 2: two-filter q1/q2 recovery in normal systems, "
            "varying spectral ratio and q2 excitation, with same-window reference "
            "versus estimated q1 deflation."
        )
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--lambda1", type=float, default=0.96)
    parser.add_argument(
        "--lambda2-values", type=parse_float_list,
        default=parse_float_list("0.95,0.94,0.92,0.90,0.88")
    )
    parser.add_argument(
        "--excitation-ratios", type=parse_float_list,
        default=parse_float_list("0,0.01,0.03,0.1,0.3,1,3")
    )
    parser.add_argument("--system-replicates", type=int, default=10)
    parser.add_argument("--initial-states-per-system", type=int, default=20)
    # Backward-compatible alias with the older q2 sweep.
    parser.add_argument("--trials-per-system", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--other-mode-scale", type=float, default=0.25)
    parser.add_argument("--tail-max", type=float, default=0.84)
    parser.add_argument("--tail-min", type=float, default=0.20)
    parser.add_argument("--tail-gap-below-lambda2", type=float, default=0.02)
    parser.add_argument("--stability-threshold-deg", type=float, default=0.2)
    parser.add_argument("--stability-patience", type=int, default=5)
    parser.add_argument("--relative-window-norm-floor", type=float, default=1e-12)
    parser.add_argument("--min-residual-energy-fraction", type=float, default=1e-10)
    parser.add_argument("--numeric-relative-residual-floor", type=float, default=1e-15)
    parser.add_argument("--min-stage-pc1-energy-fraction", type=float, default=0.80)
    parser.add_argument("--recovery-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/framework_experiment2_two_filter_normal")
    )
    args = parser.parse_args()

    n_initial = (
        args.trials_per_system
        if args.trials_per_system is not None
        else args.initial_states_per_system
    )

    cfg = ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        lambda1=args.lambda1,
        lambda2_values=tuple(args.lambda2_values),
        excitation_ratios=tuple(args.excitation_ratios),
        system_replicates=args.system_replicates,
        initial_states_per_system=n_initial,
        seed=args.seed,
        other_mode_scale=args.other_mode_scale,
        tail_max=args.tail_max,
        tail_min=args.tail_min,
        tail_gap_below_lambda2=args.tail_gap_below_lambda2,
        stability_threshold_deg=args.stability_threshold_deg,
        stability_patience=args.stability_patience,
        relative_window_norm_floor=args.relative_window_norm_floor,
        min_residual_energy_fraction=args.min_residual_energy_fraction,
        numeric_relative_residual_floor=args.numeric_relative_residual_floor,
        min_stage_pc1_energy_fraction=args.min_stage_pc1_energy_fraction,
        recovery_tolerance_deg=args.recovery_tolerance_deg,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    validate_config(cfg)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg_stage1 = make_estimator_config(cfg, 1)
    cfg_stage2 = make_estimator_config(cfg, 2)

    config_json = asdict(cfg)
    config_json["lambda2_values"] = list(cfg.lambda2_values)
    config_json["excitation_ratios"] = list(cfg.excitation_ratios)
    config_json["estimator_config_fields_detected"] = [f.name for f in fields(EstimatorConfig)]
    with (output / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    total = (
        len(cfg.lambda2_values) * len(cfg.excitation_ratios)
        * cfg.system_replicates * cfg.initial_states_per_system
    )
    print("========================================================")
    print("Framework Experiment 2: two-filter normal-system validation")
    print("========================================================")
    print("dimension:", cfg.dim)
    print("steps:", cfg.steps)
    print("window:", cfg.window)
    print("lambda1:", cfg.lambda1)
    print("lambda2 values:", cfg.lambda2_values)
    print("g values:", tuple(round(abs(v) / cfg.lambda1, 6) for v in cfg.lambda2_values))
    print("alpha values:", cfg.excitation_ratios)
    print("systems per g:", cfg.system_replicates)
    print("initial states per system per alpha:", cfg.initial_states_per_system)
    print("total trajectories:", total)
    print("The estimator receives only X and L; references are used after estimation.")
    print("Reference-vs-estimated q1 deflation uses the SAME selected raw window.")
    print("========================================================")

    trial_rows: List[Dict[str, object]] = []
    system_rows: List[Dict[str, object]] = []
    completed = 0

    for l2_idx, l2 in enumerate(cfg.lambda2_values):
        for rep in range(cfg.system_replicates):
            system_seed = cfg.seed + 1_000_000 * l2_idx + 10_000 * rep
            A, Q, eigenvalues, normality_error = build_random_normal_system(
                cfg, l2, system_seed
            )
            system_uid = "l2_%02d_rep_%03d" % (l2_idx, rep)
            system_rows.append({
                "system_uid": system_uid,
                "lambda1": cfg.lambda1,
                "lambda2": l2,
                "lambda2_abs": abs(l2),
                "g_abs_lambda2_over_lambda1": abs(l2) / cfg.lambda1,
                "system_replicate": rep,
                "system_seed": system_seed,
                "normality_error": normality_error,
                "tail_max_abs_eigenvalue": float(np.max(np.abs(eigenvalues[2:]))),
                "tail_eigenvalues_json": json.dumps(eigenvalues[2:].tolist()),
            })

            for a_idx, alpha in enumerate(cfg.excitation_ratios):
                for trial in range(cfg.initial_states_per_system):
                    trial_seed = (
                        cfg.seed + 100_000_000 * l2_idx + 1_000_000 * rep
                        + 10_000 * a_idx + trial
                    )
                    trial_rows.append(analyse_one_trial(
                        cfg=cfg,
                        cfg_stage1=cfg_stage1,
                        cfg_stage2=cfg_stage2,
                        A=A,
                        true_basis=Q,
                        eigenvalues=eigenvalues,
                        lambda2=l2,
                        lambda2_index=l2_idx,
                        system_replicate=rep,
                        system_seed=system_seed,
                        excitation_ratio=alpha,
                        excitation_index=a_idx,
                        trial_within_system=trial,
                        trial_seed=trial_seed,
                    ))
                    completed += 1

            print(
                "completed lambda2=%+.3f, g=%.3f, system %d/%d; %d/%d trajectories"
                % (l2, abs(l2) / cfg.lambda1, rep + 1, cfg.system_replicates, completed, total)
            )

    all_trials = pd.DataFrame(trial_rows)
    systems = pd.DataFrame(system_rows)
    all_trials.to_csv(output / "all_trials.csv", index=False)
    systems.to_csv(output / "systems.csv", index=False)

    summary = build_cell_summary(all_trials, cfg)
    propagation = build_propagation_summary(all_trials, cfg)
    design = build_design_table(cfg)

    summary.to_csv(output / "summary_by_g_and_alpha.csv", index=False)
    propagation.to_csv(output / "propagation_summary.csv", index=False)
    design.to_csv(output / "table1_experiment_design.csv", index=False)

    # Compact oracle comparison table, alpha>0 only.
    oracle_cols = [
        "lambda2", "spectral_ratio_g_abs_lambda2_over_lambda1",
        "excitation_ratio_abs_a2_over_a1", "n_stage2_accepted",
        "median_q2_error_deg_accepted",
        "median_reference_q1_deflation_q2_error_deg_same_window",
        "median_estimated_minus_reference_q2_error_deg",
        "reference_q1_deflation_q2_correct_rate_same_window",
        "median_estimated_vs_reference_residual_relative_fro_difference",
    ]
    summary.loc[
        summary["excitation_ratio_abs_a2_over_a1"] > 0, oracle_cols
    ].to_csv(output / "table2_reference_vs_estimated_q1_deflation.csv", index=False)

    plot_heatmap(
        summary, "acceptance_rate",
        "Experiment 2: stage-2 acceptance",
        r"$P(A_2=1)$",
        output / "01_stage2_acceptance_heatmap.png",
    )
    plot_heatmap(
        summary, "overall_recovery_rate",
        "Experiment 2: successful q2 recovery",
        r"$P(A_2=1,C_2=1)$",
        output / "02_stage2_successful_recovery_heatmap.png",
    )
    plot_heatmap(
        summary, "reliability_given_accepted_rate",
        "Experiment 2: reliability of accepted q2 estimates",
        r"$P(C_2=1\mid A_2=1)$",
        output / "03_stage2_reliability_heatmap.png",
    )
    plot_heatmap(
        summary, "false_acceptance_rate",
        "Experiment 2: stage-2 false acceptance",
        r"$P(A_2=1,C_2=0)$",
        output / "04_stage2_false_acceptance_heatmap.png",
    )
    plot_heatmap(
        summary, "median_q2_error_deg_accepted",
        "Experiment 2: q2 angle error among accepted estimates",
        "median q2 angle error (degrees)",
        output / "05_stage2_median_q2_error_heatmap.png",
        fmt=".3f",
    )
    plot_oracle_penalty(summary, output / "06_reference_vs_estimated_q1_deflation_penalty.png")
    plot_recovery_vs_excitation(summary, output / "07_successful_recovery_vs_excitation.png")

    print("\n=== Experiment 2 overall descriptive summary ===")
    print("stage-2 acceptance: %.4f" % float(all_trials["A2_accepted"].mean()))
    print("stage-2 overall successful recovery: %.4f" % float(all_trials["A2_and_C2"].mean()))
    accepted = all_trials.loc[all_trials["A2_accepted"].astype(bool)]
    if len(accepted):
        rel = pd.to_numeric(accepted["C2_correct"], errors="coerce").mean()
        print("stage-2 reliability given acceptance: %.4f" % float(rel))
        print("median accepted q2 error: %.6f degrees" % float(accepted["q2_error_deg_estimated_sequence"].median()))
    zero = all_trials.loc[np.isclose(all_trials["excitation_ratio_abs_a2_over_a1"], 0.0)]
    if len(zero):
        print("alpha=0 target-specific false-acceptance rate: %.4f" % float(zero["A2_and_not_C2"].mean()))

    print("\nPropagation summary (alpha>0 only):")
    print(propagation.to_string(index=False))
    print("\nResults written to:", output.resolve())


if __name__ == "__main__":
    main()
