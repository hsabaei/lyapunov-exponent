from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
rolling_same_window_diagnostics = ESTIMATOR.rolling_same_window_diagnostics
select_prefix_window = ESTIMATOR.select_prefix_window
simulate_trajectory = ESTIMATOR.simulate_trajectory
angle_deg = ESTIMATOR.angle_deg


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20
    n_directions: int = 5

    # All targets are real, distinct, orthogonal eigendirections of a normal system.
    leading_eigenvalues: Tuple[float, ...] = (0.96, 0.95, 0.94, 0.93, 0.92)

    system_replicates: int = 10
    initial_states_per_system: int = 20
    seed: int = 42

    # All target modes remain present (no alpha=0 target-absence confound).
    # q1 amplitude is fixed to 1; q2...qk are sampled log-uniformly in this range.
    leading_excitation_min: float = 1e-3
    leading_excitation_max: float = 3.0
    tail_coefficient_scale: float = 0.10

    # Lower modes remain spectrally below q_k.
    tail_max: float = 0.90
    tail_min: float = 0.20
    tail_gap_below_last_target: float = 0.01

    # Observation-only acceptance criteria, frozen before evaluation.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External correctness tolerance only; never used to select a window.
    recovery_tolerance_deg: float = 1.0

    # Hierarchical bootstrap: systems first, then trajectories within system.
    bootstrap_replicates: int = 2000


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number.")
    return values


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.n_directions < 2:
        raise ValueError("n_directions must be at least 2.")
    if cfg.dim < cfg.n_directions + 1:
        raise ValueError("dim must exceed n_directions so lower non-target modes exist.")
    if len(cfg.leading_eigenvalues) != cfg.n_directions:
        raise ValueError("leading_eigenvalues must contain exactly n_directions values.")
    mags = np.abs(np.asarray(cfg.leading_eigenvalues, dtype=float))
    if not np.all((mags > 0.0) & (mags < 1.0)):
        raise ValueError("All leading eigenvalue magnitudes must lie in (0,1).")
    if not np.all(mags[:-1] > mags[1:]):
        raise ValueError("Require strictly decreasing target magnitudes |lambda1|>...>|lambdak|.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if cfg.window < 2 or cfg.window > cfg.steps + 1:
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if cfg.system_replicates < 1 or cfg.initial_states_per_system < 1:
        raise ValueError("replicate counts must be positive.")
    if cfg.leading_excitation_min <= 0.0:
        raise ValueError("leading_excitation_min must be positive: Experiment 3 keeps targets identifiable.")
    if cfg.leading_excitation_max < cfg.leading_excitation_min:
        raise ValueError("leading_excitation_max must be >= leading_excitation_min.")
    if cfg.tail_coefficient_scale < 0.0:
        raise ValueError("tail_coefficient_scale must be nonnegative.")
    if not (0.0 < cfg.tail_min < cfg.tail_max < 1.0):
        raise ValueError("Require 0 < tail_min < tail_max < 1.")
    available_tail_max = min(
        cfg.tail_max,
        float(mags[-1]) - cfg.tail_gap_below_last_target,
    )
    if available_tail_max <= cfg.tail_min:
        raise ValueError("No valid lower-mode interval below the final target eigenvalue.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be at least 100.")
    if cfg.recovery_tolerance_deg <= 0.0:
        raise ValueError("recovery_tolerance_deg must be positive.")


def _estimator_config_kwargs(cfg: ExperimentConfig) -> Dict[str, object]:
    desired = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": 1,
        "window": cfg.window,
        "n_directions": cfg.n_directions,
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


def make_estimator_config(cfg: ExperimentConfig) -> EstimatorConfig:
    return EstimatorConfig(**_estimator_config_kwargs(cfg))


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        raise ValueError("Cannot normalize a non-finite or near-zero vector.")
    return v / n


def orthonormal_basis(vectors: List[np.ndarray], dim: int) -> np.ndarray:
    if not vectors:
        return np.empty((dim, 0), dtype=float)
    M = np.column_stack([normalize(v) for v in vectors])
    Q, _ = np.linalg.qr(M, mode="reduced")
    return Q


def deflate_by_basis(error_window: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.asarray(error_window, dtype=float).copy()
    B, _ = np.linalg.qr(np.asarray(basis, dtype=float), mode="reduced")
    return error_window @ (np.eye(error_window.shape[1]) - B @ B.T)


def top_right_singular_direction(matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if not np.isfinite(norm) or norm <= EPS:
        return np.full(matrix.shape[1], np.nan), np.nan, np.nan
    _u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    direction = normalize(vt[0])
    energy = float(np.sum(s * s))
    pc1 = float(s[0] ** 2 / energy) if energy > EPS else np.nan
    ratio = float(s[0] / s[1]) if len(s) > 1 and s[1] > EPS else np.inf
    return direction, pc1, ratio


def build_random_normal_system(
    cfg: ExperimentConfig, system_seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(system_seed)
    G = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(G)

    leading = np.asarray(cfg.leading_eigenvalues, dtype=float)
    remaining = cfg.dim - cfg.n_directions
    available_tail_max = min(
        cfg.tail_max,
        abs(leading[-1]) - cfg.tail_gap_below_last_target,
    )
    tail_magnitudes = rng.uniform(cfg.tail_min, available_tail_max, size=remaining)
    tail_magnitudes = np.sort(tail_magnitudes)[::-1]
    tail_signs = rng.choice(np.array([-1.0, 1.0]), size=remaining)
    tail = tail_signs * tail_magnitudes

    eigenvalues = np.concatenate([leading, tail])
    A = Q @ np.diag(eigenvalues) @ Q.T
    normality_error = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    if normality_error > 1e-10:
        raise RuntimeError("Constructed system is not numerically normal.")
    return A, Q, eigenvalues, normality_error


def construct_initial_state(
    cfg: ExperimentConfig,
    true_basis: np.ndarray,
    trial_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(trial_seed)
    coefficients = rng.normal(0.0, cfg.tail_coefficient_scale, size=cfg.dim)

    # Keep q1 as the scale reference.
    coefficients[0] = float(rng.choice(np.array([-1.0, 1.0])))

    # q2...qk are all nonzero but span an intentionally broad range.
    lo = np.log(cfg.leading_excitation_min)
    hi = np.log(cfg.leading_excitation_max)
    magnitudes = np.exp(rng.uniform(lo, hi, size=cfg.n_directions - 1))
    signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.n_directions - 1)
    coefficients[1 : cfg.n_directions] = signs * magnitudes

    x0 = true_basis @ coefficients
    return x0, coefficients


def correctness_value(accepted: bool, error_deg: float, tolerance: float) -> Optional[bool]:
    if not accepted:
        return None
    if not np.isfinite(error_deg):
        return False
    return bool(error_deg <= tolerance)


def bool_or_nan(value: Optional[bool]):
    if value is None:
        return np.nan
    return bool(value)


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
            sampled_values = rng.choice(values, size=len(values), replace=True)
            chunks.append(np.asarray(sampled_values, dtype=float))
        boot[b] = float(np.mean(np.concatenate(chunks)))

    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def analyse_one_trajectory(
    *,
    cfg: ExperimentConfig,
    estimator_cfg: EstimatorConfig,
    A: np.ndarray,
    true_basis: np.ndarray,
    eigenvalues: np.ndarray,
    system_replicate: int,
    system_seed: int,
    trial_within_system: int,
    trial_seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    x0, coefficients = construct_initial_state(cfg, true_basis, trial_seed)
    X = simulate_trajectory(A=A, x0=x0, steps=cfg.steps)
    L = np.zeros(cfg.dim, dtype=float)

    # One observation-only rolling pass extracts up to k directions per window.
    # Stage i acceptance is then defined by the prefix rule for directions 1...i.
    diagnostics = rolling_same_window_diagnostics(X=X, L=L, cfg=estimator_cfg)

    system_uid = "sys_%03d" % system_replicate
    trajectory_uid = "%s_trial_%04d" % (system_uid, trial_within_system)

    trajectory_row: Dict[str, object] = {
        "system_uid": system_uid,
        "trajectory_uid": trajectory_uid,
        "system_replicate": system_replicate,
        "system_seed": system_seed,
        "initial_state_within_system": trial_within_system,
        "trial_seed": trial_seed,
        "x0_norm": float(np.linalg.norm(x0)),
        "tail_coefficient_l2_norm": float(np.linalg.norm(coefficients[cfg.n_directions :])),
    }
    for j in range(cfg.n_directions):
        trajectory_row[f"lambda_{j+1}"] = float(eigenvalues[j])
        trajectory_row[f"a{j+1}"] = float(coefficients[j])
        trajectory_row[f"abs_a{j+1}"] = abs(float(coefficients[j]))

    stage_rows: List[Dict[str, object]] = []

    for stage in range(1, cfg.n_directions + 1):
        selected, n_candidates = select_prefix_window(
            diagnostics=diagnostics,
            cfg=estimator_cfg,
            n_stages=stage,
        )
        accepted = selected is not None

        row: Dict[str, object] = {
            "system_uid": system_uid,
            "trajectory_uid": trajectory_uid,
            "system_replicate": system_replicate,
            "initial_state_within_system": trial_within_system,
            "stage": stage,
            "target_name": f"q{stage}",
            "target_lambda": float(eigenvalues[stage - 1]),
            "target_abs_initial_excitation": abs(float(coefficients[stage - 1])),
            "A_i_accepted": bool(accepted),
            "n_acceptable_windows_for_prefix": int(n_candidates),
            "selected_window_start": np.nan,
            "selected_window_end": np.nan,
            "C_i_correct": np.nan,
            "A_i_and_C_i": False,
            "A_i_and_not_C_i": False,
            "q_i_error_deg": np.nan,
            "all_prior_correct_same_stage_window": np.nan,
            "any_prior_incorrect_same_stage_window": np.nan,
            "n_prior_incorrect_same_stage_window": np.nan,
            "reference_q_i_error_deg_same_window": np.nan,
            "reference_q_i_correct_same_window": np.nan,
            "estimated_rebuilt_q_i_error_deg_same_window": np.nan,
            "estimated_vs_reference_error_penalty_deg": np.nan,
            "reference_stage_pc1_energy_fraction_same_window": np.nan,
            "reference_stage_singular_value_ratio_same_window": np.nan,
            "estimated_vs_reference_residual_relative_fro_difference": np.nan,
        }

        # Store all target excitations to permit later propagation/excitation auditing.
        for j in range(cfg.n_directions):
            row[f"abs_a{j+1}"] = abs(float(coefficients[j]))

        if not accepted:
            stage_rows.append(row)
            trajectory_row[f"A{stage}_accepted"] = False
            trajectory_row[f"C{stage}_correct"] = np.nan
            continue

        window_start = int(selected["window_start"])
        window_end = int(selected["window_end"])
        row["selected_window_start"] = window_start
        row["selected_window_end"] = window_end

        # Correctness is evaluated for every member of the ACTUAL prefix chain
        # at this same selected stage-i window. This is what propagated into qhat_i.
        selected_dirs: List[np.ndarray] = []
        chain_correctness: List[bool] = []
        chain_errors: List[float] = []
        for j in range(1, stage + 1):
            direction_obj = selected[f"direction_{j}"]
            if direction_obj is None:
                raise RuntimeError("Prefix selection accepted a missing direction.")
            u_j = normalize(np.asarray(direction_obj, dtype=float))
            selected_dirs.append(u_j)
            err_j = angle_deg(u_j, true_basis[:, j - 1])
            chain_errors.append(float(err_j))
            chain_correctness.append(bool(err_j <= cfg.recovery_tolerance_deg))
            row[f"q{j}_error_deg_at_stage_i_window"] = float(err_j)
            row[f"C{j}_at_stage_i_window"] = bool(chain_correctness[-1])

        q_i_error = chain_errors[-1]
        C_i = correctness_value(True, q_i_error, cfg.recovery_tolerance_deg)
        row["q_i_error_deg"] = q_i_error
        row["C_i_correct"] = bool_or_nan(C_i)
        row["A_i_and_C_i"] = bool(C_i is True)
        row["A_i_and_not_C_i"] = bool(C_i is False)

        if stage == 1:
            row["all_prior_correct_same_stage_window"] = True
            row["any_prior_incorrect_same_stage_window"] = False
            row["n_prior_incorrect_same_stage_window"] = 0
        else:
            prior = chain_correctness[:-1]
            row["all_prior_correct_same_stage_window"] = bool(all(prior))
            row["any_prior_incorrect_same_stage_window"] = bool(not all(prior))
            row["n_prior_incorrect_same_stage_window"] = int(sum(not x for x in prior))

        # Same raw observation-only-selected window for estimated and reference chains.
        R = X[window_start : window_end + 1] - L
        estimated_prior_basis = orthonormal_basis(selected_dirs[:-1], cfg.dim)
        reference_prior_basis = true_basis[:, : stage - 1]
        R_est = deflate_by_basis(R, estimated_prior_basis)
        R_ref = deflate_by_basis(R, reference_prior_basis)

        rebuilt_u_i, _est_pc1, _est_ratio = top_right_singular_direction(R_est)
        reference_u_i, ref_pc1, ref_ratio = top_right_singular_direction(R_ref)

        if np.all(np.isfinite(rebuilt_u_i)):
            rebuilt_error = angle_deg(rebuilt_u_i, true_basis[:, stage - 1])
            row["estimated_rebuilt_q_i_error_deg_same_window"] = float(rebuilt_error)
            # The rebuilt cumulative-projector version should reproduce the estimator.
            mismatch = angle_deg(rebuilt_u_i, selected_dirs[-1])
            row["estimator_vs_rebuilt_direction_mismatch_deg"] = float(mismatch)
            if mismatch > 1e-4:
                raise RuntimeError(
                    "Estimated cumulative deflation did not reproduce the estimator "
                    f"at stage {stage}: mismatch={mismatch:.6g} deg"
                )

        if np.all(np.isfinite(reference_u_i)):
            reference_error = angle_deg(reference_u_i, true_basis[:, stage - 1])
            row["reference_q_i_error_deg_same_window"] = float(reference_error)
            row["reference_q_i_correct_same_window"] = bool(
                reference_error <= cfg.recovery_tolerance_deg
            )
            row["reference_stage_pc1_energy_fraction_same_window"] = float(ref_pc1)
            row["reference_stage_singular_value_ratio_same_window"] = float(ref_ratio)
            row["estimated_vs_reference_error_penalty_deg"] = float(
                q_i_error - reference_error
            )

        ref_norm = float(np.linalg.norm(R_ref, ord="fro"))
        if ref_norm > EPS:
            row["estimated_vs_reference_residual_relative_fro_difference"] = float(
                np.linalg.norm(R_est - R_ref, ord="fro") / ref_norm
            )

        # Preserve the observation-only internal diagnostics at this selected prefix window.
        for j in range(1, stage + 1):
            for name in (
                "direction_change_deg",
                "stage_pc1_energy_fraction",
                "singular_value_ratio_1_to_2",
                "residual_energy_before_fraction",
                "residual_energy_after_fraction",
                "extracted_energy_fraction_original",
            ):
                key = f"stage_{j}_{name}"
                row[f"selected_{key}"] = float(selected[key])

        stage_rows.append(row)
        trajectory_row[f"A{stage}_accepted"] = True
        trajectory_row[f"C{stage}_correct"] = bool_or_nan(C_i)
        trajectory_row[f"q{stage}_error_deg"] = q_i_error
        trajectory_row[f"stage{stage}_selected_window_end"] = window_end

    return trajectory_row, stage_rows


def summarize_by_stage(
    stage_results: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stage, group in stage_results.groupby("stage", sort=True):
        work = group.copy()
        work["reliability_value"] = np.where(
            work["A_i_accepted"].astype(bool),
            pd.to_numeric(work["C_i_correct"], errors="coerce"),
            np.nan,
        )
        metrics = {
            "acceptance": "A_i_accepted",
            "overall_recovery": "A_i_and_C_i",
            "reliability_given_accepted": "reliability_value",
            "false_acceptance": "A_i_and_not_C_i",
        }
        result: Dict[str, object] = {
            "stage": int(stage),
            "target": f"q{int(stage)}",
            "n_systems": int(group["system_uid"].nunique()),
            "n_trajectories": int(len(group)),
            "n_accepted": int(group["A_i_accepted"].astype(bool).sum()),
        }
        for j, (label, col) in enumerate(metrics.items()):
            boot = hierarchical_bootstrap_mean(
                work,
                col,
                cfg.bootstrap_replicates,
                cfg.seed + 100_000 * int(stage) + 997 * j,
            )
            result[f"{label}_rate"] = boot["estimate"]
            result[f"{label}_ci95_low"] = boot["ci95_low"]
            result[f"{label}_ci95_high"] = boot["ci95_high"]

        accepted = work.loc[work["A_i_accepted"].astype(bool)].copy()
        result["median_q_i_error_deg_accepted"] = (
            float(accepted["q_i_error_deg"].median()) if len(accepted) else np.nan
        )
        result["q25_q_i_error_deg_accepted"] = (
            float(accepted["q_i_error_deg"].quantile(0.25)) if len(accepted) else np.nan
        )
        result["q75_q_i_error_deg_accepted"] = (
            float(accepted["q_i_error_deg"].quantile(0.75)) if len(accepted) else np.nan
        )
        result["median_selected_window_end"] = (
            float(accepted["selected_window_end"].median()) if len(accepted) else np.nan
        )
        rows.append(result)
    return pd.DataFrame(rows).sort_values("stage")


def build_propagation_summary(
    stage_results: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Main propagation test.

    Compare stage-i correctness among accepted prefix chains where all preceding
    directions were correct at the SAME stage-i selected window versus accepted
    chains where at least one preceding direction was incorrect.
    """
    rows: List[Dict[str, object]] = []
    for stage in range(2, cfg.n_directions + 1):
        accepted = stage_results.loc[
            (stage_results["stage"] == stage)
            & stage_results["A_i_accepted"].astype(bool)
        ].copy()

        for condition, mask in (
            (
                "all_prior_correct",
                accepted["all_prior_correct_same_stage_window"] == True,
            ),
            (
                "any_prior_incorrect",
                accepted["any_prior_incorrect_same_stage_window"] == True,
            ),
        ):
            group = accepted.loc[mask].copy()
            row: Dict[str, object] = {
                "stage": stage,
                "target": f"q{stage}",
                "preceding_chain_condition": condition,
                "n_systems": int(group["system_uid"].nunique()) if len(group) else 0,
                "n_trajectories": int(len(group)),
                "stage_i_correct_rate": np.nan,
                "stage_i_correct_ci95_low": np.nan,
                "stage_i_correct_ci95_high": np.nan,
                "reference_correct_rate_same_window": np.nan,
                "median_q_i_error_deg": np.nan,
                "median_reference_q_i_error_deg_same_window": np.nan,
                "median_estimated_minus_reference_error_penalty_deg": np.nan,
            }
            if len(group):
                group["C_i_numeric"] = pd.to_numeric(group["C_i_correct"], errors="coerce")
                boot = hierarchical_bootstrap_mean(
                    group,
                    "C_i_numeric",
                    cfg.bootstrap_replicates,
                    cfg.seed + 1_000_000 * stage + (1 if condition == "all_prior_correct" else 2),
                )
                row["stage_i_correct_rate"] = boot["estimate"]
                row["stage_i_correct_ci95_low"] = boot["ci95_low"]
                row["stage_i_correct_ci95_high"] = boot["ci95_high"]
                row["reference_correct_rate_same_window"] = float(
                    pd.to_numeric(
                        group["reference_q_i_correct_same_window"], errors="coerce"
                    ).mean()
                )
                row["median_q_i_error_deg"] = float(group["q_i_error_deg"].median())
                row["median_reference_q_i_error_deg_same_window"] = float(
                    group["reference_q_i_error_deg_same_window"].median()
                )
                row["median_estimated_minus_reference_error_penalty_deg"] = float(
                    group["estimated_vs_reference_error_penalty_deg"].median()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_reference_comparison(stage_results: pd.DataFrame) -> pd.DataFrame:
    accepted = stage_results.loc[stage_results["A_i_accepted"].astype(bool)].copy()
    rows: List[Dict[str, object]] = []
    for stage, group in accepted.groupby("stage", sort=True):
        evaluable = group.loc[group["reference_q_i_error_deg_same_window"].notna()].copy()
        rows.append({
            "stage": int(stage),
            "target": f"q{int(stage)}",
            "n_accepted": int(len(group)),
            "n_reference_evaluable": int(len(evaluable)),
            "median_estimated_q_i_error_deg": float(group["q_i_error_deg"].median()) if len(group) else np.nan,
            "median_reference_q_i_error_deg_same_window": float(evaluable["reference_q_i_error_deg_same_window"].median()) if len(evaluable) else np.nan,
            "median_estimated_minus_reference_error_penalty_deg": float(evaluable["estimated_vs_reference_error_penalty_deg"].median()) if len(evaluable) else np.nan,
            "estimated_correct_rate": float(pd.to_numeric(group["C_i_correct"], errors="coerce").mean()) if len(group) else np.nan,
            "reference_correct_rate_same_window": float(pd.to_numeric(evaluable["reference_q_i_correct_same_window"], errors="coerce").mean()) if len(evaluable) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("stage")


def build_design_table(cfg: ExperimentConfig) -> pd.DataFrame:
    return pd.DataFrame([{
        "system_type": "linear normal; real orthogonal eigendirections",
        "dimension": cfg.dim,
        "steps": cfg.steps,
        "window_m": cfg.window,
        "n_directions": cfg.n_directions,
        "leading_eigenvalues": ",".join("%+.4f" % v for v in cfg.leading_eigenvalues),
        "system_replicates": cfg.system_replicates,
        "initial_states_per_system": cfg.initial_states_per_system,
        "total_trajectories": cfg.system_replicates * cfg.initial_states_per_system,
        "q1_excitation_magnitude": 1.0,
        "q2_to_qk_excitation_distribution": (
            "independent log-uniform[%g,%g], random sign"
            % (cfg.leading_excitation_min, cfg.leading_excitation_max)
        ),
        "tail_coefficient_scale": cfg.tail_coefficient_scale,
        "stability_threshold_deg": cfg.stability_threshold_deg,
        "stability_patience": cfg.stability_patience,
        "pc1_energy_threshold": cfg.min_stage_pc1_energy_fraction,
        "external_tolerance_deg": cfg.recovery_tolerance_deg,
    }])


def plot_stage_rates(summary: pd.DataFrame, output_path: Path) -> None:
    x = summary["stage"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, summary["acceptance_rate"], marker="o", label=r"$P(A_i=1)$")
    ax.plot(x, summary["overall_recovery_rate"], marker="o", label=r"$P(A_i=1,C_i=1)$")
    ax.plot(x, summary["reliability_given_accepted_rate"], marker="o", label=r"$P(C_i=1\mid A_i=1)$")
    ax.plot(x, summary["false_acceptance_rate"], marker="o", label=r"$P(A_i=1,C_i=0)$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"q{i}" for i in x])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Filter stage")
    ax.set_ylabel("Rate")
    ax.set_title("Experiment 3: stage-wise multi-filter outcomes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_propagation(propagation: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    stages = sorted(propagation["stage"].unique())
    for condition, label in (
        ("all_prior_correct", "all preceding accepted directions correct"),
        ("any_prior_incorrect", "at least one preceding accepted direction incorrect"),
    ):
        subset = propagation.loc[
            propagation["preceding_chain_condition"] == condition
        ].sort_values("stage")
        ax.plot(
            subset["stage"],
            subset["stage_i_correct_rate"],
            marker="o",
            label=label,
        )
    ax.set_xticks(stages)
    ax.set_xticklabels([f"q{i}" for i in stages])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Current filter stage")
    ax.set_ylabel(r"$P(C_i=1\mid A_1=\cdots=A_i=1,\;\mathrm{prior\ condition})$")
    ax.set_title("Experiment 3: propagation of preceding-direction errors")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reference_penalty(reference: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        reference["stage"],
        reference["median_estimated_minus_reference_error_penalty_deg"],
        marker="o",
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(reference["stage"])
    ax.set_xticklabels([f"q{i}" for i in reference["stage"]])
    ax.set_xlabel("Filter stage")
    ax.set_ylabel("median estimated error - reference-deflation error (degrees)")
    ax.set_title("Experiment 3: error attributable to estimated preceding deflation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_distribution(stage_results: pd.DataFrame, cfg: ExperimentConfig, output_path: Path) -> None:
    data = []
    labels = []
    for stage in range(1, cfg.n_directions + 1):
        values = stage_results.loc[
            (stage_results["stage"] == stage)
            & stage_results["A_i_accepted"].astype(bool),
            "q_i_error_deg",
        ].dropna().to_numpy(dtype=float)
        if len(values):
            data.append(values)
            labels.append(f"q{stage}")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(cfg.recovery_tolerance_deg, linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Filter stage")
    ax.set_ylabel("Angular error among accepted estimates (degrees, log scale)")
    ax.set_title("Experiment 3: accepted-direction angular errors")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Framework Experiment 3: multi-filter error propagation in normal linear "
            "systems. Stage-i propagation is evaluated using correctness of preceding "
            "directions at the same observation-only selected stage-i window."
        )
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--n-directions", type=int, default=5)
    parser.add_argument(
        "--leading-eigenvalues",
        type=parse_float_list,
        default=parse_float_list("0.96,0.95,0.94,0.93,0.92"),
    )
    parser.add_argument("--system-replicates", type=int, default=10)
    parser.add_argument("--initial-states-per-system", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leading-excitation-min", type=float, default=1e-3)
    parser.add_argument("--leading-excitation-max", type=float, default=3.0)
    parser.add_argument("--tail-coefficient-scale", type=float, default=0.10)
    parser.add_argument("--tail-max", type=float, default=0.90)
    parser.add_argument("--tail-min", type=float, default=0.20)
    parser.add_argument("--tail-gap-below-last-target", type=float, default=0.01)
    parser.add_argument("--stability-threshold-deg", type=float, default=0.2)
    parser.add_argument("--stability-patience", type=int, default=5)
    parser.add_argument("--relative-window-norm-floor", type=float, default=1e-12)
    parser.add_argument("--min-residual-energy-fraction", type=float, default=1e-10)
    parser.add_argument("--numeric-relative-residual-floor", type=float, default=1e-15)
    parser.add_argument("--min-stage-pc1-energy-fraction", type=float, default=0.80)
    parser.add_argument("--recovery-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/framework_experiment3_multifilter_propagation_normal"),
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        n_directions=args.n_directions,
        leading_eigenvalues=tuple(args.leading_eigenvalues),
        system_replicates=args.system_replicates,
        initial_states_per_system=args.initial_states_per_system,
        seed=args.seed,
        leading_excitation_min=args.leading_excitation_min,
        leading_excitation_max=args.leading_excitation_max,
        tail_coefficient_scale=args.tail_coefficient_scale,
        tail_max=args.tail_max,
        tail_min=args.tail_min,
        tail_gap_below_last_target=args.tail_gap_below_last_target,
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
    estimator_cfg = make_estimator_config(cfg)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config_json = asdict(cfg)
    config_json["leading_eigenvalues"] = list(cfg.leading_eigenvalues)
    config_json["estimator_config_fields_detected"] = [f.name for f in fields(EstimatorConfig)]
    with (output / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    total = cfg.system_replicates * cfg.initial_states_per_system
    print("============================================================")
    print("Framework Experiment 3: multi-filter propagation, normal system")
    print("============================================================")
    print("dimension:", cfg.dim)
    print("steps:", cfg.steps)
    print("window:", cfg.window)
    print("n directions:", cfg.n_directions)
    print("leading eigenvalues:", cfg.leading_eigenvalues)
    print("systems:", cfg.system_replicates)
    print("initial states per system:", cfg.initial_states_per_system)
    print("total trajectories:", total)
    print(
        "leading excitations: q1=1; q2...qk log-uniform in "
        f"[{cfg.leading_excitation_min:g}, {cfg.leading_excitation_max:g}]"
    )
    print("All target modes are nonzero; no target-absence cases are used.")
    print("References are used only after the observation-only window is selected.")
    print("============================================================")

    trajectory_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []
    system_rows: List[Dict[str, object]] = []

    completed = 0
    for rep in range(cfg.system_replicates):
        system_seed = cfg.seed + 1_000_000 * rep
        A, Q, eigenvalues, normality_error = build_random_normal_system(cfg, system_seed)
        system_uid = "sys_%03d" % rep
        system_rows.append({
            "system_uid": system_uid,
            "system_replicate": rep,
            "system_seed": system_seed,
            "normality_error": normality_error,
            "leading_eigenvalues_json": json.dumps(eigenvalues[: cfg.n_directions].tolist()),
            "tail_max_abs_eigenvalue": float(np.max(np.abs(eigenvalues[cfg.n_directions :]))),
            "tail_eigenvalues_json": json.dumps(eigenvalues[cfg.n_directions :].tolist()),
        })

        for trial in range(cfg.initial_states_per_system):
            trial_seed = cfg.seed + 10_000_000 * rep + trial
            traj_row, rows = analyse_one_trajectory(
                cfg=cfg,
                estimator_cfg=estimator_cfg,
                A=A,
                true_basis=Q,
                eigenvalues=eigenvalues,
                system_replicate=rep,
                system_seed=system_seed,
                trial_within_system=trial,
                trial_seed=trial_seed,
            )
            trajectory_rows.append(traj_row)
            stage_rows.extend(rows)
            completed += 1

        print(f"completed system {rep + 1}/{cfg.system_replicates}; {completed}/{total} trajectories")

    trajectories = pd.DataFrame(trajectory_rows)
    stage_results = pd.DataFrame(stage_rows)
    systems = pd.DataFrame(system_rows)

    trajectories.to_csv(output / "all_trajectories.csv", index=False)
    stage_results.to_csv(output / "all_stage_results.csv", index=False)
    systems.to_csv(output / "systems.csv", index=False)

    summary = summarize_by_stage(stage_results, cfg)
    propagation = build_propagation_summary(stage_results, cfg)
    reference = build_reference_comparison(stage_results)
    design = build_design_table(cfg)

    summary.to_csv(output / "table2_stagewise_results.csv", index=False)
    propagation.to_csv(output / "table3_propagation_results.csv", index=False)
    reference.to_csv(output / "table4_reference_vs_estimated_deflation.csv", index=False)
    design.to_csv(output / "table1_experiment_design.csv", index=False)

    plot_stage_rates(summary, output / "01_stagewise_outcome_rates.png")
    plot_propagation(propagation, output / "02_error_propagation.png")
    plot_reference_penalty(reference, output / "03_reference_vs_estimated_deflation_penalty.png")
    plot_error_distribution(stage_results, cfg, output / "04_accepted_angle_error_distribution.png")

    print("\n=== Stage-wise results ===")
    print(summary[[
        "stage", "n_accepted", "acceptance_rate", "overall_recovery_rate",
        "reliability_given_accepted_rate", "false_acceptance_rate",
        "median_q_i_error_deg_accepted",
    ]].to_string(index=False))

    print("\n=== Propagation test ===")
    print(propagation[[
        "stage", "preceding_chain_condition", "n_trajectories",
        "stage_i_correct_rate", "stage_i_correct_ci95_low", "stage_i_correct_ci95_high",
        "reference_correct_rate_same_window",
        "median_estimated_minus_reference_error_penalty_deg",
    ]].to_string(index=False))

    empty_incorrect = propagation.loc[
        (propagation["preceding_chain_condition"] == "any_prior_incorrect")
        & (propagation["n_trajectories"] == 0)
    ]
    if len(empty_incorrect):
        stages = ", ".join("q%d" % int(v) for v in empty_incorrect["stage"])
        print(
            "\nNOTE: No accepted chains with preceding errors were observed for: " + stages + "."
        )
        print(
            "That is a valid result; do not manufacture propagation errors. "
            "If needed later, a separate prespecified stress condition can be added."
        )

    print("\nResults written to:", output.resolve())


if __name__ == "__main__":
    main()
