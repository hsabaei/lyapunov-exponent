from __future__ import annotations

import argparse
import importlib.util
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
rolling_same_window_diagnostics = ESTIMATOR.rolling_same_window_diagnostics
prefix_window_is_stable = ESTIMATOR.prefix_window_is_stable
simulate_trajectory = ESTIMATOR.simulate_trajectory
angle_deg = ESTIMATOR.angle_deg


SPECTRUM_CASES: Dict[str, Tuple[float, ...]] = {
    "strong_gap": (0.96, 0.88, 0.80, 0.72, 0.64),
    "moderate_gap": (0.96, 0.92, 0.88, 0.84, 0.80),
    "weak_gap": (0.96, 0.95, 0.94, 0.93, 0.92),
}

EXCITATION_PROFILES: Dict[str, Tuple[float, ...]] = {
    "balanced": (1.0, 1.0, 1.0, 1.0, 1.0),
    "mild_decreasing": (1.0, 0.5, 0.25, 0.125, 0.0625),
    "strong_decreasing": (1.0, 0.3, 0.1, 0.03, 0.01),
    "increasing": (1.0, 2.0, 4.0, 8.0, 16.0),
    "later_modes_strong": (1.0, 3.0, 3.0, 3.0, 3.0),
}


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20
    n_directions: int = 5

    system_replicates: int = 10
    initial_states_per_system: int = 20
    seed: int = 42

    # Non-target coefficients q6...qd. The same random draw is reused across
    # excitation profiles for a given system/initial-state replicate.
    tail_coefficient_scale: float = 0.10
    tail_max: float = 0.90
    tail_min: float = 0.20
    tail_gap_below_last_target: float = 0.01

    # Observation-only acceptance criteria: frozen before evaluation.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External correctness only. Never used in window acceptance.
    recovery_tolerance_deg: float = 2.5

    # A sustained recovery starts at the first of this many consecutive
    # windows for which the prefix is accepted and q_i is externally correct.
    recovery_persistence_windows: int = 5

    # Hierarchical bootstrap: systems first, then initial states.
    bootstrap_replicates: int = 2000

    # Save compact per-case arrays in addition to CSV summaries.
    save_trace_npz: bool = True


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.n_directions != 5:
        raise ValueError("This experiment is prespecified for q1,...,q5 (n_directions=5).")
    if cfg.dim <= cfg.n_directions:
        raise ValueError("dim must exceed n_directions so lower non-target modes exist.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if not (2 <= cfg.window <= cfg.steps + 1):
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if cfg.system_replicates != 10:
        raise ValueError("Prespecified design requires exactly 10 systems per case.")
    if cfg.initial_states_per_system != 20:
        raise ValueError("Prespecified design requires exactly 20 initial states per system.")
    if cfg.recovery_persistence_windows < 1:
        raise ValueError("recovery_persistence_windows must be positive.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be at least 100.")
    if cfg.recovery_tolerance_deg <= 0.0:
        raise ValueError("recovery_tolerance_deg must be positive.")
    if cfg.tail_coefficient_scale < 0.0:
        raise ValueError("tail_coefficient_scale must be nonnegative.")
    if not (0.0 < cfg.tail_min < cfg.tail_max < 1.0):
        raise ValueError("Require 0 < tail_min < tail_max < 1.")

    for name, spectrum in SPECTRUM_CASES.items():
        if len(spectrum) != cfg.n_directions:
            raise ValueError(f"Spectrum {name} must have {cfg.n_directions} values.")
        mags = np.abs(np.asarray(spectrum, dtype=float))
        if not np.all((mags > 0.0) & (mags < 1.0)):
            raise ValueError(f"Spectrum {name} has an invalid eigenvalue magnitude.")
        if not np.all(mags[:-1] > mags[1:]):
            raise ValueError(f"Spectrum {name} must satisfy |lambda1|>...>|lambda5|.")
        available_tail_max = min(cfg.tail_max, float(mags[-1]) - cfg.tail_gap_below_last_target)
        if available_tail_max <= cfg.tail_min:
            raise ValueError(f"Spectrum {name} leaves no valid lower-mode interval.")

    for name, profile in EXCITATION_PROFILES.items():
        if len(profile) != cfg.n_directions:
            raise ValueError(f"Excitation profile {name} must have {cfg.n_directions} values.")
        if np.any(np.asarray(profile, dtype=float) <= 0.0):
            raise ValueError(f"Excitation profile {name} must keep all targets nonzero.")


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


def orthonormal_basis(vectors: Sequence[np.ndarray], dim: int) -> np.ndarray:
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


def top_right_singular_direction(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if not np.isfinite(norm) or norm <= EPS:
        return np.full(matrix.shape[1], np.nan)
    _u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
    return normalize(vt[0])


def build_random_normal_system(
    cfg: ExperimentConfig,
    leading_eigenvalues: Sequence[float],
    system_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Paired system geometry: same seed => same Q across spectral cases."""
    rng = np.random.default_rng(system_seed)
    G = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(G)

    leading = np.asarray(leading_eigenvalues, dtype=float)
    available_tail_max = min(
        cfg.tail_max,
        abs(float(leading[-1])) - cfg.tail_gap_below_last_target,
    )
    remaining = cfg.dim - cfg.n_directions

    # The same U(0,1) values and signs are reused across spectra, then mapped
    # into the spectrum-specific valid interval. This makes spectrum comparisons paired.
    u = rng.uniform(0.0, 1.0, size=remaining)
    tail_magnitudes = cfg.tail_min + u * (available_tail_max - cfg.tail_min)
    tail_magnitudes = np.sort(tail_magnitudes)[::-1]
    tail_signs = rng.choice(np.array([-1.0, 1.0]), size=remaining)
    tail = tail_signs * tail_magnitudes

    eigenvalues = np.concatenate([leading, tail])
    A = Q @ np.diag(eigenvalues) @ Q.T
    normality_error = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    if normality_error > 1e-10:
        raise RuntimeError("Constructed system is not numerically normal.")
    return A, Q, eigenvalues, normality_error


def construct_controlled_initial_state(
    cfg: ExperimentConfig,
    true_basis: np.ndarray,
    excitation_profile: Sequence[float],
    state_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use fixed target magnitudes for the case, random signs, and random lower-mode
    coefficients. The seed is intentionally independent of excitation profile,
    so signs/tail coefficients are paired across profile comparisons.
    """
    rng = np.random.default_rng(state_seed)
    coefficients = rng.normal(0.0, cfg.tail_coefficient_scale, size=cfg.dim)

    target_signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.n_directions)
    coefficients[: cfg.n_directions] = (
        target_signs * np.asarray(excitation_profile, dtype=float)
    )

    x0 = true_basis @ coefficients
    return x0, coefficients


def compute_prefix_acceptance_matrix(
    diagnostics: pd.DataFrame,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Exact vectorized-equivalent implementation of estimator prefix acceptance.

    accepted[i, k-1] says that at diagnostics row i the first k directions
    simultaneously satisfy the observation-only acceptance rule over the most
    recent stability_patience windows.
    """
    n_windows = len(diagnostics)
    out = np.zeros((n_windows, cfg.n_directions), dtype=bool)
    p = cfg.stability_patience

    for i in range(n_windows):
        if i < p - 1:
            continue
        recent = diagnostics.iloc[i - p + 1 : i + 1]
        if float(recent.iloc[-1]["relative_window_norm"]) < cfg.relative_window_norm_floor:
            continue

        cumulative_ok = True
        for stage in range(1, cfg.n_directions + 1):
            changes = recent[f"stage_{stage}_direction_change_deg"].to_numpy(dtype=float)
            finite_changes = changes[np.isfinite(changes)]
            stage_ok = len(finite_changes) >= p - 1
            if stage_ok:
                stage_ok = bool(np.all(finite_changes <= cfg.stability_threshold_deg))

            pc1 = recent[f"stage_{stage}_stage_pc1_energy_fraction"].to_numpy(dtype=float)
            if stage_ok:
                stage_ok = bool(
                    np.all(np.isfinite(pc1))
                    and np.all(pc1 >= cfg.min_stage_pc1_energy_fraction)
                )

            residual_before = recent[
                f"stage_{stage}_residual_energy_before_fraction"
            ].to_numpy(dtype=float)
            if stage_ok:
                stage_ok = bool(
                    np.all(np.isfinite(residual_before))
                    and np.all(residual_before >= cfg.min_residual_energy_fraction)
                )

            cumulative_ok = bool(cumulative_ok and stage_ok)
            out[i, stage - 1] = cumulative_ok

    return out


def verify_acceptance_against_estimator(
    diagnostics: pd.DataFrame,
    accepted: np.ndarray,
    estimator_cfg: EstimatorConfig,
) -> None:
    """One-time safety check that our time-resolved acceptance matches the estimator."""
    p = estimator_cfg.stability_patience
    for i in range(len(diagnostics)):
        if i < p - 1:
            expected = [False] * estimator_cfg.n_directions
        else:
            recent = diagnostics.iloc[i - p + 1 : i + 1]
            expected = [
                bool(prefix_window_is_stable(recent, estimator_cfg, stage))
                for stage in range(1, estimator_cfg.n_directions + 1)
            ]
        actual = accepted[i].tolist()
        if actual != expected:
            raise RuntimeError(
                f"Time-resolved acceptance mismatch at diagnostics row {i}: "
                f"actual={actual}, estimator={expected}"
            )


def theoretical_pointwise_dominance_time(
    eigenvalues: np.ndarray,
    coefficients: np.ndarray,
    stage_index0: int,
    competitor_stop_exclusive: int,
) -> float:
    """
    Ground-truth diagnostic only.

    Earliest continuous time t >= 0 at which mode i has amplitude at least as
    large as every specified lower mode j>i:
        |a_i lambda_i^t| >= |a_j lambda_j^t|.

    This is NOT a prediction of the SVD-window acceptance time; it is a simple
    pointwise modal-dominance reference.
    """
    ai = abs(float(coefficients[stage_index0]))
    li = abs(float(eigenvalues[stage_index0]))
    if ai <= 0.0:
        return np.inf

    required = 0.0
    for j in range(stage_index0 + 1, competitor_stop_exclusive):
        aj = abs(float(coefficients[j]))
        lj = abs(float(eigenvalues[j]))
        if aj <= 0.0:
            continue
        if not (li > lj > 0.0):
            return np.inf
        rhs = math.log(aj / ai) / math.log(li / lj)
        required = max(required, rhs)
    return float(max(0.0, required))


def first_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def last_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[-1]) if len(idx) else None


def first_consecutive_true_run(mask: np.ndarray, run_length: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (start_index, confirmation_index) of the first all-True run."""
    if run_length <= 1:
        idx = first_true_index(mask)
        return idx, idx
    count = 0
    for i, value in enumerate(mask.astype(bool)):
        count = count + 1 if value else 0
        if count >= run_length:
            start = i - run_length + 1
            return int(start), int(i)
    return None, None


def safe_log10(x: float) -> float:
    if not np.isfinite(x) or x <= 0.0:
        return np.nan
    return float(np.log10(x))


def reference_error_at_window(
    X: np.ndarray,
    L: np.ndarray,
    true_basis: np.ndarray,
    stage: int,
    window_start: int,
    window_end: int,
) -> float:
    R = X[window_start : window_end + 1] - L
    ref_prior = true_basis[:, : stage - 1]
    R_ref = deflate_by_basis(R, ref_prior)
    u_ref = top_right_singular_direction(R_ref)
    if not np.all(np.isfinite(u_ref)):
        return np.nan
    return float(angle_deg(u_ref, true_basis[:, stage - 1]))


def estimated_error_at_diag_row(
    diagnostics: pd.DataFrame,
    row_index: Optional[int],
    stage: int,
    true_basis: np.ndarray,
) -> float:
    if row_index is None:
        return np.nan
    direction = diagnostics.iloc[row_index][f"direction_{stage}"]
    if direction is None:
        return np.nan
    return float(angle_deg(normalize(np.asarray(direction, dtype=float)), true_basis[:, stage - 1]))


def event_reference_metrics(
    diagnostics: pd.DataFrame,
    X: np.ndarray,
    L: np.ndarray,
    true_basis: np.ndarray,
    stage: int,
    diag_index: Optional[int],
) -> Tuple[float, float]:
    if diag_index is None:
        return np.nan, np.nan
    row = diagnostics.iloc[diag_index]
    est_error = estimated_error_at_diag_row(diagnostics, diag_index, stage, true_basis)
    ref_error = reference_error_at_window(
        X=X,
        L=L,
        true_basis=true_basis,
        stage=stage,
        window_start=int(row["window_start"]),
        window_end=int(row["window_end"]),
    )
    penalty = est_error - ref_error if np.isfinite(est_error) and np.isfinite(ref_error) else np.nan
    return float(ref_error), float(penalty)


def analyse_one_trajectory_time_resolved(
    *,
    cfg: ExperimentConfig,
    estimator_cfg: EstimatorConfig,
    spectrum_name: str,
    excitation_name: str,
    excitation_profile: Sequence[float],
    A: np.ndarray,
    true_basis: np.ndarray,
    eigenvalues: np.ndarray,
    system_replicate: int,
    system_seed: int,
    state_index: int,
    state_seed: int,
    verify_acceptance: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, np.ndarray]]:
    x0, coefficients = construct_controlled_initial_state(
        cfg=cfg,
        true_basis=true_basis,
        excitation_profile=excitation_profile,
        state_seed=state_seed,
    )
    X = simulate_trajectory(A=A, x0=x0, steps=cfg.steps)
    L = np.zeros(cfg.dim, dtype=float)
    diagnostics = rolling_same_window_diagnostics(X=X, L=L, cfg=estimator_cfg)

    accepted = compute_prefix_acceptance_matrix(diagnostics, cfg)
    if verify_acceptance:
        verify_acceptance_against_estimator(diagnostics, accepted, estimator_cfg)

    n_windows = len(diagnostics)
    window_ends = diagnostics["window_end"].to_numpy(dtype=int)
    window_starts = diagnostics["window_start"].to_numpy(dtype=int)

    errors = np.full((cfg.n_directions, n_windows), np.nan, dtype=np.float64)
    pc1 = np.full((cfg.n_directions, n_windows), np.nan, dtype=np.float64)
    for stage in range(1, cfg.n_directions + 1):
        pc1[stage - 1] = diagnostics[
            f"stage_{stage}_stage_pc1_energy_fraction"
        ].to_numpy(dtype=float)
        for i in range(n_windows):
            direction = diagnostics.iloc[i][f"direction_{stage}"]
            if direction is None:
                continue
            errors[stage - 1, i] = angle_deg(
                normalize(np.asarray(direction, dtype=float)),
                true_basis[:, stage - 1],
            )

    correct_angle = np.isfinite(errors) & (errors <= cfg.recovery_tolerance_deg)
    accepted_T = accepted.T
    accepted_correct = accepted_T & correct_angle

    x0_norm = float(np.linalg.norm(x0 - L))
    endpoint_distance = np.linalg.norm(X[window_ends] - L, axis=1)
    relative_distance = endpoint_distance / x0_norm if x0_norm > EPS else np.full(n_windows, np.nan)

    system_uid = f"{spectrum_name}_sys_{system_replicate:02d}"
    case_name = f"{spectrum_name}__{excitation_name}"
    trajectory_uid = f"{case_name}__sys{system_replicate:02d}__state{state_index:02d}"

    trajectory_row: Dict[str, object] = {
        "case_name": case_name,
        "spectrum_case": spectrum_name,
        "excitation_case": excitation_name,
        "system_uid": system_uid,
        "paired_system_replicate": system_replicate,
        "system_seed": system_seed,
        "trajectory_uid": trajectory_uid,
        "initial_state_within_system": state_index,
        "state_seed": state_seed,
        "x0_norm": x0_norm,
        "tail_coefficient_l2_norm": float(np.linalg.norm(coefficients[cfg.n_directions :])),
    }
    for j in range(cfg.n_directions):
        trajectory_row[f"lambda_{j+1}"] = float(eigenvalues[j])
        trajectory_row[f"a{j+1}"] = float(coefficients[j])
        trajectory_row[f"abs_a{j+1}"] = abs(float(coefficients[j]))

    event_rows: List[Dict[str, object]] = []
    for stage in range(1, cfg.n_directions + 1):
        sidx = stage - 1
        A_mask = accepted_T[sidx]
        AC_mask = accepted_correct[sidx]

        first_A = first_true_index(A_mask)
        first_AC = first_true_index(AC_mask)
        sustained_start, sustained_confirm = first_consecutive_true_run(
            AC_mask, cfg.recovery_persistence_windows
        )
        last_AC = last_true_index(AC_mask)
        latest_A = last_true_index(A_mask)

        t_target = theoretical_pointwise_dominance_time(
            eigenvalues=eigenvalues,
            coefficients=coefficients,
            stage_index0=sidx,
            competitor_stop_exclusive=cfg.n_directions,
        )
        t_full = theoretical_pointwise_dominance_time(
            eigenvalues=eigenvalues,
            coefficients=coefficients,
            stage_index0=sidx,
            competitor_stop_exclusive=cfg.dim,
        )

        row: Dict[str, object] = {
            "case_name": case_name,
            "spectrum_case": spectrum_name,
            "excitation_case": excitation_name,
            "system_uid": system_uid,
            "trajectory_uid": trajectory_uid,
            "initial_state_within_system": state_index,
            "stage": stage,
            "target": f"q{stage}",
            "target_lambda": float(eigenvalues[sidx]),
            "target_abs_excitation": abs(float(coefficients[sidx])),
            "ever_accepted": bool(np.any(A_mask)),
            "ever_accepted_and_correct": bool(np.any(AC_mask)),
            "ever_sustained_recovery": sustained_start is not None,
            "n_accepted_windows": int(np.sum(A_mask)),
            "n_accepted_correct_windows": int(np.sum(AC_mask)),
            "theoretical_pointwise_dominance_time_target_modes": t_target,
            "theoretical_pointwise_dominance_time_full_modes": t_full,
            "first_accept_window_end": np.nan,
            "first_accept_absolute_distance": np.nan,
            "first_accept_relative_distance": np.nan,
            "first_accept_log10_relative_distance": np.nan,
            "first_accept_error_deg": np.nan,
            "first_recovery_window_end": np.nan,
            "first_recovery_absolute_distance": np.nan,
            "first_recovery_relative_distance": np.nan,
            "first_recovery_log10_relative_distance": np.nan,
            "first_recovery_error_deg": np.nan,
            "sustained_recovery_start_window_end": np.nan,
            "sustained_recovery_confirm_window_end": np.nan,
            "sustained_recovery_start_absolute_distance": np.nan,
            "sustained_recovery_start_relative_distance": np.nan,
            "sustained_recovery_start_log10_relative_distance": np.nan,
            "sustained_recovery_start_error_deg": np.nan,
            "last_recovery_window_end": np.nan,
            "latest_accepted_window_end": np.nan,
            "latest_accepted_error_deg": np.nan,
            "reference_error_deg_at_first_accept": np.nan,
            "estimated_minus_reference_penalty_deg_at_first_accept": np.nan,
            "reference_error_deg_at_first_recovery": np.nan,
            "estimated_minus_reference_penalty_deg_at_first_recovery": np.nan,
            "reference_error_deg_at_sustained_start": np.nan,
            "estimated_minus_reference_penalty_deg_at_sustained_start": np.nan,
            "reference_error_deg_at_latest_accept": np.nan,
            "estimated_minus_reference_penalty_deg_at_latest_accept": np.nan,
        }

        def fill_event(prefix: str, idx: Optional[int]) -> None:
            if idx is None:
                return
            t = int(window_ends[idx])
            rel = float(relative_distance[idx])
            absolute = float(endpoint_distance[idx])
            row[f"{prefix}_window_end"] = t
            if f"{prefix}_absolute_distance" in row:
                row[f"{prefix}_absolute_distance"] = absolute
            if f"{prefix}_relative_distance" in row:
                row[f"{prefix}_relative_distance"] = rel
            if f"{prefix}_log10_relative_distance" in row:
                row[f"{prefix}_log10_relative_distance"] = safe_log10(rel)
            if f"{prefix}_error_deg" in row:
                row[f"{prefix}_error_deg"] = float(errors[sidx, idx])

        fill_event("first_accept", first_A)
        fill_event("first_recovery", first_AC)
        fill_event("sustained_recovery_start", sustained_start)
        fill_event("last_recovery", last_AC)
        fill_event("latest_accepted", latest_A)
        if sustained_confirm is not None:
            row["sustained_recovery_confirm_window_end"] = int(window_ends[sustained_confirm])

        # Event-level same-window reference diagnostics. The reference never chooses a window.
        for event_key, idx in (
            ("first_accept", first_A),
            ("first_recovery", first_AC),
            ("sustained_start", sustained_start),
            ("latest_accept", latest_A),
        ):
            ref_err, penalty = event_reference_metrics(
                diagnostics=diagnostics,
                X=X,
                L=L,
                true_basis=true_basis,
                stage=stage,
                diag_index=idx,
            )
            row[f"reference_error_deg_at_{event_key}"] = ref_err
            row[f"estimated_minus_reference_penalty_deg_at_{event_key}"] = penalty

        if sustained_start is not None and np.isfinite(t_full):
            row["sustained_start_minus_theoretical_full_dominance_time"] = (
                int(window_ends[sustained_start]) - t_full
            )
        else:
            row["sustained_start_minus_theoretical_full_dominance_time"] = np.nan

        event_rows.append(row)

    traces = {
        "window_start": window_starts,
        "window_end": window_ends,
        "accepted": accepted_T.astype(np.uint8),
        "errors_deg": errors.astype(np.float32),
        "accepted_correct": accepted_correct.astype(np.uint8),
        "pc1_fraction": pc1.astype(np.float32),
        "absolute_distance": endpoint_distance.astype(np.float64),
        "relative_distance": relative_distance.astype(np.float64),
    }
    return trajectory_row, event_rows, traces


def hierarchical_bootstrap_stat(
    frame: pd.DataFrame,
    value_col: str,
    n_boot: int,
    seed: int,
    statistic: str = "mean",
) -> Dict[str, float]:
    data = frame[["system_uid", value_col]].dropna().copy()
    if data.empty:
        return {"estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    system_ids = data["system_uid"].drop_duplicates().to_numpy()
    grouped = {
        sid: data.loc[data["system_uid"] == sid, value_col].astype(float).to_numpy()
        for sid in system_ids
    }

    def stat(values: np.ndarray) -> float:
        return float(np.mean(values)) if statistic == "mean" else float(np.median(values))

    estimate = stat(data[value_col].astype(float).to_numpy())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_systems = rng.choice(system_ids, size=len(system_ids), replace=True)
        chunks: List[np.ndarray] = []
        for sid in sampled_systems:
            values = grouped[sid]
            chunks.append(rng.choice(values, size=len(values), replace=True))
        boot[b] = stat(np.concatenate(chunks))

    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def summarize_case_stage(events: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = events.groupby(
        ["spectrum_case", "excitation_case", "case_name", "stage"],
        sort=True,
    )
    for gidx, (keys, group) in enumerate(grouped):
        spectrum_name, excitation_name, case_name, stage = keys
        work = group.copy()
        for col in ("ever_accepted", "ever_accepted_and_correct", "ever_sustained_recovery"):
            work[col + "_numeric"] = work[col].astype(float)

        row: Dict[str, object] = {
            "spectrum_case": spectrum_name,
            "excitation_case": excitation_name,
            "case_name": case_name,
            "stage": int(stage),
            "target": f"q{int(stage)}",
            "n_systems": int(group["system_uid"].nunique()),
            "n_initial_states_per_system": cfg.initial_states_per_system,
            "n_trajectories": int(len(group)),
        }

        rate_specs = [
            ("ever_acceptance", "ever_accepted_numeric"),
            ("ever_recovery", "ever_accepted_and_correct_numeric"),
            ("sustained_recovery", "ever_sustained_recovery_numeric"),
        ]
        for midx, (label, col) in enumerate(rate_specs):
            boot = hierarchical_bootstrap_stat(
                work,
                col,
                cfg.bootstrap_replicates,
                cfg.seed + gidx * 10000 + midx * 101,
                statistic="mean",
            )
            row[label + "_rate"] = boot["estimate"]
            row[label + "_ci95_low"] = boot["ci95_low"]
            row[label + "_ci95_high"] = boot["ci95_high"]

        sustained = work.loc[work["ever_sustained_recovery"].astype(bool)].copy()
        accepted = work.loc[work["ever_accepted"].astype(bool)].copy()
        recovered = work.loc[work["ever_accepted_and_correct"].astype(bool)].copy()

        row["median_first_accept_window_end"] = float(accepted["first_accept_window_end"].median()) if len(accepted) else np.nan
        row["median_first_recovery_window_end"] = float(recovered["first_recovery_window_end"].median()) if len(recovered) else np.nan
        row["median_sustained_recovery_start_window_end"] = float(sustained["sustained_recovery_start_window_end"].median()) if len(sustained) else np.nan
        row["q25_sustained_recovery_start_window_end"] = float(sustained["sustained_recovery_start_window_end"].quantile(0.25)) if len(sustained) else np.nan
        row["q75_sustained_recovery_start_window_end"] = float(sustained["sustained_recovery_start_window_end"].quantile(0.75)) if len(sustained) else np.nan
        row["median_sustained_recovery_confirm_window_end"] = float(sustained["sustained_recovery_confirm_window_end"].median()) if len(sustained) else np.nan
        row["median_absolute_distance_at_sustained_start"] = float(sustained["sustained_recovery_start_absolute_distance"].median()) if len(sustained) else np.nan
        row["median_relative_distance_at_sustained_start"] = float(sustained["sustained_recovery_start_relative_distance"].median()) if len(sustained) else np.nan
        row["median_log10_relative_distance_at_sustained_start"] = float(sustained["sustained_recovery_start_log10_relative_distance"].median()) if len(sustained) else np.nan
        row["q25_log10_relative_distance_at_sustained_start"] = float(sustained["sustained_recovery_start_log10_relative_distance"].quantile(0.25)) if len(sustained) else np.nan
        row["q75_log10_relative_distance_at_sustained_start"] = float(sustained["sustained_recovery_start_log10_relative_distance"].quantile(0.75)) if len(sustained) else np.nan
        row["median_latest_accepted_error_deg"] = float(accepted["latest_accepted_error_deg"].median()) if len(accepted) else np.nan
        row["median_theoretical_full_dominance_time"] = float(work["theoretical_pointwise_dominance_time_full_modes"].replace([np.inf, -np.inf], np.nan).median())
        row["median_sustained_minus_theoretical_full_time"] = float(sustained["sustained_start_minus_theoretical_full_dominance_time"].median()) if len(sustained) else np.nan
        row["median_reference_error_at_sustained_start_deg"] = float(sustained["reference_error_deg_at_sustained_start"].median()) if len(sustained) else np.nan
        row["median_estimated_minus_reference_penalty_at_sustained_start_deg"] = float(sustained["estimated_minus_reference_penalty_deg_at_sustained_start"].median()) if len(sustained) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["spectrum_case", "excitation_case", "stage"])


def summarize_time_case(
    *,
    spectrum_name: str,
    excitation_name: str,
    window_ends: np.ndarray,
    accepted: np.ndarray,
    errors: np.ndarray,
    accepted_correct: np.ndarray,
    relative_distance: np.ndarray,
) -> pd.DataFrame:
    """
    Inputs have shapes:
      accepted/errors/accepted_correct: [n_traj, n_stage, n_window]
      relative_distance: [n_traj, n_window]
    """
    n_traj, n_stage, n_window = accepted.shape
    rows: List[Dict[str, object]] = []
    for sidx in range(n_stage):
        for widx in range(n_window):
            A = accepted[:, sidx, widx].astype(bool)
            AC = accepted_correct[:, sidx, widx].astype(bool)
            e = errors[:, sidx, widx]
            accepted_errors = e[A & np.isfinite(e)]
            reliability = float(np.mean(AC[A])) if np.any(A) else np.nan
            rows.append({
                "spectrum_case": spectrum_name,
                "excitation_case": excitation_name,
                "case_name": f"{spectrum_name}__{excitation_name}",
                "stage": sidx + 1,
                "target": f"q{sidx+1}",
                "window_end": int(window_ends[widx]),
                "acceptance_rate": float(np.mean(A)),
                "successful_recovery_rate": float(np.mean(AC)),
                "reliability_given_accepted": reliability,
                "median_error_deg_accepted": float(np.median(accepted_errors)) if len(accepted_errors) else np.nan,
                "q25_error_deg_accepted": float(np.quantile(accepted_errors, 0.25)) if len(accepted_errors) else np.nan,
                "q75_error_deg_accepted": float(np.quantile(accepted_errors, 0.75)) if len(accepted_errors) else np.nan,
                "median_relative_distance_to_limit": float(np.nanmedian(relative_distance[:, widx])),
                "median_log10_relative_distance_to_limit": float(np.nanmedian([safe_log10(v) for v in relative_distance[:, widx]])),
                "n_trajectories": n_traj,
            })
    return pd.DataFrame(rows)


def build_design_table(cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spectrum_name, spectrum in SPECTRUM_CASES.items():
        for excitation_name, excitation in EXCITATION_PROFILES.items():
            rows.append({
                "case_name": f"{spectrum_name}__{excitation_name}",
                "spectrum_case": spectrum_name,
                "leading_eigenvalues": ",".join("%g" % v for v in spectrum),
                "excitation_case": excitation_name,
                "target_excitation_magnitudes": ",".join("%g" % v for v in excitation),
                "n_systems": cfg.system_replicates,
                "initial_states_per_system": cfg.initial_states_per_system,
                "trajectories_per_case": cfg.system_replicates * cfg.initial_states_per_system,
                "window_m": cfg.window,
                "steps": cfg.steps,
                "stability_threshold_deg": cfg.stability_threshold_deg,
                "stability_patience": cfg.stability_patience,
                "pc1_energy_threshold": cfg.min_stage_pc1_energy_fraction,
                "external_correctness_tolerance_deg": cfg.recovery_tolerance_deg,
                "sustained_recovery_persistence_windows": cfg.recovery_persistence_windows,
            })
    return pd.DataFrame(rows)


def plot_case_stage_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    output_path: Path,
    fmt: str = ".2f",
) -> None:
    order = [f"{s}__{a}" for s in SPECTRUM_CASES for a in EXCITATION_PROFILES]
    pivot = summary.pivot(index="case_name", columns="stage", values=value_col).reindex(order)
    arr = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(arr, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"q{int(v)}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Direction")
    ax.set_ylabel("Spectrum / excitation case")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center")
    cb = fig.colorbar(image, ax=ax)
    cb.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_time_curves_for_case(
    time_summary: pd.DataFrame,
    case_name: str,
    output_dir: Path,
    recovery_tolerance_deg: float,
) -> None:
    case = time_summary.loc[time_summary["case_name"] == case_name].copy()
    if case.empty:
        return

    # Successful recovery probability over iteration.
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage, g in case.groupby("stage", sort=True):
        g = g.sort_values("window_end")
        ax.plot(g["window_end"], g["successful_recovery_rate"], label=f"q{int(stage)}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Window-end iteration")
    ax.set_ylabel(r"$P(A_i(t)=1, C_i(t)=1)$")
    ax.set_title(f"Time-resolved successful recovery: {case_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{case_name}__recovery_probability_vs_iteration.png", dpi=180)
    plt.close(fig)

    # Median accepted angular error over iteration.
    fig, ax = plt.subplots(figsize=(10, 6))
    for stage, g in case.groupby("stage", sort=True):
        g = g.sort_values("window_end")
        ax.plot(g["window_end"], g["median_error_deg_accepted"], label=f"q{int(stage)}")
    ax.axhline(recovery_tolerance_deg, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Window-end iteration")
    ax.set_ylabel("Median angular error among accepted estimates (degrees)")
    ax.set_title(f"Accepted-direction error over iteration: {case_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{case_name}__accepted_error_vs_iteration.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Redesigned Framework Experiment 3: time-resolved q1...q5 recovery "
            "across three spectra and five controlled excitation profiles."
        )
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--system-replicates", type=int, default=10)
    parser.add_argument("--initial-states-per-system", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--recovery-tolerance-deg", type=float, default=2.5)
    parser.add_argument("--recovery-persistence-windows", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--no-trace-npz", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/framework_experiment3_spectrum_excitation_time_resolved"),
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        n_directions=5,
        system_replicates=args.system_replicates,
        initial_states_per_system=args.initial_states_per_system,
        seed=args.seed,
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
        recovery_persistence_windows=args.recovery_persistence_windows,
        bootstrap_replicates=args.bootstrap_replicates,
        save_trace_npz=not args.no_trace_npz,
    )
    validate_config(cfg)
    estimator_cfg = make_estimator_config(cfg)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    traces_dir = output / "time_traces"
    curves_dir = output / "time_curves"
    traces_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    config_json = asdict(cfg)
    config_json["spectrum_cases"] = {k: list(v) for k, v in SPECTRUM_CASES.items()}
    config_json["excitation_profiles"] = {k: list(v) for k, v in EXCITATION_PROFILES.items()}
    config_json["trajectories_per_case"] = cfg.system_replicates * cfg.initial_states_per_system
    config_json["n_cases"] = len(SPECTRUM_CASES) * len(EXCITATION_PROFILES)
    config_json["total_trajectories"] = (
        len(SPECTRUM_CASES)
        * len(EXCITATION_PROFILES)
        * cfg.system_replicates
        * cfg.initial_states_per_system
    )
    config_json["estimator_config_fields_detected"] = [f.name for f in fields(EstimatorConfig)]
    with (output / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    design = build_design_table(cfg)
    design.to_csv(output / "table1_experiment_design.csv", index=False)

    total_cases = len(SPECTRUM_CASES) * len(EXCITATION_PROFILES)
    trajectories_per_case = cfg.system_replicates * cfg.initial_states_per_system
    total_trajectories = total_cases * trajectories_per_case

    print("================================================================")
    print("Framework Experiment 3: spectrum x excitation x recovery time")
    print("================================================================")
    print("dimension:", cfg.dim)
    print("steps:", cfg.steps)
    print("window:", cfg.window)
    print("directions: q1...q5")
    print("spectrum cases:")
    for k, v in SPECTRUM_CASES.items():
        print("  ", k, v)
    print("excitation profiles |a_i|:")
    for k, v in EXCITATION_PROFILES.items():
        print("  ", k, v)
    print("systems per case:", cfg.system_replicates)
    print("initial states per system per case:", cfg.initial_states_per_system)
    print("trajectories per case:", trajectories_per_case)
    print("number of cases:", total_cases)
    print("TOTAL trajectories:", total_trajectories)
    print("external correctness tolerance:", cfg.recovery_tolerance_deg, "degrees")
    print("sustained recovery persistence:", cfg.recovery_persistence_windows, "windows")
    print("References never select windows; they are evaluated only at observation-only event windows.")
    print("================================================================")

    all_trajectory_rows: List[Dict[str, object]] = []
    all_event_rows: List[Dict[str, object]] = []
    all_system_rows: List[Dict[str, object]] = []
    all_time_summary: List[pd.DataFrame] = []

    # Pre-build the 10 paired systems for each spectrum. System seed depends only
    # on replicate, so Q is paired across spectral cases.
    systems_by_spectrum: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]]] = {}
    for spectrum_name, spectrum in SPECTRUM_CASES.items():
        systems_by_spectrum[spectrum_name] = []
        for rep in range(cfg.system_replicates):
            system_seed = cfg.seed + 100_000 * rep
            A, Q, eigenvalues, normality_error = build_random_normal_system(
                cfg=cfg,
                leading_eigenvalues=spectrum,
                system_seed=system_seed,
            )
            systems_by_spectrum[spectrum_name].append(
                (A, Q, eigenvalues, normality_error, system_seed)
            )
            all_system_rows.append({
                "spectrum_case": spectrum_name,
                "system_uid": f"{spectrum_name}_sys_{rep:02d}",
                "paired_system_replicate": rep,
                "system_seed": system_seed,
                "leading_eigenvalues_json": json.dumps(list(spectrum)),
                "normality_error": normality_error,
                "tail_max_abs_eigenvalue": float(np.max(np.abs(eigenvalues[cfg.n_directions :]))),
                "tail_eigenvalues_json": json.dumps(eigenvalues[cfg.n_directions :].tolist()),
            })

    case_counter = 0
    completed_total = 0
    acceptance_verified_once = False

    for spectrum_name, spectrum in SPECTRUM_CASES.items():
        for excitation_name, excitation_profile in EXCITATION_PROFILES.items():
            case_counter += 1
            case_name = f"{spectrum_name}__{excitation_name}"
            print(f"\n--- Case {case_counter}/{total_cases}: {case_name} ---")

            case_accepted: List[np.ndarray] = []
            case_errors: List[np.ndarray] = []
            case_accepted_correct: List[np.ndarray] = []
            case_absolute_distance: List[np.ndarray] = []
            case_relative_distance: List[np.ndarray] = []
            case_window_ends: Optional[np.ndarray] = None
            case_trajectory_uids: List[str] = []

            for rep in range(cfg.system_replicates):
                A, Q, eigenvalues, _normality_error, system_seed = systems_by_spectrum[spectrum_name][rep]

                for state_idx in range(cfg.initial_states_per_system):
                    # Intentionally independent of spectrum/profile: paired state signs/tail coefficients.
                    state_seed = cfg.seed + 10_000_000 * rep + state_idx
                    trajectory_row, event_rows, traces = analyse_one_trajectory_time_resolved(
                        cfg=cfg,
                        estimator_cfg=estimator_cfg,
                        spectrum_name=spectrum_name,
                        excitation_name=excitation_name,
                        excitation_profile=excitation_profile,
                        A=A,
                        true_basis=Q,
                        eigenvalues=eigenvalues,
                        system_replicate=rep,
                        system_seed=system_seed,
                        state_index=state_idx,
                        state_seed=state_seed,
                        verify_acceptance=not acceptance_verified_once,
                    )
                    acceptance_verified_once = True

                    all_trajectory_rows.append(trajectory_row)
                    all_event_rows.extend(event_rows)
                    case_trajectory_uids.append(str(trajectory_row["trajectory_uid"]))
                    case_accepted.append(traces["accepted"])
                    case_errors.append(traces["errors_deg"])
                    case_accepted_correct.append(traces["accepted_correct"])
                    case_absolute_distance.append(traces["absolute_distance"])
                    case_relative_distance.append(traces["relative_distance"])
                    if case_window_ends is None:
                        case_window_ends = traces["window_end"].copy()
                    elif not np.array_equal(case_window_ends, traces["window_end"]):
                        raise RuntimeError("Window indexing changed within a case.")

                    completed_total += 1

                print(
                    f"  completed system {rep+1}/{cfg.system_replicates}; "
                    f"case trajectories {(rep+1)*cfg.initial_states_per_system}/{trajectories_per_case}; "
                    f"overall {completed_total}/{total_trajectories}"
                )

            if len(case_accepted) != trajectories_per_case:
                raise RuntimeError(
                    f"Case {case_name} produced {len(case_accepted)} trajectories; "
                    f"expected {trajectories_per_case}."
                )
            accepted_arr = np.stack(case_accepted, axis=0).astype(np.uint8)
            errors_arr = np.stack(case_errors, axis=0).astype(np.float32)
            accepted_correct_arr = np.stack(case_accepted_correct, axis=0).astype(np.uint8)
            absolute_distance_arr = np.stack(case_absolute_distance, axis=0).astype(np.float64)
            relative_distance_arr = np.stack(case_relative_distance, axis=0).astype(np.float64)
            if case_window_ends is None:
                raise RuntimeError("Case produced no trajectories.")

            time_summary_case = summarize_time_case(
                spectrum_name=spectrum_name,
                excitation_name=excitation_name,
                window_ends=case_window_ends,
                accepted=accepted_arr,
                errors=errors_arr,
                accepted_correct=accepted_correct_arr,
                relative_distance=relative_distance_arr,
            )
            all_time_summary.append(time_summary_case)

            if cfg.save_trace_npz:
                np.savez_compressed(
                    traces_dir / f"{case_name}.npz",
                    window_end=case_window_ends,
                    accepted=accepted_arr,
                    errors_deg=errors_arr,
                    accepted_correct=accepted_correct_arr,
                    absolute_distance=absolute_distance_arr,
                    relative_distance=relative_distance_arr,
                    trajectory_uid=np.asarray(case_trajectory_uids, dtype=str),
                )

    trajectories = pd.DataFrame(all_trajectory_rows)
    events = pd.DataFrame(all_event_rows)
    systems = pd.DataFrame(all_system_rows)
    time_summary = pd.concat(all_time_summary, ignore_index=True)

    trajectories.to_csv(output / "all_trajectories.csv", index=False)
    events.to_csv(output / "trajectory_stage_recovery_events.csv", index=False)
    systems.to_csv(output / "systems.csv", index=False)
    time_summary.to_csv(output / "time_resolved_summary.csv", index=False)

    case_stage_summary = summarize_case_stage(events, cfg)
    case_stage_summary.to_csv(output / "table2_case_stage_recovery_summary.csv", index=False)

    # Compact tables focused on the two new questions: when recovery occurs and
    # how close to the limit the orbit is at that time.
    time_cols = [
        "spectrum_case", "excitation_case", "case_name", "stage", "target",
        "sustained_recovery_rate", "sustained_recovery_ci95_low", "sustained_recovery_ci95_high",
        "median_sustained_recovery_start_window_end",
        "q25_sustained_recovery_start_window_end",
        "q75_sustained_recovery_start_window_end",
        "median_sustained_recovery_confirm_window_end",
        "median_theoretical_full_dominance_time",
        "median_sustained_minus_theoretical_full_time",
    ]
    case_stage_summary[time_cols].to_csv(
        output / "table3_recovery_iteration_summary.csv", index=False
    )

    distance_cols = [
        "spectrum_case", "excitation_case", "case_name", "stage", "target",
        "sustained_recovery_rate",
        "median_absolute_distance_at_sustained_start",
        "median_relative_distance_at_sustained_start",
        "median_log10_relative_distance_at_sustained_start",
        "q25_log10_relative_distance_at_sustained_start",
        "q75_log10_relative_distance_at_sustained_start",
        "median_reference_error_at_sustained_start_deg",
        "median_estimated_minus_reference_penalty_at_sustained_start_deg",
    ]
    case_stage_summary[distance_cols].to_csv(
        output / "table4_distance_to_limit_at_recovery.csv", index=False
    )

    plot_case_stage_heatmap(
        case_stage_summary,
        "sustained_recovery_rate",
        "Experiment 3: sustained recovery rate across spectrum/excitation cases",
        "sustained recovery rate",
        output / "01_sustained_recovery_rate_heatmap.png",
        fmt=".2f",
    )
    plot_case_stage_heatmap(
        case_stage_summary,
        "median_sustained_recovery_start_window_end",
        "Experiment 3: iteration of first sustained recovery",
        "median window-end iteration",
        output / "02_median_sustained_recovery_iteration_heatmap.png",
        fmt=".0f",
    )
    plot_case_stage_heatmap(
        case_stage_summary,
        "median_log10_relative_distance_at_sustained_start",
        "Experiment 3: closeness to limit at first sustained recovery",
        r"median log10(||x_t-L|| / ||x_0-L||)",
        output / "03_distance_to_limit_at_sustained_recovery_heatmap.png",
        fmt=".2f",
    )

    for case_name in design["case_name"]:
        plot_time_curves_for_case(
            time_summary, case_name, curves_dir, cfg.recovery_tolerance_deg
        )

    print("\n=== Compact case-stage summary ===")
    display_cols = [
        "spectrum_case", "excitation_case", "stage",
        "sustained_recovery_rate",
        "median_sustained_recovery_start_window_end",
        "median_log10_relative_distance_at_sustained_start",
        "median_latest_accepted_error_deg",
    ]
    print(case_stage_summary[display_cols].to_string(index=False))
    print("\nResults written to:", output.resolve())
    print("Trajectories per case:", trajectories_per_case)
    print("Total trajectories:", len(trajectories))
    print("Expected total trajectories:", total_trajectories)
    if len(trajectories) != total_trajectories:
        raise RuntimeError("Unexpected total trajectory count.")


if __name__ == "__main__":
    main()
