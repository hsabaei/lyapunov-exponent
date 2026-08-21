from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EPS = 1e-14


# -----------------------------------------------------------------------------
# Existing observation-only estimator
# -----------------------------------------------------------------------------
def _load_estimator_module():
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

    searched = "\n".join(f"  - {p}" for p in candidates)
    raise ImportError(
        "Could not locate run_observation_only_same_window_deflation_normal.py.\n"
        f"Searched:\n{searched}"
    )


ESTIMATOR = _load_estimator_module()
EstimatorConfig = ESTIMATOR.Config
rolling_same_window_diagnostics = ESTIMATOR.rolling_same_window_diagnostics
select_prefix_window = ESTIMATOR.select_prefix_window
simulate_trajectory = ESTIMATOR.simulate_trajectory
angle_deg = ESTIMATOR.angle_deg


# -----------------------------------------------------------------------------
# Prespecified experiment design
# -----------------------------------------------------------------------------
SINGLE_CASES = ("strong_gap", "weak_gap", "equal_magnitude", "rotation")
SINGLE_UNIQUE_CASES = ("strong_gap", "weak_gap")
SINGLE_NONUNIQUE_CASES = ("equal_magnitude", "rotation")
MULTI_CASE = "moderate_gap_balanced"


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500

    # The experiment varies only these two robustness factors.
    window_lengths: Tuple[int, ...] = (10, 20, 40)
    noise_levels: Tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)

    # Hierarchical units.  For rho=0 only one realization is used because
    # repeated zero-noise realizations would be duplicate observations.
    system_replicates: int = 10
    initial_states_per_system: int = 10
    noise_realizations_per_nonzero_level: int = 2
    seed: int = 42

    # Fixed estimator acceptance criteria: NOT retuned across noise/window.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External correctness only; never used for acceptance.
    recovery_tolerance_deg: float = 2.5

    # Single-filter cases: same geometry as Experiment 1.
    lambda1: float = 0.96
    strong_lambda2: float = 0.60
    weak_lambda2: float = 0.94
    rotation_angle_deg: float = 25.0
    single_tail_min: float = 0.15
    single_tail_max: float = 0.50
    single_other_mode_scale: float = 0.35

    # Multi-filter case: representative balanced normal system from the
    # earlier multi-filter experiment.  Only the first 3 filters are tested.
    multi_leading_eigenvalues: Tuple[float, ...] = (0.96, 0.92, 0.88)
    multi_tail_min: float = 0.20
    multi_tail_max: float = 0.80
    multi_tail_coefficient_scale: float = 0.10
    multi_n_directions: int = 3

    # Normalize every true initial error to the same norm.  Noise level rho is
    # then defined by E||epsilon_t||^2 = rho^2 ||x0-L||^2.
    initial_error_norm: float = 2.0

    # Hierarchical bootstrap: systems -> states -> noise realizations.
    bootstrap_replicates: int = 1000


def _parse_csv_numbers(text: str, cast: Callable[[str], object]) -> Tuple:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if part:
            vals.append(cast(part))
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Framework Experiment 6: observation-noise and window-length evaluation. "
            "Repeats representative single-filter and multi-filter tests while keeping "
            "the observation-only acceptance rule fixed."
        )
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/framework_experiment6_observation_noise_window"),
    )
    p.add_argument("--dim", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--window-lengths", type=str, default="10,20,40")
    p.add_argument("--noise-levels", type=str, default="0,1e-4,1e-3,1e-2")
    p.add_argument("--system-replicates", type=int, default=10)
    p.add_argument("--initial-states-per-system", type=int, default=10)
    p.add_argument("--noise-realizations", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recovery-tolerance-deg", type=float, default=2.5)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window_lengths=_parse_csv_numbers(args.window_lengths, int),
        noise_levels=_parse_csv_numbers(args.noise_levels, float),
        system_replicates=args.system_replicates,
        initial_states_per_system=args.initial_states_per_system,
        noise_realizations_per_nonzero_level=args.noise_realizations,
        seed=args.seed,
        recovery_tolerance_deg=args.recovery_tolerance_deg,
        bootstrap_replicates=args.bootstrap_replicates,
    )


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.dim < 6:
        raise ValueError("dim must be at least 6.")
    if cfg.steps < 10:
        raise ValueError("steps must be at least 10.")
    if not cfg.window_lengths:
        raise ValueError("At least one window length is required.")
    if len(set(cfg.window_lengths)) != len(cfg.window_lengths):
        raise ValueError("window_lengths must not contain duplicates.")
    for m in cfg.window_lengths:
        if m < 2 or m > cfg.steps + 1:
            raise ValueError(f"Invalid window length m={m}.")
    if not cfg.noise_levels:
        raise ValueError("At least one noise level is required.")
    if any(rho < 0 or not np.isfinite(rho) for rho in cfg.noise_levels):
        raise ValueError("Noise levels must be finite and nonnegative.")
    if 0.0 not in cfg.noise_levels:
        raise ValueError("Include rho=0 as the noiseless control.")
    if cfg.system_replicates < 1 or cfg.initial_states_per_system < 1:
        raise ValueError("System and initial-state counts must be positive.")
    if cfg.noise_realizations_per_nonzero_level < 1:
        raise ValueError("noise_realizations must be positive.")
    if cfg.recovery_tolerance_deg <= 0:
        raise ValueError("recovery_tolerance_deg must be positive.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be >=100.")
    if cfg.multi_n_directions != 3:
        raise ValueError("This prespecified noise experiment uses three multi-filter stages.")
    mags = np.abs(np.asarray(cfg.multi_leading_eigenvalues, dtype=float))
    if len(mags) != cfg.multi_n_directions or not np.all(mags[:-1] > mags[1:]):
        raise ValueError("multi_leading_eigenvalues must have three strictly decreasing magnitudes.")
    if cfg.multi_tail_max >= mags[-1]:
        raise ValueError("multi_tail_max must be below |lambda3|.")


def make_estimator_config(cfg: ExperimentConfig, window: int, n_directions: int) -> EstimatorConfig:
    desired = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": 1,
        "window": window,
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
    return EstimatorConfig(**{k: v for k, v in desired.items() if k in supported})


# -----------------------------------------------------------------------------
# Linear systems and initial states
# -----------------------------------------------------------------------------
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


def rotation_block(radius: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return radius * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )


def deflate_by_basis(error_window: np.ndarray, basis: np.ndarray) -> np.ndarray:
    R = np.asarray(error_window, dtype=float)
    if basis.size == 0:
        return R.copy()
    Q, _ = np.linalg.qr(np.asarray(basis, dtype=float), mode="reduced")
    return R @ (np.eye(R.shape[1]) - Q @ Q.T)


def top_right_singular_direction(matrix: np.ndarray) -> np.ndarray:
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if not np.all(np.isfinite(M)) or np.linalg.norm(M, ord="fro") <= np.finfo(float).tiny:
        return np.full(M.shape[1], np.nan)
    _u, _s, vt = np.linalg.svd(M, full_matrices=False)
    return normalize(vt[0])


def _normalize_x0(x0: np.ndarray, target_norm: float) -> np.ndarray:
    return normalize(x0) * float(target_norm)


def build_single_system(
    cfg: ExperimentConfig,
    case: str,
    system_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if case not in SINGLE_CASES:
        raise ValueError(f"Unknown single-filter case {case}")
    rng = np.random.default_rng(system_seed)
    G = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(G)

    B = np.zeros((cfg.dim, cfg.dim), dtype=float)
    tail_mag = np.sort(
        rng.uniform(cfg.single_tail_min, cfg.single_tail_max, size=cfg.dim - 2)
    )[::-1]
    tail_sign = rng.choice(np.array([-1.0, 1.0]), size=cfg.dim - 2)
    B[2:, 2:] = np.diag(tail_mag * tail_sign)

    if case == "strong_gap":
        B[0, 0] = cfg.lambda1
        B[1, 1] = cfg.strong_lambda2
        target_defined = True
        target_type = "unique_q1"
    elif case == "weak_gap":
        B[0, 0] = cfg.lambda1
        B[1, 1] = cfg.weak_lambda2
        target_defined = True
        target_type = "unique_q1"
    elif case == "equal_magnitude":
        B[0, 0] = cfg.lambda1
        B[1, 1] = -cfg.lambda1
        target_defined = False
        target_type = "no_unique_1d_target"
    else:
        B[:2, :2] = rotation_block(cfg.lambda1, cfg.rotation_angle_deg)
        target_defined = False
        target_type = "no_unique_1d_target"

    A = Q @ B @ Q.T
    normality_error = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    return A, Q, {
        "evaluation_type": "single_filter",
        "case_name": case,
        "target_defined": target_defined,
        "target_type": target_type,
        "normality_error_fro": normality_error,
    }


def build_multi_system(
    cfg: ExperimentConfig,
    system_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(system_seed)
    G = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(G)

    leading = np.asarray(cfg.multi_leading_eigenvalues, dtype=float)
    n_tail = cfg.dim - cfg.multi_n_directions
    tail_mag = np.sort(
        rng.uniform(cfg.multi_tail_min, cfg.multi_tail_max, size=n_tail)
    )[::-1]
    tail_sign = rng.choice(np.array([-1.0, 1.0]), size=n_tail)
    eigenvalues = np.concatenate([leading, tail_mag * tail_sign])
    A = Q @ np.diag(eigenvalues) @ Q.T
    normality_error = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    return A, Q, eigenvalues, {
        "evaluation_type": "multi_filter",
        "case_name": MULTI_CASE,
        "target_defined": True,
        "target_type": "q1_q2_q3",
        "normality_error_fro": normality_error,
    }


def construct_single_x0(
    cfg: ExperimentConfig,
    case: str,
    Q: np.ndarray,
    state_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(state_seed)
    coeff = rng.normal(0.0, cfg.single_other_mode_scale, size=cfg.dim)
    if case in SINGLE_UNIQUE_CASES:
        coeff[0] = float(rng.choice(np.array([-1.0, 1.0])))
        coeff[1] = rng.uniform(0.50, 1.50) * float(rng.choice(np.array([-1.0, 1.0])))
    else:
        leading = normalize(rng.normal(size=2))
        coeff[:2] = leading
    x0 = Q @ coeff
    x0 = _normalize_x0(x0, cfg.initial_error_norm)
    # Scaling x0 rescales all modal coefficients equally.
    scale = cfg.initial_error_norm / np.linalg.norm(Q @ coeff)
    return x0, coeff * scale


def construct_multi_x0(
    cfg: ExperimentConfig,
    Q: np.ndarray,
    state_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(state_seed)
    coeff = rng.normal(0.0, cfg.multi_tail_coefficient_scale, size=cfg.dim)
    signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.multi_n_directions)
    coeff[: cfg.multi_n_directions] = signs  # balanced excitation
    raw = Q @ coeff
    scale = cfg.initial_error_norm / np.linalg.norm(raw)
    return raw * scale, coeff * scale


# -----------------------------------------------------------------------------
# Observation noise
# -----------------------------------------------------------------------------
def add_observation_noise(
    X: np.ndarray,
    x0_norm: float,
    rho: float,
    standard_normal_draw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    y_t = x_t + epsilon_t,
    epsilon_t ~ N(0, sigma^2 I),
    sigma = rho * ||x0-L|| / sqrt(d).

    Thus E||epsilon_t||^2 = rho^2 ||x0-L||^2.  The reported rho is the
    RMS observation-noise norm relative to the initial true error norm.
    """
    d = X.shape[1]
    sigma = float(rho) * float(x0_norm) / math.sqrt(d)
    E = sigma * np.asarray(standard_normal_draw, dtype=float)
    return X + E, E, sigma


# -----------------------------------------------------------------------------
# Per-stage evaluation
# -----------------------------------------------------------------------------
def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= EPS:
        return np.nan
    return float(num / den)


def _reference_metrics_same_window(
    raw_window: np.ndarray,
    true_basis: np.ndarray,
    stage: int,
    tolerance_deg: float,
) -> Dict[str, float | bool]:
    prior = true_basis[:, : stage - 1]
    current = true_basis[:, stage - 1]

    R_ref_before = deflate_by_basis(raw_window, prior)
    u_ref = top_right_singular_direction(R_ref_before)
    if not np.all(np.isfinite(u_ref)):
        return {
            "reference_error_deg": np.nan,
            "reference_correct": False,
            "reference_eta_local": np.nan,
            "reference_remaining_cumulative_fraction": np.nan,
        }

    ref_error = float(angle_deg(u_ref, current))

    # Reference-filter sequence uses the exact preceding/current reference
    # subspace when advancing to the next stage.
    through_current = true_basis[:, :stage]
    R_ref_after_exact = deflate_by_basis(raw_window, through_current)

    e_before = float(np.linalg.norm(R_ref_before, ord="fro") ** 2)
    e_after = float(np.linalg.norm(R_ref_after_exact, ord="fro") ** 2)
    e_original = float(np.linalg.norm(raw_window, ord="fro") ** 2)

    return {
        "reference_error_deg": ref_error,
        "reference_correct": bool(ref_error <= tolerance_deg),
        "reference_eta_local": _safe_ratio(e_after, e_before),
        "reference_remaining_cumulative_fraction": _safe_ratio(e_after, e_original),
    }


def analyse_observed_sequence(
    *,
    cfg: ExperimentConfig,
    evaluation_type: str,
    case_name: str,
    Y: np.ndarray,
    X_true: np.ndarray,
    noise: np.ndarray,
    true_basis: np.ndarray,
    target_defined_single: bool,
    window: int,
    rho: float,
    noise_realization: int,
    system_uid: str,
    system_replicate: int,
    state_index: int,
    trajectory_uid_base: str,
) -> List[Dict[str, object]]:
    if evaluation_type == "single_filter":
        n_directions = 1
    elif evaluation_type == "multi_filter":
        n_directions = cfg.multi_n_directions
    else:
        raise ValueError(evaluation_type)

    est_cfg = make_estimator_config(cfg, window, n_directions)
    L = np.zeros(cfg.dim, dtype=float)
    diagnostics = rolling_same_window_diagnostics(X=Y, L=L, cfg=est_cfg)

    rows: List[Dict[str, object]] = []
    for stage in range(1, n_directions + 1):
        selected, n_candidates = select_prefix_window(
            diagnostics=diagnostics,
            cfg=est_cfg,
            n_stages=stage,
        )

        target_defined = bool(target_defined_single) if evaluation_type == "single_filter" else True
        target = "q1" if evaluation_type == "single_filter" else f"q{stage}"
        primary_metric = "direction_angle_deg" if target_defined else "false_unique_acceptance"

        row: Dict[str, object] = {
            "evaluation_type": evaluation_type,
            "case_name": case_name,
            "system_uid": system_uid,
            "system_replicate": system_replicate,
            "initial_state_within_system": state_index,
            "noise_realization": noise_realization,
            "noise_level_rho": rho,
            "window_length": window,
            "trajectory_uid": f"{trajectory_uid_base}__noise{noise_realization:02d}__rho{rho:g}__m{window}",
            "stage": stage,
            "target": target,
            "target_defined": target_defined,
            "primary_metric": primary_metric,
            "n_stable_candidate_windows": int(n_candidates),
            "accepted": False,
            "correct": np.nan,
            "accepted_and_correct": False,
            "false_acceptance": False,
            "false_unique_acceptance": False if target_defined else False,
            "selected_window_start": np.nan,
            "selected_window_end": np.nan,
            "estimated_error_deg": np.nan,
            "reference_error_deg": np.nan,
            "reference_correct": np.nan,
            "estimated_minus_reference_error_deg": np.nan,
            "estimated_eta_local": np.nan,
            "reference_eta_local": np.nan,
            "estimated_remaining_cumulative_fraction": np.nan,
            "reference_remaining_cumulative_fraction": np.nan,
            "eta_local_estimated_minus_reference": np.nan,
            "remaining_cumulative_estimated_minus_reference": np.nan,
            "selected_noise_to_true_signal_fro_ratio": np.nan,
            "selected_snr_db": np.nan,
            "selected_relative_window_norm": np.nan,
            "selected_stage_pc1_fraction": np.nan,
        }

        if selected is None:
            # Rejection is not an incorrect estimate.  C_i remains unevaluated.
            rows.append(row)
            continue

        direction_obj = selected.get(f"direction_{stage}")
        if direction_obj is None:
            rows.append(row)
            continue
        u_est = normalize(np.asarray(direction_obj, dtype=float))

        row["accepted"] = True
        row["selected_window_start"] = int(selected["window_start"])
        row["selected_window_end"] = int(selected["window_end"])
        row["selected_relative_window_norm"] = float(selected.get("relative_window_norm", np.nan))
        row["selected_stage_pc1_fraction"] = float(
            selected.get(f"stage_{stage}_stage_pc1_energy_fraction", np.nan)
        )

        ws = int(selected["window_start"])
        we = int(selected["window_end"])
        R = Y[ws : we + 1] - L
        Xw = X_true[ws : we + 1] - L
        Ew = noise[ws : we + 1]
        signal_norm = float(np.linalg.norm(Xw, ord="fro"))
        noise_norm = float(np.linalg.norm(Ew, ord="fro"))
        row["selected_noise_to_true_signal_fro_ratio"] = _safe_ratio(noise_norm, signal_norm)
        if noise_norm <= EPS and signal_norm > EPS:
            row["selected_snr_db"] = np.inf
        elif signal_norm > EPS and noise_norm > EPS:
            row["selected_snr_db"] = float(20.0 * np.log10(signal_norm / noise_norm))

        before_frac = float(selected.get(f"stage_{stage}_residual_energy_before_fraction", np.nan))
        after_frac = float(selected.get(f"stage_{stage}_residual_energy_after_fraction", np.nan))
        row["estimated_eta_local"] = _safe_ratio(after_frac, before_frac)
        row["estimated_remaining_cumulative_fraction"] = after_frac

        if not target_defined:
            row["false_unique_acceptance"] = True
            rows.append(row)
            continue

        q = normalize(true_basis[:, stage - 1])
        est_error = float(angle_deg(u_est, q))
        est_correct = bool(est_error <= cfg.recovery_tolerance_deg)
        row["estimated_error_deg"] = est_error
        row["correct"] = est_correct
        row["accepted_and_correct"] = est_correct  # accepted is already true
        row["false_acceptance"] = not est_correct

        ref = _reference_metrics_same_window(
            raw_window=R,
            true_basis=true_basis,
            stage=stage,
            tolerance_deg=cfg.recovery_tolerance_deg,
        )
        row.update(ref)
        if np.isfinite(float(row["reference_error_deg"])):
            row["estimated_minus_reference_error_deg"] = (
                float(row["estimated_error_deg"]) - float(row["reference_error_deg"])
            )
        row["eta_local_estimated_minus_reference"] = (
            float(row["estimated_eta_local"]) - float(row["reference_eta_local"])
            if np.isfinite(float(row["estimated_eta_local"]))
            and np.isfinite(float(row["reference_eta_local"]))
            else np.nan
        )
        row["remaining_cumulative_estimated_minus_reference"] = (
            float(row["estimated_remaining_cumulative_fraction"])
            - float(row["reference_remaining_cumulative_fraction"])
            if np.isfinite(float(row["estimated_remaining_cumulative_fraction"]))
            and np.isfinite(float(row["reference_remaining_cumulative_fraction"]))
            else np.nan
        )
        rows.append(row)

    return rows


# -----------------------------------------------------------------------------
# Hierarchical bootstrap
# -----------------------------------------------------------------------------
def hierarchical_bootstrap_metric(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if frame.empty:
        return {"estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}

    estimate = float(metric(frame))
    systems = frame["system_uid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan, dtype=float)

    # Pre-index once for speed.
    by_system: Dict[object, Dict[object, pd.DataFrame]] = {}
    for sid, sg in frame.groupby("system_uid", sort=False):
        by_system[sid] = {
            state: st.copy()
            for state, st in sg.groupby("initial_state_within_system", sort=False)
        }

    for b in range(n_boot):
        sampled_systems = rng.choice(systems, size=len(systems), replace=True)
        chunks: List[pd.DataFrame] = []
        for draw_index, sid in enumerate(sampled_systems):
            state_map = by_system[sid]
            states = np.asarray(list(state_map.keys()), dtype=object)
            sampled_states = rng.choice(states, size=len(states), replace=True)
            for state_draw_index, state in enumerate(sampled_states):
                st = state_map[state]
                # Noise is the innermost experimental unit.  Resample the rows
                # within this state; for rho=0 there is only one row.
                take = rng.integers(0, len(st), size=len(st))
                sampled = st.iloc[take].copy()
                sampled["_bootstrap_system"] = draw_index
                sampled["_bootstrap_state"] = state_draw_index
                chunks.append(sampled)
        sample = pd.concat(chunks, ignore_index=True)
        boot[b] = float(metric(sample))

    finite = boot[np.isfinite(boot)]
    if len(finite) == 0:
        lo = hi = np.nan
    else:
        lo, hi = np.quantile(finite, [0.025, 0.975])
    return {"estimate": estimate, "ci95_low": float(lo), "ci95_high": float(hi)}


def _mean_bool(col: str) -> Callable[[pd.DataFrame], float]:
    def fn(df: pd.DataFrame) -> float:
        return float(df[col].astype(bool).mean())
    return fn


def _reliability(df: pd.DataFrame) -> float:
    accepted = df["accepted"].astype(bool)
    if not accepted.any():
        return np.nan
    return float(df.loc[accepted, "accepted_and_correct"].astype(bool).mean())


def _reference_reliability(df: pd.DataFrame) -> float:
    accepted = df["accepted"].astype(bool)
    if not accepted.any():
        return np.nan
    vals = df.loc[accepted, "reference_correct"]
    vals = vals[vals.notna()].astype(bool)
    return float(vals.mean()) if len(vals) else np.nan


def _reference_overall_success(df: pd.DataFrame) -> float:
    accepted = df["accepted"].astype(bool)
    ref_correct = df["reference_correct"].map(lambda x: bool(x) if pd.notna(x) else False)
    return float((accepted & ref_correct).mean())


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------
def summarize_outcomes(rows: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    out: List[Dict[str, object]] = []
    keys = ["evaluation_type", "case_name", "noise_level_rho", "window_length", "stage", "target", "target_defined"]
    for gidx, (group_keys, g) in enumerate(rows.groupby(keys, sort=True, dropna=False)):
        evaluation_type, case_name, rho, window, stage, target, target_defined = group_keys
        target_defined = bool(target_defined)
        base = {
            "evaluation_type": evaluation_type,
            "case_name": case_name,
            "noise_level_rho": float(rho),
            "window_length": int(window),
            "stage": int(stage),
            "target": target,
            "target_defined": target_defined,
            "n_systems": int(g["system_uid"].nunique()),
            "n_initial_states_per_system": cfg.initial_states_per_system,
            "n_noise_realizations_per_state": int(g.groupby(["system_uid", "initial_state_within_system"]).size().max()),
            "n_trajectories": int(len(g)),
        }

        for midx, (label, metric) in enumerate([
            ("acceptance", _mean_bool("accepted")),
            ("successful_recovery", _mean_bool("accepted_and_correct")),
            ("false_acceptance", _mean_bool("false_acceptance")),
        ]):
            if not target_defined and label != "acceptance":
                base[f"{label}_rate"] = np.nan
                base[f"{label}_ci95_low"] = np.nan
                base[f"{label}_ci95_high"] = np.nan
                continue
            boot = hierarchical_bootstrap_metric(
                g, metric, cfg.bootstrap_replicates,
                cfg.seed + 100_000 * gidx + 1009 * midx,
            )
            base[f"{label}_rate"] = boot["estimate"]
            base[f"{label}_ci95_low"] = boot["ci95_low"]
            base[f"{label}_ci95_high"] = boot["ci95_high"]

        if target_defined:
            rel = hierarchical_bootstrap_metric(
                g, _reliability, cfg.bootstrap_replicates,
                cfg.seed + 100_000 * gidx + 4007,
            )
            base["reliability_given_accepted"] = rel["estimate"]
            base["reliability_ci95_low"] = rel["ci95_low"]
            base["reliability_ci95_high"] = rel["ci95_high"]
            accepted_errors = pd.to_numeric(
                g.loc[g["accepted"].astype(bool), "estimated_error_deg"], errors="coerce"
            ).dropna()
            base["median_estimated_error_deg_accepted"] = float(accepted_errors.median()) if len(accepted_errors) else np.nan
            base["q25_estimated_error_deg_accepted"] = float(accepted_errors.quantile(0.25)) if len(accepted_errors) else np.nan
            base["q75_estimated_error_deg_accepted"] = float(accepted_errors.quantile(0.75)) if len(accepted_errors) else np.nan
            base["false_unique_acceptance_rate"] = np.nan
        else:
            fu = hierarchical_bootstrap_metric(
                g, _mean_bool("false_unique_acceptance"), cfg.bootstrap_replicates,
                cfg.seed + 100_000 * gidx + 5003,
            )
            base["reliability_given_accepted"] = np.nan
            base["reliability_ci95_low"] = np.nan
            base["reliability_ci95_high"] = np.nan
            base["median_estimated_error_deg_accepted"] = np.nan
            base["q25_estimated_error_deg_accepted"] = np.nan
            base["q75_estimated_error_deg_accepted"] = np.nan
            base["false_unique_acceptance_rate"] = fu["estimate"]
            base["false_unique_acceptance_ci95_low"] = fu["ci95_low"]
            base["false_unique_acceptance_ci95_high"] = fu["ci95_high"]

        out.append(base)
    return pd.DataFrame(out)


def summarize_energy(rows: pd.DataFrame) -> pd.DataFrame:
    use = rows[rows["accepted"].astype(bool)].copy()
    result: List[Dict[str, object]] = []
    keys = ["evaluation_type", "case_name", "noise_level_rho", "window_length", "stage", "target"]
    for group_keys, g in use.groupby(keys, sort=True):
        evaluation_type, case_name, rho, window, stage, target = group_keys
        row: Dict[str, object] = {
            "evaluation_type": evaluation_type,
            "case_name": case_name,
            "noise_level_rho": float(rho),
            "window_length": int(window),
            "stage": int(stage),
            "target": target,
            "n_accepted": int(len(g)),
        }
        for col in [
            "estimated_eta_local",
            "reference_eta_local",
            "estimated_remaining_cumulative_fraction",
            "reference_remaining_cumulative_fraction",
            "eta_local_estimated_minus_reference",
            "remaining_cumulative_estimated_minus_reference",
            "selected_noise_to_true_signal_fro_ratio",
            "selected_snr_db",
        ]:
            vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row[f"median_{col}"] = float(vals.median()) if len(vals) else np.nan
            row[f"q25_{col}"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            row[f"q75_{col}"] = float(vals.quantile(0.75)) if len(vals) else np.nan
        result.append(row)
    return pd.DataFrame(result)


def summarize_reference_vs_estimated(rows: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    use = rows[rows["target_defined"].astype(bool)].copy()
    result: List[Dict[str, object]] = []
    keys = ["evaluation_type", "case_name", "noise_level_rho", "window_length", "stage", "target"]
    for gidx, (group_keys, g) in enumerate(use.groupby(keys, sort=True)):
        evaluation_type, case_name, rho, window, stage, target = group_keys
        est_success = float(g["accepted_and_correct"].astype(bool).mean())
        ref_success = _reference_overall_success(g)
        est_rel = _reliability(g)
        ref_rel = _reference_reliability(g)
        accepted = g[g["accepted"].astype(bool)].copy()
        penalty = pd.to_numeric(accepted["estimated_minus_reference_error_deg"], errors="coerce").dropna()
        result.append({
            "evaluation_type": evaluation_type,
            "case_name": case_name,
            "noise_level_rho": float(rho),
            "window_length": int(window),
            "stage": int(stage),
            "target": target,
            "n_trajectories": int(len(g)),
            "n_accepted": int(g["accepted"].astype(bool).sum()),
            "estimated_successful_recovery_rate": est_success,
            "reference_successful_recovery_rate_same_selected_windows": ref_success,
            "estimated_minus_reference_success_rate": est_success - ref_success,
            "estimated_reliability_given_accepted": est_rel,
            "reference_reliability_same_selected_windows": ref_rel,
            "estimated_minus_reference_reliability": (
                est_rel - ref_rel if np.isfinite(est_rel) and np.isfinite(ref_rel) else np.nan
            ),
            "median_estimated_minus_reference_error_deg": float(penalty.median()) if len(penalty) else np.nan,
            "q25_estimated_minus_reference_error_deg": float(penalty.quantile(0.25)) if len(penalty) else np.nan,
            "q75_estimated_minus_reference_error_deg": float(penalty.quantile(0.75)) if len(penalty) else np.nan,
        })
    return pd.DataFrame(result)


def build_design_table(cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for case in SINGLE_CASES:
        rows.append({
            "evaluation_type": "single_filter",
            "case_name": case,
            "stages_evaluated": "1",
            "primary_target": "q1" if case in SINGLE_UNIQUE_CASES else "no unique 1-D target",
            "noise_levels_rho": ", ".join(f"{x:g}" for x in cfg.noise_levels),
            "window_lengths": ", ".join(str(x) for x in cfg.window_lengths),
            "n_systems": cfg.system_replicates,
            "n_initial_states_per_system": cfg.initial_states_per_system,
            "n_noise_realizations_nonzero": cfg.noise_realizations_per_nonzero_level,
            "noise_definition": "E||eps_t||^2 = rho^2 ||x0-L||^2",
        })
    rows.append({
        "evaluation_type": "multi_filter",
        "case_name": MULTI_CASE,
        "stages_evaluated": "1,2,3",
        "primary_target": "q1, q2, q3",
        "noise_levels_rho": ", ".join(f"{x:g}" for x in cfg.noise_levels),
        "window_lengths": ", ".join(str(x) for x in cfg.window_lengths),
        "n_systems": cfg.system_replicates,
        "n_initial_states_per_system": cfg.initial_states_per_system,
        "n_noise_realizations_nonzero": cfg.noise_realizations_per_nonzero_level,
        "noise_definition": "E||eps_t||^2 = rho^2 ||x0-L||^2",
    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def _heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    value_col: str,
    title: str,
    noise_levels: Sequence[float],
    windows: Sequence[int],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".2f",
):
    mat = np.full((len(noise_levels), len(windows)), np.nan)
    for i, rho in enumerate(noise_levels):
        for j, m in enumerate(windows):
            sub = table[np.isclose(table["noise_level_rho"], rho) & table["window_length"].eq(m)]
            if len(sub):
                mat[i, j] = float(sub.iloc[0][value_col])
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(windows)), [str(m) for m in windows])
    ax.set_yticks(range(len(noise_levels)), [f"{r:g}" for r in noise_levels])
    ax.set_xlabel("window length m")
    ax.set_ylabel("noise level rho")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8)
    return im


def make_figures(out_dir: Path, outcomes: pd.DataFrame, energy: pd.DataFrame, refcomp: pd.DataFrame, cfg: ExperimentConfig) -> None:
    # Figure 1: unique single-filter recovery.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    last_im = None
    for ax, case in zip(axes, SINGLE_UNIQUE_CASES):
        sub = outcomes[(outcomes["case_name"] == case) & (outcomes["stage"] == 1)]
        last_im = _heatmap(
            ax, sub, "successful_recovery_rate", case.replace("_", " ").title(),
            cfg.noise_levels, cfg.window_lengths, vmin=0, vmax=1,
        )
    fig.suptitle("Experiment 6: single-filter successful recovery")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="P(A1=1, C1=1)")
    fig.savefig(out_dir / "figure1_single_filter_successful_recovery_heatmaps.png", dpi=220)
    plt.close(fig)

    # Figure 2: non-unique negative controls.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    last_im = None
    for ax, case in zip(axes, SINGLE_NONUNIQUE_CASES):
        sub = outcomes[(outcomes["case_name"] == case) & (outcomes["stage"] == 1)]
        last_im = _heatmap(
            ax, sub, "false_unique_acceptance_rate", case.replace("_", " ").title(),
            cfg.noise_levels, cfg.window_lengths, vmin=0, vmax=1,
        )
    fig.suptitle("Experiment 6: false unique-direction acceptance")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="false uniqueness acceptance")
    fig.savefig(out_dir / "figure2_single_filter_false_uniqueness_heatmaps.png", dpi=220)
    plt.close(fig)

    # Figure 3: multi-filter successful recovery by stage.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    last_im = None
    for stage, ax in enumerate(axes, start=1):
        sub = outcomes[(outcomes["case_name"] == MULTI_CASE) & (outcomes["stage"] == stage)]
        last_im = _heatmap(
            ax, sub, "successful_recovery_rate", f"q{stage}",
            cfg.noise_levels, cfg.window_lengths, vmin=0, vmax=1,
        )
    fig.suptitle("Experiment 6: multi-filter successful recovery")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="P(Ai=1, Ci=1)")
    fig.savefig(out_dir / "figure3_multifilter_successful_recovery_heatmaps.png", dpi=220)
    plt.close(fig)

    # Figure 4: multi-filter reliability.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    last_im = None
    for stage, ax in enumerate(axes, start=1):
        sub = outcomes[(outcomes["case_name"] == MULTI_CASE) & (outcomes["stage"] == stage)]
        last_im = _heatmap(
            ax, sub, "reliability_given_accepted", f"q{stage}",
            cfg.noise_levels, cfg.window_lengths, vmin=0, vmax=1,
        )
    fig.suptitle("Experiment 6: multi-filter reliability")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="P(Ci=1 | Ai=1)")
    fig.savefig(out_dir / "figure4_multifilter_reliability_heatmaps.png", dpi=220)
    plt.close(fig)

    # Figure 5: remaining squared error, eta_i, estimated sequence.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    last_im = None
    for stage, ax in enumerate(axes, start=1):
        sub = energy[(energy["case_name"] == MULTI_CASE) & (energy["stage"] == stage)]
        last_im = _heatmap(
            ax, sub, "median_estimated_eta_local", f"stage {stage}: eta_{stage}",
            cfg.noise_levels, cfg.window_lengths, vmin=0, vmax=1,
        )
    fig.suptitle("Experiment 6: remaining squared error fraction after each filter")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="median eta_i")
    fig.savefig(out_dir / "figure5_remaining_squared_error_heatmaps.png", dpi=220)
    plt.close(fig)

    # Figure 6: performance difference between estimated and reference sequences.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    vals = refcomp[refcomp["case_name"] == MULTI_CASE]["estimated_minus_reference_success_rate"].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    lim = max(0.05, float(np.max(np.abs(finite))) if len(finite) else 0.05)
    last_im = None
    for stage, ax in enumerate(axes, start=1):
        sub = refcomp[(refcomp["case_name"] == MULTI_CASE) & (refcomp["stage"] == stage)]
        last_im = _heatmap(
            ax, sub, "estimated_minus_reference_success_rate", f"q{stage}",
            cfg.noise_levels, cfg.window_lengths, vmin=-lim, vmax=lim,
            fmt="+.2f",
        )
    fig.suptitle("Experiment 6: estimated-filter minus reference-filter recovery rate")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="Delta successful recovery rate")
    fig.savefig(out_dir / "figure6_reference_vs_estimated_recovery_difference_heatmaps.png", dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main simulation
# -----------------------------------------------------------------------------
def run_experiment(cfg: ExperimentConfig, out_dir: Path) -> None:
    validate_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_rows: List[Dict[str, object]] = []
    trajectory_rows: List[Dict[str, object]] = []

    branch_names = list(SINGLE_CASES) + [MULTI_CASE]
    total_base_orbits = len(branch_names) * cfg.system_replicates * cfg.initial_states_per_system
    base_done = 0

    for system_replicate in range(cfg.system_replicates):
        # Same random orientation seed is deliberately reused across cases.
        system_seed = cfg.seed + 100_000 * system_replicate + 17

        single_systems = {
            case: build_single_system(cfg, case, system_seed)
            for case in SINGLE_CASES
        }
        A_multi, Q_multi, eig_multi, meta_multi = build_multi_system(cfg, system_seed)

        for case_index, branch in enumerate(branch_names):
            if branch in SINGLE_CASES:
                A, Q, meta = single_systems[branch]
                evaluation_type = "single_filter"
                true_basis = Q[:, :1]
                target_defined_single = branch in SINGLE_UNIQUE_CASES
            else:
                A, Q, meta = A_multi, Q_multi, meta_multi
                evaluation_type = "multi_filter"
                true_basis = Q[:, : cfg.multi_n_directions]
                target_defined_single = True

            for state_index in range(cfg.initial_states_per_system):
                # Paired state seed across all branch comparisons.
                state_seed = cfg.seed + 10_000_000 + system_replicate * 10_000 + state_index
                if evaluation_type == "single_filter":
                    x0, coeff = construct_single_x0(cfg, branch, Q, state_seed)
                else:
                    x0, coeff = construct_multi_x0(cfg, Q, state_seed)

                X = simulate_trajectory(A=A, x0=x0, steps=cfg.steps)
                x0_norm = float(np.linalg.norm(x0))
                system_uid = f"{branch}__sys{system_replicate:02d}"
                trajectory_uid_base = f"{branch}__sys{system_replicate:02d}__state{state_index:02d}"

                trajectory_rows.append({
                    "evaluation_type": evaluation_type,
                    "case_name": branch,
                    "system_uid": system_uid,
                    "system_replicate": system_replicate,
                    "system_seed": system_seed,
                    "initial_state_within_system": state_index,
                    "state_seed": state_seed,
                    "base_trajectory_uid": trajectory_uid_base,
                    "x0_norm": x0_norm,
                    "normality_error_fro": meta["normality_error_fro"],
                    "target_defined_single": target_defined_single,
                    "a1": float(coeff[0]),
                    "a2": float(coeff[1]),
                    "a3": float(coeff[2]) if len(coeff) > 2 else np.nan,
                })

                # A standard-normal draw is paired across rho and window length.
                # For rho=0 only realization 0 is used to avoid duplicate controls.
                noise_draws: Dict[int, np.ndarray] = {}
                max_noise_rep = cfg.noise_realizations_per_nonzero_level
                for noise_rep in range(max_noise_rep):
                    noise_seed = (
                        cfg.seed
                        + 20_000_000
                        + system_replicate * 100_000
                        + state_index * 100
                        + noise_rep
                    )
                    noise_draws[noise_rep] = np.random.default_rng(noise_seed).normal(size=X.shape)

                for rho in cfg.noise_levels:
                    n_noise = 1 if rho == 0.0 else cfg.noise_realizations_per_nonzero_level
                    for noise_rep in range(n_noise):
                        Y, E, sigma = add_observation_noise(
                            X=X,
                            x0_norm=x0_norm,
                            rho=rho,
                            standard_normal_draw=noise_draws[noise_rep],
                        )
                        for window in cfg.window_lengths:
                            stage_rows = analyse_observed_sequence(
                                cfg=cfg,
                                evaluation_type=evaluation_type,
                                case_name=branch,
                                Y=Y,
                                X_true=X,
                                noise=E,
                                true_basis=true_basis,
                                target_defined_single=target_defined_single,
                                window=window,
                                rho=rho,
                                noise_realization=noise_rep,
                                system_uid=system_uid,
                                system_replicate=system_replicate,
                                state_index=state_index,
                                trajectory_uid_base=trajectory_uid_base,
                            )
                            for r in stage_rows:
                                r["noise_sigma_per_coordinate"] = sigma
                                r["x0_norm"] = x0_norm
                            all_rows.extend(stage_rows)

                base_done += 1
                if base_done % 20 == 0 or base_done == total_base_orbits:
                    print(f"completed {base_done}/{total_base_orbits} base orbits")

    stage_df = pd.DataFrame(all_rows)
    traj_df = pd.DataFrame(trajectory_rows)
    stage_df.to_csv(out_dir / "trajectory_stage_outcomes.csv", index=False)
    traj_df.to_csv(out_dir / "base_trajectories.csv", index=False)

    table1 = build_design_table(cfg)
    table2 = summarize_outcomes(stage_df, cfg)
    table3 = summarize_energy(stage_df)
    table4 = summarize_reference_vs_estimated(stage_df, cfg)

    table1.to_csv(out_dir / "table1_experiment_design.csv", index=False)
    table2.to_csv(out_dir / "table2_acceptance_recovery_reliability.csv", index=False)
    table3.to_csv(out_dir / "table3_remaining_squared_error.csv", index=False)
    table4.to_csv(out_dir / "table4_reference_vs_estimated_filter.csv", index=False)

    make_figures(out_dir, table2, table3, table4, cfg)

    manifest = {
        "experiment": "Framework Experiment 6: observation-noise/window-length evaluation",
        "noise_model": "y_t=x_t+epsilon_t, epsilon_t~N(0,sigma^2 I), sigma=rho||x0-L||/sqrt(d)",
        "single_filter_cases": list(SINGLE_CASES),
        "multi_filter_case": MULTI_CASE,
        "tables": [
            "table1_experiment_design.csv",
            "table2_acceptance_recovery_reliability.csv",
            "table3_remaining_squared_error.csv",
            "table4_reference_vs_estimated_filter.csv",
        ],
        "figures": [
            "figure1_single_filter_successful_recovery_heatmaps.png",
            "figure2_single_filter_false_uniqueness_heatmaps.png",
            "figure3_multifilter_successful_recovery_heatmaps.png",
            "figure4_multifilter_reliability_heatmaps.png",
            "figure5_remaining_squared_error_heatmaps.png",
            "figure6_reference_vs_estimated_recovery_difference_heatmaps.png",
        ],
        "notes": [
            "Acceptance criteria are fixed across all noise levels and window lengths.",
            "C_i is unevaluated when A_i=0; rejection is not counted as incorrectness.",
            "Equal-magnitude and rotation controls have no unique 1-D correctness target; acceptance is reported as false uniqueness acceptance.",
            "Reference-filter candidates are evaluated at the same window selected by the estimated-filter sequence; ground truth never selects a window.",
            "eta_i = ||R^(i+1)||_F^2 / ||R^(i)||_F^2 is reported for estimated and reference sequences.",
            "Hierarchical bootstrap resamples systems, then initial states, then noise realizations.",
        ],
    }
    with (out_dir / "report_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nGenerated:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    run_experiment(cfg, args.output.resolve())


if __name__ == "__main__":
    main()
