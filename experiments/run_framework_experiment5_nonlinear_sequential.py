from __future__ import annotations

import argparse
import importlib.util
import json
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
    """Load the existing same-window observation-only sequential estimator."""
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
prefix_window_is_stable = ESTIMATOR.prefix_window_is_stable
angle_deg = ESTIMATOR.angle_deg


# -----------------------------------------------------------------------------
# Prespecified nonlinear cases
# -----------------------------------------------------------------------------
# Experiment 5 isolates nonlinear/Jacobian-variation effects rather than
# repeating the full spectral-gap and excitation sweeps from Experiments 3-4.
# All cases use the same spectrally distinguishable leading contraction
# magnitudes unless a pair structure requires a 2x2 block.  The spectrum is
# fixed here because Experiments 1-4 already isolate spectral-gap effects;
# Experiment 5 isolates nonlinear/Jacobian-variation effects.
STRUCTURE_CASES = (
    "normal_real",
    "nonnormal_real",
    "rotation_pair",
    "equal_magnitude_pair",
)

NONLINEARITY_LEVELS: Dict[str, float] = {
    "mild_nonlinearity": 0.10,
    "moderate_nonlinearity": 0.50,
    "strong_nonlinearity": 1.00,
}

REAL_LEADING_EIGENVALUES: Tuple[float, ...] = (0.96, 0.93, 0.90, 0.86, 0.82)
TARGET_EXCITATION_MAGNITUDES: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
PAIR_TARGET_EXCITATION_MAGNITUDES: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20
    n_directions: int = 5

    system_replicates: int = 10
    initial_states_per_system: int = 20
    seed: int = 42

    initial_radius: float = 2.0
    tail_coefficient_scale: float = 0.10
    tail_min: float = 0.20
    tail_max: float = 0.78

    # Controlled non-normal geometry for the non-normal nonlinear family.
    nonnormal_shear_strength: float = 0.35

    # Rotation used in J_L for the rotational-pair family.
    rotation_radius: float = 0.90
    rotation_angle_deg: float = 25.0

    # Observation-only acceptance criteria, kept unchanged.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External validation criterion only.
    recovery_tolerance_deg: float = 2.5

    # Nonlinear matrices B,C are scaled to this spectral norm in
    # g(x)=tanh(Bx)^2 * tanh(Cx), an O(||x||^3) nonlinearity near L.
    # Therefore Dg(0)=0 and J_L=A exactly.
    nonlinear_matrix_spectral_norm: float = 1.0

    bootstrap_replicates: int = 2000
    save_trace_npz: bool = True

    # Safety checks.  These do not select an estimate; they only ensure a
    # synthetic trajectory is usable for the validation experiment.
    maximum_allowed_state_norm: float = 1e4
    final_relative_distance_max: float = 1e-5


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.dim <= cfg.n_directions:
        raise ValueError("dim must exceed n_directions.")
    if cfg.n_directions != 5:
        raise ValueError("Experiment 5 is prespecified for five filters.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if not (2 <= cfg.window <= cfg.steps + 1):
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if cfg.system_replicates < 1 or cfg.initial_states_per_system < 1:
        raise ValueError("system/state replicate counts must be positive.")
    if cfg.initial_radius <= 0:
        raise ValueError("initial_radius must be positive.")
    if cfg.recovery_tolerance_deg <= 0:
        raise ValueError("recovery_tolerance_deg must be positive.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be at least 100.")
    if not (0 < cfg.tail_min < cfg.tail_max < 0.80):
        raise ValueError("Require 0 < tail_min < tail_max < 0.80.")
    for name, beta in NONLINEARITY_LEVELS.items():
        if beta <= 0:
            raise ValueError(f"Nonlinearity strength must be positive: {name}")


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


def orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("matrix must be 2-D with at least one column.")
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if len(s) == 0 or s[0] <= EPS:
        raise ValueError("Cannot build a basis from a near-zero matrix.")
    tol = max(matrix.shape) * np.finfo(float).eps * float(s[0])
    rank = int(np.sum(s > tol))
    if rank < matrix.shape[1]:
        raise ValueError("Reference/estimated vectors are numerically rank deficient.")
    return u[:, : matrix.shape[1]]


def largest_principal_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    QA = orthonormal_basis(A)
    QB = orthonormal_basis(B)
    if QA.shape[1] != QB.shape[1]:
        raise ValueError("Subspaces must have equal dimension.")
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.max(np.degrees(np.arccos(s))))


def safe_log10(x: float) -> float:
    if not np.isfinite(x) or x <= 0.0:
        return np.nan
    return float(np.log10(x))


def rotation_block(radius: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return radius * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )


def scale_to_spectral_norm(M: np.ndarray, target_norm: float) -> np.ndarray:
    current = float(np.linalg.norm(M, ord=2))
    if current <= EPS:
        raise ValueError("Cannot scale near-zero matrix.")
    return M * (target_norm / current)


def make_random_shear_pattern(dim: int, n_target: int, rng: np.random.Generator) -> np.ndarray:
    R = np.zeros((dim, dim), dtype=float)
    block = np.triu(rng.normal(size=(n_target, n_target)), k=1)
    max_abs = float(np.max(np.abs(block)))
    if max_abs <= EPS:
        block[0, 1] = 1.0
        max_abs = 1.0
    R[:n_target, :n_target] = block / max_abs
    return R


def build_tail_eigenvalues(cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    n_tail = cfg.dim - cfg.n_directions
    magnitudes = rng.uniform(cfg.tail_min, cfg.tail_max, size=n_tail)
    magnitudes = np.sort(magnitudes)[::-1]
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_tail)
    return signs * magnitudes


def build_system_components(
    cfg: ExperimentConfig,
    structure_case: str,
    system_replicate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Build the limit Jacobian J_L=A, its reference basis, and nonlinear B,C.

    The random orthogonal base and nonlinear matrices are paired across all
    four structure cases for the same system replicate.
    """
    system_seed = cfg.seed + 100_000 * system_replicate
    rng = np.random.default_rng(system_seed)

    G = rng.normal(size=(cfg.dim, cfg.dim))
    Q0, _ = np.linalg.qr(G)
    shear_pattern = make_random_shear_pattern(cfg.dim, cfg.n_directions, rng)
    tail = build_tail_eigenvalues(cfg, rng)

    Bnl = scale_to_spectral_norm(
        rng.normal(size=(cfg.dim, cfg.dim)), cfg.nonlinear_matrix_spectral_norm
    )
    Cnl = scale_to_spectral_norm(
        rng.normal(size=(cfg.dim, cfg.dim)), cfg.nonlinear_matrix_spectral_norm
    )

    reference_basis = Q0.copy()
    structure_metrics: Dict[str, object] = {}

    if structure_case == "normal_real":
        eigenvalues = np.concatenate([np.asarray(REAL_LEADING_EIGENVALUES), tail])
        A = Q0 @ np.diag(eigenvalues) @ Q0.T
        reference_basis = Q0
        structure_metrics["geometry_type"] = "normal_orthogonal_real"
        structure_metrics["shear_strength"] = 0.0

    elif structure_case == "nonnormal_real":
        mixing = np.eye(cfg.dim) + cfg.nonnormal_shear_strength * shear_pattern
        V = Q0 @ mixing
        V = V / np.linalg.norm(V, axis=0, keepdims=True)
        cond_V = float(np.linalg.cond(V))
        if not np.isfinite(cond_V) or cond_V > 1e3:
            raise RuntimeError(f"Eigenvector basis too ill-conditioned: cond(V)={cond_V:.3e}")
        eigenvalues = np.concatenate([np.asarray(REAL_LEADING_EIGENVALUES), tail])
        A = V @ np.diag(eigenvalues) @ np.linalg.inv(V)
        reference_basis = V
        structure_metrics["geometry_type"] = "nonnormal_nonorthogonal_real"
        structure_metrics["shear_strength"] = float(cfg.nonnormal_shear_strength)
        structure_metrics["eigenvector_condition_number"] = cond_V

    elif structure_case == "rotation_pair":
        # q1=0.96, q2=0.93, span(q3,q4)=0.90 R(25deg), q5=0.82.
        block = np.zeros((cfg.dim, cfg.dim), dtype=float)
        block[0, 0] = 0.96
        block[1, 1] = 0.93
        block[2:4, 2:4] = rotation_block(cfg.rotation_radius, cfg.rotation_angle_deg)
        block[4, 4] = 0.82
        block[5:, 5:] = np.diag(tail)
        A = Q0 @ block @ Q0.T
        eigenvalues = np.concatenate([
            np.array([0.96, 0.93, cfg.rotation_radius, cfg.rotation_radius, 0.82]),
            tail,
        ])
        reference_basis = Q0
        structure_metrics["geometry_type"] = "normal_rotation_pair"
        structure_metrics["rotation_radius"] = float(cfg.rotation_radius)
        structure_metrics["rotation_angle_deg"] = float(cfg.rotation_angle_deg)
        structure_metrics["shear_strength"] = 0.0

    elif structure_case == "equal_magnitude_pair":
        # q3 and q4 have equal modulus, so only their 2-D invariant subspace is
        # identifiable as an asymptotic pair target.
        leading = np.array([0.96, 0.93, 0.90, -0.90, 0.82], dtype=float)
        eigenvalues = np.concatenate([leading, tail])
        A = Q0 @ np.diag(eigenvalues) @ Q0.T
        reference_basis = Q0
        structure_metrics["geometry_type"] = "normal_equal_magnitude_pair"
        structure_metrics["shear_strength"] = 0.0

    else:
        raise ValueError(f"Unknown structure_case: {structure_case}")

    normality_fro = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    A_norm_sq = float(np.linalg.norm(A, ord="fro") ** 2)
    relative_nonnormality = normality_fro / A_norm_sq if A_norm_sq > EPS else np.nan

    leading_basis = reference_basis[:, : cfg.n_directions]
    gram_defect = float(
        np.linalg.norm(leading_basis.T @ leading_basis - np.eye(cfg.n_directions), ord="fro")
    )

    pairwise_angles: List[float] = []
    for i in range(cfg.n_directions):
        for j in range(i + 1, cfg.n_directions):
            pairwise_angles.append(angle_deg(leading_basis[:, i], leading_basis[:, j]))

    structure_metrics.update({
        "normality_fro": normality_fro,
        "relative_nonnormality": relative_nonnormality,
        "leading_gram_nonorthogonality_fro": gram_defect,
        "min_pairwise_reference_basis_angle_deg": float(np.min(pairwise_angles)),
        "median_pairwise_reference_basis_angle_deg": float(np.median(pairwise_angles)),
    })

    return A, reference_basis, Bnl, Cnl, {"eigenvalues": eigenvalues, **structure_metrics}


def nonlinear_term(x: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Bounded cubic-order nonlinear term near L=0."""
    a = np.tanh(B @ x)
    b = np.tanh(C @ x)
    return (a * a) * b


def map_U(x: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, beta: float) -> np.ndarray:
    return A @ x + beta * nonlinear_term(x, B, C)


def jacobian_U(x: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, beta: float) -> np.ndarray:
    a = np.tanh(B @ x)
    b = np.tanh(C @ x)
    da = 1.0 - a**2
    db = 1.0 - b**2
    Dg = np.diag(2.0 * a * da * b) @ B + np.diag((a * a) * db) @ C
    return A + beta * Dg


def simulate_nonlinear_trajectory(
    cfg: ExperimentConfig,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beta: float,
    x0: np.ndarray,
    v0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate X and the transported tangent direction V.

        v_{t+1} = J_{x_t} v_t,

    with normalization after each multiplication.  Normalization changes only
    magnitude, not the tangent direction represented by the Jacobian product.
    """
    X = np.empty((cfg.steps + 1, cfg.dim), dtype=float)
    V = np.empty_like(X)
    Jdiff = np.empty(cfg.steps + 1, dtype=float)

    L = np.zeros(cfg.dim, dtype=float)
    J_L = jacobian_U(L, A, B, C, beta)
    denom = max(float(np.linalg.norm(J_L, ord="fro")), EPS)

    X[0] = x0
    V[0] = normalize(v0)

    for t in range(cfg.steps + 1):
        Jx = jacobian_U(X[t], A, B, C, beta)
        Jdiff[t] = float(np.linalg.norm(Jx - J_L, ord="fro") / denom)

        if t == cfg.steps:
            break

        v_next = Jx @ V[t]
        V[t + 1] = normalize(v_next)
        X[t + 1] = map_U(X[t], A, B, C, beta)

        if not np.all(np.isfinite(X[t + 1])):
            raise RuntimeError(f"Non-finite nonlinear state at iteration {t+1}.")
        if float(np.linalg.norm(X[t + 1])) > cfg.maximum_allowed_state_norm:
            raise RuntimeError(
                f"Nonlinear trajectory exceeded safety norm at iteration {t+1}: "
                f"||x||={np.linalg.norm(X[t+1]):.3e}"
            )

    return X, V, Jdiff


def construct_initial_state(
    cfg: ExperimentConfig,
    reference_basis: np.ndarray,
    structure_case: str,
    state_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(state_seed)

    coefficients = np.empty(cfg.dim, dtype=float)
    signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.n_directions)
    magnitudes = (
        PAIR_TARGET_EXCITATION_MAGNITUDES
        if structure_case in {"rotation_pair", "equal_magnitude_pair"}
        else TARGET_EXCITATION_MAGNITUDES
    )
    coefficients[: cfg.n_directions] = signs * np.asarray(magnitudes)
    coefficients[cfg.n_directions :] = rng.normal(
        0.0, cfg.tail_coefficient_scale, size=cfg.dim - cfg.n_directions
    )

    # For a non-normal eigenbasis this directly sets modal coefficients.  For
    # normal pair cases, reference_basis is orthogonal and the same expression
    # gives coordinates in the orthogonal reference basis.
    raw_x0 = reference_basis @ coefficients
    x0 = cfg.initial_radius * normalize(raw_x0)
    v0 = normalize(rng.normal(size=cfg.dim))
    return x0, v0, coefficients


def prefix_acceptance_mask(
    diagnostics: pd.DataFrame,
    est_cfg: EstimatorConfig,
    stage: int,
) -> np.ndarray:
    n = len(diagnostics)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < est_cfg.stability_patience - 1:
            continue
        recent = diagnostics.iloc[i - est_cfg.stability_patience + 1 : i + 1]
        mask[i] = bool(prefix_window_is_stable(recent, est_cfg, stage))
    return mask


def estimated_prefix_matrix(row: pd.Series, stage: int) -> Optional[np.ndarray]:
    vectors: List[np.ndarray] = []
    for j in range(1, stage + 1):
        value = row.get(f"direction_{j}", None)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            return None
        vectors.append(normalize(arr))
    return np.column_stack(vectors)


def pair_u4_window_is_stable(
    recent: pd.DataFrame,
    est_cfg: EstimatorConfig,
) -> bool:
    """
    Observation-only acceptance for a leading 4-D block containing a 2-D
    equal-modulus/rotational pair.

    The individual basis vectors inside the block need not stabilize.  We
    therefore require: (i) the unique q1 filter to satisfy its ordinary
    criteria, (ii) four directions to be present with measurable fourth-stage
    residual signal, (iii) the first four extracted directions jointly to
    explain at least the usual 0.8 energy fraction, and (iv) the accumulated
    U4 estimate itself to be stable over the patience window.

    Importantly, q2 does not have to be individually stable before U4 can be
    identified: a higher-dimensional invariant subspace can become stable
    before its internal one-dimensional modes separate.
    """
    if recent.empty or len(recent) < est_cfg.stability_patience:
        return False
    if not prefix_window_is_stable(recent, est_cfg, 1):
        return False

    last = recent.iloc[-1]
    if float(last["relative_window_norm"]) < est_cfg.relative_window_norm_floor:
        return False

    for _, row in recent.iterrows():
        if int(row.get("n_extracted_directions", 0)) < 4:
            return False
        e4_before = float(row.get("stage_4_residual_energy_before_fraction", np.nan))
        e4_after = float(row.get("stage_4_residual_energy_after_fraction", np.nan))
        if not (np.isfinite(e4_before) and np.isfinite(e4_after)):
            return False
        if e4_before < est_cfg.min_residual_energy_fraction:
            return False
        joint_top4_fraction = max(1.0 - e4_after, 0.0)
        if joint_top4_fraction < est_cfg.min_stage_pc1_energy_fraction:
            return False

    changes: List[float] = []
    for i in range(1, len(recent)):
        A = estimated_prefix_matrix(recent.iloc[i - 1], 4)
        B = estimated_prefix_matrix(recent.iloc[i], 4)
        if A is None or B is None:
            return False
        changes.append(largest_principal_angle_deg(A, B))
    if len(changes) < est_cfg.stability_patience - 1:
        return False
    return bool(np.all(np.asarray(changes) <= est_cfg.stability_threshold_deg))


def pair_u4_acceptance_mask(
    diagnostics: pd.DataFrame,
    est_cfg: EstimatorConfig,
) -> np.ndarray:
    mask = np.zeros(len(diagnostics), dtype=bool)
    for i in range(len(diagnostics)):
        if i < est_cfg.stability_patience - 1:
            continue
        recent = diagnostics.iloc[i - est_cfg.stability_patience + 1 : i + 1]
        mask[i] = pair_u4_window_is_stable(recent, est_cfg)
    return mask


def pair_q5_window_is_stable(
    recent: pd.DataFrame,
    est_cfg: EstimatorConfig,
) -> bool:
    if not pair_u4_window_is_stable(recent, est_cfg):
        return False
    changes = recent["stage_5_direction_change_deg"].to_numpy(dtype=float)
    finite_changes = changes[np.isfinite(changes)]
    if len(finite_changes) < est_cfg.stability_patience - 1:
        return False
    if not np.all(finite_changes <= est_cfg.stability_threshold_deg):
        return False
    pc1 = recent["stage_5_stage_pc1_energy_fraction"].to_numpy(dtype=float)
    if not np.all(np.isfinite(pc1)) or not np.all(pc1 >= est_cfg.min_stage_pc1_energy_fraction):
        return False
    residual = recent["stage_5_residual_energy_before_fraction"].to_numpy(dtype=float)
    if not np.all(np.isfinite(residual)) or not np.all(residual >= est_cfg.min_residual_energy_fraction):
        return False
    return True


def pair_q5_acceptance_mask(
    diagnostics: pd.DataFrame,
    est_cfg: EstimatorConfig,
) -> np.ndarray:
    mask = np.zeros(len(diagnostics), dtype=bool)
    for i in range(len(diagnostics)):
        if i < est_cfg.stability_patience - 1:
            continue
        recent = diagnostics.iloc[i - est_cfg.stability_patience + 1 : i + 1]
        mask[i] = pair_q5_window_is_stable(recent, est_cfg)
    return mask


def direction_from_row(row: pd.Series, stage: int) -> Optional[np.ndarray]:
    value = row.get(f"direction_{stage}", None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        return None
    return normalize(arr)


def primary_target_spec(structure_case: str, stage: int) -> Tuple[str, str, bool]:
    """
    Return (target_name, metric_name, correctness_evaluated).

    Pair cases deliberately leave stage 3 without an individual correctness
    target.  At stage 4 the complete pair is evaluated as U4.
    """
    if structure_case == "normal_real":
        return f"q{stage}", "direction_angle_deg", True

    if structure_case == "nonnormal_real":
        if stage == 1:
            return "q1", "direction_angle_deg", True
        return f"U{stage}", "largest_principal_angle_deg", True

    if structure_case in {"rotation_pair", "equal_magnitude_pair"}:
        if stage == 1:
            return "q1", "direction_angle_deg", True
        if stage == 2:
            return "q2", "direction_angle_deg", True
        if stage == 3:
            return "pair_partial_NA", "not_evaluated", False
        if stage == 4:
            return "U4", "largest_principal_angle_deg", True
        if stage == 5:
            return "q5", "direction_angle_deg", True

    raise ValueError(f"Unknown structure/stage combination: {structure_case}, {stage}")


def primary_error_for_row(
    row: pd.Series,
    reference_basis: np.ndarray,
    structure_case: str,
    stage: int,
) -> Tuple[float, float]:
    """
    Return (primary error, individual q_i diagnostic error).

    For non-normal real systems, accumulated U_k is primary for k>=2.
    For rotation/equal-magnitude pair cases, stage 3 has no primary 1-D target;
    stage 4 uses accumulated U4; stage 5 returns to the distinct q5 direction.
    """
    estimated: List[np.ndarray] = []
    for j in range(1, stage + 1):
        u = direction_from_row(row, j)
        if u is None:
            return np.nan, np.nan
        estimated.append(u)

    qi = normalize(reference_basis[:, stage - 1])
    individual = angle_deg(estimated[-1], qi)
    target_name, metric_name, evaluated = primary_target_spec(structure_case, stage)
    if not evaluated:
        return np.nan, float(individual)

    if metric_name == "direction_angle_deg":
        return float(individual), float(individual)

    est_matrix = np.column_stack(estimated)
    true_matrix = reference_basis[:, :stage]
    return largest_principal_angle_deg(est_matrix, true_matrix), float(individual)


def first_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def last_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[-1]) if len(idx) else None


def analyse_trajectory(
    cfg: ExperimentConfig,
    structure_case: str,
    nonlinearity_name: str,
    beta: float,
    system_replicate: int,
    state_index: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, object]]:
    system_seed = cfg.seed + 100_000 * system_replicate
    state_seed = cfg.seed + 1_000_000 + 10_000 * system_replicate + state_index

    A, reference_basis, Bnl, Cnl, system_meta = build_system_components(
        cfg, structure_case, system_replicate
    )
    x0, v0, coefficients = construct_initial_state(
        cfg, reference_basis, structure_case, state_seed
    )

    X, Vtangent, Jdiff = simulate_nonlinear_trajectory(
        cfg, A, Bnl, Cnl, beta, x0, v0
    )
    L = np.zeros(cfg.dim, dtype=float)
    J_L = jacobian_U(L, A, Bnl, Cnl, beta)
    jl_error = float(np.linalg.norm(J_L - A, ord="fro"))

    x0_norm = float(np.linalg.norm(x0 - L))
    final_relative_distance = float(np.linalg.norm(X[-1] - L) / x0_norm)
    if final_relative_distance > cfg.final_relative_distance_max:
        raise RuntimeError(
            f"Trajectory did not approach L sufficiently: final relative distance "
            f"{final_relative_distance:.3e} > {cfg.final_relative_distance_max:.3e}. "
            f"case={structure_case} beta={beta} system={system_replicate} state={state_index}"
        )

    est_cfg = make_estimator_config(cfg)
    diagnostics = rolling_same_window_diagnostics(X, L, est_cfg)
    window_ends = diagnostics["window_end"].to_numpy(dtype=int)
    n_windows = len(diagnostics)

    accepted = np.zeros((cfg.n_directions, n_windows), dtype=bool)
    primary_errors = np.full((cfg.n_directions, n_windows), np.nan, dtype=float)
    individual_errors = np.full((cfg.n_directions, n_windows), np.nan, dtype=float)
    correctness_evaluated = np.zeros(cfg.n_directions, dtype=bool)

    for stage in range(1, cfg.n_directions + 1):
        if structure_case in {"rotation_pair", "equal_magnitude_pair"}:
            if stage <= 2:
                accepted[stage - 1] = prefix_acceptance_mask(diagnostics, est_cfg, stage)
            elif stage == 3:
                # No unique one-dimensional object exists for the first half
                # of the pair, so stage-3 acceptance is intentionally N/A.
                accepted[stage - 1] = False
            elif stage == 4:
                accepted[stage - 1] = pair_u4_acceptance_mask(diagnostics, est_cfg)
            else:
                accepted[stage - 1] = pair_q5_acceptance_mask(diagnostics, est_cfg)
        else:
            accepted[stage - 1] = prefix_acceptance_mask(diagnostics, est_cfg, stage)

        _, _, evaluated = primary_target_spec(structure_case, stage)
        correctness_evaluated[stage - 1] = evaluated
        for idx in range(n_windows):
            p_err, i_err = primary_error_for_row(
                diagnostics.iloc[idx], reference_basis, structure_case, stage
            )
            primary_errors[stage - 1, idx] = p_err
            individual_errors[stage - 1, idx] = i_err

    correct = np.isfinite(primary_errors) & (primary_errors <= cfg.recovery_tolerance_deg)
    accepted_correct = accepted & correct

    endpoint_distance = np.linalg.norm(X[window_ends] - L, axis=1)
    relative_distance = endpoint_distance / x0_norm
    log10_relative_distance = np.array([safe_log10(x) for x in relative_distance])
    jacobian_difference_at_window = Jdiff[window_ends]

    q1 = normalize(reference_basis[:, 0])
    tangent_vs_q1 = np.array([angle_deg(Vtangent[t], q1) for t in window_ends])
    estimator_q1_vs_tangent = np.full(n_windows, np.nan, dtype=float)
    for idx in range(n_windows):
        u1 = direction_from_row(diagnostics.iloc[idx], 1)
        if u1 is not None:
            estimator_q1_vs_tangent[idx] = angle_deg(u1, Vtangent[window_ends[idx]])

    case_name = f"{structure_case}__{nonlinearity_name}"
    system_uid = f"{case_name}__sys{system_replicate:02d}"
    trajectory_uid = f"{system_uid}__state{state_index:02d}"

    trajectory_row: Dict[str, object] = {
        "case_name": case_name,
        "structure_case": structure_case,
        "nonlinearity_case": nonlinearity_name,
        "beta": float(beta),
        "system_uid": system_uid,
        "system_replicate": system_replicate,
        "system_seed": system_seed,
        "trajectory_uid": trajectory_uid,
        "initial_state_within_system": state_index,
        "state_seed": state_seed,
        "x0_norm": x0_norm,
        "final_relative_distance": final_relative_distance,
        "initial_jacobian_relative_difference": float(Jdiff[0]),
        "final_jacobian_relative_difference": float(Jdiff[-1]),
        "max_jacobian_relative_difference": float(np.max(Jdiff)),
        "median_jacobian_relative_difference": float(np.median(Jdiff)),
        "JL_minus_A_fro": jl_error,
    }
    eigenvalues = np.asarray(system_meta["eigenvalues"], dtype=float)
    for j in range(cfg.n_directions):
        trajectory_row[f"lambda_or_modulus_{j+1}"] = float(eigenvalues[j])
        trajectory_row[f"a{j+1}"] = float(coefficients[j])

    event_rows: List[Dict[str, object]] = []
    for stage in range(1, cfg.n_directions + 1):
        sidx = stage - 1
        A_mask = accepted[sidx]
        target_name, metric_name, evaluated = primary_target_spec(structure_case, stage)
        AC_mask = accepted_correct[sidx] if evaluated else np.zeros(n_windows, dtype=bool)

        first_A = first_true_index(A_mask)
        first_AC = first_true_index(AC_mask) if evaluated else None
        last_AC = last_true_index(AC_mask) if evaluated else None
        latest_A = last_true_index(A_mask)

        row: Dict[str, object] = {
            "case_name": case_name,
            "structure_case": structure_case,
            "nonlinearity_case": nonlinearity_name,
            "beta": float(beta),
            "system_uid": system_uid,
            "trajectory_uid": trajectory_uid,
            "initial_state_within_system": state_index,
            "stage": stage,
            "primary_target": target_name,
            "primary_metric": metric_name,
            "correctness_evaluated": bool(evaluated),
            "ever_accepted": bool(np.any(A_mask)),
            "ever_recovered_primary_target": (bool(np.any(AC_mask)) if evaluated else np.nan),
            "n_accepted_windows": int(np.sum(A_mask)),
            "n_accepted_correct_windows": (int(np.sum(AC_mask)) if evaluated else np.nan),
            "first_accept_window_end": np.nan,
            "first_recovery_window_end": np.nan,
            "last_recovery_window_end": np.nan,
            "latest_accepted_window_end": np.nan,
            "first_recovery_relative_distance": np.nan,
            "first_recovery_log10_relative_distance": np.nan,
            "first_recovery_primary_error_deg": np.nan,
            "first_recovery_jacobian_relative_difference": np.nan,
            "latest_accepted_primary_error_deg": np.nan,
            "latest_accepted_individual_qi_error_deg_diagnostic": np.nan,
            "latest_accepted_jacobian_relative_difference": np.nan,
            "latest_accepted_relative_distance": np.nan,
            "latest_accepted_log10_relative_distance": np.nan,
            "latest_accepted_tangent_vs_q1_deg_diagnostic": np.nan,
            "latest_accepted_estimator_q1_vs_tangent_deg_diagnostic": np.nan,
        }

        if first_A is not None:
            row["first_accept_window_end"] = int(window_ends[first_A])
        if first_AC is not None:
            row["first_recovery_window_end"] = int(window_ends[first_AC])
            row["first_recovery_relative_distance"] = float(relative_distance[first_AC])
            row["first_recovery_log10_relative_distance"] = float(log10_relative_distance[first_AC])
            row["first_recovery_primary_error_deg"] = float(primary_errors[sidx, first_AC])
            row["first_recovery_jacobian_relative_difference"] = float(
                jacobian_difference_at_window[first_AC]
            )
        if last_AC is not None:
            row["last_recovery_window_end"] = int(window_ends[last_AC])
        if latest_A is not None:
            row["latest_accepted_window_end"] = int(window_ends[latest_A])
            row["latest_accepted_individual_qi_error_deg_diagnostic"] = float(
                individual_errors[sidx, latest_A]
            )
            row["latest_accepted_jacobian_relative_difference"] = float(
                jacobian_difference_at_window[latest_A]
            )
            row["latest_accepted_relative_distance"] = float(relative_distance[latest_A])
            row["latest_accepted_log10_relative_distance"] = float(
                log10_relative_distance[latest_A]
            )
            row["latest_accepted_tangent_vs_q1_deg_diagnostic"] = float(
                tangent_vs_q1[latest_A]
            )
            row["latest_accepted_estimator_q1_vs_tangent_deg_diagnostic"] = float(
                estimator_q1_vs_tangent[latest_A]
            )
            if evaluated:
                row["latest_accepted_primary_error_deg"] = float(
                    primary_errors[sidx, latest_A]
                )

        event_rows.append(row)

    traces = {
        "window_end": window_ends.astype(np.int32),
        "accepted": accepted.astype(np.uint8),
        "primary_errors_deg": primary_errors.astype(np.float32),
        "individual_errors_deg": individual_errors.astype(np.float32),
        "accepted_correct": accepted_correct.astype(np.uint8),
        "correctness_evaluated": correctness_evaluated.astype(np.uint8),
        "relative_distance": relative_distance.astype(np.float64),
        "log10_relative_distance": log10_relative_distance.astype(np.float64),
        "jacobian_relative_difference": jacobian_difference_at_window.astype(np.float64),
        "tangent_vs_q1_deg": tangent_vs_q1.astype(np.float32),
        "estimator_q1_vs_tangent_deg": estimator_q1_vs_tangent.astype(np.float32),
    }

    system_row: Dict[str, object] = {
        "case_name": case_name,
        "structure_case": structure_case,
        "nonlinearity_case": nonlinearity_name,
        "beta": float(beta),
        "system_uid": system_uid,
        "system_replicate": system_replicate,
        "system_seed": system_seed,
        "JL_minus_A_fro": jl_error,
        **{k: v for k, v in system_meta.items() if k != "eigenvalues"},
    }

    return trajectory_row, event_rows, traces, system_row


def hierarchical_bootstrap_rate(
    frame: pd.DataFrame,
    value_col: str,
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    if len(frame) == 0:
        return {"estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    estimate = float(frame[value_col].mean())
    systems = frame["system_uid"].unique()
    if len(systems) == 0:
        return {"estimate": estimate, "ci95_low": np.nan, "ci95_high": np.nan}

    rng = np.random.default_rng(seed)
    grouped = {s: frame[frame["system_uid"] == s] for s in systems}
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_systems = rng.choice(systems, size=len(systems), replace=True)
        vals: List[float] = []
        for s in sampled_systems:
            g = grouped[s]
            idx = rng.integers(0, len(g), size=len(g))
            vals.extend(g.iloc[idx][value_col].astype(float).tolist())
        boot[b] = float(np.mean(vals)) if vals else np.nan
    boot = boot[np.isfinite(boot)]
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
        "ci95_high": float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def clean(x: object) -> str:
        try:
            if pd.isna(x):
                return ""
        except (TypeError, ValueError):
            pass
        return str(x).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(dataframe_to_markdown(df), encoding="utf-8")


def summarize_primary(events: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = events.groupby(
        ["structure_case", "nonlinearity_case", "case_name", "stage"], sort=True
    )
    for (structure, nonlinearity, case_name, stage), g in grouped:
        g = g.copy()
        evaluated = bool(g["correctness_evaluated"].iloc[0])
        g["ever_accepted_float"] = g["ever_accepted"].astype(float)
        acc = hierarchical_bootstrap_rate(
            g, "ever_accepted_float", cfg.bootstrap_replicates, cfg.seed + 11 * int(stage)
        )

        row: Dict[str, object] = {
            "structure_case": structure,
            "nonlinearity_case": nonlinearity,
            "case_name": case_name,
            "beta": float(g["beta"].iloc[0]),
            "stage": int(stage),
            "primary_target": g["primary_target"].iloc[0],
            "primary_metric": g["primary_metric"].iloc[0],
            "correctness_evaluated": evaluated,
            "n_systems": int(g["system_uid"].nunique()),
            "n_trajectories": int(len(g)),
            "acceptance_rate": acc["estimate"],
            "acceptance_ci95_low": acc["ci95_low"],
            "acceptance_ci95_high": acc["ci95_high"],
            "ever_recovery_rate": np.nan,
            "ever_recovery_ci95_low": np.nan,
            "ever_recovery_ci95_high": np.nan,
            "overall_recovery_latest_rate": np.nan,
            "reliability_latest_given_accepted_rate": np.nan,
            "false_acceptance_latest_rate": np.nan,
            "median_latest_primary_error_deg_accepted": np.nan,
        }

        if evaluated:
            g["ever_recovered_float"] = g["ever_recovered_primary_target"].astype(float)
            g["latest_correct_float"] = (
                g["latest_accepted_primary_error_deg"].notna()
                & (g["latest_accepted_primary_error_deg"] <= cfg.recovery_tolerance_deg)
            ).astype(float)
            g["latest_false_accept_float"] = (
                g["latest_accepted_primary_error_deg"].notna()
                & (g["latest_accepted_primary_error_deg"] > cfg.recovery_tolerance_deg)
            ).astype(float)
            ever = hierarchical_bootstrap_rate(
                g, "ever_recovered_float", cfg.bootstrap_replicates, cfg.seed + 13 * int(stage)
            )
            latest = hierarchical_bootstrap_rate(
                g, "latest_correct_float", cfg.bootstrap_replicates, cfg.seed + 17 * int(stage)
            )
            false = hierarchical_bootstrap_rate(
                g, "latest_false_accept_float", cfg.bootstrap_replicates, cfg.seed + 19 * int(stage)
            )
            accepted_latest = g[g["latest_accepted_primary_error_deg"].notna()]
            reliability = (
                float(np.mean(accepted_latest["latest_accepted_primary_error_deg"] <= cfg.recovery_tolerance_deg))
                if len(accepted_latest)
                else np.nan
            )
            row.update({
                "ever_recovery_rate": ever["estimate"],
                "ever_recovery_ci95_low": ever["ci95_low"],
                "ever_recovery_ci95_high": ever["ci95_high"],
                "overall_recovery_latest_rate": latest["estimate"],
                "reliability_latest_given_accepted_rate": reliability,
                "false_acceptance_latest_rate": false["estimate"],
                "median_latest_primary_error_deg_accepted": (
                    float(accepted_latest["latest_accepted_primary_error_deg"].median())
                    if len(accepted_latest)
                    else np.nan
                ),
            })

        rows.append(row)
    return pd.DataFrame(rows)


def summarize_recovery_intervals(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = events.groupby(
        ["structure_case", "nonlinearity_case", "case_name", "stage"], sort=True
    )
    for (structure, nonlinearity, case_name, stage), g in grouped:
        evaluated = bool(g["correctness_evaluated"].iloc[0])
        row: Dict[str, object] = {
            "structure_case": structure,
            "nonlinearity_case": nonlinearity,
            "case_name": case_name,
            "beta": float(g["beta"].iloc[0]),
            "stage": int(stage),
            "primary_target": g["primary_target"].iloc[0],
            "correctness_evaluated": evaluated,
            "n_trajectories": int(len(g)),
            "n_ever_recovered": np.nan,
            "ever_recovery_rate": np.nan,
            "median_first_recovery_window_end": np.nan,
            "median_last_recovery_window_end": np.nan,
            "median_recovery_span_inclusive": np.nan,
            "median_n_accepted_correct_windows": np.nan,
            "median_recovery_continuity_fraction": np.nan,
            "median_log10_relative_distance_at_first_recovery": np.nan,
            "median_jacobian_relative_difference_at_first_recovery": np.nan,
            "median_first_recovery_primary_error_deg": np.nan,
        }
        if evaluated:
            recovered = g[g["first_recovery_window_end"].notna()].copy()
            row["n_ever_recovered"] = int(len(recovered))
            row["ever_recovery_rate"] = float(len(recovered) / len(g))
            if len(recovered):
                span = (
                    recovered["last_recovery_window_end"]
                    - recovered["first_recovery_window_end"]
                    + 1
                )
                continuity = recovered["n_accepted_correct_windows"] / span.replace(0, np.nan)
                row.update({
                    "median_first_recovery_window_end": float(recovered["first_recovery_window_end"].median()),
                    "median_last_recovery_window_end": float(recovered["last_recovery_window_end"].median()),
                    "median_recovery_span_inclusive": float(span.median()),
                    "median_n_accepted_correct_windows": float(recovered["n_accepted_correct_windows"].median()),
                    "median_recovery_continuity_fraction": float(continuity.median()),
                    "median_log10_relative_distance_at_first_recovery": float(
                        recovered["first_recovery_log10_relative_distance"].median()
                    ),
                    "median_jacobian_relative_difference_at_first_recovery": float(
                        recovered["first_recovery_jacobian_relative_difference"].median()
                    ),
                    "median_first_recovery_primary_error_deg": float(
                        recovered["first_recovery_primary_error_deg"].median()
                    ),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_nonlinearity(trajectories: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "initial_jacobian_relative_difference",
        "max_jacobian_relative_difference",
        "median_jacobian_relative_difference",
        "final_jacobian_relative_difference",
        "final_relative_distance",
        "JL_minus_A_fro",
    ]
    rows: List[Dict[str, object]] = []
    for (structure, nonlinearity, beta), g in trajectories.groupby(
        ["structure_case", "nonlinearity_case", "beta"], sort=True
    ):
        row: Dict[str, object] = {
            "structure_case": structure,
            "nonlinearity_case": nonlinearity,
            "beta": float(beta),
            "n_trajectories": int(len(g)),
        }
        for metric in metrics:
            row[f"median_{metric}"] = float(g[metric].median())
            row[f"q25_{metric}"] = float(g[metric].quantile(0.25))
            row[f"q75_{metric}"] = float(g[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_tangent_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    q1 = events[events["stage"] == 1]
    for (structure, nonlinearity, beta), g in q1.groupby(
        ["structure_case", "nonlinearity_case", "beta"], sort=True
    ):
        accepted = g[g["latest_accepted_window_end"].notna()]
        rows.append({
            "structure_case": structure,
            "nonlinearity_case": nonlinearity,
            "beta": float(beta),
            "n_latest_q1_accepted": int(len(accepted)),
            "median_tangent_vs_q1_deg_at_latest_q1_accept": (
                float(accepted["latest_accepted_tangent_vs_q1_deg_diagnostic"].median())
                if len(accepted) else np.nan
            ),
            "median_estimator_q1_vs_tangent_deg_at_latest_q1_accept": (
                float(accepted["latest_accepted_estimator_q1_vs_tangent_deg_diagnostic"].median())
                if len(accepted) else np.nan
            ),
            "median_jacobian_relative_difference_at_latest_q1_accept": (
                float(accepted["latest_accepted_jacobian_relative_difference"].median())
                if len(accepted) else np.nan
            ),
            "median_log10_relative_distance_at_latest_q1_accept": (
                float(accepted["latest_accepted_log10_relative_distance"].median())
                if len(accepted) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def case_order() -> List[str]:
    return [f"{s}__{n}" for s in STRUCTURE_CASES for n in NONLINEARITY_LEVELS]


def target_label(structure: str, stage: int) -> str:
    return primary_target_spec(structure, stage)[0]


def plot_ever_recovery_heatmap(summary: pd.DataFrame, output: Path) -> None:
    cases = case_order()
    arr = np.full((len(cases), 5), np.nan, dtype=float)
    labels = ["stage1", "stage2", "stage3", "stage4", "stage5"]
    for i, case in enumerate(cases):
        for stage in range(1, 6):
            m = summary[(summary["case_name"] == case) & (summary["stage"] == stage)]
            if len(m) and bool(m["correctness_evaluated"].iloc[0]):
                arr[i, stage - 1] = float(m["ever_recovery_rate"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(5), labels=labels)
    ax.set_yticks(range(len(cases)), labels=cases)
    ax.set_xlabel("Sequential filter position (target depends on structure)")
    ax.set_ylabel("Nonlinear structure / nonlinearity case")
    ax.set_title("Experiment 5: probability of ever recovering each valid primary target")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Ever-recovery rate")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            text = "N/A" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_first_recovery_heatmap(intervals: pd.DataFrame, output: Path) -> None:
    cases = case_order()
    arr = np.full((len(cases), 5), np.nan, dtype=float)
    for i, case in enumerate(cases):
        for stage in range(1, 6):
            m = intervals[(intervals["case_name"] == case) & (intervals["stage"] == stage)]
            if len(m):
                arr[i, stage - 1] = float(m["median_first_recovery_window_end"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(5), labels=["stage1", "stage2", "stage3", "stage4", "stage5"])
    ax.set_yticks(range(len(cases)), labels=cases)
    ax.set_xlabel("Sequential filter position")
    ax.set_ylabel("Nonlinear structure / nonlinearity case")
    ax.set_title("Experiment 5: median first-recovery iteration")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Window-end iteration")
    finite = arr[np.isfinite(arr)]
    midpoint = float(np.nanmedian(finite)) if len(finite) else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v < midpoint else "black")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distance_at_recovery_heatmap(intervals: pd.DataFrame, output: Path) -> None:
    cases = case_order()
    arr = np.full((len(cases), 5), np.nan, dtype=float)
    for i, case in enumerate(cases):
        for stage in range(1, 6):
            m = intervals[(intervals["case_name"] == case) & (intervals["stage"] == stage)]
            if len(m):
                arr[i, stage - 1] = float(m["median_log10_relative_distance_at_first_recovery"].iloc[0])

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(5), labels=["stage1", "stage2", "stage3", "stage4", "stage5"])
    ax.set_yticks(range(len(cases)), labels=cases)
    ax.set_xlabel("Sequential filter position")
    ax.set_ylabel("Nonlinear structure / nonlinearity case")
    ax.set_title("Experiment 5: closeness to L at first recovery")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("median log10(||x_t-L|| / ||x_0-L||)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_jacobian_variation_vs_distance_from_traces(
    output_dir: Path,
    trace_dir: Path,
    structure_case: str,
) -> None:
    """One plot per structure: median Jacobian variation vs distance, by beta."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for nonlinearity_name, beta in NONLINEARITY_LEVELS.items():
        files = sorted(trace_dir.glob(f"{structure_case}__{nonlinearity_name}__sys*_state*.npz"))
        if not files:
            continue
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        for path in files:
            with np.load(path) as z:
                all_x.append(z["log10_relative_distance"].astype(float))
                all_y.append(z["jacobian_relative_difference"].astype(float))
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < 10:
            continue
        edges = np.linspace(np.nanmin(x), np.nanmax(x), 30)
        centers, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x < hi)
            if np.sum(mask) >= 10:
                centers.append(0.5 * (lo + hi))
                medians.append(float(np.median(y[mask])))
        if centers:
            ax.plot(centers, medians, marker="o", markersize=3,
                    label=f"{nonlinearity_name} (beta={beta:g})")
            plotted = True
    if plotted:
        ax.set_xlabel("log10(||x_t-L|| / ||x_0-L||)")
        ax.set_ylabel("Median ||J_xt - J_L||_F / ||J_L||_F")
        ax.set_title(f"Experiment 5: Jacobian variation as orbit approaches L — {structure_case}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"figure4_jacobian_variation_vs_distance__{structure_case}.png", dpi=180)
    plt.close(fig)


def plot_primary_error_vs_distance_from_traces(
    output_dir: Path,
    trace_dir: Path,
    structure_case: str,
    stage: int,
) -> None:
    target_name, _, evaluated = primary_target_spec(structure_case, stage)
    if not evaluated:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for nonlinearity_name, beta in NONLINEARITY_LEVELS.items():
        files = sorted(trace_dir.glob(f"{structure_case}__{nonlinearity_name}__sys*_state*.npz"))
        if not files:
            continue
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for path in files:
            with np.load(path) as z:
                x = z["log10_relative_distance"].astype(float)
                y = z["primary_errors_deg"].astype(float)[stage - 1]
                accepted = z["accepted"].astype(bool)[stage - 1]
                mask = np.isfinite(x) & np.isfinite(y) & accepted
                xs.append(x[mask])
                ys.append(y[mask])
        x = np.concatenate(xs) if xs else np.array([])
        y = np.concatenate(ys) if ys else np.array([])
        if len(x) < 10:
            continue
        edges = np.linspace(np.nanmin(x), np.nanmax(x), 28)
        centers, medians = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x < hi)
            if np.sum(mask) >= 10:
                centers.append(0.5 * (lo + hi))
                medians.append(float(np.median(y[mask])))
        if centers:
            ax.plot(centers, medians, marker="o", markersize=3,
                    label=f"{nonlinearity_name} (beta={beta:g})")
            plotted = True
    if plotted:
        ax.axhline(cfg_global_for_plot.recovery_tolerance_deg, linestyle="--", linewidth=1.2,
                   label=f"{cfg_global_for_plot.recovery_tolerance_deg:g} deg tolerance")
        ax.set_xlabel("log10(||x_t-L|| / ||x_0-L||)")
        ax.set_ylabel("Median primary error among accepted windows (degrees)")
        ax.set_title(f"Experiment 5: {target_name} error vs distance — {structure_case}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"figure5_primary_error_vs_distance__{structure_case}__stage{stage}_{target_name}.png",
            dpi=180,
        )
    plt.close(fig)


# Set at runtime only so plotting helper can draw the configured threshold.
cfg_global_for_plot: ExperimentConfig


def run_experiment(cfg: ExperimentConfig, output_dir: Path) -> None:
    global cfg_global_for_plot
    cfg_global_for_plot = cfg

    validate_config(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "time_traces"
    if cfg.save_trace_npz:
        trace_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(cfg)
    config_dict["structure_cases"] = list(STRUCTURE_CASES)
    config_dict["nonlinearity_levels"] = NONLINEARITY_LEVELS
    config_dict["real_leading_eigenvalues"] = list(REAL_LEADING_EIGENVALUES)
    config_dict["target_excitation_magnitudes"] = list(TARGET_EXCITATION_MAGNITUDES)
    config_dict["pair_target_excitation_magnitudes"] = list(PAIR_TARGET_EXCITATION_MAGNITUDES)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config_dict, indent=2), encoding="utf-8"
    )

    n_cases = len(STRUCTURE_CASES) * len(NONLINEARITY_LEVELS)
    trajectories_per_case = cfg.system_replicates * cfg.initial_states_per_system
    total_trajectories = n_cases * trajectories_per_case

    print("=" * 68)
    print("Framework Experiment 5: nonlinear sequential direction/subspace recovery")
    print("=" * 68)
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"window: {cfg.window}")
    print(f"filters: {cfg.n_directions}")
    print(f"structures: {STRUCTURE_CASES}")
    print(f"nonlinearity levels beta: {NONLINEARITY_LEVELS}")
    print(f"systems per case: {cfg.system_replicates}")
    print(f"initial states per system: {cfg.initial_states_per_system}")
    print(f"trajectories per case: {trajectories_per_case}")
    print(f"total cases: {n_cases}")
    print(f"total trajectories: {total_trajectories}")
    print(f"external correctness tolerance: {cfg.recovery_tolerance_deg} degrees")
    print(f"target excitation magnitudes: {TARGET_EXCITATION_MAGNITUDES}")
    print("Reference objects come only from J_L=D_L U and are never used for window selection.")
    print(
        "For pair cases, stage 3 correctness is intentionally not evaluated; U4 is the pair target. "
        "Pair acceptance is observation-only and uses U4 subspace stability plus joint pair residual energy."
    )
    print("=" * 68)

    trajectory_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    system_rows_map: Dict[Tuple[str, str, int], Dict[str, object]] = {}

    completed = 0
    for structure_case in STRUCTURE_CASES:
        for nonlinearity_name, beta in NONLINEARITY_LEVELS.items():
            case_name = f"{structure_case}__{nonlinearity_name}"
            for system_replicate in range(cfg.system_replicates):
                for state_index in range(cfg.initial_states_per_system):
                    trajectory_row, events, traces, system_row = analyse_trajectory(
                        cfg=cfg,
                        structure_case=structure_case,
                        nonlinearity_name=nonlinearity_name,
                        beta=beta,
                        system_replicate=system_replicate,
                        state_index=state_index,
                    )
                    trajectory_rows.append(trajectory_row)
                    event_rows.extend(events)
                    system_rows_map[(structure_case, nonlinearity_name, system_replicate)] = system_row

                    if cfg.save_trace_npz:
                        np.savez_compressed(
                            trace_dir / f"{case_name}__sys{system_replicate:02d}_state{state_index:02d}.npz",
                            **traces,
                        )
                    completed += 1
                print(
                    f"completed {case_name}: system {system_replicate + 1}/{cfg.system_replicates}; "
                    f"{completed}/{total_trajectories} trajectories"
                )

    trajectories = pd.DataFrame(trajectory_rows)
    events = pd.DataFrame(event_rows)
    systems = pd.DataFrame(list(system_rows_map.values()))

    expected_events = total_trajectories * cfg.n_directions
    if len(trajectories) != total_trajectories:
        raise RuntimeError(f"Expected {total_trajectories} trajectories, got {len(trajectories)}")
    if len(events) != expected_events:
        raise RuntimeError(f"Expected {expected_events} event rows, got {len(events)}")

    trajectories.to_csv(output_dir / "all_trajectories.csv", index=False)
    events.to_csv(output_dir / "trajectory_stage_recovery_events.csv", index=False)
    systems.to_csv(output_dir / "systems.csv", index=False)

    primary = summarize_primary(events, cfg)
    intervals = summarize_recovery_intervals(events)
    nonlinearity = summarize_nonlinearity(trajectories)
    tangent = summarize_tangent_diagnostics(events)

    design_rows: List[Dict[str, object]] = []
    for structure_case in STRUCTURE_CASES:
        for nonlinearity_name, beta in NONLINEARITY_LEVELS.items():
            target_map = [primary_target_spec(structure_case, s)[0] for s in range(1, 6)]
            design_rows.append({
                "case_name": f"{structure_case}__{nonlinearity_name}",
                "structure_case": structure_case,
                "nonlinearity_case": nonlinearity_name,
                "beta": beta,
                "primary_targets_stage1_to_5": str(target_map),
                "n_systems": cfg.system_replicates,
                "initial_states_per_system": cfg.initial_states_per_system,
                "trajectories_per_case": trajectories_per_case,
                "window_m": cfg.window,
                "steps": cfg.steps,
                "stability_threshold_deg": cfg.stability_threshold_deg,
                "stability_patience": cfg.stability_patience,
                "pc1_energy_threshold": cfg.min_stage_pc1_energy_fraction,
                "external_correctness_tolerance_deg": cfg.recovery_tolerance_deg,
            })
    design = pd.DataFrame(design_rows)

    save_table(design, output_dir, "table1_experiment_design")
    save_table(primary, output_dir, "table2_primary_stagewise_performance")
    save_table(intervals, output_dir, "table3_primary_recovery_intervals")
    save_table(nonlinearity, output_dir, "table4_jacobian_variation_summary")
    save_table(tangent, output_dir, "table5_tangent_diagnostics")

    plot_ever_recovery_heatmap(primary, output_dir / "figure1_ever_recovery_heatmap.png")
    plot_first_recovery_heatmap(intervals, output_dir / "figure2_first_recovery_iteration_heatmap.png")
    plot_distance_at_recovery_heatmap(
        intervals, output_dir / "figure3_distance_to_limit_at_first_recovery_heatmap.png"
    )

    if cfg.save_trace_npz:
        for structure_case in STRUCTURE_CASES:
            plot_jacobian_variation_vs_distance_from_traces(output_dir, trace_dir, structure_case)
            # Plot all primary stages that have a defined target in that structure.
            for stage in range(1, cfg.n_directions + 1):
                if primary_target_spec(structure_case, stage)[2]:
                    plot_primary_error_vs_distance_from_traces(
                        output_dir, trace_dir, structure_case, stage
                    )

    print("\n=== Primary stage-wise summary ===")
    print(
        primary[
            [
                "structure_case",
                "nonlinearity_case",
                "stage",
                "primary_target",
                "correctness_evaluated",
                "acceptance_rate",
                "ever_recovery_rate",
                "reliability_latest_given_accepted_rate",
                "median_latest_primary_error_deg_accepted",
            ]
        ].to_string(index=False)
    )

    print("\n=== Jacobian-variation summary ===")
    print(
        nonlinearity[
            [
                "structure_case",
                "nonlinearity_case",
                "beta",
                "median_initial_jacobian_relative_difference",
                "median_final_jacobian_relative_difference",
                "median_final_relative_distance",
                "median_JL_minus_A_fro",
            ]
        ].to_string(index=False)
    )

    print(f"\nResults written to: {output_dir.resolve()}")
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Framework Experiment 5: nonlinear sequential direction/subspace validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/framework_experiment5_nonlinear_sequential"),
    )
    parser.add_argument("--dim", type=int, default=ExperimentConfig.dim)
    parser.add_argument("--steps", type=int, default=ExperimentConfig.steps)
    parser.add_argument("--window", type=int, default=ExperimentConfig.window)
    parser.add_argument("--systems", type=int, default=ExperimentConfig.system_replicates)
    parser.add_argument(
        "--initial-states", type=int, default=ExperimentConfig.initial_states_per_system
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--bootstrap", type=int, default=ExperimentConfig.bootstrap_replicates)
    parser.add_argument("--no-traces", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        system_replicates=args.systems,
        initial_states_per_system=args.initial_states,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap,
        save_trace_npz=not args.no_traces,
    )
    run_experiment(cfg, args.output)


if __name__ == "__main__":
    main()
