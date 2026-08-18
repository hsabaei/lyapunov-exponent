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

# All three levels are genuinely non-normal.  The same random shear pattern is
# reused within a system replicate, scaled by these strengths.
NONNORMALITY_LEVELS: Dict[str, float] = {
    "mild_nonnormal": 0.15,
    "moderate_nonnormal": 0.35,
    "strong_nonnormal": 0.60,
}

# Experiment 4 isolates geometry, so target excitation is fixed and balanced.
TARGET_EXCITATION_MAGNITUDES: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20
    n_directions: int = 5

    system_replicates: int = 10
    initial_states_per_system: int = 20
    seed: int = 42

    # q6,...,qd remain lower modes and are only nuisance signal.
    tail_coefficient_scale: float = 0.10
    tail_max: float = 0.90
    tail_min: float = 0.20
    tail_gap_below_last_target: float = 0.01

    # Observation-only acceptance criteria, unchanged from Experiments 1-3.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External validation tolerance only.
    recovery_tolerance_deg: float = 2.5

    bootstrap_replicates: int = 2000
    save_trace_npz: bool = True


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.n_directions != 5:
        raise ValueError("Experiment 4 is prespecified for five sequential filters.")
    if cfg.dim <= cfg.n_directions:
        raise ValueError("dim must exceed n_directions.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if not (2 <= cfg.window <= cfg.steps + 1):
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if cfg.system_replicates < 1 or cfg.initial_states_per_system < 1:
        raise ValueError("system and initial-state counts must be positive.")
    if cfg.recovery_tolerance_deg <= 0.0:
        raise ValueError("recovery_tolerance_deg must be positive.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be at least 100.")

    for name, spectrum in SPECTRUM_CASES.items():
        if len(spectrum) != cfg.n_directions:
            raise ValueError(f"Spectrum {name} must have five target eigenvalues.")
        mags = np.abs(np.asarray(spectrum, dtype=float))
        if not np.all((mags > 0.0) & (mags < 1.0)):
            raise ValueError(f"Invalid eigenvalue magnitude in {name}.")
        if not np.all(mags[:-1] > mags[1:]):
            raise ValueError(f"{name} must satisfy |lambda1|>...>|lambda5|.")


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
        raise ValueError("Cannot build basis from a near-zero matrix.")
    tol = max(matrix.shape) * np.finfo(float).eps * float(s[0])
    rank = int(np.sum(s > tol))
    if rank < matrix.shape[1]:
        raise ValueError("Target/estimated vectors are numerically rank deficient.")
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


def first_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def last_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.flatnonzero(mask)
    return int(idx[-1]) if len(idx) else None


def make_random_shear_pattern(dim: int, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a unit-scaled strictly upper-triangular shear pattern among the
    first five eigenvectors.  The same pattern is reused at all non-normality
    levels for a given system replicate.
    """
    R = np.zeros((dim, dim), dtype=float)
    block = np.triu(rng.normal(size=(n_target, n_target)), k=1)
    max_abs = float(np.max(np.abs(block)))
    if max_abs <= EPS:
        block[0, 1] = 1.0
        max_abs = 1.0
    block = block / max_abs
    R[:n_target, :n_target] = block
    return R


def build_controlled_nonnormal_system(
    cfg: ExperimentConfig,
    leading_eigenvalues: Sequence[float],
    shear_strength: float,
    system_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Construct A = V diag(lambda) V^{-1} with real non-orthogonal right
    eigenvectors.  V is generated by applying a controlled shear to a random
    orthogonal basis.  Columns are normalized after shearing.
    """
    rng = np.random.default_rng(system_seed)

    G = rng.normal(size=(cfg.dim, cfg.dim))
    Q0, _ = np.linalg.qr(G)

    shear_pattern = make_random_shear_pattern(cfg.dim, cfg.n_directions, rng)
    mixing = np.eye(cfg.dim) + float(shear_strength) * shear_pattern
    V = Q0 @ mixing
    V = V / np.linalg.norm(V, axis=0, keepdims=True)

    cond_V = float(np.linalg.cond(V))
    if not np.isfinite(cond_V) or cond_V > 1e3:
        raise RuntimeError(f"Eigenvector basis too ill-conditioned: cond(V)={cond_V:.3e}")

    leading = np.asarray(leading_eigenvalues, dtype=float)
    available_tail_max = min(
        cfg.tail_max,
        abs(float(leading[-1])) - cfg.tail_gap_below_last_target,
    )
    if available_tail_max <= cfg.tail_min:
        raise ValueError("No valid interval for lower eigenvalues.")

    remaining = cfg.dim - cfg.n_directions
    u = rng.uniform(0.0, 1.0, size=remaining)
    tail_magnitudes = cfg.tail_min + u * (available_tail_max - cfg.tail_min)
    tail_magnitudes = np.sort(tail_magnitudes)[::-1]
    tail_signs = rng.choice(np.array([-1.0, 1.0]), size=remaining)
    tail = tail_signs * tail_magnitudes
    eigenvalues = np.concatenate([leading, tail])

    A = V @ np.diag(eigenvalues) @ np.linalg.inv(V)

    normality_fro = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    A_norm_sq = float(np.linalg.norm(A, ord="fro") ** 2)
    relative_nonnormality = normality_fro / A_norm_sq if A_norm_sq > EPS else np.nan

    leading_V = V[:, : cfg.n_directions]
    gram = leading_V.T @ leading_V
    gram_defect = float(np.linalg.norm(gram - np.eye(cfg.n_directions), ord="fro"))

    pairwise_angles: List[float] = []
    for i in range(cfg.n_directions):
        for j in range(i + 1, cfg.n_directions):
            pairwise_angles.append(angle_deg(leading_V[:, i], leading_V[:, j]))

    # Diagnostic lower bound for an individual orthogonal sequential output.
    # If the previous true accumulated subspace U_{i-1} were exact, the next
    # estimator output must lie in U_{i-1}^perp.  This is the closest possible
    # angle from q_i to that orthogonal complement.
    orthogonality_floors = [0.0]
    for stage in range(2, cfg.n_directions + 1):
        prev = orthonormal_basis(V[:, : stage - 1])
        qi = normalize(V[:, stage - 1])
        residual = qi - prev @ (prev.T @ qi)
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= EPS:
            floor = 90.0
        else:
            floor = angle_deg(qi, residual / residual_norm)
        orthogonality_floors.append(float(floor))

    metrics: Dict[str, float] = {
        "eigenvector_condition_number": cond_V,
        "leading_gram_nonorthogonality_fro": gram_defect,
        "nonnormality_fro": normality_fro,
        "relative_nonnormality": relative_nonnormality,
        "min_pairwise_target_eigenvector_angle_deg": float(np.min(pairwise_angles)),
        "median_pairwise_target_eigenvector_angle_deg": float(np.median(pairwise_angles)),
        "max_pairwise_target_eigenvector_angle_deg": float(np.max(pairwise_angles)),
    }
    for i, floor in enumerate(orthogonality_floors, start=1):
        metrics[f"q{i}_individual_orthogonality_floor_deg"] = float(floor)

    return A, V, eigenvalues, metrics


def construct_initial_state(
    cfg: ExperimentConfig,
    true_eigenvectors: np.ndarray,
    state_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(state_seed)
    coefficients = np.empty(cfg.dim, dtype=float)

    signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.n_directions)
    coefficients[: cfg.n_directions] = signs * np.asarray(TARGET_EXCITATION_MAGNITUDES)
    coefficients[cfg.n_directions :] = rng.normal(
        loc=0.0,
        scale=cfg.tail_coefficient_scale,
        size=cfg.dim - cfg.n_directions,
    )

    x0 = true_eigenvectors @ coefficients
    return x0, coefficients


def prefix_acceptance_mask(diagnostics: pd.DataFrame, est_cfg: EstimatorConfig, stage: int) -> np.ndarray:
    n = len(diagnostics)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if i < est_cfg.stability_patience - 1:
            continue
        recent = diagnostics.iloc[i - est_cfg.stability_patience + 1 : i + 1]
        mask[i] = bool(prefix_window_is_stable(recent, est_cfg, stage))
    return mask


def direction_from_row(row: pd.Series, stage: int) -> Optional[np.ndarray]:
    value = row.get(f"direction_{stage}", None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        return None
    return normalize(arr)


def primary_error_for_row(
    row: pd.Series,
    true_eigenvectors: np.ndarray,
    stage: int,
) -> Tuple[float, float]:
    """
    Returns
    -------
    primary_error
        q1 angular error for stage 1; largest principal angle between
        accumulated estimated and true subspaces for stage >=2.
    individual_stage_error
        Angle between qhat_stage and q_stage.  Diagnostic only for stage >=2.
    """
    estimated: List[np.ndarray] = []
    for j in range(1, stage + 1):
        u = direction_from_row(row, j)
        if u is None:
            return np.nan, np.nan
        estimated.append(u)

    q_stage = normalize(true_eigenvectors[:, stage - 1])
    individual = angle_deg(estimated[-1], q_stage)

    if stage == 1:
        return individual, individual

    est_matrix = np.column_stack(estimated)
    true_matrix = true_eigenvectors[:, :stage]
    primary = largest_principal_angle_deg(est_matrix, true_matrix)
    return float(primary), float(individual)


def analyse_trajectory(
    cfg: ExperimentConfig,
    spectrum_name: str,
    geometry_name: str,
    shear_strength: float,
    system_replicate: int,
    state_index: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, np.ndarray], Dict[str, object]]:
    system_seed = cfg.seed + 100_000 * system_replicate
    state_seed = cfg.seed + 1_000_000 + 10_000 * system_replicate + state_index

    A, V, eigenvalues, geometry_metrics = build_controlled_nonnormal_system(
        cfg=cfg,
        leading_eigenvalues=SPECTRUM_CASES[spectrum_name],
        shear_strength=shear_strength,
        system_seed=system_seed,
    )
    x0, coefficients = construct_initial_state(cfg, V, state_seed)
    X = simulate_trajectory(A, x0, cfg.steps)
    L = np.zeros(cfg.dim, dtype=float)

    est_cfg = make_estimator_config(cfg)
    diagnostics = rolling_same_window_diagnostics(X, L, est_cfg)
    window_ends = diagnostics["window_end"].to_numpy(dtype=int)
    n_windows = len(diagnostics)

    accepted = np.zeros((cfg.n_directions, n_windows), dtype=bool)
    primary_errors = np.full((cfg.n_directions, n_windows), np.nan, dtype=float)
    individual_errors = np.full((cfg.n_directions, n_windows), np.nan, dtype=float)

    for stage in range(1, cfg.n_directions + 1):
        accepted[stage - 1] = prefix_acceptance_mask(diagnostics, est_cfg, stage)
        for idx in range(n_windows):
            p_err, i_err = primary_error_for_row(diagnostics.iloc[idx], V, stage)
            primary_errors[stage - 1, idx] = p_err
            individual_errors[stage - 1, idx] = i_err

    correct = np.isfinite(primary_errors) & (primary_errors <= cfg.recovery_tolerance_deg)
    accepted_correct = accepted & correct

    x0_norm = float(np.linalg.norm(x0 - L))
    endpoint_distance = np.linalg.norm(X[window_ends] - L, axis=1)
    relative_distance = endpoint_distance / x0_norm if x0_norm > EPS else np.full(n_windows, np.nan)

    case_name = f"{spectrum_name}__{geometry_name}"
    system_uid = f"{case_name}__sys{system_replicate:02d}"
    trajectory_uid = f"{system_uid}__state{state_index:02d}"

    trajectory_row: Dict[str, object] = {
        "case_name": case_name,
        "spectrum_case": spectrum_name,
        "geometry_case": geometry_name,
        "shear_strength": float(shear_strength),
        "system_uid": system_uid,
        "system_replicate": system_replicate,
        "system_seed": system_seed,
        "trajectory_uid": trajectory_uid,
        "initial_state_within_system": state_index,
        "state_seed": state_seed,
        "x0_norm": x0_norm,
    }
    for j in range(cfg.n_directions):
        trajectory_row[f"lambda_{j+1}"] = float(eigenvalues[j])
        trajectory_row[f"a{j+1}"] = float(coefficients[j])

    event_rows: List[Dict[str, object]] = []
    for stage in range(1, cfg.n_directions + 1):
        sidx = stage - 1
        A_mask = accepted[sidx]
        AC_mask = accepted_correct[sidx]
        first_A = first_true_index(A_mask)
        first_AC = first_true_index(AC_mask)
        last_AC = last_true_index(AC_mask)
        latest_A = last_true_index(A_mask)

        target_name = "q1" if stage == 1 else f"U{stage}"
        primary_metric = "direction_angle_deg" if stage == 1 else "largest_principal_angle_deg"

        row: Dict[str, object] = {
            "case_name": case_name,
            "spectrum_case": spectrum_name,
            "geometry_case": geometry_name,
            "shear_strength": float(shear_strength),
            "system_uid": system_uid,
            "trajectory_uid": trajectory_uid,
            "initial_state_within_system": state_index,
            "stage": stage,
            "primary_target": target_name,
            "primary_metric": primary_metric,
            "target_lambda": float(eigenvalues[sidx]),
            "ever_accepted": bool(np.any(A_mask)),
            "ever_recovered_primary_target": bool(np.any(AC_mask)),
            "n_accepted_windows": int(np.sum(A_mask)),
            "n_accepted_correct_windows": int(np.sum(AC_mask)),
            "first_accept_window_end": np.nan,
            "first_recovery_window_end": np.nan,
            "last_recovery_window_end": np.nan,
            "latest_accepted_window_end": np.nan,
            "first_recovery_relative_distance": np.nan,
            "first_recovery_log10_relative_distance": np.nan,
            "first_recovery_primary_error_deg": np.nan,
            "latest_accepted_primary_error_deg": np.nan,
            "latest_accepted_individual_qi_error_deg_diagnostic": np.nan,
            "individual_qi_orthogonality_floor_deg_diagnostic": float(
                geometry_metrics[f"q{stage}_individual_orthogonality_floor_deg"]
            ),
        }

        if first_A is not None:
            row["first_accept_window_end"] = int(window_ends[first_A])
        if first_AC is not None:
            t = int(window_ends[first_AC])
            rel = float(relative_distance[first_AC])
            row["first_recovery_window_end"] = t
            row["first_recovery_relative_distance"] = rel
            row["first_recovery_log10_relative_distance"] = safe_log10(rel)
            row["first_recovery_primary_error_deg"] = float(primary_errors[sidx, first_AC])
        if last_AC is not None:
            row["last_recovery_window_end"] = int(window_ends[last_AC])
        if latest_A is not None:
            row["latest_accepted_window_end"] = int(window_ends[latest_A])
            row["latest_accepted_primary_error_deg"] = float(primary_errors[sidx, latest_A])
            row["latest_accepted_individual_qi_error_deg_diagnostic"] = float(
                individual_errors[sidx, latest_A]
            )

        event_rows.append(row)

    traces = {
        "window_end": window_ends.astype(np.int32),
        "accepted": accepted.astype(np.uint8),
        "primary_errors_deg": primary_errors.astype(np.float32),
        "individual_errors_deg": individual_errors.astype(np.float32),
        "accepted_correct": accepted_correct.astype(np.uint8),
        "relative_distance": relative_distance.astype(np.float64),
    }

    system_row: Dict[str, object] = {
        "case_name": case_name,
        "spectrum_case": spectrum_name,
        "geometry_case": geometry_name,
        "shear_strength": float(shear_strength),
        "system_uid": system_uid,
        "system_replicate": system_replicate,
        **geometry_metrics,
    }
    for j in range(cfg.n_directions):
        system_row[f"lambda_{j+1}"] = float(eigenvalues[j])

    return trajectory_row, event_rows, traces, system_row


def hierarchical_bootstrap_rate(frame: pd.DataFrame, value_col: str, n_boot: int, seed: int) -> Dict[str, float]:
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
            chunks.append(rng.choice(values, size=len(values), replace=True))
        boot[b] = float(np.concatenate(chunks).mean())
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    def clean(x: object) -> str:
        if isinstance(x, (list, tuple, np.ndarray)):
            return str(x).replace("|", "\\|").replace("\n", " ")
        try:
            if pd.isna(x):
                return ""
        except (TypeError, ValueError):
            pass
        return str(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(dataframe_to_markdown(df), encoding="utf-8")


def summarize_primary(events: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = events.groupby(["spectrum_case", "geometry_case", "case_name", "stage"], sort=True)
    for (spectrum, geometry, case_name, stage), g in grouped:
        g = g.copy()
        g["ever_accepted_float"] = g["ever_accepted"].astype(float)
        g["ever_recovered_float"] = g["ever_recovered_primary_target"].astype(float)
        latest_correct = (
            g["latest_accepted_primary_error_deg"].notna()
            & (g["latest_accepted_primary_error_deg"] <= cfg.recovery_tolerance_deg)
        ).astype(float)
        g["latest_correct_float"] = latest_correct
        g["latest_false_accept_float"] = (
            g["latest_accepted_primary_error_deg"].notna()
            & (g["latest_accepted_primary_error_deg"] > cfg.recovery_tolerance_deg)
        ).astype(float)

        acc = hierarchical_bootstrap_rate(g, "ever_accepted_float", cfg.bootstrap_replicates, cfg.seed + 11 * int(stage))
        ever = hierarchical_bootstrap_rate(g, "ever_recovered_float", cfg.bootstrap_replicates, cfg.seed + 13 * int(stage))
        latest = hierarchical_bootstrap_rate(g, "latest_correct_float", cfg.bootstrap_replicates, cfg.seed + 17 * int(stage))
        false = hierarchical_bootstrap_rate(g, "latest_false_accept_float", cfg.bootstrap_replicates, cfg.seed + 19 * int(stage))

        accepted_latest = g[g["latest_accepted_primary_error_deg"].notna()]
        reliability = float(
            np.mean(accepted_latest["latest_accepted_primary_error_deg"] <= cfg.recovery_tolerance_deg)
        ) if len(accepted_latest) else np.nan

        rows.append({
            "spectrum_case": spectrum,
            "geometry_case": geometry,
            "case_name": case_name,
            "stage": int(stage),
            "primary_target": g["primary_target"].iloc[0],
            "primary_metric": g["primary_metric"].iloc[0],
            "n_systems": int(g["system_uid"].nunique()),
            "n_trajectories": int(len(g)),
            "acceptance_rate": acc["estimate"],
            "acceptance_ci95_low": acc["ci95_low"],
            "acceptance_ci95_high": acc["ci95_high"],
            "ever_recovery_rate": ever["estimate"],
            "ever_recovery_ci95_low": ever["ci95_low"],
            "ever_recovery_ci95_high": ever["ci95_high"],
            "overall_recovery_latest_rate": latest["estimate"],
            "overall_recovery_latest_ci95_low": latest["ci95_low"],
            "overall_recovery_latest_ci95_high": latest["ci95_high"],
            "reliability_latest_given_accepted_rate": reliability,
            "false_acceptance_latest_rate": false["estimate"],
            "median_latest_primary_error_deg_accepted": float(accepted_latest["latest_accepted_primary_error_deg"].median()) if len(accepted_latest) else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_recovery_intervals(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = events.groupby(["spectrum_case", "geometry_case", "case_name", "stage"], sort=True)
    for (spectrum, geometry, case_name, stage), g in grouped:
        recovered = g[g["first_recovery_window_end"].notna()].copy()
        if len(recovered):
            span = recovered["last_recovery_window_end"] - recovered["first_recovery_window_end"] + 1
            continuity = recovered["n_accepted_correct_windows"] / span.replace(0, np.nan)
            row = {
                "spectrum_case": spectrum,
                "geometry_case": geometry,
                "case_name": case_name,
                "stage": int(stage),
                "primary_target": g["primary_target"].iloc[0],
                "n_trajectories": int(len(g)),
                "n_ever_recovered": int(len(recovered)),
                "ever_recovery_rate": float(len(recovered) / len(g)),
                "median_first_recovery_window_end": float(recovered["first_recovery_window_end"].median()),
                "median_last_recovery_window_end": float(recovered["last_recovery_window_end"].median()),
                "median_recovery_span_inclusive": float(span.median()),
                "median_n_accepted_correct_windows": float(recovered["n_accepted_correct_windows"].median()),
                "median_recovery_continuity_fraction": float(continuity.median()),
                "median_log10_relative_distance_at_first_recovery": float(recovered["first_recovery_log10_relative_distance"].median()),
                "median_first_recovery_primary_error_deg": float(recovered["first_recovery_primary_error_deg"].median()),
            }
        else:
            row = {
                "spectrum_case": spectrum,
                "geometry_case": geometry,
                "case_name": case_name,
                "stage": int(stage),
                "primary_target": g["primary_target"].iloc[0],
                "n_trajectories": int(len(g)),
                "n_ever_recovered": 0,
                "ever_recovery_rate": 0.0,
                "median_first_recovery_window_end": np.nan,
                "median_last_recovery_window_end": np.nan,
                "median_recovery_span_inclusive": np.nan,
                "median_n_accepted_correct_windows": np.nan,
                "median_recovery_continuity_fraction": np.nan,
                "median_log10_relative_distance_at_first_recovery": np.nan,
                "median_first_recovery_primary_error_deg": np.nan,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_individual_diagnostics(events: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = events.groupby(["spectrum_case", "geometry_case", "case_name", "stage"], sort=True)
    for (spectrum, geometry, case_name, stage), g in grouped:
        accepted = g[g["latest_accepted_primary_error_deg"].notna()].copy()
        rows.append({
            "spectrum_case": spectrum,
            "geometry_case": geometry,
            "case_name": case_name,
            "stage": int(stage),
            "primary_target": g["primary_target"].iloc[0],
            "n_latest_accepted": int(len(accepted)),
            "median_primary_error_deg_latest_accepted": float(accepted["latest_accepted_primary_error_deg"].median()) if len(accepted) else np.nan,
            "median_individual_qi_error_deg_latest_accepted_diagnostic": float(accepted["latest_accepted_individual_qi_error_deg_diagnostic"].median()) if len(accepted) else np.nan,
            "median_individual_qi_orthogonality_floor_deg_diagnostic": float(g["individual_qi_orthogonality_floor_deg_diagnostic"].median()),
        })
    return pd.DataFrame(rows)


def summarize_geometry(systems: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "eigenvector_condition_number",
        "leading_gram_nonorthogonality_fro",
        "relative_nonnormality",
        "min_pairwise_target_eigenvector_angle_deg",
        "median_pairwise_target_eigenvector_angle_deg",
    ]
    rows: List[Dict[str, object]] = []
    for geometry, g in systems.groupby("geometry_case", sort=True):
        row: Dict[str, object] = {
            "geometry_case": geometry,
            "shear_strength": float(g["shear_strength"].iloc[0]),
            "n_system_case_rows": int(len(g)),
        }
        for metric in metrics:
            row[f"median_{metric}"] = float(g[metric].median())
            row[f"min_{metric}"] = float(g[metric].min())
            row[f"max_{metric}"] = float(g[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(table: pd.DataFrame, value_col: str, output: Path, title: str, cbar_label: str, fmt: str) -> None:
    case_order = [f"{s}__{g}" for s in SPECTRUM_CASES for g in NONNORMALITY_LEVELS]
    targets = ["q1", "U2", "U3", "U4", "U5"]
    arr = np.full((len(case_order), 5), np.nan, dtype=float)
    for i, case in enumerate(case_order):
        for stage in range(1, 6):
            m = table[(table["case_name"] == case) & (table["stage"] == stage)]
            if len(m):
                arr[i, stage - 1] = float(m[value_col].iloc[0])

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(5), labels=targets)
    ax.set_yticks(range(len(case_order)), labels=case_order)
    ax.set_xlabel("Primary target")
    ax.set_ylabel("Spectrum / non-normality case")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)
    finite = arr[np.isfinite(arr)]
    midpoint = float(np.nanmedian(finite)) if finite.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                color = "white" if v < midpoint else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_diagnostic_comparison(table: pd.DataFrame, output: Path) -> None:
    # Aggregate across spectra so the geometry message is easy to see.
    stages = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    x = np.arange(len(stages), dtype=float)
    width = 0.22
    for gi, geometry in enumerate(NONNORMALITY_LEVELS):
        vals_primary = []
        vals_individual = []
        for stage in stages:
            g = table[(table["geometry_case"] == geometry) & (table["stage"] == stage)]
            vals_primary.append(float(g["median_primary_error_deg_latest_accepted"].median()) if g["median_primary_error_deg_latest_accepted"].notna().any() else np.nan)
            vals_individual.append(float(g["median_individual_qi_error_deg_latest_accepted_diagnostic"].median()) if g["median_individual_qi_error_deg_latest_accepted_diagnostic"].notna().any() else np.nan)
        ax.plot(x + (gi - 1) * 0.04, vals_primary, marker="o", label=f"{geometry}: subspace primary")
        ax.plot(x + (gi - 1) * 0.04, vals_individual, marker="x", linestyle="--", label=f"{geometry}: individual q_i diagnostic")
    ax.axhline(2.5, linestyle=":", linewidth=1.2)
    ax.set_xticks(x, labels=[f"stage {s}" for s in stages])
    ax.set_ylabel("Median error at latest accepted window (degrees)")
    ax.set_title("Experiment 4: primary accumulated-subspace error vs individual eigenvector diagnostic")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Framework Experiment 4: non-normal accumulated-subspace recovery")
    p.add_argument("--output", type=Path, default=Path("results/framework_experiment4_nonnormal_subspace"))
    p.add_argument("--systems", type=int, default=10)
    p.add_argument("--states-per-system", type=int, default=20)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        steps=args.steps,
        window=args.window,
        system_replicates=args.systems,
        initial_states_per_system=args.states_per_system,
        bootstrap_replicates=args.bootstrap,
        seed=args.seed,
    )
    validate_config(cfg)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "time_traces"
    if cfg.save_trace_npz:
        trace_dir.mkdir(parents=True, exist_ok=True)

    n_cases = len(SPECTRUM_CASES) * len(NONNORMALITY_LEVELS)
    trajectories_per_case = cfg.system_replicates * cfg.initial_states_per_system
    total_trajectories = n_cases * trajectories_per_case

    print("=" * 68)
    print("Framework Experiment 4: non-normal accumulated-subspace recovery")
    print("=" * 68)
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"window: {cfg.window}")
    print(f"directions: {cfg.n_directions}")
    print(f"spectra: {list(SPECTRUM_CASES.keys())}")
    print(f"non-normality levels: {NONNORMALITY_LEVELS}")
    print(f"target |a_i|: {TARGET_EXCITATION_MAGNITUDES}")
    print(f"systems per case: {cfg.system_replicates}")
    print(f"initial states per system: {cfg.initial_states_per_system}")
    print(f"trajectories per case: {trajectories_per_case}")
    print(f"total cases: {n_cases}")
    print(f"total trajectories: {total_trajectories}")
    print(f"external correctness tolerance: {cfg.recovery_tolerance_deg} degrees")
    print("Primary target: q1 angle for stage 1; accumulated invariant subspace U_k for k>=2.")
    print("Individual q2...q5 angles are diagnostic only.")
    print("=" * 68)

    trajectory_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    system_rows: Dict[str, Dict[str, object]] = {}

    completed = 0
    for spectrum_name in SPECTRUM_CASES:
        for geometry_name, shear_strength in NONNORMALITY_LEVELS.items():
            case_name = f"{spectrum_name}__{geometry_name}"
            case_trace_accumulator: Dict[str, List[np.ndarray]] = {
                "accepted": [],
                "primary_errors_deg": [],
                "individual_errors_deg": [],
                "accepted_correct": [],
                "relative_distance": [],
            }
            case_window_end: Optional[np.ndarray] = None

            for system_replicate in range(cfg.system_replicates):
                for state_index in range(cfg.initial_states_per_system):
                    tr, ev, traces, sysrow = analyse_trajectory(
                        cfg=cfg,
                        spectrum_name=spectrum_name,
                        geometry_name=geometry_name,
                        shear_strength=shear_strength,
                        system_replicate=system_replicate,
                        state_index=state_index,
                    )
                    trajectory_rows.append(tr)
                    event_rows.extend(ev)
                    system_rows[sysrow["system_uid"]] = sysrow

                    if cfg.save_trace_npz:
                        if case_window_end is None:
                            case_window_end = traces["window_end"]
                        for key in case_trace_accumulator:
                            case_trace_accumulator[key].append(traces[key])

                    completed += 1
                    if completed % 20 == 0 or completed == total_trajectories:
                        print(f"completed {completed}/{total_trajectories} trajectories")

            if cfg.save_trace_npz and case_window_end is not None:
                np.savez_compressed(
                    trace_dir / f"{case_name}.npz",
                    window_end=case_window_end,
                    accepted=np.stack(case_trace_accumulator["accepted"], axis=0),
                    primary_errors_deg=np.stack(case_trace_accumulator["primary_errors_deg"], axis=0),
                    individual_errors_deg=np.stack(case_trace_accumulator["individual_errors_deg"], axis=0),
                    accepted_correct=np.stack(case_trace_accumulator["accepted_correct"], axis=0),
                    relative_distance=np.stack(case_trace_accumulator["relative_distance"], axis=0),
                )

    trajectories = pd.DataFrame(trajectory_rows)
    events = pd.DataFrame(event_rows)
    systems = pd.DataFrame(list(system_rows.values()))

    expected_events = total_trajectories * cfg.n_directions
    if len(trajectories) != total_trajectories:
        raise RuntimeError(f"Expected {total_trajectories} trajectories, got {len(trajectories)}")
    if len(events) != expected_events:
        raise RuntimeError(f"Expected {expected_events} event rows, got {len(events)}")

    trajectories.to_csv(output_dir / "all_trajectories.csv", index=False)
    events.to_csv(output_dir / "trajectory_stage_events.csv", index=False)
    systems.to_csv(output_dir / "systems.csv", index=False)

    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(cfg),
                "spectrum_cases": {k: list(v) for k, v in SPECTRUM_CASES.items()},
                "nonnormality_levels": NONNORMALITY_LEVELS,
                "target_excitation_magnitudes": list(TARGET_EXCITATION_MAGNITUDES),
                "n_cases": n_cases,
                "trajectories_per_case": trajectories_per_case,
                "total_trajectories": total_trajectories,
                "primary_evaluation": {
                    "stage_1": "sign-invariant angular error to q1",
                    "stage_k_ge_2": "largest principal angle between span(qhat1,...,qhatk) and span(q1,...,qk)",
                    "individual_higher_eigenvector_angles": "diagnostic only",
                },
            },
            f,
            indent=2,
        )

    design_rows: List[Dict[str, object]] = []
    for spectrum_name, spectrum in SPECTRUM_CASES.items():
        for geometry_name, shear_strength in NONNORMALITY_LEVELS.items():
            design_rows.append({
                "case_name": f"{spectrum_name}__{geometry_name}",
                "spectrum_case": spectrum_name,
                "leading_eigenvalues": list(spectrum),
                "geometry_case": geometry_name,
                "shear_strength": shear_strength,
                "target_excitation_magnitudes": list(TARGET_EXCITATION_MAGNITUDES),
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
    table1 = pd.DataFrame(design_rows)
    table2 = summarize_primary(events, cfg)
    table3 = summarize_recovery_intervals(events)
    table4 = summarize_individual_diagnostics(events)
    table5 = summarize_geometry(systems)

    save_table(table1, output_dir, "table1_experiment_design")
    save_table(table2, output_dir, "table2_primary_stagewise_performance")
    save_table(table3, output_dir, "table3_primary_recovery_intervals")
    save_table(table4, output_dir, "table4_individual_eigenvector_diagnostics")
    save_table(table5, output_dir, "table5_nonnormal_geometry_summary")

    plot_heatmap(
        table2,
        "ever_recovery_rate",
        output_dir / "figure1_ever_recovery_primary_target_heatmap.png",
        "Experiment 4: probability of ever recovering q1 / accumulated subspace U_k",
        "Ever-recovery rate",
        ".2f",
    )
    plot_heatmap(
        table2,
        "median_latest_primary_error_deg_accepted",
        output_dir / "figure2_latest_primary_error_heatmap.png",
        "Experiment 4: latest accepted primary-target error",
        "Median error (degrees)",
        ".2f",
    )
    plot_heatmap(
        table3,
        "median_first_recovery_window_end",
        output_dir / "figure3_first_recovery_iteration_heatmap.png",
        "Experiment 4: median first-recovery iteration",
        "Window-end iteration",
        ".0f",
    )
    plot_heatmap(
        table3,
        "median_log10_relative_distance_at_first_recovery",
        output_dir / "figure4_distance_to_limit_first_recovery_heatmap.png",
        "Experiment 4: closeness to limit at first recovery",
        "median log10(||x_t-L|| / ||x_0-L||)",
        ".2f",
    )
    plot_diagnostic_comparison(
        table4,
        output_dir / "figure5_subspace_vs_individual_eigenvector_diagnostic.png",
    )

    manifest = {
        "tables": [
            "table1_experiment_design.csv",
            "table2_primary_stagewise_performance.csv",
            "table3_primary_recovery_intervals.csv",
            "table4_individual_eigenvector_diagnostics.csv",
            "table5_nonnormal_geometry_summary.csv",
        ],
        "figures": [
            "figure1_ever_recovery_primary_target_heatmap.png",
            "figure2_latest_primary_error_heatmap.png",
            "figure3_first_recovery_iteration_heatmap.png",
            "figure4_distance_to_limit_first_recovery_heatmap.png",
            "figure5_subspace_vs_individual_eigenvector_diagnostic.png",
        ],
        "notes": [
            "Stage 1 primary target is q1 and uses angular error.",
            "Stages k>=2 use the accumulated subspace U_k=span(q1,...,qk) and largest principal angle.",
            "Individual q2,...,q5 angles are diagnostic only and never define correctness.",
            "All acceptance decisions are observation-only; known eigenvectors are used only for synthetic validation.",
        ],
    }
    (output_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== Geometry summary ===")
    print(table5.to_string(index=False))
    print("\n=== Primary stage-wise results ===")
    display_cols = [
        "spectrum_case", "geometry_case", "stage", "primary_target",
        "acceptance_rate", "ever_recovery_rate", "overall_recovery_latest_rate",
        "reliability_latest_given_accepted_rate", "false_acceptance_latest_rate",
        "median_latest_primary_error_deg_accepted",
    ]
    print(table2[display_cols].to_string(index=False))

    print(f"\nResults written to: {output_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
