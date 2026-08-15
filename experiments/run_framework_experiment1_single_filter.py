from __future__ import annotations

import argparse
import importlib.util
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
EPS = 1e-14


def _load_estimator_module():
    """Load the observation-only same-window estimator without renaming it.

    In the repository the expected filename is
    ``run_observation_only_same_window_deflation_normal.py``.  The second
    candidate exists only so this standalone artifact can be smoke-tested
    against the uploaded conversation copy.
    """
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

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise ImportError(
        "Could not locate run_observation_only_same_window_deflation_normal.py.\n"
        f"Searched:\n{searched}"
    )


ESTIMATOR = _load_estimator_module()
EstimatorConfig = ESTIMATOR.Config
estimate_from_observations_only = ESTIMATOR.estimate_from_observations_only
simulate_trajectory = ESTIMATOR.simulate_trajectory
angle_deg = ESTIMATOR.angle_deg
max_principal_angle_deg = ESTIMATOR.max_principal_angle_deg
common_window_is_stable = ESTIMATOR.common_window_is_stable


@dataclass(frozen=True)
class ExperimentConfig:
    dim: int = 20
    steps: int = 500
    window: int = 20
    system_replicates: int = 10
    initial_states_per_system: int = 20
    seed: int = 42

    # Estimator-only acceptance criteria.  These do not use ground truth.
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12
    min_residual_energy_fraction: float = 1e-10
    numeric_relative_residual_floor: float = 1e-15
    min_stage_pc1_energy_fraction: float = 0.80

    # External validation convention only.
    recovery_tolerance_deg: float = 1.0
    subspace_tolerance_deg: float = 1.0

    # Synthetic-system construction.
    lambda1: float = 0.96
    strong_lambda2: float = 0.60
    weak_lambda2: float = 0.94
    rotation_angle_deg: float = 25.0
    tail_max: float = 0.50
    tail_min: float = 0.15
    other_mode_scale: float = 0.35

    # Optional two-filter diagnostic for the two non-unique controls.
    include_two_filter_plane_diagnostic: bool = False

    # Hierarchical uncertainty.
    bootstrap_replicates: int = 2000


CASES = ("strong_gap", "weak_gap", "equal_magnitude", "rotation")
UNIQUE_CASES = ("strong_gap", "weak_gap")
NONUNIQUE_CASES = ("equal_magnitude", "rotation")


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= EPS:
        raise ValueError("Cannot normalize a non-finite or near-zero vector.")
    return v / n


def rotation_block(radius: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return radius * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=float,
    )


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.dim < 3:
        raise ValueError("dim must be at least 3.")
    if cfg.steps < 1:
        raise ValueError("steps must be positive.")
    if cfg.window < 2 or cfg.window > cfg.steps + 1:
        raise ValueError("window must satisfy 2 <= window <= steps + 1.")
    if cfg.system_replicates < 1 or cfg.initial_states_per_system < 1:
        raise ValueError("system and initial-state replicate counts must be positive.")
    if not (0.0 < cfg.lambda1 < 1.0):
        raise ValueError("lambda1 must lie in (0,1).")
    if not (0.0 < cfg.strong_lambda2 < cfg.lambda1):
        raise ValueError("strong_lambda2 must lie in (0,lambda1).")
    if not (0.0 < cfg.weak_lambda2 < cfg.lambda1):
        raise ValueError("weak_lambda2 must lie in (0,lambda1).")
    if cfg.strong_lambda2 >= cfg.weak_lambda2:
        raise ValueError("strong_lambda2 should be smaller than weak_lambda2.")
    if not (0.0 < cfg.tail_min < cfg.tail_max):
        raise ValueError("Require 0 < tail_min < tail_max.")
    if cfg.tail_max >= min(cfg.strong_lambda2, cfg.weak_lambda2):
        raise ValueError("tail_max must be below both controlled second eigenvalues.")
    if cfg.recovery_tolerance_deg <= 0.0 or cfg.subspace_tolerance_deg <= 0.0:
        raise ValueError("validation tolerances must be positive.")
    if cfg.bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates should be at least 100.")


def make_estimator_config(cfg: ExperimentConfig, n_directions: int) -> EstimatorConfig:
    return EstimatorConfig(
        dim=cfg.dim,
        steps=cfg.steps,
        trials=1,
        window=cfg.window,
        n_directions=n_directions,
        seed=cfg.seed,
        stability_threshold_deg=cfg.stability_threshold_deg,
        stability_patience=cfg.stability_patience,
        relative_window_norm_floor=cfg.relative_window_norm_floor,
        min_residual_energy_fraction=cfg.min_residual_energy_fraction,
        numeric_relative_residual_floor=cfg.numeric_relative_residual_floor,
        min_stage_pc1_energy_fraction=cfg.min_stage_pc1_energy_fraction,
        recovery_tolerance_deg=cfg.recovery_tolerance_deg,
    )


def random_tail_eigenvalues(cfg: ExperimentConfig, rng: np.random.Generator) -> np.ndarray:
    magnitudes = rng.uniform(cfg.tail_min, cfg.tail_max, size=cfg.dim - 2)
    magnitudes = np.sort(magnitudes)[::-1]
    signs = rng.choice(np.array([-1.0, 1.0]), size=cfg.dim - 2)
    return signs * magnitudes


def build_system(
    cfg: ExperimentConfig,
    case: str,
    system_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Build one independently rotated real normal linear system.

    Returns ``A``, the orthogonal coordinate matrix ``Q``, and metadata.
    For the unique cases, ``Q[:,0]`` is the reference q1.  For the two
    non-unique controls, ``Q[:,:2]`` is the reference dominant plane.
    """
    if case not in CASES:
        raise ValueError(f"Unknown case: {case}")

    rng = np.random.default_rng(system_seed)
    gaussian = rng.normal(size=(cfg.dim, cfg.dim))
    Q, _ = np.linalg.qr(gaussian)
    B = np.zeros((cfg.dim, cfg.dim), dtype=float)
    tail = random_tail_eigenvalues(cfg, rng)
    B[2:, 2:] = np.diag(tail)

    if case == "strong_gap":
        B[0, 0] = cfg.lambda1
        B[1, 1] = cfg.strong_lambda2
        target_type = "unique_q1"
        lambda2_abs = abs(cfg.strong_lambda2)
    elif case == "weak_gap":
        B[0, 0] = cfg.lambda1
        B[1, 1] = cfg.weak_lambda2
        target_type = "unique_q1"
        lambda2_abs = abs(cfg.weak_lambda2)
    elif case == "equal_magnitude":
        # Opposite signs make the lack of a unique one-dimensional asymptotic
        # direction especially visible while retaining a known 2-D invariant plane.
        B[0, 0] = cfg.lambda1
        B[1, 1] = -cfg.lambda1
        target_type = "no_unique_1d_target"
        lambda2_abs = cfg.lambda1
    else:  # rotation
        B[:2, :2] = rotation_block(cfg.lambda1, cfg.rotation_angle_deg)
        target_type = "no_unique_1d_target"
        lambda2_abs = cfg.lambda1

    A = Q @ B @ Q.T
    normality_error = float(np.linalg.norm(A.T @ A - A @ A.T, ord="fro"))
    if normality_error > 1e-10:
        raise RuntimeError(f"Constructed system is not numerically normal: {normality_error:.3e}")

    metadata: dict[str, float | str] = {
        "case": case,
        "target_type": target_type,
        "lambda1_abs": cfg.lambda1,
        "lambda2_abs": lambda2_abs,
        "spectral_ratio_abs_lambda2_over_lambda1": lambda2_abs / cfg.lambda1,
        "normality_error_fro": normality_error,
        "tail_max_abs_eigenvalue": float(np.max(np.abs(tail))) if len(tail) else np.nan,
    }
    return A, Q, metadata


def construct_initial_state(
    cfg: ExperimentConfig,
    case: str,
    Q: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct an initial state with the relevant leading structure excited."""
    coefficients = rng.normal(0.0, cfg.other_mode_scale, size=cfg.dim)

    if case in UNIQUE_CASES:
        # q1 is intentionally present so Experiment 1 isolates gap effects rather
        # than mixing them with the unexcited-target negative control of Exp. 2.
        coefficients[0] = float(rng.choice(np.array([-1.0, 1.0])))
        coefficients[1] = rng.uniform(0.50, 1.50) * float(
            rng.choice(np.array([-1.0, 1.0]))
        )
    else:
        # Both coordinates of the dominant two-dimensional object are forced to
        # be present.  This avoids accidentally reducing a non-unique control to
        # a one-dimensional trajectory.
        leading = rng.normal(size=2)
        leading = normalize(leading)
        coefficients[:2] = leading

    return Q @ coefficients, coefficients


def _selected_diagnostic_values(info: dict, stage: int = 1) -> dict[str, float]:
    names = (
        "direction_change_deg",
        "stage_pc1_energy_fraction",
        "singular_value_ratio_1_to_2",
        "residual_energy_before_fraction",
        "residual_energy_after_fraction",
        "extracted_energy_fraction_original",
    )
    return {
        f"stage_{stage}_{name}": float(info.get(f"stage_{stage}_{name}", np.nan))
        for name in names
    }


def first_acceptable_window(
    diagnostics: pd.DataFrame,
    cfg: EstimatorConfig,
) -> pd.Series | None:
    """Return the earliest window satisfying the estimator's acceptance rule."""
    for i in range(len(diagnostics)):
        if i < cfg.stability_patience - 1:
            continue
        recent = diagnostics.iloc[i - cfg.stability_patience + 1 : i + 1]
        if common_window_is_stable(recent, cfg):
            return diagnostics.iloc[i]
    return None


def analyse_trial(
    *,
    cfg: ExperimentConfig,
    one_filter_cfg: EstimatorConfig,
    two_filter_cfg: EstimatorConfig,
    case: str,
    A: np.ndarray,
    Q: np.ndarray,
    system_metadata: dict[str, float | str],
    system_replicate: int,
    system_seed: int,
    initial_state_index: int,
    trial_seed: int,
) -> tuple[dict, dict | None]:
    rng = np.random.default_rng(trial_seed)
    x0, modal_coefficients = construct_initial_state(cfg, case, Q, rng)
    X = simulate_trajectory(A=A, x0=x0, steps=cfg.steps)
    L = np.zeros(cfg.dim, dtype=float)

    directions, info, diagnostics = estimate_from_observations_only(
        X=X, L=L, cfg=one_filter_cfg
    )
    accepted = bool(info.get("success", False) and len(directions) == 1)
    first_selected = first_acceptable_window(diagnostics, one_filter_cfg)

    row: dict = {
        **system_metadata,
        "system_replicate": system_replicate,
        "system_seed": system_seed,
        "initial_state_index": initial_state_index,
        "trial_seed": trial_seed,
        "accepted_stage_1": accepted,
        "first_acceptable_window_start": (
            float(first_selected["window_start"]) if first_selected is not None else np.nan
        ),
        "first_acceptable_window_end": (
            float(first_selected["window_end"]) if first_selected is not None else np.nan
        ),
        "selected_window_start": info.get("window_start", np.nan),
        "selected_window_end": info.get("window_end", np.nan),
        "relative_window_norm": info.get("relative_window_norm", np.nan),
        "n_common_stable_candidates": info.get("n_common_stable_candidates", 0),
        "x0_norm": float(np.linalg.norm(x0)),
        "abs_initial_q1_coefficient": abs(float(modal_coefficients[0])),
        "abs_initial_q2_coefficient": abs(float(modal_coefficients[1])),
    }
    row.update(_selected_diagnostic_values(info, stage=1))

    q1_error = np.nan
    candidate_correct: bool | None = None
    accepted_and_correct = False
    false_acceptance = False
    false_unique_direction_acceptance = False

    if case in UNIQUE_CASES:
        if accepted:
            q1_error = float(angle_deg(directions[0], Q[:, 0]))
            candidate_correct = bool(q1_error <= cfg.recovery_tolerance_deg)
            accepted_and_correct = bool(candidate_correct)
            false_acceptance = not bool(candidate_correct)
    else:
        # There is intentionally no unique 1-D external correctness target.
        # An accepted single direction is therefore a false *uniqueness* claim.
        false_unique_direction_acceptance = accepted

    row.update(
        {
            "qhat1_vs_reference_q1_deg": q1_error,
            "candidate_correct_stage_1": candidate_correct,
            "accepted_and_correct_stage_1": accepted_and_correct,
            "false_acceptance_stage_1": false_acceptance,
            "false_unique_direction_acceptance": false_unique_direction_acceptance,
            "recovery_tolerance_deg": cfg.recovery_tolerance_deg,
        }
    )

    plane_row: dict | None = None
    if cfg.include_two_filter_plane_diagnostic and case in NONUNIQUE_CASES:
        plane_directions, plane_info, _ = estimate_from_observations_only(
            X=X, L=L, cfg=two_filter_cfg
        )
        plane_accepted = bool(
            plane_info.get("success", False) and len(plane_directions) == 2
        )
        plane_error = np.nan
        plane_correct: bool | None = None
        if plane_accepted:
            estimated_plane = np.column_stack(plane_directions)
            plane_error = float(max_principal_angle_deg(estimated_plane, Q[:, :2]))
            plane_correct = bool(plane_error <= cfg.subspace_tolerance_deg)

        plane_row = {
            "case": case,
            "system_replicate": system_replicate,
            "system_seed": system_seed,
            "initial_state_index": initial_state_index,
            "trial_seed": trial_seed,
            "accepted_two_filter_plane": plane_accepted,
            "plane_correct": plane_correct,
            "accepted_and_correct_plane": bool(plane_accepted and plane_correct is True),
            "false_acceptance_plane": bool(plane_accepted and plane_correct is False),
            "leading_2_subspace_error_deg": plane_error,
            "subspace_tolerance_deg": cfg.subspace_tolerance_deg,
            "selected_window_start": plane_info.get("window_start", np.nan),
            "selected_window_end": plane_info.get("window_end", np.nan),
            "relative_window_norm": plane_info.get("relative_window_norm", np.nan),
        }

    return row, plane_row


def _rates_from_sample(sample: pd.DataFrame, *, accepted_col: str, correct_col: str) -> dict[str, float]:
    accepted = sample[accepted_col].astype(bool).to_numpy()
    correct_series = sample[correct_col].astype("boolean")
    correct_true = correct_series.fillna(False).astype(bool).to_numpy()
    correct_false = ((correct_series == False).fillna(False)).astype(bool).to_numpy()  # noqa: E712

    success = accepted & correct_true
    false_accept = accepted & correct_false
    n_accepted = int(accepted.sum())
    return {
        "acceptance_rate": float(np.mean(accepted)) if len(accepted) else np.nan,
        "overall_successful_recovery_rate": float(np.mean(success)) if len(success) else np.nan,
        "reliability_given_accepted": float(success.sum() / n_accepted) if n_accepted else np.nan,
        "false_acceptance_rate": float(np.mean(false_accept)) if len(false_accept) else np.nan,
    }


def hierarchical_bootstrap_acceptance_correctness(
    frame: pd.DataFrame,
    *,
    system_col: str,
    accepted_col: str,
    correct_col: str,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Resample systems first, then initial states within each selected system."""
    observed = _rates_from_sample(frame, accepted_col=accepted_col, correct_col=correct_col)
    systems = frame[system_col].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in observed}

    groups = {sid: frame.loc[frame[system_col] == sid].copy() for sid in systems}
    for _ in range(n_boot):
        sampled_systems = rng.choice(systems, size=len(systems), replace=True)
        parts: list[pd.DataFrame] = []
        for sid in sampled_systems:
            group = groups[sid]
            positions = rng.integers(0, len(group), size=len(group))
            parts.append(group.iloc[positions])
        sample = pd.concat(parts, ignore_index=True)
        values = _rates_from_sample(sample, accepted_col=accepted_col, correct_col=correct_col)
        for name, value in values.items():
            if np.isfinite(value):
                boot[name].append(value)

    result: dict[str, dict[str, float]] = {}
    for name, estimate in observed.items():
        values = np.asarray(boot[name], dtype=float)
        result[name] = {
            "estimate": estimate,
            "ci95_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
            "ci95_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
        }
    return result


def hierarchical_bootstrap_binary_rate(
    frame: pd.DataFrame,
    *,
    system_col: str,
    value_col: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    systems = frame[system_col].drop_duplicates().to_numpy()
    observed = float(frame[value_col].astype(bool).mean())
    rng = np.random.default_rng(seed)
    groups = {sid: frame.loc[frame[system_col] == sid].copy() for sid in systems}
    values = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_systems = rng.choice(systems, size=len(systems), replace=True)
        sample_values: list[np.ndarray] = []
        for sid in sampled_systems:
            group = groups[sid]
            positions = rng.integers(0, len(group), size=len(group))
            sample_values.append(group.iloc[positions][value_col].astype(bool).to_numpy())
        values[b] = float(np.mean(np.concatenate(sample_values)))
    return {
        "estimate": observed,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def summarize_experiment1(all_trials: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for case_index, case in enumerate(CASES):
        group = all_trials.loc[all_trials["case"] == case].copy()
        accepted_group = group.loc[group["accepted_stage_1"]].copy()
        first_accept = accepted_group["first_acceptable_window_end"].dropna()
        common = {
            "case": case,
            "target_type": group["target_type"].iloc[0],
            "n_systems": int(group["system_replicate"].nunique()),
            "n_initial_states_per_system": int(cfg.initial_states_per_system),
            "n_trajectories": int(len(group)),
            "median_first_acceptable_window_end": (
                float(first_accept.median()) if len(first_accept) else np.nan
            ),
            "q25_first_acceptable_window_end": (
                float(first_accept.quantile(0.25)) if len(first_accept) else np.nan
            ),
            "q75_first_acceptable_window_end": (
                float(first_accept.quantile(0.75)) if len(first_accept) else np.nan
            ),
            "median_selected_window_end_accepted": float(
                accepted_group["selected_window_end"].median()
            )
            if len(accepted_group)
            else np.nan,
        }

        if case in UNIQUE_CASES:
            rates = hierarchical_bootstrap_acceptance_correctness(
                group,
                system_col="system_replicate",
                accepted_col="accepted_stage_1",
                correct_col="candidate_correct_stage_1",
                n_boot=cfg.bootstrap_replicates,
                seed=cfg.seed + 10_000 * case_index,
            )
            accepted_errors = group.loc[
                group["accepted_stage_1"], "qhat1_vs_reference_q1_deg"
            ].dropna()
            row = dict(common)
            for name, result in rates.items():
                row[name] = result["estimate"]
                row[f"{name}_ci95_low"] = result["ci95_low"]
                row[f"{name}_ci95_high"] = result["ci95_high"]
            row.update(
                {
                    "false_unique_direction_acceptance_rate": np.nan,
                    "median_q1_error_deg_accepted": float(accepted_errors.median())
                    if len(accepted_errors)
                    else np.nan,
                    "q25_q1_error_deg_accepted": float(accepted_errors.quantile(0.25))
                    if len(accepted_errors)
                    else np.nan,
                    "q75_q1_error_deg_accepted": float(accepted_errors.quantile(0.75))
                    if len(accepted_errors)
                    else np.nan,
                }
            )
        else:
            false_unique = hierarchical_bootstrap_binary_rate(
                group,
                system_col="system_replicate",
                value_col="false_unique_direction_acceptance",
                n_boot=cfg.bootstrap_replicates,
                seed=cfg.seed + 10_000 * case_index,
            )
            row = dict(common)
            row.update(
                {
                    "acceptance_rate": float(group["accepted_stage_1"].mean()),
                    "acceptance_rate_ci95_low": false_unique["ci95_low"],
                    "acceptance_rate_ci95_high": false_unique["ci95_high"],
                    "overall_successful_recovery_rate": np.nan,
                    "reliability_given_accepted": np.nan,
                    "false_acceptance_rate": np.nan,
                    "false_unique_direction_acceptance_rate": false_unique["estimate"],
                    "false_unique_direction_acceptance_rate_ci95_low": false_unique["ci95_low"],
                    "false_unique_direction_acceptance_rate_ci95_high": false_unique["ci95_high"],
                    "median_q1_error_deg_accepted": np.nan,
                    "q25_q1_error_deg_accepted": np.nan,
                    "q75_q1_error_deg_accepted": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_plane_diagnostic(plane_trials: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if plane_trials.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for case_index, case in enumerate(NONUNIQUE_CASES):
        group = plane_trials.loc[plane_trials["case"] == case].copy()
        rates = hierarchical_bootstrap_acceptance_correctness(
            group,
            system_col="system_replicate",
            accepted_col="accepted_two_filter_plane",
            correct_col="plane_correct",
            n_boot=cfg.bootstrap_replicates,
            seed=cfg.seed + 70_000 + 10_000 * case_index,
        )
        errors = group.loc[
            group["accepted_two_filter_plane"], "leading_2_subspace_error_deg"
        ].dropna()
        row: dict = {
            "case": case,
            "n_systems": int(group["system_replicate"].nunique()),
            "n_initial_states": int(len(group)),
            "median_subspace_error_deg_accepted": float(errors.median()) if len(errors) else np.nan,
            "q25_subspace_error_deg_accepted": float(errors.quantile(0.25)) if len(errors) else np.nan,
            "q75_subspace_error_deg_accepted": float(errors.quantile(0.75)) if len(errors) else np.nan,
        }
        for name, result in rates.items():
            row[name] = result["estimate"]
            row[f"{name}_ci95_low"] = result["ci95_low"]
            row[f"{name}_ci95_high"] = result["ci95_high"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_single_filter_rates(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(summary))
    values = []
    labels = []
    for _, row in summary.iterrows():
        if row["case"] in UNIQUE_CASES:
            values.append(100.0 * float(row["overall_successful_recovery_rate"]))
            labels.append(f"{row['case']}\nsuccessful recovery")
        else:
            values.append(100.0 * float(row["false_unique_direction_acceptance_rate"]))
            labels.append(f"{row['case']}\nfalse uniqueness acceptance")
    bars = ax.bar(x, values)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Percentage of trajectories")
    ax.set_title("Experiment 1: correct direction recovery and non-uniqueness controls")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_q1_errors(all_trials: pd.DataFrame, output_path: Path) -> None:
    data: list[np.ndarray] = []
    labels: list[str] = []
    for case in UNIQUE_CASES:
        values = all_trials.loc[
            (all_trials["case"] == case) & all_trials["accepted_stage_1"],
            "qhat1_vs_reference_q1_deg",
        ].dropna().to_numpy(dtype=float)
        if len(values):
            labels.append(case)
            data.append(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    if data:
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.axhline(1.0, linestyle="--", label="1 degree reporting threshold")
        ax.legend()
    ax.set_ylabel("Sign-invariant q1 angle error (degrees)")
    ax.set_title("Experiment 1: q1 accuracy among accepted estimates")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def build_report_tables(
    all_trials: pd.DataFrame,
    systems: pd.DataFrame,
    summary: pd.DataFrame,
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact report tables for Experiment 1."""
    design_rows: list[dict] = []
    case_labels = {
        "strong_gap": "Strong gap",
        "weak_gap": "Weak gap",
        "equal_magnitude": "Equal magnitude",
        "rotation": "Contracting rotation",
    }
    target_labels = {
        "strong_gap": "Unique q1",
        "weak_gap": "Unique q1",
        "equal_magnitude": "No unique 1-D target",
        "rotation": "No unique 1-D target",
    }

    for case in CASES:
        sg = systems.loc[systems["case"] == case]
        if case == "rotation":
            mode = f"{cfg.lambda1:.2f} R({cfg.rotation_angle_deg:.0f} deg)"
            lambda2_display = mode
        elif case == "equal_magnitude":
            lambda2_display = f"{-abs(cfg.lambda1):.2f}"
        else:
            lambda2_display = f"{float(sg['lambda2_abs'].iloc[0]):.2f}"

        ratio = float(sg["spectral_ratio_abs_lambda2_over_lambda1"].iloc[0])
        design_rows.append(
            {
                "case": case_labels[case],
                "target": target_labels[case],
                "lambda1_abs": float(sg["lambda1_abs"].iloc[0]),
                "second_leading_structure": lambda2_display,
                "abs_lambda2_over_lambda1": ratio,
                "n_systems": int(sg["system_replicate"].nunique()),
                "initial_states_per_system": int(cfg.initial_states_per_system),
                "total_trajectories": int(
                    len(all_trials.loc[all_trials["case"] == case])
                ),
                "window_m": int(cfg.window),
                "stability_threshold_deg": float(cfg.stability_threshold_deg),
                "stability_patience": int(cfg.stability_patience),
                "pc1_energy_threshold": float(cfg.min_stage_pc1_energy_fraction),
                "external_tolerance_deg": float(cfg.recovery_tolerance_deg),
            }
        )

    table1 = pd.DataFrame(design_rows)

    result_rows: list[dict] = []
    for _, row in summary.iterrows():
        case = str(row["case"])
        group = all_trials.loc[all_trials["case"] == case]
        n_total = len(group)
        n_accepted = int(group["accepted_stage_1"].astype(bool).sum())

        if case in UNIQUE_CASES:
            n_correct = int(group["accepted_and_correct_stage_1"].astype(bool).sum())
            n_false = int(group["false_acceptance_stage_1"].astype(bool).sum())
            error_text = (
                f"{row['median_q1_error_deg_accepted']:.6f} "
                f"[{row['q25_q1_error_deg_accepted']:.6f}, "
                f"{row['q75_q1_error_deg_accepted']:.6f}]"
            )
            success_text = f"{n_correct}/{n_total} ({100*row['overall_successful_recovery_rate']:.1f}%)"
            reliability_text = f"{100*row['reliability_given_accepted']:.1f}%"
            false_text = f"{n_false}/{n_total} ({100*row['false_acceptance_rate']:.1f}%)"
        else:
            error_text = "N/A"
            success_text = "N/A"
            reliability_text = "N/A"
            n_false_unique = int(group["false_unique_direction_acceptance"].astype(bool).sum())
            false_text = (
                f"{n_false_unique}/{n_total} "
                f"({100*row['false_unique_direction_acceptance_rate']:.1f}%)"
            )

        first_text = "N/A"
        if "median_first_acceptable_window_end" in row.index and pd.notna(
            row["median_first_acceptable_window_end"]
        ):
            first_text = (
                f"{row['median_first_acceptable_window_end']:.1f} "
                f"[{row['q25_first_acceptable_window_end']:.1f}, "
                f"{row['q75_first_acceptable_window_end']:.1f}]"
            )

        result_rows.append(
            {
                "case": case_labels[case],
                "accepted": f"{n_accepted}/{n_total} ({100*n_accepted/n_total:.1f}%)",
                "overall_successful_recovery": success_text,
                "reliability_given_accepted": reliability_text,
                "false_acceptance_or_false_unique_acceptance": false_text,
                "median_q1_error_deg_IQR": error_text,
                "first_acceptable_window_end_median_IQR": first_text,
            }
        )

    table2 = pd.DataFrame(result_rows)
    return table1, table2


def save_markdown_table(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Save a lightweight Markdown table without requiring optional packages."""
    def cell(v):
        if pd.isna(v):
            return "N/A"
        return str(v).replace("|", "\\|")

    headers = [cell(c) for c in frame.columns]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_first_acceptable_window(all_trials: pd.DataFrame, output_path: Path) -> None:
    """Compare how quickly strong- and weak-gap cases first satisfy acceptance."""
    if "first_acceptable_window_end" not in all_trials.columns:
        return

    data: list[np.ndarray] = []
    labels: list[str] = []
    for case, label in (("strong_gap", "Strong gap"), ("weak_gap", "Weak gap")):
        values = all_trials.loc[
            (all_trials["case"] == case) & all_trials["accepted_stage_1"],
            "first_acceptable_window_end",
        ].dropna().to_numpy(dtype=float)
        if len(values):
            data.append(values)
            labels.append(label)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("First acceptable window end")
    ax.set_title("Experiment 1: time to first internally acceptable q1 estimate")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def plot_plane_diagnostic(plane_summary: pd.DataFrame, output_path: Path) -> None:
    if plane_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    values = 100.0 * plane_summary["overall_successful_recovery_rate"].to_numpy(dtype=float)
    bars = ax.bar(plane_summary["case"], values)
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Accepted and correct 2-D subspace (%)")
    ax.set_title("Experiment 1 secondary diagnostic: two-filter invariant-plane recovery")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Framework Experiment 1: observation-only single-filter validation "
            "with unique-direction cases and non-uniqueness controls."
        )
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--system-replicates", type=int, default=10)
    parser.add_argument("--initial-states-per-system", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-threshold-deg", type=float, default=0.2)
    parser.add_argument("--stability-patience", type=int, default=5)
    parser.add_argument("--relative-window-norm-floor", type=float, default=1e-12)
    parser.add_argument("--min-residual-energy-fraction", type=float, default=1e-10)
    parser.add_argument("--numeric-relative-residual-floor", type=float, default=1e-15)
    parser.add_argument("--min-stage-pc1-energy-fraction", type=float, default=0.80)
    parser.add_argument("--recovery-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--subspace-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--lambda1", type=float, default=0.96)
    parser.add_argument("--strong-lambda2", type=float, default=0.60)
    parser.add_argument("--weak-lambda2", type=float, default=0.94)
    parser.add_argument("--rotation-angle-deg", type=float, default=25.0)
    parser.add_argument("--tail-max", type=float, default=0.50)
    parser.add_argument("--tail-min", type=float, default=0.15)
    parser.add_argument("--other-mode-scale", type=float, default=0.35)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--include-two-filter-plane-diagnostic",
        action="store_true",
        help=(
            "Run an exploratory two-filter plane diagnostic for the non-unique controls. "
            "This is not part of the primary Experiment-1 analysis because the current "
            "stagewise acceptance rule is designed for one-dimensional dominance."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/framework_experiment1_single_filter"),
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        system_replicates=args.system_replicates,
        initial_states_per_system=args.initial_states_per_system,
        seed=args.seed,
        stability_threshold_deg=args.stability_threshold_deg,
        stability_patience=args.stability_patience,
        relative_window_norm_floor=args.relative_window_norm_floor,
        min_residual_energy_fraction=args.min_residual_energy_fraction,
        numeric_relative_residual_floor=args.numeric_relative_residual_floor,
        min_stage_pc1_energy_fraction=args.min_stage_pc1_energy_fraction,
        recovery_tolerance_deg=args.recovery_tolerance_deg,
        subspace_tolerance_deg=args.subspace_tolerance_deg,
        lambda1=args.lambda1,
        strong_lambda2=args.strong_lambda2,
        weak_lambda2=args.weak_lambda2,
        rotation_angle_deg=args.rotation_angle_deg,
        tail_max=args.tail_max,
        tail_min=args.tail_min,
        other_mode_scale=args.other_mode_scale,
        include_two_filter_plane_diagnostic=args.include_two_filter_plane_diagnostic,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    validate_config(cfg)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    with (output / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    one_filter_cfg = make_estimator_config(cfg, n_directions=1)
    two_filter_cfg = make_estimator_config(cfg, n_directions=2)

    trial_rows: list[dict] = []
    plane_rows: list[dict] = []
    system_rows: list[dict] = []

    total = len(CASES) * cfg.system_replicates * cfg.initial_states_per_system
    completed = 0
    for case_index, case in enumerate(CASES):
        for system_replicate in range(cfg.system_replicates):
            system_seed = cfg.seed + 10_000_000 * case_index + 100_000 * system_replicate
            A, Q, metadata = build_system(cfg, case, system_seed)
            system_rows.append(
                {
                    **metadata,
                    "system_replicate": system_replicate,
                    "system_seed": system_seed,
                }
            )
            for initial_state_index in range(cfg.initial_states_per_system):
                trial_seed = system_seed + initial_state_index + 1
                row, plane_row = analyse_trial(
                    cfg=cfg,
                    one_filter_cfg=one_filter_cfg,
                    two_filter_cfg=two_filter_cfg,
                    case=case,
                    A=A,
                    Q=Q,
                    system_metadata=metadata,
                    system_replicate=system_replicate,
                    system_seed=system_seed,
                    initial_state_index=initial_state_index,
                    trial_seed=trial_seed,
                )
                trial_rows.append(row)
                if plane_row is not None:
                    plane_rows.append(plane_row)
                completed += 1
                if completed % max(total // 20, 1) == 0 or completed == total:
                    print(f"completed {completed}/{total} trajectories")

    all_trials = pd.DataFrame(trial_rows)
    systems = pd.DataFrame(system_rows)
    plane_trials = pd.DataFrame(plane_rows)
    summary = summarize_experiment1(all_trials, cfg)
    plane_summary = summarize_plane_diagnostic(plane_trials, cfg)

    all_trials.to_csv(output / "all_trials.csv", index=False)
    systems.to_csv(output / "systems.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    table1, table2 = build_report_tables(all_trials, systems, summary, cfg)
    table1.to_csv(output / "table1_experiment_design.csv", index=False)
    table2.to_csv(output / "table2_main_results.csv", index=False)
    save_markdown_table(
        table1,
        output / "table1_experiment_design.md",
        "Experiment 1 - Experimental design",
    )
    save_markdown_table(
        table2,
        output / "table2_main_results.md",
        "Experiment 1 - Main results",
    )

    if not plane_trials.empty:
        plane_trials.to_csv(output / "two_filter_plane_trials.csv", index=False)
        plane_summary.to_csv(output / "two_filter_plane_summary.csv", index=False)

    plot_single_filter_rates(summary, output / "01_single_filter_outcomes.png")
    plot_q1_errors(all_trials, output / "02_q1_error_distribution.png")
    plot_first_acceptable_window(all_trials, output / "03_first_acceptable_window.png")
    if not plane_summary.empty:
        plot_plane_diagnostic(plane_summary, output / "04_two_filter_plane_recovery.png")

    print("\n=== Experiment 1 summary ===")
    print(summary.to_string(index=False))
    if not plane_summary.empty:
        print("\n=== Optional two-filter plane diagnostic ===")
        print(plane_summary.to_string(index=False))
    print(f"\nResults written to: {output.resolve()}")


if __name__ == "__main__":
    main()
