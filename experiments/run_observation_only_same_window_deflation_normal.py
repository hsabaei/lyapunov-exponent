from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-14


@dataclass(frozen=True)
class Config:
    dim: int = 20
    steps: int = 500
    trials: int = 100
    window: int = 20
    n_directions: int = 3
    seed: int = 42

    stability_threshold_deg: float = 0.2
    stability_patience: int = 5

    # A selected common window must still contain measurable absolute signal.
    # This is measured relative to the first observation window.
    relative_window_norm_floor: float = 1e-12

    # A later direction is accepted only if this fraction of the CURRENT
    # window energy remains before that stage. This is a relative criterion.
    min_residual_energy_fraction: float = 1e-10

    # Purely numerical stopping rule after each window has been normalised
    # to unit Frobenius norm. This replaces the old absolute-energy EPS test.
    numeric_relative_residual_floor: float = 1e-15

    # Dominance of the extracted direction inside each stage residual.
    min_stage_pc1_energy_fraction: float = 0.80


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / norm


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant angle: u and -u represent the same direction."""
    a = normalize(a)
    b = normalize(b)
    cosine = np.clip(abs(float(a @ b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def max_principal_angle_deg(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> float:
    qa, _ = np.linalg.qr(basis_a, mode="reduced")
    qb, _ = np.linalg.qr(basis_b, mode="reduced")

    singular_values = np.linalg.svd(
        qa.T @ qb,
        compute_uv=False,
    )
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(np.max(np.degrees(np.arccos(singular_values))))


def rotation_block(radius: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return radius * np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )


def build_spectrum(dim: int) -> np.ndarray:
    """
    Leading modes:
      q1             :  0.96
      q2             :  0.92
      q3             : -0.88
      span(q4, q5)   :  0.84 R(25 degrees)
    """
    if dim < 5:
        raise ValueError("dim must be at least 5.")

    blocks: list[np.ndarray] = [
        np.array([[0.96]], dtype=float),
        np.array([[0.92]], dtype=float),
        np.array([[-0.88]], dtype=float),
        rotation_block(0.84, 25.0),
    ]

    used = sum(block.shape[0] for block in blocks)
    remaining = dim - used

    if remaining > 0:
        smaller = np.linspace(0.78, 0.20, remaining)
        blocks.extend(
            np.array([[value]], dtype=float)
            for value in smaller
        )

    B = np.zeros((dim, dim), dtype=float)
    start = 0

    for block in blocks:
        size = block.shape[0]
        B[start : start + size, start : start + size] = block
        start += size

    return B


def build_normal_system(
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A and Q are used only to generate and validate synthetic data.
    The estimator itself receives only X and L.
    """
    B = build_spectrum(dim)

    gaussian = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(gaussian)

    A = Q @ B @ Q.T

    normality_error = float(
        np.linalg.norm(A.T @ A - A @ A.T, ord="fro")
    )
    if normality_error > 1e-10:
        raise RuntimeError(
            f"Constructed A is not numerically normal: "
            f"{normality_error:.3e}"
        )

    return A, Q, B


def simulate_trajectory(
    A: np.ndarray,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    X = np.empty((steps + 1, len(x0)), dtype=float)
    X[0] = x0

    for t in range(steps):
        X[t + 1] = A @ X[t]

    return X


def extract_directions_from_same_window(
    error_window: np.ndarray,
    n_directions: int,
    numeric_relative_residual_floor: float,
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    """
    Sequential deflation inside ONE fixed observation window.

    First normalise the raw window:

        E_tilde = E / ||E||_F.

    Therefore the original normalised-window energy is exactly one.
    All residual-energy checks below are relative quantities and are
    independent of the absolute size of the converging trajectory.

    Stage 1:
        qhat_1 = dominant right singular vector of E_tilde

        E_tilde^(2)
        =
        E_tilde
        -
        (E_tilde qhat_1) qhat_1^T

    Stage 2:
        qhat_2 = dominant right singular vector of E_tilde^(2)

    and similarly for later stages.

    Notes
    -----
    * This is mathematically equivalent to obtaining successive right
      singular vectors of the same window, but the explicit deflation keeps
      the observation-only construction visible.
    * The function no longer stops because the RAW trajectory energy falls
      below a fixed absolute EPS value.
    """
    if error_window.ndim != 2:
        raise ValueError("error_window must be 2-D.")

    raw_window_norm = float(
        np.linalg.norm(error_window, ord="fro")
    )

    if (
        not np.isfinite(raw_window_norm)
        or raw_window_norm <= np.finfo(float).tiny
    ):
        return [], []

    # Scale-invariant SVD input.
    normalised_window = error_window / raw_window_norm

    # Up to floating-point error, this is one.
    original_energy = float(
        np.linalg.norm(normalised_window, ord="fro") ** 2
    )

    residual = normalised_window.copy()
    directions: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []

    for _stage in range(1, n_directions + 1):
        energy_before = float(
            np.linalg.norm(residual, ord="fro") ** 2
        )

        # This is a RELATIVE numerical floor because original_energy ~= 1.
        if (
            not np.isfinite(energy_before)
            or energy_before
            < numeric_relative_residual_floor * original_energy
        ):
            break

        _, singular_values, vt = np.linalg.svd(
            residual,
            full_matrices=False,
        )

        direction = normalize(vt[0])

        stage_pc1_fraction = float(
            singular_values[0] ** 2 / energy_before
        )

        if (
            len(singular_values) >= 2
            and singular_values[1]
            > np.finfo(float).eps
        ):
            sv_ratio = float(
                singular_values[0] / singular_values[1]
            )
        else:
            sv_ratio = np.inf

        projection = (
            (residual @ direction)[:, None]
            * direction[None, :]
        )
        residual_after = residual - projection

        energy_after = float(
            np.linalg.norm(residual_after, ord="fro") ** 2
        )

        directions.append(direction)
        metrics.append(
            {
                "stage_pc1_energy_fraction": (
                    stage_pc1_fraction
                ),
                "singular_value_ratio_1_to_2": (
                    sv_ratio
                ),
                "residual_energy_before_fraction": float(
                    energy_before / original_energy
                ),
                "residual_energy_after_fraction": float(
                    energy_after / original_energy
                ),
                "extracted_energy_fraction_original": float(
                    max(
                        energy_before - energy_after,
                        0.0,
                    )
                    / original_energy
                ),
                "raw_window_fro_norm": raw_window_norm,
            }
        )

        residual = residual_after

    return directions, metrics


def rolling_same_window_diagnostics(
    X: np.ndarray,
    L: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """
    For every rolling window, qhat_1, qhat_2, qhat_3 are all extracted
    from that same window.
    """
    errors = X - L
    n_observations = len(errors)

    if cfg.window < 2:
        raise ValueError("window must be at least 2.")
    if cfg.window > n_observations:
        raise ValueError("window exceeds the trajectory length.")

    first_norm = float(
        np.linalg.norm(errors[: cfg.window], ord="fro")
    )
    if first_norm <= EPS:
        first_norm = 1.0

    previous: list[np.ndarray | None] = [
        None for _ in range(cfg.n_directions)
    ]

    rows: list[dict] = []

    for window_end in range(cfg.window - 1, n_observations):
        window_start = window_end - cfg.window + 1
        error_window = errors[window_start : window_end + 1]

        current_norm = float(
            np.linalg.norm(error_window, ord="fro")
        )

        directions, metrics = extract_directions_from_same_window(
            error_window=error_window,
            n_directions=cfg.n_directions,
            numeric_relative_residual_floor=(
                cfg.numeric_relative_residual_floor
            ),
        )

        row: dict = {
            "window_start": window_start,
            "window_end": window_end,
            "raw_window_fro_norm": current_norm,
            "relative_window_norm": current_norm / first_norm,
            "n_extracted_directions": len(directions),
        }

        for stage in range(1, cfg.n_directions + 1):
            if stage <= len(directions):
                direction = directions[stage - 1]
                stage_metrics = metrics[stage - 1]

                if previous[stage - 1] is None:
                    change = np.nan
                else:
                    change = angle_deg(
                        direction,
                        previous[stage - 1],
                    )

                row[f"direction_{stage}"] = direction
                row[f"stage_{stage}_direction_change_deg"] = change

                for name, value in stage_metrics.items():
                    row[f"stage_{stage}_{name}"] = value

                previous[stage - 1] = direction
            else:
                row[f"direction_{stage}"] = None
                row[f"stage_{stage}_direction_change_deg"] = np.nan
                row[
                    f"stage_{stage}_stage_pc1_energy_fraction"
                ] = np.nan
                row[
                    f"stage_{stage}_singular_value_ratio_1_to_2"
                ] = np.nan
                row[
                    f"stage_{stage}_residual_energy_before_fraction"
                ] = np.nan
                row[
                    f"stage_{stage}_residual_energy_after_fraction"
                ] = np.nan
                row[
                    f"stage_{stage}_extracted_energy_fraction_original"
                ] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def common_window_is_stable(
    recent: pd.DataFrame,
    cfg: Config,
) -> bool:
    """
    All requested directions must be simultaneously stable and supported
    by measurable residual signal in the same sequence of windows.
    """
    if recent.empty:
        return False

    last = recent.iloc[-1]

    if (
        float(last["relative_window_norm"])
        < cfg.relative_window_norm_floor
    ):
        return False

    for stage in range(1, cfg.n_directions + 1):
        changes = recent[
            f"stage_{stage}_direction_change_deg"
        ].to_numpy(dtype=float)

        finite_changes = changes[np.isfinite(changes)]

        if len(finite_changes) < cfg.stability_patience - 1:
            return False

        if not np.all(
            finite_changes <= cfg.stability_threshold_deg
        ):
            return False

        pc1 = recent[
            f"stage_{stage}_stage_pc1_energy_fraction"
        ].to_numpy(dtype=float)

        if not np.all(np.isfinite(pc1)):
            return False

        if not np.all(
            pc1 >= cfg.min_stage_pc1_energy_fraction
        ):
            return False

        residual_before = recent[
            f"stage_{stage}_residual_energy_before_fraction"
        ].to_numpy(dtype=float)

        if not np.all(np.isfinite(residual_before)):
            return False

        if not np.all(
            residual_before >= cfg.min_residual_energy_fraction
        ):
            return False

    return True


def select_common_window(
    diagnostics: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.Series | None, int]:
    candidates: list[int] = []

    for i in range(len(diagnostics)):
        if i < cfg.stability_patience - 1:
            continue

        recent = diagnostics.iloc[
            i - cfg.stability_patience + 1 : i + 1
        ]

        if common_window_is_stable(recent, cfg):
            candidates.append(i)

    if not candidates:
        return None, 0

    # Latest window for which all directions are still stable and measurable.
    selected_index = candidates[-1]
    return diagnostics.iloc[selected_index], len(candidates)


def estimate_from_observations_only(
    X: np.ndarray,
    L: np.ndarray,
    cfg: Config,
) -> tuple[list[np.ndarray], dict, pd.DataFrame]:
    """
    Estimator inputs are only X and L.
    All estimated directions come from one selected common window.
    """
    diagnostics = rolling_same_window_diagnostics(
        X=X,
        L=L,
        cfg=cfg,
    )

    selected, n_candidates = select_common_window(
        diagnostics=diagnostics,
        cfg=cfg,
    )

    if selected is None:
        return (
            [],
            {
                "success": False,
                "window_start": np.nan,
                "window_end": np.nan,
                "n_common_stable_candidates": 0,
            },
            diagnostics,
        )

    directions: list[np.ndarray] = []

    for stage in range(1, cfg.n_directions + 1):
        direction = selected[f"direction_{stage}"]
        if direction is None:
            return (
                [],
                {
                    "success": False,
                    "window_start": np.nan,
                    "window_end": np.nan,
                    "n_common_stable_candidates": n_candidates,
                },
                diagnostics,
            )

        directions.append(normalize(direction))

    info: dict = {
        "success": True,
        "window_start": int(selected["window_start"]),
        "window_end": int(selected["window_end"]),
        "relative_window_norm": float(
            selected["relative_window_norm"]
        ),
        "n_common_stable_candidates": int(n_candidates),
    }

    for stage in range(1, cfg.n_directions + 1):
        for name in [
            "direction_change_deg",
            "stage_pc1_energy_fraction",
            "singular_value_ratio_1_to_2",
            "residual_energy_before_fraction",
            "residual_energy_after_fraction",
            "extracted_energy_fraction_original",
        ]:
            info[f"stage_{stage}_{name}"] = float(
                selected[f"stage_{stage}_{name}"]
            )

    return directions, info, diagnostics


def analyse_trial(
    A: np.ndarray,
    true_basis: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    trial_id: int,
    save_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    x0 = rng.normal(size=cfg.dim)

    X = simulate_trajectory(
        A=A,
        x0=x0,
        steps=cfg.steps,
    )

    L = np.zeros(cfg.dim, dtype=float)

    directions, info, diagnostics = estimate_from_observations_only(
        X=X,
        L=L,
        cfg=cfg,
    )

    row: dict = {
        "trial": trial_id,
        "success": bool(info["success"]),
        "window_start": info.get("window_start", np.nan),
        "window_end": info.get("window_end", np.nan),
        "relative_window_norm": info.get(
            "relative_window_norm",
            np.nan,
        ),
        "n_common_stable_candidates": info.get(
            "n_common_stable_candidates",
            0,
        ),
    }

    if len(directions) == cfg.n_directions:
        estimated_basis = np.column_stack(directions)
        true_basis_k = true_basis[:, : cfg.n_directions]

        row["leading_subspace_error_deg"] = (
            max_principal_angle_deg(
                estimated_basis,
                true_basis_k,
            )
        )

        row["estimated_orthogonality_fro_error"] = float(
            np.linalg.norm(
                estimated_basis.T @ estimated_basis
                - np.eye(cfg.n_directions),
                ord="fro",
            )
        )

        for stage in range(1, cfg.n_directions + 1):
            row[
                f"estimated_qhat{stage}_vs_true_q{stage}_deg"
            ] = angle_deg(
                directions[stage - 1],
                true_basis[:, stage - 1],
            )

            row[
                f"abs_initial_component_q{stage}"
            ] = abs(
                float(
                    x0 @ normalize(true_basis[:, stage - 1])
                )
            )

            for name in [
                "direction_change_deg",
                "stage_pc1_energy_fraction",
                "singular_value_ratio_1_to_2",
                "residual_energy_before_fraction",
                "residual_energy_after_fraction",
                "extracted_energy_fraction_original",
            ]:
                row[f"stage_{stage}_{name}"] = info[
                    f"stage_{stage}_{name}"
                ]
    else:
        row["leading_subspace_error_deg"] = np.nan
        row["estimated_orthogonality_fro_error"] = np.nan

        for stage in range(1, cfg.n_directions + 1):
            row[
                f"estimated_qhat{stage}_vs_true_q{stage}_deg"
            ] = np.nan
            row[
                f"abs_initial_component_q{stage}"
            ] = abs(
                float(
                    x0 @ normalize(true_basis[:, stage - 1])
                )
            )

    return (
        pd.DataFrame([row]),
        diagnostics if save_diagnostics else None,
    )


def summarise_trials(
    all_trials: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    successful = all_trials[
        all_trials["success"] == True  # noqa: E712
    ]

    rows: list[dict] = []

    for stage in range(1, cfg.n_directions + 1):
        values = successful[
            f"estimated_qhat{stage}_vs_true_q{stage}_deg"
        ].dropna()

        rows.append(
            {
                "stage": stage,
                "n_trials": len(all_trials),
                "n_successful_common_windows": len(successful),
                "common_window_success_rate": float(
                    len(successful) / len(all_trials)
                ),
                "median_angle_error_deg": (
                    float(values.median())
                    if not values.empty
                    else np.nan
                ),
                "q25_angle_error_deg": (
                    float(values.quantile(0.25))
                    if not values.empty
                    else np.nan
                ),
                "q75_angle_error_deg": (
                    float(values.quantile(0.75))
                    if not values.empty
                    else np.nan
                ),
                "median_selected_window_end": (
                    float(successful["window_end"].median())
                    if not successful.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_angle_errors(
    all_trials: pd.DataFrame,
    cfg: Config,
    output_path: Path,
) -> None:
    labels: list[str] = []
    data: list[np.ndarray] = []

    for stage in range(1, cfg.n_directions + 1):
        values = all_trials[
            f"estimated_qhat{stage}_vs_true_q{stage}_deg"
        ].dropna().to_numpy(dtype=float)

        if len(values) > 0:
            labels.append(f"qhat{stage}")
            data.append(values)

    fig, ax = plt.subplots(figsize=(8, 5))

    if data:
        ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=True,
        )

    ax.set_xlabel("Direction extracted from the same selected window")
    ax.set_ylabel("Sign-invariant angle error (degrees)")
    ax.set_title("Observation-only same-window sequential deflation")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_first_trial_diagnostics(
    diagnostics: pd.DataFrame,
    cfg: Config,
    outdir: Path,
) -> None:
    x = diagnostics["window_end"]

    for stage in range(1, cfg.n_directions + 1):
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            x,
            diagnostics[
                f"stage_{stage}_direction_change_deg"
            ],
        )
        ax.axhline(
            cfg.stability_threshold_deg,
            linestyle="--",
            label="Stability threshold",
        )

        ax.set_xlabel("Common window end iteration")
        ax.set_ylabel("Angle to previous rolling estimate (degrees)")
        ax.set_title(
            f"First trial: qhat{stage} stability from the same window"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            outdir / f"first_trial_qhat{stage}_stability.png",
            dpi=180,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            x,
            diagnostics[
                f"stage_{stage}_stage_pc1_energy_fraction"
            ],
        )
        ax.axhline(
            cfg.min_stage_pc1_energy_fraction,
            linestyle="--",
            label="Minimum accepted stage PC1 fraction",
        )

        ax.set_xlabel("Common window end iteration")
        ax.set_ylabel("PC1 energy fraction in the stage residual")
        ax.set_title(
            f"First trial: qhat{stage} residual dominance"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            outdir / f"first_trial_qhat{stage}_pc1_energy.png",
            dpi=180,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            x,
            diagnostics[
                f"stage_{stage}_residual_energy_before_fraction"
            ],
        )
        ax.axhline(
            cfg.min_residual_energy_fraction,
            linestyle="--",
            label="Minimum residual-energy fraction",
        )

        ax.set_yscale("log")
        ax.set_xlabel("Common window end iteration")
        ax.set_ylabel(
            "Residual energy before stage / original-window energy"
        )
        ax.set_title(
            f"First trial: signal remaining before qhat{stage}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            outdir / f"first_trial_qhat{stage}_residual_energy.png",
            dpi=180,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Observation-only extraction of qhat1, qhat2, qhat3 "
            "from one common rolling window."
        )
    )

    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--n-directions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

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
        help=(
            "Purely numerical residual floor after each observation window "
            "has been normalised to unit Frobenius norm."
        ),
    )
    parser.add_argument(
        "--min-stage-pc1-energy-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/observation_only_same_window_deflation_normalized"
        ),
    )

    args = parser.parse_args()

    cfg = Config(
        dim=args.dim,
        steps=args.steps,
        trials=args.trials,
        window=args.window,
        n_directions=args.n_directions,
        seed=args.seed,
        stability_threshold_deg=args.stability_threshold_deg,
        stability_patience=args.stability_patience,
        relative_window_norm_floor=args.relative_window_norm_floor,
        min_residual_energy_fraction=args.min_residual_energy_fraction,
        numeric_relative_residual_floor=(
            args.numeric_relative_residual_floor
        ),
        min_stage_pc1_energy_fraction=(
            args.min_stage_pc1_energy_fraction
        ),
    )

    if cfg.n_directions < 1:
        raise ValueError("n_directions must be at least 1.")
    if cfg.n_directions > 3:
        raise ValueError(
            "For this test use at most 3 directions; "
            "modes 4-5 form a rotating block."
        )
    if cfg.window > cfg.steps + 1:
        raise ValueError("window must not exceed steps + 1.")

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    rng_system = np.random.default_rng(cfg.seed)
    A, true_basis, B = build_normal_system(
        dim=cfg.dim,
        rng=rng_system,
    )

    normality_error = float(
        np.linalg.norm(A.T @ A - A @ A.T, ord="fro")
    )

    print("========================================================")
    print("Step 3A revised: normalised same-window observation-only deflation")
    print("========================================================")
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"trials: {cfg.trials}")
    print(f"common window length: {cfg.window}")
    print(f"directions extracted from each window: {cfg.n_directions}")
    print(f"stability threshold: {cfg.stability_threshold_deg} deg")
    print(f"stability patience: {cfg.stability_patience}")
    print(
        "minimum residual-energy fraction: "
        f"{cfg.min_residual_energy_fraction:.3e}"
    )
    print(
        "numeric relative residual floor: "
        f"{cfg.numeric_relative_residual_floor:.3e}"
    )
    print(
        "window normalisation: "
        "E_tilde = E_t / ||E_t||_F before every SVD"
    )
    print(
        "minimum stage PC1 energy fraction: "
        f"{cfg.min_stage_pc1_energy_fraction:.3f}"
    )
    print(
        "normality error ||A^T A - A A^T||_F: "
        f"{normality_error:.6e}"
    )
    print("Estimator inputs: X and L only")
    print("All qhat_i come from the SAME selected window")
    print("No perturbations, Jacobians, or update-map queries")
    print("========================================================")

    trial_frames: list[pd.DataFrame] = []
    first_diagnostics: pd.DataFrame | None = None

    for trial_id in range(cfg.trials):
        trial_rng = np.random.default_rng(
            cfg.seed + 1000 + trial_id
        )

        trial_df, diagnostics = analyse_trial(
            A=A,
            true_basis=true_basis,
            cfg=cfg,
            rng=trial_rng,
            trial_id=trial_id,
            save_diagnostics=(trial_id == 0),
        )

        trial_frames.append(trial_df)

        if diagnostics is not None:
            first_diagnostics = diagnostics

    all_trials = pd.concat(
        trial_frames,
        ignore_index=True,
    )
    all_trials.to_csv(
        outdir / "all_trial_metrics.csv",
        index=False,
    )

    stage_summary = summarise_trials(
        all_trials=all_trials,
        cfg=cfg,
    )
    stage_summary.to_csv(
        outdir / "experiment_summary_by_stage.csv",
        index=False,
    )

    np.savez(
        outdir / "system_definition_validation_only.npz",
        A=A,
        B=B,
        true_basis=true_basis,
        normality_error=normality_error,
    )

    if first_diagnostics is not None:
        direction_columns = [
            f"direction_{stage}"
            for stage in range(1, cfg.n_directions + 1)
        ]

        first_diagnostics.drop(
            columns=direction_columns,
        ).to_csv(
            outdir / "first_trial_common_window_diagnostics.csv",
            index=False,
        )

        plot_first_trial_diagnostics(
            diagnostics=first_diagnostics,
            cfg=cfg,
            outdir=outdir,
        )

    plot_angle_errors(
        all_trials=all_trials,
        cfg=cfg,
        output_path=(
            outdir / "01_angle_errors_same_selected_window.png"
        ),
    )

    successful = all_trials[
        all_trials["success"] == True  # noqa: E712
    ]

    overall_summary: dict = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": cfg.trials,
        "window": cfg.window,
        "n_directions": cfg.n_directions,
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
        "common_window_success_rate": float(
            len(successful) / len(all_trials)
        ),
        "median_selected_window_start": (
            float(successful["window_start"].median())
            if not successful.empty
            else np.nan
        ),
        "median_selected_window_end": (
            float(successful["window_end"].median())
            if not successful.empty
            else np.nan
        ),
        "median_leading_subspace_error_deg": (
            float(
                successful["leading_subspace_error_deg"].median()
            )
            if not successful.empty
            else np.nan
        ),
        "median_estimated_orthogonality_fro_error": (
            float(
                successful[
                    "estimated_orthogonality_fro_error"
                ].median()
            )
            if not successful.empty
            else np.nan
        ),
    }

    for stage in range(1, cfg.n_directions + 1):
        values = successful[
            f"estimated_qhat{stage}_vs_true_q{stage}_deg"
        ].dropna()

        overall_summary[
            f"median_qhat{stage}_vs_q{stage}_deg"
        ] = (
            float(values.median())
            if not values.empty
            else np.nan
        )

    pd.DataFrame([overall_summary]).to_csv(
        outdir / "experiment_summary.csv",
        index=False,
    )

    print("\nOverall summary:")
    for key, value in overall_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8e}")
        else:
            print(f"{key}: {value}")

    print(
        "\nInterpretation:"
        "\n- One rolling window E_t is selected."
        "\n- Before SVD, E_t is divided by ||E_t||_F."
        "\n- qhat1 is extracted from the normalised E_t."
        "\n- qhat1 is removed only from that same E_t."
        "\n- qhat2 is extracted from that same-window residual."
        "\n- qhat2 is removed from the same residual before qhat3."
        "\n- All qhat_i therefore refer to the same local observation window."
        "\n- The true q_i are used only afterward for validation."
    )

    print(f"\nResults written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
