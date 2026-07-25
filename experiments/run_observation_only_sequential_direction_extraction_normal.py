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
    seed: int = 42

    # Observation-only estimator settings
    window: int = 20
    n_directions: int = 3
    pc1_energy_threshold: float = 0.995
    stability_threshold_deg: float = 0.2
    stability_patience: int = 5
    relative_window_norm_floor: float = 1e-12


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """
    Sign-invariant angle between two 1-D directions.

    u and -u represent the same one-dimensional direction.
    """
    a = normalize(a)
    b = normalize(b)

    c = np.clip(
        abs(float(a @ b)),
        0.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


def rotation_block(
    radius: float,
    theta_deg: float,
) -> np.ndarray:
    theta = np.deg2rad(
        theta_deg
    )

    return radius * np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta),
            ],
            [
                np.sin(theta),
                np.cos(theta),
            ],
        ],
        dtype=float,
    )


def build_real_block_diagonal_spectrum(
    dim: int,
) -> np.ndarray:
    """
    Same leading dynamics used in the previous experiments:

      q1            :  0.96
      q2            :  0.92
      q3            : -0.88
      span(q4,q5)   :  0.84 R(25 degrees)

    Remaining modes have smaller magnitudes.
    """
    if dim < 5:
        raise ValueError(
            "dim must be at least 5."
        )

    blocks: list[np.ndarray] = [
        np.array(
            [[0.96]],
            dtype=float,
        ),
        np.array(
            [[0.92]],
            dtype=float,
        ),
        np.array(
            [[-0.88]],
            dtype=float,
        ),
        rotation_block(
            0.84,
            25.0,
        ),
    ]

    used = sum(
        block.shape[0]
        for block in blocks
    )

    remaining = dim - used

    if remaining > 0:
        smaller = np.linspace(
            0.78,
            0.20,
            remaining,
        )

        blocks.extend(
            np.array(
                [[value]],
                dtype=float,
            )
            for value in smaller
        )

    B = np.zeros(
        (dim, dim),
        dtype=float,
    )

    start = 0

    for block in blocks:
        size = block.shape[0]

        B[
            start : start + size,
            start : start + size,
        ] = block

        start += size

    return B


def build_normal_system(
    dim: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build the Step-3A validation system:

        A = Q B Q^T,

    where Q is orthogonal.

    IMPORTANT:
    ----------
    A and Q are used ONLY to generate synthetic observations and to validate
    the estimator afterward.

    The observation-only estimator itself receives only:
        X = {x_0, ..., x_T}
        L

    It does not receive A, Q, eigenvalues, Jacobians, or perturbations.
    """
    B = build_real_block_diagonal_spectrum(
        dim
    )

    G = rng.normal(
        size=(dim, dim)
    )

    Q, _ = np.linalg.qr(
        G
    )

    A = Q @ B @ Q.T

    normality_error = float(
        np.linalg.norm(
            A.T @ A
            - A @ A.T,
            ord="fro",
        )
    )

    if normality_error > 1e-10:
        raise RuntimeError(
            "Constructed A is not numerically normal: "
            f"{normality_error:.3e}"
        )

    return (
        A,
        Q,
        B,
    )


def simulate_observed_trajectory(
    A: np.ndarray,
    x0: np.ndarray,
    steps: int,
) -> np.ndarray:
    """
    Generate one observed trajectory:

        x_{t+1} = A x_t.

    No perturbation trajectory is generated.
    """
    X = np.empty(
        (
            steps + 1,
            len(x0),
        ),
        dtype=float,
    )

    X[0] = x0

    for t in range(steps):
        X[t + 1] = (
            A @ X[t]
        )

    return X


def first_uncentered_pca_direction(
    rows: np.ndarray,
) -> tuple[
    np.ndarray,
    float,
]:
    """
    Estimate the dominant one-dimensional direction of a collection of
    observed row vectors using uncentered PCA / SVD.

    Returns
    -------
    direction
        First right singular vector.
    energy_fraction
        sigma_1^2 / sum_j sigma_j^2.
    """
    if rows.ndim != 2:
        raise ValueError(
            "rows must be a 2-D array."
        )

    if float(
        np.linalg.norm(rows)
    ) <= EPS:
        raise ValueError(
            "Cannot estimate a direction "
            "from a near-zero window."
        )

    _, singular_values, vt = (
        np.linalg.svd(
            rows,
            full_matrices=False,
        )
    )

    direction = normalize(
        vt[0]
    )

    denom = float(
        np.sum(
            singular_values**2
        )
    )

    energy_fraction = (
        float(
            singular_values[0] ** 2
        )
        / denom
        if denom > EPS
        else np.nan
    )

    return (
        direction,
        energy_fraction,
    )


def rolling_observation_diagnostics(
    residual_errors: np.ndarray,
    window: int,
) -> pd.DataFrame:
    """
    Compute rolling observation-only direction diagnostics.

    For every window, use ONLY the residual observed error vectors.

    The returned table contains:
      - the estimated dominant direction,
      - PC1 energy fraction,
      - change from the previous rolling direction,
      - relative window norm.

    No true q_i is used here.
    """
    n_steps, dim = (
        residual_errors.shape
    )

    if window < 2:
        raise ValueError(
            "window must be >= 2."
        )

    if window > n_steps:
        raise ValueError(
            "window is longer than "
            "the observation sequence."
        )

    base_window = (
        residual_errors[
            :window
        ]
    )

    base_norm = float(
        np.linalg.norm(
            base_window
        )
    )

    if base_norm <= EPS:
        base_norm = 1.0

    rows: list[dict] = []

    previous_direction: (
        np.ndarray | None
    ) = None

    for end in range(
        window - 1,
        n_steps,
    ):
        start = (
            end - window + 1
        )

        current_window = (
            residual_errors[
                start : end + 1
            ]
        )

        window_norm = float(
            np.linalg.norm(
                current_window
            )
        )

        if window_norm <= EPS:
            rows.append(
                {
                    "window_start": start,
                    "window_end": end,
                    "pc1_energy_fraction": np.nan,
                    "direction_change_deg": np.nan,
                    "relative_window_norm": (
                        window_norm
                        / base_norm
                    ),
                    "direction": None,
                }
            )

            continue

        direction, energy = (
            first_uncentered_pca_direction(
                current_window
            )
        )

        if (
            previous_direction
            is None
        ):
            change = np.nan
        else:
            change = angle_deg(
                direction,
                previous_direction,
            )

        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "pc1_energy_fraction": energy,
                "direction_change_deg": change,
                "relative_window_norm": (
                    window_norm
                    / base_norm
                ),
                "direction": direction,
            }
        )

        previous_direction = (
            direction
        )

    return pd.DataFrame(rows)


def find_stable_candidates(
    diagnostics: pd.DataFrame,
    energy_threshold: float,
    stability_threshold_deg: float,
    stability_patience: int,
    relative_window_norm_floor: float,
) -> list[int]:
    """
    Return row indices that satisfy a purely observation-based stability rule.

    A candidate is accepted only when, for `stability_patience` consecutive
    rolling windows:

      PC1 energy >= energy_threshold
      direction change <= stability_threshold_deg

    and the current residual window remains above a relative norm floor.

    Ground-truth q_i values are NOT used.
    """
    if stability_patience < 1:
        raise ValueError(
            "stability_patience must "
            "be at least 1."
        )

    candidates: list[int] = []

    for i in range(
        len(diagnostics)
    ):
        if (
            i
            < stability_patience
        ):
            continue

        row = diagnostics.iloc[i]

        if (
            row[
                "relative_window_norm"
            ]
            < relative_window_norm_floor
        ):
            continue

        start_i = (
            i
            - stability_patience
            + 1
        )

        recent = diagnostics.iloc[
            start_i : i + 1
        ]

        energies = recent[
            "pc1_energy_fraction"
        ].to_numpy(
            dtype=float
        )

        changes = recent[
            "direction_change_deg"
        ].to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(
                energies
            )
        ):
            continue

        finite_changes = changes[
            np.isfinite(
                changes
            )
        ]

        if (
            len(finite_changes)
            < stability_patience - 1
        ):
            continue

        energy_ok = bool(
            np.all(
                energies
                >= energy_threshold
            )
        )

        stability_ok = bool(
            np.all(
                finite_changes
                <= stability_threshold_deg
            )
        )

        if (
            energy_ok
            and stability_ok
        ):
            candidates.append(i)

    return candidates


def estimate_one_direction_from_observations(
    residual_errors: np.ndarray,
    cfg: Config,
    stage_index: int,
) -> tuple[
    np.ndarray | None,
    dict,
    pd.DataFrame,
]:
    """
    Estimate one direction from residual observed errors.

    Selection policy
    ----------------
    Stage 1:
        Use the LATEST stable usable window.

        Reason: the first dominant direction becomes increasingly isolated
        late in the convergent trajectory, so this gives the cleanest q1
        estimate while the observed signal is still numerically usable.

    Stages 2 and later:
        After deflating previously estimated directions, use the FIRST stable
        window satisfying the same observation-only criteria.

        Reason: later residual modes decay faster. Waiting until the very end
        can make them disappear numerically or allow tiny deflation errors to
        dominate the residual.

    This policy uses only the observed residual sequence and numerical
    stability diagnostics. It does NOT use the true q_i values.
    """
    diagnostics = (
        rolling_observation_diagnostics(
            residual_errors,
            cfg.window,
        )
    )

    candidates = (
        find_stable_candidates(
            diagnostics=diagnostics,
            energy_threshold=(
                cfg.pc1_energy_threshold
            ),
            stability_threshold_deg=(
                cfg.stability_threshold_deg
            ),
            stability_patience=(
                cfg.stability_patience
            ),
            relative_window_norm_floor=(
                cfg.relative_window_norm_floor
            ),
        )
    )

    if not candidates:
        info = {
            "success": False,
            "window_start": np.nan,
            "window_end": np.nan,
            "pc1_energy_fraction": np.nan,
            "direction_change_deg": np.nan,
            "relative_window_norm": np.nan,
            "n_stable_candidates": 0,
            "selection_policy": (
                "latest"
                if stage_index == 1
                else "first"
            ),
        }

        return (
            None,
            info,
            diagnostics,
        )

    if stage_index == 1:
        selected_index = (
            candidates[-1]
        )
        selection_policy = (
            "latest"
        )
    else:
        selected_index = (
            candidates[0]
        )
        selection_policy = (
            "first"
        )

    selected = diagnostics.iloc[
        selected_index
    ]

    direction = selected[
        "direction"
    ]

    if direction is None:
        raise RuntimeError(
            "Selected candidate has "
            "no direction."
        )

    info = {
        "success": True,
        "window_start": int(
            selected[
                "window_start"
            ]
        ),
        "window_end": int(
            selected[
                "window_end"
            ]
        ),
        "pc1_energy_fraction": float(
            selected[
                "pc1_energy_fraction"
            ]
        ),
        "direction_change_deg": float(
            selected[
                "direction_change_deg"
            ]
        ),
        "relative_window_norm": float(
            selected[
                "relative_window_norm"
            ]
        ),
        "n_stable_candidates": int(
            len(candidates)
        ),
        "selection_policy": (
            selection_policy
        ),
    }

    return (
        normalize(direction),
        info,
        diagnostics,
    )


def deflate_observed_errors(
    residual_errors: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Remove the estimated component along `direction` from every observed
    residual error vector:

        R_new = R - (R u) u^T.

    This is observation-space deflation.

    It does NOT propagate a new perturbation and does NOT use A or a Jacobian.
    """
    u = normalize(
        direction
    )

    coefficients = (
        residual_errors @ u
    )

    return (
        residual_errors
        - coefficients[:, None]
        * u[None, :]
    )


def sequential_observation_only_extraction(
    X: np.ndarray,
    L: np.ndarray,
    cfg: Config,
) -> tuple[
    list[np.ndarray],
    list[dict],
    list[pd.DataFrame],
]:
    """
    Step 3A estimator.

    Inputs available to the estimator:
        observed trajectory X
        limiting state L

    The estimator does NOT receive:
        A
        Q
        eigenvalues
        true q_i
        perturbation directions
        Jacobians
        update-map queries at perturbed states

    Procedure:
        r_t^(1) = x_t - L

        estimate u_1 from r_t^(1)

        r_t^(2) =
            r_t^(1)
            - <r_t^(1), u_1> u_1

        estimate u_2 from r_t^(2)

        continue recursively.
    """
    residual_errors = (
        X - L
    )

    estimated_directions: (
        list[np.ndarray]
    ) = []

    extraction_info: (
        list[dict]
    ) = []

    diagnostics_by_stage: (
        list[pd.DataFrame]
    ) = []

    for stage_index in range(
        1,
        cfg.n_directions + 1,
    ):
        (
            direction,
            info,
            diagnostics,
        ) = (
            estimate_one_direction_from_observations(
                residual_errors=(
                    residual_errors
                ),
                cfg=cfg,
                stage_index=(
                    stage_index
                ),
            )
        )

        info = dict(info)
        info[
            "stage"
        ] = stage_index

        extraction_info.append(
            info
        )

        diagnostics_by_stage.append(
            diagnostics
        )

        if direction is None:
            break

        estimated_directions.append(
            direction
        )

        residual_errors = (
            deflate_observed_errors(
                residual_errors,
                direction,
            )
        )

    return (
        estimated_directions,
        extraction_info,
        diagnostics_by_stage,
    )


def analyze_trial(
    A: np.ndarray,
    true_basis: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    trial_id: int,
    save_diagnostics: bool = False,
) -> tuple[
    pd.DataFrame,
    list[pd.DataFrame] | None,
]:
    """
    Generate one trajectory, run the observation-only estimator, then use the
    known synthetic q_i ONLY AFTERWARD for validation.
    """
    x0 = rng.normal(
        size=cfg.dim
    )

    X = (
        simulate_observed_trajectory(
            A=A,
            x0=x0,
            steps=cfg.steps,
        )
    )

    L = np.zeros(
        cfg.dim,
        dtype=float,
    )

    (
        estimated_directions,
        extraction_info,
        diagnostics_by_stage,
    ) = (
        sequential_observation_only_extraction(
            X=X,
            L=L,
            cfg=cfg,
        )
    )

    row: dict[
        str,
        float | int | bool | str,
    ] = {
        "trial": trial_id,
        "x0_norm": float(
            np.linalg.norm(x0)
        ),
        "n_extracted_directions": int(
            len(
                estimated_directions
            )
        ),
    }

    for stage in range(
        1,
        cfg.n_directions + 1,
    ):
        if (
            stage - 1
            < len(
                extraction_info
            )
        ):
            info = (
                extraction_info[
                    stage - 1
                ]
            )
        else:
            info = {
                "success": False,
            }

        row[
            f"stage_{stage}_success"
        ] = bool(
            info.get(
                "success",
                False,
            )
        )

        for key in [
            "window_start",
            "window_end",
            "pc1_energy_fraction",
            "direction_change_deg",
            "relative_window_norm",
            "n_stable_candidates",
            "selection_policy",
        ]:
            row[
                f"stage_{stage}_{key}"
            ] = info.get(
                key,
                np.nan,
            )

        if (
            stage - 1
            < len(
                estimated_directions
            )
        ):
            estimated = (
                estimated_directions[
                    stage - 1
                ]
            )

            true_direction = (
                true_basis[
                    :,
                    stage - 1,
                ]
            )

            row[
                f"estimated_q{stage}"
                f"_vs_true_q{stage}_deg"
            ] = angle_deg(
                estimated,
                true_direction,
            )

            # Also record components of x0 in the true direction.
            # This is validation only; it is NOT used by the estimator.
            row[
                f"abs_initial_component_q{stage}"
            ] = abs(
                float(
                    x0
                    @ normalize(
                        true_direction
                    )
                )
            )

        else:
            row[
                f"estimated_q{stage}"
                f"_vs_true_q{stage}_deg"
            ] = np.nan

            row[
                f"abs_initial_component_q{stage}"
            ] = abs(
                float(
                    x0
                    @ normalize(
                        true_basis[
                            :,
                            stage - 1,
                        ]
                    )
                )
            )

    diagnostics_out = (
        diagnostics_by_stage
        if save_diagnostics
        else None
    )

    return (
        pd.DataFrame([row]),
        diagnostics_out,
    )


def summarize_trials(
    all_trials: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    rows: list[dict] = []

    for stage in range(
        1,
        cfg.n_directions + 1,
    ):
        angle_col = (
            f"estimated_q{stage}"
            f"_vs_true_q{stage}_deg"
        )

        success_col = (
            f"stage_{stage}_success"
        )

        end_col = (
            f"stage_{stage}_window_end"
        )

        successful = all_trials[
            all_trials[
                success_col
            ]
            == True  # noqa: E712
        ]

        angles = successful[
            angle_col
        ].dropna()

        ends = successful[
            end_col
        ].dropna()

        row = {
            "stage": stage,
            "n_trials": len(
                all_trials
            ),
            "n_successful_extractions": int(
                len(successful)
            ),
            "extraction_success_rate": float(
                len(successful)
                / len(all_trials)
            ),
            "median_angle_error_deg": (
                float(
                    angles.median()
                )
                if not angles.empty
                else np.nan
            ),
            "q25_angle_error_deg": (
                float(
                    angles.quantile(
                        0.25
                    )
                )
                if not angles.empty
                else np.nan
            ),
            "q75_angle_error_deg": (
                float(
                    angles.quantile(
                        0.75
                    )
                )
                if not angles.empty
                else np.nan
            ),
            "median_selected_window_end": (
                float(
                    ends.median()
                )
                if not ends.empty
                else np.nan
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def plot_angle_error_summary(
    all_trials: pd.DataFrame,
    cfg: Config,
    output_path: Path,
) -> None:
    labels: list[str] = []
    data: list[np.ndarray] = []

    for stage in range(
        1,
        cfg.n_directions + 1,
    ):
        col = (
            f"estimated_q{stage}"
            f"_vs_true_q{stage}_deg"
        )

        values = (
            all_trials[col]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(values) > 0:
            labels.append(
                f"q{stage}"
            )
            data.append(
                values
            )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
    )

    ax.set_xlabel(
        "Sequentially extracted direction"
    )

    ax.set_ylabel(
        "Sign-invariant angle error (degrees)"
    )

    ax.set_title(
        "Observation-only sequential direction extraction"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_selected_window_summary(
    all_trials: pd.DataFrame,
    cfg: Config,
    output_path: Path,
) -> None:
    labels: list[str] = []
    data: list[np.ndarray] = []

    for stage in range(
        1,
        cfg.n_directions + 1,
    ):
        col = (
            f"stage_{stage}"
            "_window_end"
        )

        values = (
            all_trials[col]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(values) > 0:
            labels.append(
                f"q{stage}"
            )
            data.append(
                values
            )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
    )

    ax.set_xlabel(
        "Sequential extraction stage"
    )

    ax.set_ylabel(
        "Selected window end iteration"
    )

    ax.set_title(
        "Where the observation-only estimator selected each direction"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def plot_first_trial_stage_diagnostics(
    diagnostics_by_stage: list[pd.DataFrame],
    cfg: Config,
    outdir: Path,
) -> None:
    for stage_index, diagnostics in enumerate(
        diagnostics_by_stage,
        start=1,
    ):
        if diagnostics.empty:
            continue

        x = diagnostics[
            "window_end"
        ]

        # Direction stability
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            x,
            diagnostics[
                "direction_change_deg"
            ],
        )

        ax.axhline(
            cfg.stability_threshold_deg,
            linestyle="--",
            label="Stability threshold",
        )

        ax.set_xlabel(
            "Window end iteration"
        )

        ax.set_ylabel(
            "Angle to previous rolling estimate (degrees)"
        )

        ax.set_title(
            f"First trial: stage {stage_index} rolling direction stability"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            outdir
            / (
                f"first_trial_stage_"
                f"{stage_index}_"
                "direction_stability.png"
            ),
            dpi=180,
        )

        plt.close(fig)

        # PC1 energy
        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            x,
            diagnostics[
                "pc1_energy_fraction"
            ],
        )

        ax.axhline(
            cfg.pc1_energy_threshold,
            linestyle="--",
            label="Energy threshold",
        )

        ax.set_xlabel(
            "Window end iteration"
        )

        ax.set_ylabel(
            "PC1 energy fraction"
        )

        ax.set_title(
            f"First trial: stage {stage_index} rolling PC1 energy"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            outdir
            / (
                f"first_trial_stage_"
                f"{stage_index}_"
                "pc1_energy.png"
            ),
            dpi=180,
        )

        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3A: extract multiple directions using only one observed "
            "trajectory in a controlled normal linear system."
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
        "--trials",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n-directions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--pc1-energy-threshold",
        type=float,
        default=0.995,
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
        "--output",
        type=Path,
        default=Path(
            "results/"
            "observation_only_"
            "sequential_normal"
        ),
    )

    args = parser.parse_args()

    cfg = Config(
        dim=args.dim,
        steps=args.steps,
        trials=args.trials,
        seed=args.seed,
        window=args.window,
        n_directions=(
            args.n_directions
        ),
        pc1_energy_threshold=(
            args.pc1_energy_threshold
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
    )

    if cfg.n_directions < 1:
        raise ValueError(
            "n_directions must be >= 1."
        )

    if cfg.n_directions > 3:
        raise ValueError(
            "For Step 3A, use n_directions <= 3. "
            "The fourth/fifth modes form a rotational block and are not "
            "individual fixed directions."
        )

    if cfg.window > cfg.steps + 1:
        raise ValueError(
            "window must be <= steps + 1."
        )

    outdir = args.output

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng_system = (
        np.random.default_rng(
            cfg.seed
        )
    )

    (
        A,
        true_basis,
        B,
    ) = build_normal_system(
        dim=cfg.dim,
        rng=rng_system,
    )

    normality_error = float(
        np.linalg.norm(
            A.T @ A
            - A @ A.T,
            ord="fro",
        )
    )

    print(
        "========================================================"
    )
    print(
        "Step 3A: observation-only sequential direction extraction"
    )
    print(
        "========================================================"
    )
    print(
        f"dimension: {cfg.dim}"
    )
    print(
        f"steps: {cfg.steps}"
    )
    print(
        f"trials: {cfg.trials}"
    )
    print(
        f"window: {cfg.window}"
    )
    print(
        "number of sequential directions: "
        f"{cfg.n_directions}"
    )
    print(
        "PC1 energy threshold: "
        f"{cfg.pc1_energy_threshold}"
    )
    print(
        "stability threshold: "
        f"{cfg.stability_threshold_deg} deg"
    )
    print(
        "stability patience: "
        f"{cfg.stability_patience}"
    )
    print(
        "relative window norm floor: "
        f"{cfg.relative_window_norm_floor:.3e}"
    )
    print(
        "normality error "
        "||A^T A - A A^T||_F: "
        f"{normality_error:.6e}"
    )
    print(
        "Estimator inputs: observed X and L only"
    )
    print(
        "No perturbation trajectories"
    )
    print(
        "No Jacobian or matrix-free perturbation queries"
    )
    print(
        "No A/Q/eigenvalues supplied to the estimator"
    )
    print(
        "========================================================"
    )

    trial_frames: list[
        pd.DataFrame
    ] = []

    first_trial_diagnostics: (
        list[pd.DataFrame] | None
    ) = None

    for trial_id in range(
        cfg.trials
    ):
        trial_rng = (
            np.random.default_rng(
                cfg.seed
                + 1000
                + trial_id
            )
        )

        (
            trial_df,
            diagnostics,
        ) = analyze_trial(
            A=A,
            true_basis=(
                true_basis
            ),
            cfg=cfg,
            rng=trial_rng,
            trial_id=trial_id,
            save_diagnostics=(
                trial_id == 0
            ),
        )

        trial_frames.append(
            trial_df
        )

        if diagnostics is not None:
            first_trial_diagnostics = (
                diagnostics
            )

    all_trials = pd.concat(
        trial_frames,
        ignore_index=True,
    )

    all_trials.to_csv(
        outdir
        / "all_trial_metrics.csv",
        index=False,
    )

    summary = summarize_trials(
        all_trials,
        cfg,
    )

    summary.to_csv(
        outdir
        / "experiment_summary_by_stage.csv",
        index=False,
    )

    np.savez(
        outdir
        / "system_definition_validation_only.npz",
        A=A,
        B=B,
        true_basis=true_basis,
        normality_error=(
            normality_error
        ),
    )

    if (
        first_trial_diagnostics
        is not None
    ):
        for (
            stage,
            diagnostics,
        ) in enumerate(
            first_trial_diagnostics,
            start=1,
        ):
            save_df = (
                diagnostics.drop(
                    columns=[
                        "direction"
                    ]
                )
            )

            save_df.to_csv(
                outdir
                / (
                    "first_trial_"
                    f"stage_{stage}_"
                    "rolling_diagnostics.csv"
                ),
                index=False,
            )

        plot_first_trial_stage_diagnostics(
            diagnostics_by_stage=(
                first_trial_diagnostics
            ),
            cfg=cfg,
            outdir=outdir,
        )

    plot_angle_error_summary(
        all_trials=all_trials,
        cfg=cfg,
        output_path=(
            outdir
            / "01_angle_error_by_extracted_direction.png"
        ),
    )

    plot_selected_window_summary(
        all_trials=all_trials,
        cfg=cfg,
        output_path=(
            outdir
            / "02_selected_window_end_by_stage.png"
        ),
    )

    print(
        "\nStage-wise summary:"
    )

    for _, row in summary.iterrows():
        stage = int(
            row["stage"]
        )

        print(
            f"\nStage {stage}:"
        )
        print(
            "  extraction success rate: "
            f"{row['extraction_success_rate']:.3f}"
        )
        print(
            "  median angle error: "
            f"{row['median_angle_error_deg']:.6f} deg"
        )
        print(
            "  25%-75% angle error: "
            f"[{row['q25_angle_error_deg']:.6f}, "
            f"{row['q75_angle_error_deg']:.6f}] deg"
        )
        print(
            "  median selected window end: "
            f"{row['median_selected_window_end']:.1f}"
        )

    print(
        "\nInterpretation:"
        "\n- Stage 1 tests whether q1 can be estimated from observations only."
        "\n- After estimating q1, its projection is removed from every observed "
        "error vector."
        "\n- Stage 2 tests whether the dominant direction of those residual "
        "observations recovers q2."
        "\n- Stage 3 repeats the same idea for q3."
        "\n- The estimator never uses the true q_i values for extraction; "
        "they are used only after extraction to compute validation angles."
        "\n- Failure to recover a later direction is a meaningful result: "
        "a single trajectory may not contain enough observable signal for "
        "that mode, especially when its initial component is small or it "
        "has already decayed."
        "\n- This Step 3A experiment is deliberately normal. Only after "
        "understanding this case should the same observation-only idea be "
        "tested in a non-normal system."
    )

    print(
        "\nResults written to: "
        f"{outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
