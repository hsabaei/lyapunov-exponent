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
    n_directions: int = 5
    seed: int = 42
    epsilon: float = 1e-5
    transport: str = "matrix_free"


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """
    Sign-invariant angle between two 1-D directions.

    a and -a are treated as the same direction.
    """
    a = normalize(a)
    b = normalize(b)
    c = np.clip(abs(float(a @ b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def orthonormal_basis(
    matrix: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Return an orthonormal basis for the column space of `matrix`.

    SVD is used because the columns of the true non-normal eigenbasis
    are generally not orthogonal.
    """
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)

    if len(s) == 0:
        raise ValueError("Cannot build a basis from an empty matrix.")

    threshold = tol * max(matrix.shape) * float(s[0])
    rank = int(np.sum(s > threshold))

    if rank == 0:
        raise ValueError("Input matrix is numerically rank deficient.")

    return u[:, :rank]


def max_principal_angle_deg(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> float:
    """
    Largest principal angle between equal-dimensional subspaces.

    The input columns do not need to be orthonormal.
    """
    qa = orthonormal_basis(basis_a)
    qb = orthonormal_basis(basis_b)

    if qa.shape[1] != qb.shape[1]:
        raise ValueError(
            "The two subspaces must have the same dimension. "
            f"Got {qa.shape[1]} and {qb.shape[1]}."
        )

    s = np.linalg.svd(
        qa.T @ qb,
        compute_uv=False,
    )
    s = np.clip(s, 0.0, 1.0)

    return float(
        np.max(
            np.degrees(
                np.arccos(s)
            )
        )
    )


def rotation_block(
    radius: float,
    theta_deg: float,
) -> np.ndarray:
    theta = np.deg2rad(theta_deg)

    return radius * np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )


def build_real_block_diagonal_spectrum(
    dim: int,
) -> np.ndarray:
    """
    Same leading spectrum as the successful normal experiment:

      q1              :  0.96
      q2              :  0.92
      q3              : -0.88
      span(q4, q5)    :  0.84 R(25 degrees)

    Remaining modes have smaller magnitudes.

    The only major change in this Step-2 experiment is that the mixing
    matrix Q is now non-orthogonal, so A = Q B Q^{-1} is non-normal.
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
        B[
            start : start + size,
            start : start + size,
        ] = block
        start += size

    return B


def build_nonnormal_system(
    dim: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    Build

        A = Q B Q^{-1}

    with a non-orthogonal but moderately conditioned Q.

    Returns
    -------
    A
        Non-normal system matrix.
    Q
        Raw similarity basis. For scalar blocks:
          Q[:, 0] is q1,
          Q[:, 1] is q2,
          Q[:, 2] is q3.
        Q[:, 3:5] spans the true 2-D rotational invariant plane.
    B
        Block-diagonal dynamics.
    cond_Q
        Condition number of Q.
    nonnormality
        ||A^T A - A A^T||_F.
    """
    B = build_real_block_diagonal_spectrum(dim)

    # Start from an orthogonal matrix.
    G = rng.normal(size=(dim, dim))
    Q_orth, _ = np.linalg.qr(G)

    # Make the basis non-orthogonal.
    shear = np.eye(dim)

    # Deliberately couple the leading columns so q1, q2, q3 are
    # genuinely non-orthogonal. This makes Step 2 meaningfully different
    # from the normal experiment and prevents an accidental nearly
    # orthogonal leading eigenbasis.
    if dim >= 5:
        shear[0, 1] += 0.35
        shear[0, 2] -= 0.25
        shear[1, 2] += 0.30

        # Also mix the rotational block with earlier coordinates so its
        # invariant plane is not orthogonal to the leading 3-D subspace.
        shear[0, 3] += 0.20
        shear[1, 4] -= 0.20
        shear[2, 3] += 0.15

    for _ in range(dim * 2):
        i, j = rng.integers(
            low=0,
            high=dim,
            size=2,
        )

        if i != j:
            shear[i, j] += rng.uniform(
                -0.35,
                0.35,
            )

    scaling = np.diag(
        np.linspace(
            0.7,
            1.4,
            dim,
        )
    )

    Q = Q_orth @ shear @ scaling

    cond_Q = float(
        np.linalg.cond(Q)
    )

    if cond_Q > 1e4:
        raise RuntimeError(
            "Generated Q is too ill-conditioned: "
            f"cond(Q)={cond_Q:.3e}"
        )

    A = Q @ B @ np.linalg.inv(Q)

    nonnormality = float(
        np.linalg.norm(
            A.T @ A - A @ A.T,
            ord="fro",
        )
    )

    return (
        A,
        Q,
        B,
        cond_Q,
        nonnormality,
    )


def initialize_orthonormal_probes(
    dim: int,
    n_directions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Random orthonormal probe basis.

    Important:
    probe 2 is NOT assumed to be q2.
    probe 3 is NOT assumed to be q3.

    We explicitly measure individual-direction errors and subspace errors.
    """
    if n_directions > dim:
        raise ValueError(
            "n_directions cannot exceed dim."
        )

    G = rng.normal(
        size=(dim, n_directions)
    )

    probes, _ = np.linalg.qr(
        G,
        mode="reduced",
    )

    return probes


def direct_transport(
    A: np.ndarray,
    probes: np.ndarray,
) -> np.ndarray:
    """
    Exact controlled transport:
        Z_{t+1} = A W_t.
    """
    return A @ probes


def matrix_free_transport(
    A: np.ndarray,
    x_t: np.ndarray,
    probes: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transport probes without explicitly forming or applying a Jacobian:

        z_j =
        [U(x_t + epsilon w_j) - U(x_t)] / epsilon.

    For this linear test U(x)=Ax.

    This is Jacobian-free, but NOT observation-only:
    it still requires evaluating U at perturbed states.
    """
    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    def update_map(x: np.ndarray) -> np.ndarray:
        return A @ x

    x_next = update_map(x_t)

    transported = np.empty_like(
        probes
    )

    for j in range(
        probes.shape[1]
    ):
        perturbed_next = update_map(
            x_t
            + epsilon * probes[:, j]
        )

        transported[:, j] = (
            perturbed_next - x_next
        ) / epsilon

    return (
        x_next,
        transported,
    )


def enforce_orthogonality(
    transported: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    QR re-orthogonalizes the transported probes.

    In a non-normal system this does NOT imply that probe 2 must converge
    to the second right eigenvector q2. The meaningful quantities are also
    the nested leading invariant subspaces.
    """
    Q_probe, R = np.linalg.qr(
        transported,
        mode="reduced",
    )

    return (
        Q_probe,
        R,
    )


def build_true_validation_objects(
    similarity_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build the known reference directions/subspaces from the synthetic system.

    The raw columns of the similarity matrix are generally non-orthogonal.

    q1, q2, q3:
        known individual right eigendirections for the scalar blocks.

    leading_2 / leading_3 / leading_5:
        true invariant subspaces spanned by the first 2, 3, and 5 columns.

    rotation_plane:
        the true invariant plane span(q4, q5).

    qr_residual_plane:
        the orthogonal complement of the leading-3 subspace inside the
        leading-5 subspace.

        This is the 2-D object that QR columns 4-5 can naturally converge to
        in the non-normal case. It is generally NOT identical to the raw
        invariant rotation plane span(q4, q5).
    """
    if similarity_basis.shape[1] < 5:
        raise ValueError(
            "Need at least 5 basis columns."
        )

    q1 = normalize(
        similarity_basis[:, 0]
    )
    q2 = normalize(
        similarity_basis[:, 1]
    )
    q3 = normalize(
        similarity_basis[:, 2]
    )

    leading_1 = orthonormal_basis(
        similarity_basis[:, :1]
    )
    leading_2 = orthonormal_basis(
        similarity_basis[:, :2]
    )
    leading_3 = orthonormal_basis(
        similarity_basis[:, :3]
    )
    leading_5 = orthonormal_basis(
        similarity_basis[:, :5]
    )

    rotation_plane = orthonormal_basis(
        similarity_basis[:, 3:5]
    )

    projector_perp_leading_3 = (
        np.eye(
            similarity_basis.shape[0]
        )
        - leading_3 @ leading_3.T
    )

    projected_rotation_plane = (
        projector_perp_leading_3
        @ similarity_basis[:, 3:5]
    )

    qr_residual_plane = orthonormal_basis(
        projected_rotation_plane
    )

    if qr_residual_plane.shape[1] != 2:
        raise RuntimeError(
            "Expected a 2-D QR residual plane."
        )

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "leading_1": leading_1,
        "leading_2": leading_2,
        "leading_3": leading_3,
        "leading_5": leading_5,
        "rotation_plane": rotation_plane,
        "qr_residual_plane": qr_residual_plane,
    }


def run_trial(
    A: np.ndarray,
    references: dict[str, np.ndarray],
    cfg: Config,
    rng: np.random.Generator,
    trial_id: int,
    save_history: bool = False,
) -> tuple[
    pd.DataFrame,
    np.ndarray | None,
]:
    k = cfg.n_directions

    probes = initialize_orthonormal_probes(
        dim=cfg.dim,
        n_directions=k,
        rng=rng,
    )

    x_t = rng.normal(
        size=cfg.dim
    )

    history = None

    if save_history:
        history = np.empty(
            (
                cfg.steps + 1,
                cfg.dim,
                k,
            ),
            dtype=float,
        )

        history[0] = probes

    previous_probes: (
        np.ndarray | None
    ) = None

    rows: list[
        dict[str, float | int]
    ] = []

    for t in range(
        cfg.steps + 1
    ):
        row: dict[
            str,
            float | int,
        ] = {
            "trial": trial_id,
            "iteration": t,
        }

        # -------------------------------------------------------------
        # 1. Numerical orthogonality after QR
        # -------------------------------------------------------------
        gram_error = (
            probes.T @ probes
            - np.eye(k)
        )

        row[
            "orthogonality_fro_error"
        ] = float(
            np.linalg.norm(
                gram_error,
                ord="fro",
            )
        )

        pairwise_deviations: list[
            float
        ] = []

        for i in range(k):
            for j in range(
                i + 1,
                k,
            ):
                pair_angle = angle_deg(
                    probes[:, i],
                    probes[:, j],
                )

                row[
                    f"probe_{i + 1}"
                    f"_vs_probe_{j + 1}_deg"
                ] = pair_angle

                pairwise_deviations.append(
                    abs(
                        pair_angle - 90.0
                    )
                )

        row[
            "max_pairwise_deviation_from_90_deg"
        ] = (
            float(
                max(
                    pairwise_deviations
                )
            )
            if pairwise_deviations
            else 0.0
        )

        # -------------------------------------------------------------
        # 2. Iteration-to-iteration change of each probe
        # -------------------------------------------------------------
        for j in range(k):
            if previous_probes is None:
                row[
                    f"probe_{j + 1}"
                    "_change_deg"
                ] = np.nan

            else:
                row[
                    f"probe_{j + 1}"
                    "_change_deg"
                ] = angle_deg(
                    probes[:, j],
                    previous_probes[:, j],
                )

        # -------------------------------------------------------------
        # 3. Individual direction diagnostics
        #
        # q1 should be recoverable.
        # In a non-normal system probe 2 and probe 3 are NOT guaranteed
        # to converge individually to q2 and q3.
        # -------------------------------------------------------------
        if k >= 1:
            row[
                "probe_1_vs_q1_deg"
            ] = angle_deg(
                probes[:, 0],
                references["q1"],
            )

        if k >= 2:
            row[
                "probe_2_vs_q2_deg"
            ] = angle_deg(
                probes[:, 1],
                references["q2"],
            )

        if k >= 3:
            row[
                "probe_3_vs_q3_deg"
            ] = angle_deg(
                probes[:, 2],
                references["q3"],
            )

        # -------------------------------------------------------------
        # 4. Leading invariant-subspace recovery
        #
        # These are the main Step-2 metrics.
        # -------------------------------------------------------------
        if k >= 1:
            row[
                "leading_1_subspace_error_deg"
            ] = max_principal_angle_deg(
                probes[:, :1],
                references["leading_1"],
            )

        if k >= 2:
            row[
                "leading_2_subspace_error_deg"
            ] = max_principal_angle_deg(
                probes[:, :2],
                references["leading_2"],
            )

        if k >= 3:
            row[
                "leading_3_subspace_error_deg"
            ] = max_principal_angle_deg(
                probes[:, :3],
                references["leading_3"],
            )

        if k >= 5:
            row[
                "leading_5_subspace_error_deg"
            ] = max_principal_angle_deg(
                probes[:, :5],
                references["leading_5"],
            )

            # Raw comparison to the actual rotational invariant plane.
            # In a non-normal system this is NOT expected necessarily to
            # converge to zero because probes 4-5 are forced to be
            # orthogonal to probes 1-3.
            row[
                "probe45_vs_true_rotation_plane_error_deg"
            ] = max_principal_angle_deg(
                probes[:, 3:5],
                references[
                    "rotation_plane"
                ],
            )

            # This is the QR-compatible 2-D residual plane:
            # orthogonal complement of leading-3 inside leading-5.
            row[
                "probe45_vs_qr_residual_plane_error_deg"
            ] = max_principal_angle_deg(
                probes[:, 3:5],
                references[
                    "qr_residual_plane"
                ],
            )

        rows.append(row)

        if t == cfg.steps:
            break

        previous_probes = (
            probes.copy()
        )

        if cfg.transport == "direct":
            transported = direct_transport(
                A=A,
                probes=probes,
            )

            x_t = A @ x_t

        elif (
            cfg.transport
            == "matrix_free"
        ):
            (
                x_t,
                transported,
            ) = matrix_free_transport(
                A=A,
                x_t=x_t,
                probes=probes,
                epsilon=cfg.epsilon,
            )

        else:
            raise ValueError(
                "transport must be "
                "'direct' or "
                "'matrix_free'."
            )

        # Key operation:
        # transport first, then QR.
        probes, _ = (
            enforce_orthogonality(
                transported
            )
        )

        if history is not None:
            history[
                t + 1
            ] = probes

    return (
        pd.DataFrame(rows),
        history,
    )


def summarize_by_iteration(
    all_trials: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        c
        for c in all_trials.columns
        if c
        not in {
            "trial",
            "iteration",
        }
    ]

    rows: list[
        dict[str, float | int]
    ] = []

    for (
        iteration,
        group,
    ) in all_trials.groupby(
        "iteration"
    ):
        row: dict[
            str,
            float | int,
        ] = {
            "iteration": int(
                iteration
            )
        }

        for metric in metric_columns:
            values = (
                group[metric]
                .dropna()
            )

            if values.empty:
                row[
                    f"{metric}_median"
                ] = np.nan

                row[
                    f"{metric}_q25"
                ] = np.nan

                row[
                    f"{metric}_q75"
                ] = np.nan

            else:
                row[
                    f"{metric}_median"
                ] = float(
                    values.median()
                )

                row[
                    f"{metric}_q25"
                ] = float(
                    values.quantile(
                        0.25
                    )
                )

                row[
                    f"{metric}_q75"
                ] = float(
                    values.quantile(
                        0.75
                    )
                )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def plot_summary_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    x = summary[
        "iteration"
    ]

    median = summary[
        f"{metric}_median"
    ]

    q25 = summary[
        f"{metric}_q25"
    ]

    q75 = summary[
        f"{metric}_q75"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        x,
        median,
        label="Median",
    )

    ax.fill_between(
        x,
        q25,
        q75,
        alpha=0.2,
        label="25%-75%",
    )

    if log_scale:
        ax.set_yscale(
            "log"
        )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2: controlled multi-direction tracking "
            "in a non-normal linear system."
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
        "--n-directions",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--transport",
        choices=(
            "direct",
            "matrix_free",
        ),
        default="matrix_free",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "complex_linear_"
            "multidirection_nonnormal"
        ),
    )

    args = parser.parse_args()

    cfg = Config(
        dim=args.dim,
        steps=args.steps,
        trials=args.trials,
        n_directions=(
            args.n_directions
        ),
        seed=args.seed,
        epsilon=args.epsilon,
        transport=args.transport,
    )

    if cfg.n_directions < 1:
        raise ValueError(
            "n_directions must "
            "be at least 1."
        )

    if (
        cfg.n_directions
        > cfg.dim
    ):
        raise ValueError(
            "n_directions cannot "
            "exceed dim."
        )

    if cfg.n_directions < 5:
        print(
            "Warning: use at least "
            "5 directions to test "
            "the first 3 modes and "
            "the 2-D rotational block."
        )

    outdir = args.output

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Build one fixed non-normal system shared by all trials.
    # -------------------------------------------------------------
    rng_system = np.random.default_rng(
        cfg.seed
    )

    (
        A,
        similarity_basis,
        B,
        cond_Q,
        nonnormality,
    ) = build_nonnormal_system(
        dim=cfg.dim,
        rng=rng_system,
    )

    references = (
        build_true_validation_objects(
            similarity_basis
        )
    )

    # Diagnostics showing that the true right eigenvectors
    # are not generally orthogonal.
    q1_q2_angle = angle_deg(
        references["q1"],
        references["q2"],
    )

    q1_q3_angle = angle_deg(
        references["q1"],
        references["q3"],
    )

    q2_q3_angle = angle_deg(
        references["q2"],
        references["q3"],
    )

    print(
        "========================================================"
    )
    print(
        "Step 2: non-normal multi-direction validation"
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
        "tracked probes: "
        f"{cfg.n_directions}"
    )
    print(
        f"transport: {cfg.transport}"
    )
    print(
        f"cond(Q): {cond_Q:.6f}"
    )
    print(
        "non-normality "
        "||A^T A - A A^T||_F: "
        f"{nonnormality:.6f}"
    )
    print(
        "known leading dynamics:"
    )
    print(
        "  q1:  0.96"
    )
    print(
        "  q2:  0.92"
    )
    print(
        "  q3: -0.88"
    )
    print(
        "  span(q4,q5): "
        "0.84 R(25 degrees)"
    )
    print(
        "angles among true right eigendirections:"
    )
    print(
        "  angle(q1,q2): "
        f"{q1_q2_angle:.6f} deg"
    )
    print(
        "  angle(q1,q3): "
        f"{q1_q3_angle:.6f} deg"
    )
    print(
        "  angle(q2,q3): "
        f"{q2_q3_angle:.6f} deg"
    )
    print(
        "========================================================"
    )

    trial_frames: list[
        pd.DataFrame
    ] = []

    first_history: (
        np.ndarray | None
    ) = None

    for trial_id in range(
        cfg.trials
    ):
        trial_rng = np.random.default_rng(
            cfg.seed
            + 1000
            + trial_id
        )

        (
            trial_df,
            history,
        ) = run_trial(
            A=A,
            references=references,
            cfg=cfg,
            rng=trial_rng,
            trial_id=trial_id,
            save_history=(
                trial_id == 0
            ),
        )

        trial_frames.append(
            trial_df
        )

        if history is not None:
            first_history = (
                history
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

    summary = (
        summarize_by_iteration(
            all_trials
        )
    )

    summary.to_csv(
        outdir
        / "iteration_summary.csv",
        index=False,
    )

    np.savez(
        outdir
        / "system_definition.npz",
        A=A,
        B=B,
        similarity_basis=(
            similarity_basis
        ),
        q1=references["q1"],
        q2=references["q2"],
        q3=references["q3"],
        leading_2=(
            references["leading_2"]
        ),
        leading_3=(
            references["leading_3"]
        ),
        leading_5=(
            references["leading_5"]
        ),
        rotation_plane=(
            references[
                "rotation_plane"
            ]
        ),
        qr_residual_plane=(
            references[
                "qr_residual_plane"
            ]
        ),
        cond_Q=cond_Q,
        nonnormality=nonnormality,
    )

    if first_history is not None:
        np.savez(
            outdir
            / "first_trial_probe_history.npz",
            probes=first_history,
        )

    # -------------------------------------------------------------
    # Plots: individual directions
    # -------------------------------------------------------------
    if cfg.n_directions >= 1:
        plot_summary_metric(
            summary=summary,
            metric=(
                "probe_1_vs_q1_deg"
            ),
            title=(
                "Probe 1 versus true "
                "dominant direction q1"
            ),
            ylabel=(
                "Sign-invariant angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "01_probe_1_vs_q1.png"
            ),
        )

    if cfg.n_directions >= 2:
        plot_summary_metric(
            summary=summary,
            metric=(
                "probe_2_vs_q2_deg"
            ),
            title=(
                "Probe 2 versus true q2 "
                "(diagnostic; not guaranteed "
                "in non-normal case)"
            ),
            ylabel=(
                "Sign-invariant angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "02_probe_2_vs_q2.png"
            ),
        )

    if cfg.n_directions >= 3:
        plot_summary_metric(
            summary=summary,
            metric=(
                "probe_3_vs_q3_deg"
            ),
            title=(
                "Probe 3 versus true q3 "
                "(diagnostic; not guaranteed "
                "in non-normal case)"
            ),
            ylabel=(
                "Sign-invariant angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "03_probe_3_vs_q3.png"
            ),
        )

    # -------------------------------------------------------------
    # Plots: main non-normal subspace metrics
    # -------------------------------------------------------------
    if cfg.n_directions >= 2:
        plot_summary_metric(
            summary=summary,
            metric=(
                "leading_2_"
                "subspace_error_deg"
            ),
            title=(
                "Tracked leading 2-D "
                "subspace versus "
                "span(q1,q2)"
            ),
            ylabel=(
                "Largest principal angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "04_leading_2_"
                "subspace_error.png"
            ),
        )

    if cfg.n_directions >= 3:
        plot_summary_metric(
            summary=summary,
            metric=(
                "leading_3_"
                "subspace_error_deg"
            ),
            title=(
                "Tracked leading 3-D "
                "subspace versus "
                "span(q1,q2,q3)"
            ),
            ylabel=(
                "Largest principal angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "05_leading_3_"
                "subspace_error.png"
            ),
        )

    if cfg.n_directions >= 5:
        plot_summary_metric(
            summary=summary,
            metric=(
                "leading_5_"
                "subspace_error_deg"
            ),
            title=(
                "Tracked leading 5-D "
                "subspace versus true "
                "leading 5-D invariant subspace"
            ),
            ylabel=(
                "Largest principal angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "06_leading_5_"
                "subspace_error.png"
            ),
        )

        plot_summary_metric(
            summary=summary,
            metric=(
                "probe45_vs_true_"
                "rotation_plane_"
                "error_deg"
            ),
            title=(
                "Probes 4-5 versus raw "
                "true rotational invariant plane"
            ),
            ylabel=(
                "Largest principal angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "07_probe45_vs_true_"
                "rotation_plane.png"
            ),
        )

        plot_summary_metric(
            summary=summary,
            metric=(
                "probe45_vs_qr_"
                "residual_plane_"
                "error_deg"
            ),
            title=(
                "Probes 4-5 versus QR-compatible "
                "residual 2-D plane"
            ),
            ylabel=(
                "Largest principal angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / "08_probe45_vs_qr_"
                "residual_plane.png"
            ),
        )

    # -------------------------------------------------------------
    # Probe stability
    # -------------------------------------------------------------
    for j in range(
        1,
        cfg.n_directions + 1,
    ):
        plot_summary_metric(
            summary=summary,
            metric=(
                f"probe_{j}_"
                "change_deg"
            ),
            title=(
                "Iteration-to-iteration "
                f"change of probe {j}"
            ),
            ylabel=(
                "Sign-invariant angle "
                "(degrees)"
            ),
            output_path=(
                outdir
                / f"1{j}_probe_{j}_"
                "change.png"
            ),
        )

    # -------------------------------------------------------------
    # Orthogonality
    # -------------------------------------------------------------
    plot_summary_metric(
        summary=summary,
        metric=(
            "orthogonality_"
            "fro_error"
        ),
        title=(
            "Numerical orthogonality "
            "error after QR"
        ),
        ylabel=(
            "||W^T W - I||_F"
        ),
        output_path=(
            outdir
            / "21_orthogonality_"
            "error.png"
        ),
        log_scale=True,
    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------
    final_rows = all_trials[
        all_trials[
            "iteration"
        ]
        == cfg.steps
    ]

    final_summary: dict[
        str,
        float | int | str,
    ] = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": cfg.trials,
        "n_directions": (
            cfg.n_directions
        ),
        "transport": (
            cfg.transport
        ),
        "cond_Q": cond_Q,
        "nonnormality": (
            nonnormality
        ),
        "true_q1_q2_angle_deg": (
            q1_q2_angle
        ),
        "true_q1_q3_angle_deg": (
            q1_q3_angle
        ),
        "true_q2_q3_angle_deg": (
            q2_q3_angle
        ),
        "median_final_"
        "orthogonality_fro_error":
            float(
                final_rows[
                    "orthogonality_"
                    "fro_error"
                ].median()
            ),
    }

    metric_map = {
        "median_final_probe_1_vs_q1_deg":
            "probe_1_vs_q1_deg",
        "median_final_probe_2_vs_q2_deg":
            "probe_2_vs_q2_deg",
        "median_final_probe_3_vs_q3_deg":
            "probe_3_vs_q3_deg",
        "median_final_leading_2_subspace_error_deg":
            "leading_2_subspace_error_deg",
        "median_final_leading_3_subspace_error_deg":
            "leading_3_subspace_error_deg",
        "median_final_leading_5_subspace_error_deg":
            "leading_5_subspace_error_deg",
        "median_final_probe45_vs_true_rotation_plane_error_deg":
            "probe45_vs_true_rotation_plane_error_deg",
        "median_final_probe45_vs_qr_residual_plane_error_deg":
            "probe45_vs_qr_residual_plane_error_deg",
    }

    for (
        output_key,
        metric,
    ) in metric_map.items():
        if metric in final_rows.columns:
            final_summary[
                output_key
            ] = float(
                final_rows[
                    metric
                ].median()
            )

    pd.DataFrame(
        [final_summary]
    ).to_csv(
        outdir
        / "experiment_summary.csv",
        index=False,
    )

    print(
        "\nFinal median validation metrics:"
    )

    for (
        key,
        value,
    ) in final_summary.items():
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key}: "
                f"{value:.8e}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        "\nInterpretation:"
        "\n- probe 1 should still recover q1."
        "\n- probe 2 and probe 3 are NOT expected necessarily "
        "to equal q2 and q3 individually in a non-normal system."
        "\n- the main tests are whether the leading 2-D, 3-D, "
        "and 5-D invariant subspace errors approach zero."
        "\n- probe45_vs_true_rotation_plane_error_deg may remain "
        "nonzero because QR forces probes 4-5 to be orthogonal "
        "to the first three probes."
        "\n- probe45_vs_qr_residual_plane_error_deg tests the "
        "QR-compatible 2-D complement inside the leading 5-D subspace."
        "\n- matrix_free transport is Jacobian-free but still "
        "uses controlled perturbations; it is not observation-only."
    )

    print(
        "\nResults written to: "
        f"{outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
