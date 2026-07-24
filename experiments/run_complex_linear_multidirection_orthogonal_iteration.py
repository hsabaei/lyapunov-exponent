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
    steps: int = 300
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
    """Sign-invariant angle: a and -a represent the same direction."""
    a = normalize(a)
    b = normalize(b)
    c = np.clip(abs(float(a @ b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def max_principal_angle_deg(
    estimated_basis: np.ndarray,
    true_basis: np.ndarray,
) -> float:
    """Largest principal angle between equal-dimensional subspaces."""
    if estimated_basis.shape != true_basis.shape:
        raise ValueError("The two bases must have the same shape.")

    s = np.linalg.svd(
        true_basis.T @ estimated_basis,
        compute_uv=False,
    )
    s = np.clip(s, 0.0, 1.0)
    return float(np.max(np.degrees(np.arccos(s))))


def rotation_block(radius: float, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return radius * np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )


def build_real_block_diagonal_spectrum(dim: int) -> np.ndarray:
    """
    Leading modes:
      q1: eigenvalue  0.96
      q2: eigenvalue  0.92
      q3: eigenvalue -0.88
      span(q4,q5): contracting rotation 0.84 R(25 degrees)
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
    Build A = Q B Q^T with orthogonal Q.

    We deliberately start with a normal system because then q1, q2, q3
    are orthogonal, so recovery of the second and third eigendirections
    can be tested unambiguously.
    """
    B = build_real_block_diagonal_spectrum(dim)
    G = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(G)
    A = Q @ B @ Q.T

    normality_error = float(
        np.linalg.norm(A.T @ A - A @ A.T, ord="fro")
    )
    if normality_error > 1e-10:
        raise RuntimeError(
            f"A is not numerically normal: {normality_error:.3e}"
        )

    return A, Q, B


def initialize_orthonormal_probes(
    dim: int,
    n_directions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create random orthonormal probes.

    Column 2 is only the second probe. It is not assumed to be q2.
    We say q2 is recovered only if angle(probe_2, q2) tends to zero.
    """
    if n_directions > dim:
        raise ValueError("n_directions cannot exceed dim.")

    G = rng.normal(size=(dim, n_directions))
    probes, _ = np.linalg.qr(G, mode="reduced")
    return probes


def direct_transport(A: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Exact synthetic transport Z = A Q."""
    return A @ probes


def matrix_free_transport(
    A: np.ndarray,
    x_t: np.ndarray,
    probes: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transport each probe without forming a Jacobian:

        z_j = [U(x_t + epsilon w_j) - U(x_t)] / epsilon.

    Here U(x)=Ax. This still requires evaluating U at perturbed states;
    it cannot be obtained from one fixed recorded trajectory alone.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    def update_map(x: np.ndarray) -> np.ndarray:
        return A @ x

    x_next = update_map(x_t)
    transported = np.empty_like(probes)

    for j in range(probes.shape[1]):
        perturbed_next = update_map(
            x_t + epsilon * probes[:, j]
        )
        transported[:, j] = (
            perturbed_next - x_next
        ) / epsilon

    return x_next, transported


def enforce_orthogonality(
    transported: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    QR removes from each later probe its components along earlier probes.

    This prevents probe 2, probe 3, ... from all collapsing onto probe 1.
    """
    Q, R = np.linalg.qr(transported, mode="reduced")
    return Q, R


def run_trial(
    A: np.ndarray,
    true_basis: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    trial_id: int,
    save_history: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    k = cfg.n_directions

    probes = initialize_orthonormal_probes(
        dim=cfg.dim,
        n_directions=k,
        rng=rng,
    )
    x_t = rng.normal(size=cfg.dim)

    history = None
    if save_history:
        history = np.empty(
            (cfg.steps + 1, cfg.dim, k),
            dtype=float,
        )
        history[0] = probes

    rows: list[dict[str, float | int]] = []
    previous_probes: np.ndarray | None = None

    for t in range(cfg.steps + 1):
        row: dict[str, float | int] = {
            "trial": trial_id,
            "iteration": t,
        }

        gram_error = probes.T @ probes - np.eye(k)
        row["orthogonality_fro_error"] = float(
            np.linalg.norm(gram_error, ord="fro")
        )

        pairwise_deviations: list[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_angle = angle_deg(
                    probes[:, i],
                    probes[:, j],
                )
                row[
                    f"probe_{i + 1}_vs_probe_{j + 1}_deg"
                ] = pair_angle
                pairwise_deviations.append(
                    abs(pair_angle - 90.0)
                )

        row["max_pairwise_deviation_from_90_deg"] = (
            float(max(pairwise_deviations))
            if pairwise_deviations
            else 0.0
        )

        for j in range(k):
            if previous_probes is None:
                row[f"probe_{j + 1}_change_deg"] = np.nan
            else:
                row[f"probe_{j + 1}_change_deg"] = angle_deg(
                    probes[:, j],
                    previous_probes[:, j],
                )

        # Cross-angle table for the first three probes and q1,q2,q3.
        # The diagonal entries test actual recovery.
        n_fixed = min(3, k)
        for probe_index in range(n_fixed):
            for true_index in range(3):
                row[
                    f"probe_{probe_index + 1}_vs_q"
                    f"{true_index + 1}_deg"
                ] = angle_deg(
                    probes[:, probe_index],
                    true_basis[:, true_index],
                )

        for subspace_dim in (1, 2, 3):
            if subspace_dim <= k:
                row[
                    f"leading_{subspace_dim}_subspace_error_deg"
                ] = max_principal_angle_deg(
                    probes[:, :subspace_dim],
                    true_basis[:, :subspace_dim],
                )

        if k >= 5:
            # q4 and q5 need not stabilize individually because they rotate.
            # Their shared two-dimensional plane is the meaningful object.
            row["rotation_plane_error_deg"] = (
                max_principal_angle_deg(
                    probes[:, 3:5],
                    true_basis[:, 3:5],
                )
            )

        rows.append(row)

        if t == cfg.steps:
            break

        previous_probes = probes.copy()

        if cfg.transport == "direct":
            transported = direct_transport(A, probes)
            x_t = A @ x_t
        elif cfg.transport == "matrix_free":
            x_t, transported = matrix_free_transport(
                A=A,
                x_t=x_t,
                probes=probes,
                epsilon=cfg.epsilon,
            )
        else:
            raise ValueError(
                "transport must be 'direct' or 'matrix_free'."
            )

        # Key operation: propagate first, then orthogonalize again.
        probes, _ = enforce_orthogonality(transported)

        if history is not None:
            history[t + 1] = probes

    return pd.DataFrame(rows), history


def summarize_by_iteration(
    all_trials: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        c for c in all_trials.columns
        if c not in {"trial", "iteration"}
    ]

    rows: list[dict[str, float | int]] = []

    for iteration, group in all_trials.groupby("iteration"):
        row: dict[str, float | int] = {
            "iteration": int(iteration)
        }

        for metric in metric_columns:
            values = group[metric].dropna()
            if values.empty:
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q25"] = np.nan
                row[f"{metric}_q75"] = np.nan
            else:
                row[f"{metric}_median"] = float(values.median())
                row[f"{metric}_q25"] = float(
                    values.quantile(0.25)
                )
                row[f"{metric}_q75"] = float(
                    values.quantile(0.75)
                )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_summary_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    x = summary["iteration"]
    median = summary[f"{metric}_median"]
    q25 = summary[f"{metric}_q25"]
    q75 = summary[f"{metric}_q75"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, median, label="Median")
    ax.fill_between(
        x,
        q25,
        q75,
        alpha=0.2,
        label="25%-75%",
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate q1, q2, q3 recovery by repeated transport "
            "and QR orthogonalization."
        )
    )
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--n-directions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--transport",
        choices=("direct", "matrix_free"),
        default="matrix_free",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/complex_linear_multidirection_normal"
        ),
    )
    args = parser.parse_args()

    cfg = Config(
        dim=args.dim,
        steps=args.steps,
        trials=args.trials,
        n_directions=args.n_directions,
        seed=args.seed,
        epsilon=args.epsilon,
        transport=args.transport,
    )

    if cfg.n_directions < 1:
        raise ValueError("n_directions must be at least 1.")
    if cfg.n_directions > cfg.dim:
        raise ValueError("n_directions cannot exceed dim.")

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

    print("===================================================")
    print("Multiple-direction orthogonal-iteration validation")
    print("===================================================")
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"trials: {cfg.trials}")
    print(f"tracked probes: {cfg.n_directions}")
    print(f"transport: {cfg.transport}")
    print(f"normality error: {normality_error:.6e}")
    print("q1: 0.96")
    print("q2: 0.92")
    print("q3: -0.88")
    print("span(q4,q5): 0.84 R(25 degrees)")
    print("===================================================")

    trial_frames: list[pd.DataFrame] = []
    first_history: np.ndarray | None = None

    for trial_id in range(cfg.trials):
        trial_rng = np.random.default_rng(
            cfg.seed + 1000 + trial_id
        )
        trial_df, history = run_trial(
            A=A,
            true_basis=true_basis,
            cfg=cfg,
            rng=trial_rng,
            trial_id=trial_id,
            save_history=(trial_id == 0),
        )
        trial_frames.append(trial_df)
        if history is not None:
            first_history = history

    all_trials = pd.concat(trial_frames, ignore_index=True)
    all_trials.to_csv(
        outdir / "all_trial_metrics.csv",
        index=False,
    )

    summary = summarize_by_iteration(all_trials)
    summary.to_csv(
        outdir / "iteration_summary.csv",
        index=False,
    )

    np.savez(
        outdir / "system_definition.npz",
        A=A,
        B=B,
        true_basis=true_basis,
        normality_error=normality_error,
    )

    if first_history is not None:
        np.savez(
            outdir / "first_trial_probe_history.npz",
            probes=first_history,
            true_basis=true_basis,
        )

    for j in range(1, min(3, cfg.n_directions) + 1):
        metric = f"probe_{j}_vs_q{j}_deg"
        plot_summary_metric(
            summary=summary,
            metric=metric,
            title=f"Tracked probe {j} versus true direction q{j}",
            ylabel="Sign-invariant angle (degrees)",
            output_path=outdir / f"0{j}_probe_{j}_vs_q{j}.png",
        )

    for j in range(1, cfg.n_directions + 1):
        metric = f"probe_{j}_change_deg"
        plot_summary_metric(
            summary=summary,
            metric=metric,
            title=f"Iteration-to-iteration change of probe {j}",
            ylabel="Sign-invariant angle (degrees)",
            output_path=outdir / f"1{j}_probe_{j}_change.png",
        )

    if cfg.n_directions >= 5:
        plot_summary_metric(
            summary=summary,
            metric="rotation_plane_error_deg",
            title="Estimated rotating plane versus true span(q4,q5)",
            ylabel="Largest principal angle (degrees)",
            output_path=outdir / "20_rotation_plane_error.png",
        )

    plot_summary_metric(
        summary=summary,
        metric="orthogonality_fro_error",
        title="Numerical orthogonality error after QR",
        ylabel="||Q^T Q - I||_F",
        output_path=outdir / "21_orthogonality_error.png",
        log_scale=True,
    )

    final_rows = all_trials[
        all_trials["iteration"] == cfg.steps
    ]

    final_summary: dict[str, float | int | str] = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "trials": cfg.trials,
        "n_directions": cfg.n_directions,
        "transport": cfg.transport,
        "normality_error": normality_error,
        "median_final_orthogonality_fro_error": float(
            final_rows["orthogonality_fro_error"].median()
        ),
    }

    for j in range(1, min(3, cfg.n_directions) + 1):
        metric = f"probe_{j}_vs_q{j}_deg"
        final_summary[
            f"median_final_probe_{j}_vs_q{j}_deg"
        ] = float(final_rows[metric].median())

    if cfg.n_directions >= 5:
        final_summary[
            "median_final_rotation_plane_error_deg"
        ] = float(
            final_rows["rotation_plane_error_deg"].median()
        )

    pd.DataFrame([final_summary]).to_csv(
        outdir / "experiment_summary.csv",
        index=False,
    )

    print("\nFinal median validation metrics:")
    for key, value in final_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.8e}")
        else:
            print(f"{key}: {value}")

    print(
        "\nInterpretation:"
        "\n- probe 2 is not assumed to be q2."
        "\n- q2 is recovered only if probe_2_vs_q2_deg approaches zero."
        "\n- q4 and q5 may keep rotating individually."
        "\n- rotation_plane_error_deg tests their shared 2-D plane."
    )
    print(f"\nResults written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
