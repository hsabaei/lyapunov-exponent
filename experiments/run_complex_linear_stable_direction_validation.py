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
    window: int = 20
    trials: int = 100
    seed: int = 42
    perturbation_scale: float = 1e-4


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """
    Sign-invariant angle between two 1-D directions.
    u and -u are treated as the same direction.
    """
    a = normalize(a)
    b = normalize(b)

    c = np.clip(abs(float(a @ b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


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
    Construct a real block-diagonal matrix B with:
      - one unique dominant real eigenvalue: 0.96
      - one nearby real eigenvalue: 0.92
      - one negative real eigenvalue: -0.88
      - several complex-conjugate pairs represented by 2x2 rotation blocks
      - remaining smaller real eigenvalues

    The dominant asymptotic eigendirection is the first coordinate of B.
    """
    if dim < 10:
        raise ValueError("Use dim >= 10 for this experiment.")

    blocks: list[np.ndarray] = [
        np.array([[0.96]]),
        np.array([[0.92]]),
        np.array([[-0.88]]),
        rotation_block(0.84, 25.0),
        rotation_block(0.72, 50.0),
        rotation_block(0.65, 80.0),
    ]

    used = sum(block.shape[0] for block in blocks)
    remaining = dim - used

    if remaining < 0:
        raise ValueError("Dimension too small for requested spectrum.")

    if remaining > 0:
        smaller = np.linspace(0.78, 0.20, remaining)
        blocks.extend(np.array([[lam]]) for lam in smaller)

    B = np.zeros((dim, dim), dtype=float)

    start = 0
    for block in blocks:
        size = block.shape[0]
        B[start : start + size, start : start + size] = block
        start += size

    return B


def build_nonnormal_matrix(
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Build A = Q B Q^{-1}, where:
      - B has a known unique dominant eigenvalue 0.96
      - Q is non-orthogonal
      - A is real and non-normal

    Returns:
      A       : system matrix
      q1      : exact dominant asymptotic direction in original coordinates
      cond_Q  : condition number of Q
    """
    B = build_real_block_diagonal_spectrum(dim)

    # Start with an orthogonal basis.
    G = rng.normal(size=(dim, dim))
    Q_orth, _ = np.linalg.qr(G)

    # Make the basis non-orthogonal but not extremely ill-conditioned.
    shear = np.eye(dim)

    for _ in range(dim * 2):
        i, j = rng.integers(0, dim, size=2)
        if i != j:
            shear[i, j] += rng.uniform(-0.35, 0.35)

    scaling = np.diag(np.linspace(0.7, 1.4, dim))
    Q = Q_orth @ shear @ scaling

    cond_Q = float(np.linalg.cond(Q))

    if cond_Q > 1e4:
        raise RuntimeError(
            f"Generated Q is too ill-conditioned: cond(Q)={cond_Q:.3e}"
        )

    A = Q @ B @ np.linalg.inv(Q)

    # Dominant eigenvector of B is e1.
    e1 = np.zeros(dim)
    e1[0] = 1.0

    q1 = normalize(Q @ e1)

    return A, q1, cond_Q


def simulate(
    A: np.ndarray,
    x0: np.ndarray,
    v0: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    x_{t+1} = A x_t
    v_{t+1} = A v_t
    """
    dim = len(x0)

    X = np.empty((steps + 1, dim), dtype=float)
    V = np.empty_like(X)

    X[0] = x0
    V[0] = v0

    for t in range(steps):
        X[t + 1] = A @ X[t]
        V[t + 1] = A @ V[t]

    return X, V


def uncentered_pca_direction(
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SVD of a window of error vectors.

    Returns:
      uhat          : first right singular vector
      singular_vals : all singular values
    """
    _, s, vt = np.linalg.svd(errors, full_matrices=False)

    uhat = normalize(vt[0])

    return uhat, s


def rolling_directions(
    X: np.ndarray,
    L: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate one direction per time using only observed X.

    Also return the fraction of window energy explained
    by the first singular direction:

        R_t = sigma_1^2 / sum_i sigma_i^2
    """
    n_steps, dim = X.shape

    Uhat = np.full((n_steps, dim), np.nan, dtype=float)
    explained = np.full(n_steps, np.nan, dtype=float)

    for t in range(window - 1, n_steps):
        errors = X[t - window + 1 : t + 1] - L

        uhat, s = uncentered_pca_direction(errors)

        Uhat[t] = uhat

        denom = float(np.sum(s**2))
        if denom > EPS:
            explained[t] = float((s[0] ** 2) / denom)

    return Uhat, explained


def analyze_trial(
    A: np.ndarray,
    q1: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    trial_id: int,
) -> pd.DataFrame:
    dim = cfg.dim

    x0 = rng.normal(size=dim)

    v0 = rng.normal(size=dim)
    v0 = cfg.perturbation_scale * normalize(v0)

    L = np.zeros(dim)

    X, V = simulate(
        A=A,
        x0=x0,
        v0=v0,
        steps=cfg.steps,
    )

    Utrue = np.array([normalize(v) for v in V])

    Uhat, explained = rolling_directions(
        X=X,
        L=L,
        window=cfg.window,
    )

    rows = []

    for t in range(cfg.steps + 1):
        true_change = np.nan
        est_change = np.nan
        est_vs_true = np.nan
        true_vs_q1 = np.nan
        est_vs_q1 = np.nan

        if t >= 1:
            true_change = angle_deg(
                Utrue[t],
                Utrue[t - 1],
            )

        true_vs_q1 = angle_deg(
            Utrue[t],
            q1,
        )

        if not np.isnan(Uhat[t]).any():
            est_vs_true = angle_deg(
                Uhat[t],
                Utrue[t],
            )

            est_vs_q1 = angle_deg(
                Uhat[t],
                q1,
            )

        if (
            t >= 1
            and not np.isnan(Uhat[t]).any()
            and not np.isnan(Uhat[t - 1]).any()
        ):
            est_change = angle_deg(
                Uhat[t],
                Uhat[t - 1],
            )

        rows.append(
            {
                "trial": trial_id,
                "iteration": t,
                "x_norm": np.linalg.norm(X[t]),
                "v_norm": np.linalg.norm(V[t]),
                "true_direction_change_deg": true_change,
                "estimated_direction_change_deg": est_change,
                "estimated_vs_true_deg": est_vs_true,
                "true_vs_q1_deg": true_vs_q1,
                "estimated_vs_q1_deg": est_vs_q1,
                "pc1_energy_fraction": explained[t],
            }
        )

    return pd.DataFrame(rows)


def summarize_by_iteration(
    all_df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "true_direction_change_deg",
        "estimated_direction_change_deg",
        "estimated_vs_true_deg",
        "true_vs_q1_deg",
        "estimated_vs_q1_deg",
        "pc1_energy_fraction",
    ]

    grouped = all_df.groupby("iteration")

    rows = []

    for iteration, group in grouped:
        row = {"iteration": iteration}

        for metric in metrics:
            values = group[metric].dropna()

            if len(values) == 0:
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_q25"] = np.nan
                row[f"{metric}_q75"] = np.nan
                continue

            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_q25"] = float(values.quantile(0.25))
            row[f"{metric}_q75"] = float(values.quantile(0.75))

        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    x = summary["iteration"]

    median = summary[f"{metric}_median"]
    q25 = summary[f"{metric}_q25"]
    q75 = summary[f"{metric}_q75"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Round only for visualization.
    median_plot = median.round(1)
    q25_plot = q25.round(1)
    q75_plot = q75.round(1)

    ax.plot(
        x,
        median_plot,
        label="Median",
    )

    ax.fill_between(
        x,
        q25_plot,
        q75_plot,
        alpha=0.2,
        label="25%-75%",
    )

    ax.ticklabel_format(
        axis="y",
        style="plain",
        useOffset=False,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/complex_linear"),
    )

    args = parser.parse_args()

    cfg = Config(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        trials=args.trials,
        seed=args.seed,
    )

    if cfg.window < 2:
        raise ValueError("window must be >= 2")

    if cfg.window > cfg.steps + 1:
        raise ValueError("window must be <= steps + 1")

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    A, q1, cond_Q = build_nonnormal_matrix(
        dim=cfg.dim,
        rng=rng,
    )

    nonnormality = float(
        np.linalg.norm(
            A.T @ A - A @ A.T,
            ord="fro",
        )
    )

    print("========================================")
    print("Complex linear stable-direction experiment")
    print("========================================")
    print(f"dimension: {cfg.dim}")
    print(f"steps: {cfg.steps}")
    print(f"window: {cfg.window}")
    print(f"trials: {cfg.trials}")
    print(f"cond(Q): {cond_Q:.6f}")
    print(f"non-normality ||A^T A - A A^T||_F: {nonnormality:.6f}")
    print("dominant eigenvalue magnitude: 0.96")
    print("second eigenvalue magnitude: 0.92")
    print("========================================")

    trial_frames = []

    for trial in range(cfg.trials):
        df_trial = analyze_trial(
            A=A,
            q1=q1,
            cfg=cfg,
            rng=rng,
            trial_id=trial,
        )

        trial_frames.append(df_trial)

    all_df = pd.concat(
        trial_frames,
        ignore_index=True,
    )

    all_df.to_csv(
        outdir / "all_trial_metrics.csv",
        index=False,
    )

    summary = summarize_by_iteration(all_df)

    summary.to_csv(
        outdir / "iteration_summary.csv",
        index=False,
    )

    np.savez(
        outdir / "system_definition.npz",
        A=A,
        q1=q1,
        cond_Q=cond_Q,
        nonnormality=nonnormality,
    )

    plot_metric(
        summary,
        metric="true_direction_change_deg",
        title="True transported perturbation direction change",
        ylabel="Angle (degrees)",
        output_path=outdir / "01_true_direction_stability.png",
    )

    plot_metric(
        summary,
        metric="estimated_direction_change_deg",
        title="Estimated direction change",
        ylabel="Angle (degrees)",
        output_path=outdir / "02_estimated_direction_stability.png",
    )

    plot_metric(
        summary,
        metric="estimated_vs_true_deg",
        title="Estimated direction vs true perturbation direction",
        ylabel="Angle error (degrees)",
        output_path=outdir / "03_estimated_vs_true.png",
    )

    plot_metric(
        summary,
        metric="true_vs_q1_deg",
        title="True perturbation direction vs known asymptotic direction q1",
        ylabel="Angle error (degrees)",
        output_path=outdir / "04_true_vs_q1.png",
    )

    plot_metric(
        summary,
        metric="estimated_vs_q1_deg",
        title="Estimated direction vs known asymptotic direction q1",
        ylabel="Angle error (degrees)",
        output_path=outdir / "05_estimated_vs_q1.png",
    )

    plot_metric(
        summary,
        metric="pc1_energy_fraction",
        title="Fraction of window energy explained by first PCA direction",
        ylabel="PC1 energy fraction",
        output_path=outdir / "06_pc1_energy_fraction.png",
    )

    final_rows = all_df[
        all_df["iteration"] == cfg.steps
    ].copy()

    final_summary = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "window": cfg.window,
        "trials": cfg.trials,
        "cond_Q": cond_Q,
        "nonnormality": nonnormality,
        "median_final_estimated_vs_true_deg":
            float(final_rows["estimated_vs_true_deg"].median()),
        "median_final_true_vs_q1_deg":
            float(final_rows["true_vs_q1_deg"].median()),
        "median_final_estimated_vs_q1_deg":
            float(final_rows["estimated_vs_q1_deg"].median()),
        "median_final_pc1_energy_fraction":
            float(final_rows["pc1_energy_fraction"].median()),
    }

    pd.DataFrame([final_summary]).to_csv(
        outdir / "experiment_summary.csv",
        index=False,
    )

    print("\nFinal median metrics:")
    for key, value in final_summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nResults written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()