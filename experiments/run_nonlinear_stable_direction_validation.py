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
    steps: int = 400
    window: int = 20
    trials: int = 100
    seed: int = 42
    beta: float = 0.10
    perturbation_scale: float = 1e-4
    finite_difference_scale: float = 1e-7


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
    Real block-diagonal matrix with:
      - one unique dominant real eigenvalue: 0.96
      - nearby real eigenvalue: 0.92
      - negative real eigenvalue: -0.88
      - several complex-conjugate pairs represented by 2x2 blocks
      - smaller remaining modes
    """
    if dim < 10:
        raise ValueError("Use dim >= 10.")

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
        B[start:start + size, start:start + size] = block
        start += size

    return B


def build_nonnormal_linear_part(
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Construct A = Q B Q^{-1}.

    q1 is the known dominant direction corresponding to eigenvalue 0.96.
    """
    B = build_real_block_diagonal_spectrum(dim)

    G = rng.normal(size=(dim, dim))
    Q_orth, _ = np.linalg.qr(G)

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

    e1 = np.zeros(dim)
    e1[0] = 1.0

    q1 = normalize(Q @ e1)

    return A, q1, cond_Q


def scale_to_spectral_norm(
    M: np.ndarray,
    target_norm: float = 1.0,
) -> np.ndarray:
    current = np.linalg.norm(M, ord=2)

    if current <= EPS:
        raise ValueError("Cannot scale a near-zero matrix.")

    return M * (target_norm / current)


def build_nonlinear_matrices(
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random fixed matrices B and C used in

        g(x) = tanh(Bx) ⊙ tanh(Cx)

    They are scaled to moderate spectral norm.
    """
    B = rng.normal(size=(dim, dim))
    C = rng.normal(size=(dim, dim))

    B = scale_to_spectral_norm(B, 1.0)
    C = scale_to_spectral_norm(C, 1.0)

    return B, C


def nonlinear_term(
    x: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    a = np.tanh(B @ x)
    b = np.tanh(C @ x)

    return a * b


def map_U(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beta: float,
) -> np.ndarray:
    return A @ x + beta * nonlinear_term(x, B, C)


def jacobian_U(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Exact Jacobian of:

        U(x) = A x + beta [tanh(Bx) ⊙ tanh(Cx)]

    Let:
        a = tanh(Bx)
        b = tanh(Cx)

    Then:
        Dg(x)
        =
        diag(b) diag(1-a^2) B
        +
        diag(a) diag(1-b^2) C
    """
    a = np.tanh(B @ x)
    b = np.tanh(C @ x)

    da = 1.0 - a**2
    db = 1.0 - b**2

    Dg = (
        np.diag(b * da) @ B
        +
        np.diag(a * db) @ C
    )

    return A + beta * Dg


def simulate_nonlinear(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    beta: float,
    x0: np.ndarray,
    v0: np.ndarray,
    steps: int,
    finite_difference_scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Simulate:

        x_{t+1} = U(x_t)

    Exact infinitesimal perturbation:

        v_{t+1} = J_{x_t} v_t

    Also simulate a nearby nonlinear trajectory:

        x_tilde_0 = x_0 + eps * normalize(v0)

    to validate tangent transport.

    Returns:
      X
      V
      Xtilde
      Jdist
    """
    dim = len(x0)

    X = np.empty((steps + 1, dim), dtype=float)
    V = np.empty_like(X)
    Xtilde = np.empty_like(X)
    Jdist = np.empty(steps + 1, dtype=float)

    L = np.zeros(dim)
    J_L = jacobian_U(
        L,
        A,
        B,
        C,
        beta,
    )

    X[0] = x0
    V[0] = v0

    Xtilde[0] = (
        x0
        + finite_difference_scale * normalize(v0)
    )

    denom = np.linalg.norm(J_L, ord="fro")

    for t in range(steps + 1):
        Jx = jacobian_U(
            X[t],
            A,
            B,
            C,
            beta,
        )

        Jdist[t] = (
            np.linalg.norm(
                Jx - J_L,
                ord="fro",
            )
            / max(denom, EPS)
        )

        if t == steps:
            break

        V[t + 1] = Jx @ V[t]

        X[t + 1] = map_U(
            X[t],
            A,
            B,
            C,
            beta,
        )

        Xtilde[t + 1] = map_U(
            Xtilde[t],
            A,
            B,
            C,
            beta,
        )

    return X, V, Xtilde, Jdist


def uncentered_pca_direction(
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, s, vt = np.linalg.svd(
        errors,
        full_matrices=False,
    )

    return normalize(vt[0]), s


def rolling_directions(
    X: np.ndarray,
    L: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps, dim = X.shape

    Uhat = np.full(
        (n_steps, dim),
        np.nan,
        dtype=float,
    )

    explained = np.full(
        n_steps,
        np.nan,
        dtype=float,
    )

    for t in range(window - 1, n_steps):
        errors = (
            X[t - window + 1:t + 1]
            - L
        )

        uhat, s = uncentered_pca_direction(
            errors
        )

        Uhat[t] = uhat

        denom = float(np.sum(s**2))

        if denom > EPS:
            explained[t] = float(
                (s[0] ** 2) / denom
            )

    return Uhat, explained


def analyze_trial(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    q1: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    trial_id: int,
) -> pd.DataFrame:
    dim = cfg.dim

    # Start farther from the limit so nonlinear effects are stronger.
    initial_radius = 2.0

    x0 = initial_radius * normalize(
        rng.normal(size=dim)
    )

    v0 = cfg.perturbation_scale * normalize(
        rng.normal(size=dim)
    )

    L = np.zeros(dim)

    X, V, Xtilde, Jdist = simulate_nonlinear(
        A=A,
        B=B,
        C=C,
        beta=cfg.beta,
        x0=x0,
        v0=v0,
        steps=cfg.steps,
        finite_difference_scale=cfg.finite_difference_scale,
    )

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"Non-finite trajectory in trial {trial_id}."
        )

    # Reject trajectories that clearly diverge.
    max_norm = float(np.max(np.linalg.norm(X, axis=1)))
    final_norm = float(np.linalg.norm(X[-1]))

    if max_norm > 1e4:
        raise RuntimeError(
            f"Trial {trial_id} diverged: max ||x_t|| = {max_norm:.3e}"
        )

    # Require convergence close to L by the end.
    if final_norm > 1e-4:
        raise RuntimeError(
            f"Trial {trial_id} did not converge sufficiently: "
            f"final ||x_T|| = {final_norm:.3e}"
        )

    Utrue = np.array(
        [normalize(v) for v in V]
    )

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
        tangent_vs_finite = np.nan

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

        finite_delta = Xtilde[t] - X[t]
        delta_norm = np.linalg.norm(finite_delta)

        scale = max(
            np.linalg.norm(X[t]),
            np.linalg.norm(Xtilde[t]),
            1.0,
        )

        if delta_norm > 100.0 * np.finfo(float).eps * scale:
            tangent_vs_finite = angle_deg(
                Utrue[t],
                finite_delta,
            )

        rows.append(
            {
                "trial": trial_id,
                "iteration": t,
                "x_norm": float(
                    np.linalg.norm(X[t])
                ),
                "v_norm": float(
                    np.linalg.norm(V[t])
                ),
                "jacobian_relative_difference":
                    float(Jdist[t]),
                "true_direction_change_deg":
                    true_change,
                "estimated_direction_change_deg":
                    est_change,
                "estimated_vs_true_deg":
                    est_vs_true,
                "true_vs_q1_deg":
                    true_vs_q1,
                "estimated_vs_q1_deg":
                    est_vs_q1,
                "pc1_energy_fraction":
                    explained[t],
                "tangent_vs_finite_difference_deg":
                    tangent_vs_finite,
            }
        )

    return pd.DataFrame(rows)


def summarize_by_iteration(
    all_df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "x_norm",
        "jacobian_relative_difference",
        "true_direction_change_deg",
        "estimated_direction_change_deg",
        "estimated_vs_true_deg",
        "true_vs_q1_deg",
        "estimated_vs_q1_deg",
        "pc1_energy_fraction",
        "tangent_vs_finite_difference_deg",
    ]

    rows = []

    for iteration, group in all_df.groupby(
        "iteration"
    ):
        row = {
            "iteration": iteration
        }

        for metric in metrics:
            values = group[metric].dropna()

            if len(values) == 0:
                row[
                    f"{metric}_median"
                ] = np.nan
                row[
                    f"{metric}_q25"
                ] = np.nan
                row[
                    f"{metric}_q75"
                ] = np.nan
                continue

            row[
                f"{metric}_median"
            ] = float(values.median())

            row[
                f"{metric}_q25"
            ] = float(
                values.quantile(0.25)
            )

            row[
                f"{metric}_q75"
            ] = float(
                values.quantile(0.75)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    round_digits: int | None = None,
) -> None:
    x = summary["iteration"]

    median = summary[
        f"{metric}_median"
    ]

    q25 = summary[
        f"{metric}_q25"
    ]

    q75 = summary[
        f"{metric}_q75"
    ]

    if round_digits is not None:
        median = median.round(
            round_digits
        )
        q25 = q25.round(
            round_digits
        )
        q75 = q75.round(
            round_digits
        )

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
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def run_experiment(
    cfg: Config,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        cfg.seed
    )

    A, q1, cond_Q = (
        build_nonnormal_linear_part(
            cfg.dim,
            rng,
        )
    )

    B, C = build_nonlinear_matrices(
        cfg.dim,
        rng,
    )

    L = np.zeros(cfg.dim)

    J_L = jacobian_U(
        L,
        A,
        B,
        C,
        cfg.beta,
    )

    # Since Dg(0)=0, this should equal A numerically.
    jacobian_at_limit_error = float(
        np.linalg.norm(
            J_L - A,
            ord="fro",
        )
    )

    nonnormality = float(
        np.linalg.norm(
            A.T @ A - A @ A.T,
            ord="fro",
        )
    )

    print(
        "========================================"
    )
    print(
        "Nonlinear stable-direction experiment"
    )
    print(
        "========================================"
    )
    print(
        f"dimension: {cfg.dim}"
    )
    print(
        f"steps: {cfg.steps}"
    )
    print(
        f"window: {cfg.window}"
    )
    print(
        f"trials: {cfg.trials}"
    )
    print(
        f"beta: {cfg.beta}"
    )
    print(
        f"cond(Q): {cond_Q:.6f}"
    )
    print(
        "non-normality "
        f"||A^T A - A A^T||_F: "
        f"{nonnormality:.6f}"
    )
    print(
        "||J_L - A||_F: "
        f"{jacobian_at_limit_error:.6e}"
    )
    print(
        "dominant eigenvalue magnitude: 0.96"
    )
    print(
        "second eigenvalue magnitude: 0.92"
    )
    print(
        "========================================"
    )

    trial_frames = []

    for trial in range(
        cfg.trials
    ):
        df_trial = analyze_trial(
            A=A,
            B=B,
            C=C,
            q1=q1,
            cfg=cfg,
            rng=rng,
            trial_id=trial,
        )

        trial_frames.append(
            df_trial
        )

    all_df = pd.concat(
        trial_frames,
        ignore_index=True,
    )

    all_df.to_csv(
        output_dir
        / "all_trial_metrics.csv",
        index=False,
    )

    summary = summarize_by_iteration(
        all_df
    )

    summary.to_csv(
        output_dir
        / "iteration_summary.csv",
        index=False,
    )

    np.savez(
        output_dir
        / "system_definition.npz",
        A=A,
        B=B,
        C=C,
        q1=q1,
        J_L=J_L,
        beta=cfg.beta,
        cond_Q=cond_Q,
        nonnormality=nonnormality,
    )

    plot_metric(
        summary,
        metric="x_norm",
        title="Distance to the limit",
        ylabel="||x_t - L||",
        output_path=(
            output_dir
            / "01_distance_to_limit.png"
        ),
    )

    plot_metric(
        summary,
        metric=(
            "jacobian_relative_difference"
        ),
        title=(
            "Convergence of local Jacobian "
            "to J_L"
        ),
        ylabel=(
            "||J_xt - J_L||_F / ||J_L||_F"
        ),
        output_path=(
            output_dir
            / "02_jacobian_convergence.png"
        ),
    )

    plot_metric(
        summary,
        metric=(
            "true_direction_change_deg"
        ),
        title=(
            "True transported perturbation "
            "direction change"
        ),
        ylabel="Angle (degrees)",
        output_path=(
            output_dir
            / "03_true_direction_stability.png"
        ),
        round_digits=1,
    )

    plot_metric(
        summary,
        metric=(
            "estimated_direction_change_deg"
        ),
        title=(
            "Estimated direction change"
        ),
        ylabel="Angle (degrees)",
        output_path=(
            output_dir
            / "04_estimated_direction_stability.png"
        ),
        round_digits=1,
    )

    plot_metric(
        summary,
        metric="estimated_vs_true_deg",
        title=(
            "Estimated direction vs true "
            "perturbation direction"
        ),
        ylabel="Angle error (degrees)",
        output_path=(
            output_dir
            / "05_estimated_vs_true.png"
        ),
        round_digits=1,
    )

    plot_metric(
        summary,
        metric="true_vs_q1_deg",
        title=(
            "True perturbation direction vs "
            "known asymptotic direction q1"
        ),
        ylabel="Angle error (degrees)",
        output_path=(
            output_dir
            / "06_true_vs_q1.png"
        ),
        round_digits=1,
    )

    plot_metric(
        summary,
        metric="estimated_vs_q1_deg",
        title=(
            "Estimated direction vs known "
            "asymptotic direction q1"
        ),
        ylabel="Angle error (degrees)",
        output_path=(
            output_dir
            / "07_estimated_vs_q1.png"
        ),
        round_digits=1,
    )

    plot_metric(
        summary,
        metric="pc1_energy_fraction",
        title=(
            "Fraction of window energy explained "
            "by first PCA direction"
        ),
        ylabel="PC1 energy fraction",
        output_path=(
            output_dir
            / "08_pc1_energy_fraction.png"
        ),
        round_digits=3,
    )

    plot_metric(
        summary,
        metric=(
            "tangent_vs_finite_difference_deg"
        ),
        title=(
            "Jacobian-transported perturbation vs "
            "actual nearby-trajectory difference"
        ),
        ylabel="Angle error (degrees)",
        output_path=(
            output_dir
            / "09_tangent_vs_finite_difference.png"
        ),
        round_digits=2,
    )

    final_rows = all_df[
        all_df["iteration"]
        == cfg.steps
    ].copy()

    final_summary = {
        "dim": cfg.dim,
        "steps": cfg.steps,
        "window": cfg.window,
        "trials": cfg.trials,
        "beta": cfg.beta,
        "cond_Q": cond_Q,
        "nonnormality": nonnormality,
        "jacobian_at_limit_error":
            jacobian_at_limit_error,
        "median_final_x_norm":
            float(
                final_rows[
                    "x_norm"
                ].median()
            ),
        "median_final_jacobian_relative_difference":
            float(
                final_rows[
                    "jacobian_relative_difference"
                ].median()
            ),
        "median_final_estimated_vs_true_deg":
            float(
                final_rows[
                    "estimated_vs_true_deg"
                ].median()
            ),
        "median_final_true_vs_q1_deg":
            float(
                final_rows[
                    "true_vs_q1_deg"
                ].median()
            ),
        "median_final_estimated_vs_q1_deg":
            float(
                final_rows[
                    "estimated_vs_q1_deg"
                ].median()
            ),
        "median_final_pc1_energy_fraction":
            float(
                final_rows[
                    "pc1_energy_fraction"
                ].median()
            ),
        "median_final_tangent_vs_finite_difference_deg":
            float(
                final_rows[
                    "tangent_vs_finite_difference_deg"
                ].median()
            ),
    }

    pd.DataFrame(
        [final_summary]
    ).to_csv(
        output_dir
        / "experiment_summary.csv",
        index=False,
    )

    print(
        "\nFinal median metrics:"
    )

    for key, value in final_summary.items():
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key}: {value:.8f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print(
        "\nResults written to: "
        f"{output_dir.resolve()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dim",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=20,
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
        "--beta",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--finite-difference-scale",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/nonlinear"
        ),
    )

    args = parser.parse_args()

    cfg = Config(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        trials=args.trials,
        seed=args.seed,
        beta=args.beta,
        finite_difference_scale=(
            args.finite_difference_scale
        ),
    )

    if cfg.window < 2:
        raise ValueError(
            "window must be >= 2"
        )

    if cfg.window > cfg.steps + 1:
        raise ValueError(
            "window must be <= steps + 1"
        )

    run_experiment(
        cfg=cfg,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()