from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-15

@dataclass(frozen=True)
class Config:
    name: str
    A: np.ndarray
    x0: np.ndarray
    v0: np.ndarray
    steps: int
    window: int

def normalize(v):
    n = np.linalg.norm(v)
    if n <= EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n

def angle_deg(a, b):
    a, b = normalize(a), normalize(b)
    c = np.clip(abs(float(a @ b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def simulate(A, x0, v0, steps):
    X = np.empty((steps + 1, len(x0)))
    V = np.empty_like(X)
    X[0], V[0] = x0, v0
    for t in range(steps):
        X[t+1] = A @ X[t]
        V[t+1] = A @ V[t]
    return X, V

def pca_direction(errors):
    # Uncentered PCA: errors are already referenced to the fixed point L.
    _, _, vt = np.linalg.svd(errors, full_matrices=False)
    return normalize(vt[0])

def rolling_directions(X, L, window):
    U = np.full_like(X, np.nan, dtype=float)
    for t in range(window - 1, len(X)):
        U[t] = pca_direction(X[t-window+1:t+1] - L)
    return U

def run(cfg, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    L = np.zeros_like(cfg.x0)
    X, V = simulate(cfg.A, cfg.x0, cfg.v0, cfg.steps)
    Utrue = np.array([normalize(v) for v in V])
    Uhat = rolling_directions(X, L, cfg.window)

    rows = []
    for t in range(cfg.steps + 1):
        rows.append({
            "iteration": t,
            "x_norm": np.linalg.norm(X[t]),
            "v_norm": np.linalg.norm(V[t]),
            "true_direction_change_deg": np.nan if t == 0 else angle_deg(Utrue[t], Utrue[t-1]),
            "estimated_direction_change_deg":
                np.nan if t == 0 or np.isnan(Uhat[t]).any() or np.isnan(Uhat[t-1]).any()
                else angle_deg(Uhat[t], Uhat[t-1]),
            "estimation_error_deg":
                np.nan if np.isnan(Uhat[t]).any()
                else angle_deg(Uhat[t], Utrue[t]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "direction_metrics.csv", index=False)
    np.savez(outdir / "trajectories.npz", X=X, V=V, Utrue=Utrue, Uhat=Uhat, A=cfg.A)

    plots = [
        ("true_direction_change_deg", "01_true_direction_stability.png",
         "True transported perturbation direction change"),
        ("estimated_direction_change_deg", "02_estimated_direction_stability.png",
         "Estimated stable-direction change"),
        ("estimation_error_deg", "03_estimation_accuracy.png",
         "Estimated vs true perturbation direction"),
    ]
    for col, fname, title in plots:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_values = df[col].round(1)
        ax.plot(df["iteration"], plot_values)

        # Prevent misleading scientific-offset notation such as "1e-11 + 2e1"
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"{cfg.name}: {title}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)

    valid = df["estimation_error_deg"].dropna()
    return {
        "experiment": cfg.name,
        "final_estimation_error_deg": float(valid.iloc[-1]),
        "mean_last_10_estimation_error_deg": float(valid.tail(10).mean()),
    }

def configs(steps, window):
    x0 = np.array([1.0, 1.0])
    v0 = 1e-4 * np.array([1.0, 0.7])
    theta = np.deg2rad(20.0)
    r = 0.8
    R = r * np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    return {
        "strong_gap": Config("strong_gap", np.array([[0.8,0],[0,0.2]]), x0, v0, steps, window),
        "weak_gap": Config("weak_gap", np.array([[0.8,0],[0,0.75]]), x0, v0, steps, window),
        "equal_magnitude": Config("equal_magnitude", np.array([[0.8,0],[0,-0.8]]), x0, v0, steps, window),
        "rotation": Config("rotation", R, x0, v0, steps, window),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["strong_gap","weak_gap","equal_magnitude","rotation","all"], default="all")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results"))
    a = p.parse_args()

    if a.window < 2 or a.window > a.steps + 1:
        raise ValueError("Require 2 <= window <= steps + 1.")

    exps = configs(a.steps, a.window)
    names = list(exps) if a.experiment == "all" else [a.experiment]
    summaries = [run(exps[name], a.output / name) for name in names]
    pd.DataFrame(summaries).to_csv(a.output / "experiment_summary.csv", index=False)

    for s in summaries:
        print(s)
    print(f"Results: {a.output.resolve()}")

if __name__ == "__main__":
    main()
