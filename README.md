# Lyapunov Exponent

Controlled validation of an observation-only stable-direction estimator.

## Goal

Test whether a direction estimated only from a late window of the base trajectory
approximates the exact transported perturbation direction

\[
u_t^{true}=v_t/\|v_t\|,\qquad v_{t+1}=A v_t.
\]

The estimated direction is obtained using **uncentered PCA** on late error vectors
\(e_t=x_t-L\). The perturbed trajectory is used only as ground truth.

## Experiments

- `strong_gap`: eigenvalues 0.8 and 0.2 — expected fast stabilization.
- `weak_gap`: eigenvalues 0.8 and 0.75 — expected slow stabilization.
- `equal_magnitude`: eigenvalues 0.8 and -0.8 — expected failure of a unique stable direction.
- `rotation`: contraction plus rotation — expected failure of the fixed-direction assumption.

## Run

```bash
pip install -r requirements.txt
python experiments/run_stable_direction_validation.py --experiment all
```

Or start with the simplest positive control:

```bash
python experiments/run_stable_direction_validation.py --experiment strong_gap --steps 60 --window 10
```

Results are written to `results/`.

## Outputs

Each experiment produces:

1. `01_true_direction_stability.png`
2. `02_estimated_direction_stability.png`
3. `03_estimation_accuracy.png`
4. `direction_metrics.csv`
5. `trajectories.npz`

The main validation metric is the sign-invariant subspace angle

\[
\theta_t=\arccos\left(|\hat u_t^T u_t^{true}|\right).
\]

This repository does **not** assume PCA can recover an arbitrary perturbation in general.
It tests a narrower hypothesis: near a fixed point with a dominant real spectral mode,
late state errors and a generic transported perturbation may align with the same 1-D subspace.
