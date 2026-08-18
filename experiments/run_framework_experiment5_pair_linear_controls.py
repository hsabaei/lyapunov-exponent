#!/usr/bin/env python3
"""
Linear controls for Framework Experiment 5 pair structures.

Purpose
-------
Run ONLY beta=0 controls for the two pair structures used in Experiment 5:
  * rotation_pair
  * equal_magnitude_pair

This gives an apples-to-apples comparison against beta = 0.1, 0.5, 1.0 while
keeping the same structure, observation-only acceptance rule, estimator
hyperparameters, systems/states design, and external 2.5 degree tolerance.

The script intentionally does NOT rerun the 2400 nonlinear trajectories.
With the default design it adds only:
    2 pair structures x 10 systems x 20 initial states = 400 trajectories.

It imports the existing Experiment 5 implementation so the control uses exactly
the same synthetic systems, initial-state construction, filter logic, pair-U4
acceptance rule, validation metrics, and summaries.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PAIR_STRUCTURES = ("rotation_pair", "equal_magnitude_pair")
CONTROL_NAME = "linear_control"
CONTROL_BETA = 0.0


def _load_experiment5_module():
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "run_framework_experiment5_nonlinear_sequential.py",
        Path.cwd() / "experiments" / "run_framework_experiment5_nonlinear_sequential.py",
        Path.cwd() / "run_framework_experiment5_nonlinear_sequential.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        name = "framework_experiment5_base"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise ImportError(
        "Could not locate run_framework_experiment5_nonlinear_sequential.py.\n"
        f"Searched:\n{searched}"
    )


E5 = _load_experiment5_module()
ExperimentConfig = E5.ExperimentConfig


def run_controls(
    cfg: ExperimentConfig,
    output_dir: Path,
    nonlinear_results_dir: Path | None,
) -> None:
    E5.validate_config(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "time_traces"
    if cfg.save_trace_npz:
        trace_dir.mkdir(parents=True, exist_ok=True)

    trajectories_per_case = cfg.system_replicates * cfg.initial_states_per_system
    n_cases = len(PAIR_STRUCTURES)
    total_trajectories = n_cases * trajectories_per_case

    print("=" * 72)
    print("Framework Experiment 5: beta=0 linear controls for pair structures")
    print("=" * 72)
    print(f"structures: {PAIR_STRUCTURES}")
    print(f"beta: {CONTROL_BETA:g}")
    print(f"systems per case: {cfg.system_replicates}")
    print(f"initial states per system: {cfg.initial_states_per_system}")
    print(f"trajectories per case: {trajectories_per_case}")
    print(f"total trajectories: {total_trajectories}")
    print(f"window: {cfg.window}")
    print(f"external correctness tolerance: {cfg.recovery_tolerance_deg} degrees")
    print(
        "The same pair acceptance rule is used as in nonlinear Experiment 5; "
        "only beta is set to zero."
    )
    print("=" * 72)

    trajectory_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    system_rows_map: Dict[Tuple[str, str, int], Dict[str, object]] = {}

    completed = 0
    for structure_case in PAIR_STRUCTURES:
        case_name = f"{structure_case}__{CONTROL_NAME}"
        for system_replicate in range(cfg.system_replicates):
            for state_index in range(cfg.initial_states_per_system):
                trajectory_row, events, traces, system_row = E5.analyse_trajectory(
                    cfg=cfg,
                    structure_case=structure_case,
                    nonlinearity_name=CONTROL_NAME,
                    beta=CONTROL_BETA,
                    system_replicate=system_replicate,
                    state_index=state_index,
                )
                trajectory_rows.append(trajectory_row)
                event_rows.extend(events)
                system_rows_map[(structure_case, CONTROL_NAME, system_replicate)] = system_row

                if cfg.save_trace_npz:
                    np.savez_compressed(
                        trace_dir
                        / f"{case_name}__sys{system_replicate:02d}_state{state_index:02d}.npz",
                        **traces,
                    )
                completed += 1
            print(
                f"completed {case_name}: system {system_replicate + 1}/{cfg.system_replicates}; "
                f"{completed}/{total_trajectories} trajectories"
            )

    trajectories = pd.DataFrame(trajectory_rows)
    events = pd.DataFrame(event_rows)
    systems = pd.DataFrame(list(system_rows_map.values()))

    if len(trajectories) != total_trajectories:
        raise RuntimeError(
            f"Expected {total_trajectories} control trajectories, got {len(trajectories)}"
        )
    expected_events = total_trajectories * cfg.n_directions
    if len(events) != expected_events:
        raise RuntimeError(f"Expected {expected_events} event rows, got {len(events)}")

    trajectories.to_csv(output_dir / "all_trajectories.csv", index=False)
    events.to_csv(output_dir / "trajectory_stage_recovery_events.csv", index=False)
    systems.to_csv(output_dir / "systems.csv", index=False)

    primary = E5.summarize_primary(events, cfg)
    intervals = E5.summarize_recovery_intervals(events)
    nonlinear_summary = E5.summarize_nonlinearity(trajectories)
    tangent = E5.summarize_tangent_diagnostics(events)

    primary.to_csv(output_dir / "table_control_primary_performance.csv", index=False)
    intervals.to_csv(output_dir / "table_control_recovery_intervals.csv", index=False)
    nonlinear_summary.to_csv(output_dir / "table_control_jacobian_summary.csv", index=False)
    tangent.to_csv(output_dir / "table_control_tangent_diagnostics.csv", index=False)

    config = asdict(cfg)
    config.update(
        {
            "control_beta": CONTROL_BETA,
            "control_name": CONTROL_NAME,
            "pair_structures": list(PAIR_STRUCTURES),
            "trajectories_per_case": trajectories_per_case,
            "total_control_trajectories": total_trajectories,
        }
    )
    (output_dir / "control_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Optional comparison with the already-completed nonlinear run.
    if nonlinear_results_dir is not None:
        nonlinear_events_path = nonlinear_results_dir / "trajectory_stage_recovery_events.csv"
        nonlinear_trajectories_path = nonlinear_results_dir / "all_trajectories.csv"
        if nonlinear_events_path.exists():
            nonlinear_events = pd.read_csv(nonlinear_events_path)
            nonlinear_events = nonlinear_events[
                nonlinear_events["structure_case"].isin(PAIR_STRUCTURES)
            ].copy()
            combined_events = pd.concat([events, nonlinear_events], ignore_index=True)

            comparison_primary = E5.summarize_primary(combined_events, cfg)
            comparison_primary = comparison_primary[
                comparison_primary["structure_case"].isin(PAIR_STRUCTURES)
            ].sort_values(["structure_case", "beta", "stage"])
            comparison_primary.to_csv(
                output_dir / "table_pair_beta0_to_1_primary_comparison.csv", index=False
            )

            comparison_intervals = E5.summarize_recovery_intervals(combined_events)
            comparison_intervals = comparison_intervals[
                comparison_intervals["structure_case"].isin(PAIR_STRUCTURES)
            ].sort_values(["structure_case", "beta", "stage"])
            comparison_intervals.to_csv(
                output_dir / "table_pair_beta0_to_1_interval_comparison.csv", index=False
            )

        if nonlinear_trajectories_path.exists():
            nonlinear_traj = pd.read_csv(nonlinear_trajectories_path)
            nonlinear_traj = nonlinear_traj[
                nonlinear_traj["structure_case"].isin(PAIR_STRUCTURES)
            ].copy()
            combined_traj = pd.concat([trajectories, nonlinear_traj], ignore_index=True)
            comparison_jac = E5.summarize_nonlinearity(combined_traj)
            comparison_jac = comparison_jac[
                comparison_jac["structure_case"].isin(PAIR_STRUCTURES)
            ].sort_values(["structure_case", "beta"])
            comparison_jac.to_csv(
                output_dir / "table_pair_beta0_to_1_jacobian_comparison.csv", index=False
            )

    print("\nGenerated linear pair-control results in:")
    print(output_dir.resolve())
    print("Main control files:")
    print("  table_control_primary_performance.csv")
    print("  table_control_recovery_intervals.csv")
    print("  table_control_jacobian_summary.csv")
    if nonlinear_results_dir is not None:
        print("Comparison files (if nonlinear results were found):")
        print("  table_pair_beta0_to_1_primary_comparison.csv")
        print("  table_pair_beta0_to_1_interval_comparison.csv")
        print("  table_pair_beta0_to_1_jacobian_comparison.csv")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run beta=0 linear controls for Experiment 5 rotation/equal-magnitude pairs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/framework_experiment5_pair_linear_controls"),
    )
    parser.add_argument(
        "--nonlinear-results-dir",
        type=Path,
        default=Path("results/framework_experiment5_nonlinear_sequential"),
        help="Existing Experiment 5 nonlinear results; used only to make beta=0..1 comparison tables.",
    )
    parser.add_argument("--dim", type=int, default=ExperimentConfig.dim)
    parser.add_argument("--steps", type=int, default=ExperimentConfig.steps)
    parser.add_argument("--window", type=int, default=ExperimentConfig.window)
    parser.add_argument("--systems", type=int, default=ExperimentConfig.system_replicates)
    parser.add_argument(
        "--initial-states", type=int, default=ExperimentConfig.initial_states_per_system
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--bootstrap", type=int, default=ExperimentConfig.bootstrap_replicates)
    parser.add_argument("--no-traces", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        dim=args.dim,
        steps=args.steps,
        window=args.window,
        system_replicates=args.systems,
        initial_states_per_system=args.initial_states,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap,
        save_trace_npz=not args.no_traces,
    )
    run_controls(cfg, args.output, args.nonlinear_results_dir)


if __name__ == "__main__":
    main()
