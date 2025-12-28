# Jet Clustering Example

This example demonstrates the application of quantum optimization for jet clustering in particle physics using QUBO models and the Kaiwu quantum computing platform.

## Overview

This example implements jet clustering algorithms using:
- QUBO (Quadratic Unconstrained Binary Optimization) models
- Classical solvers (Simulated Annealing, KMeans, Random Search)
- Quantum solvers via Kaiwu platform

## Files Structure

- **Baseline solvers**: `baseline_sa.py`, `baseline_kmeans.py`, `baseline_random.py`, `baseline_kaiwu_sa.py`
- **QUBO models**: `qubo_model.py`, `real_qubo_model.py`
- **Analysis scripts**: `analyze_results.py`, `compare_solvers.py`, `make_paper_plots.py`
- **Plotting scripts**: `plot_energy_vs_time_sa_vs_kaiwu.py`, `plot_jet_efficiency_hist.py`
- **Data generation**: `generate_data.py`
- **Configuration**: `kaiwu_config.py`
- **Run scripts**: `run_real_event0_cpqc550.py`, `run_real_event1_cpqc550.py`, `run_real_event2_cpqc550.py`

## Results

Results are stored in the `results/` directory:
- `results_baseline.csv`
- `results_baseline_sa.csv`
- `results_baseline_random.csv`
- `results_kaiwu_sa.csv`

## Figures

Generated figures are in the `figures/` directory, including:
- Energy comparisons
- Efficiency histograms
- Pareto plots
- Flowcharts

## Documentation

Documentation files are in the `docs/` directory.

## Outputs

CPQC-550 outputs are stored in the `outputs/` directory.

