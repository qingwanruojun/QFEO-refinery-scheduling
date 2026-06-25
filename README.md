# QFEO — QUBO-based Flexibility-Enhanced Optimization for Refinery Scheduling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A block-QUBO annealing heuristic for long-horizon refinery scheduling with near-zero mode switching.

## Key Result

QFEO matches the **exact MILP optimum (Gurobi)** across all seven tested industrial scenarios while producing **1.0 average mode switches** versus **59–166 for heuristic baselines** (GA, PSO, MTCEA, RL).

## Features

- **Block-wise QUBO formulation** — decomposes 360-day horizon into 30-day blocks
- **Smoothness coupling penalty** — suppresses unnecessary mode switches within QUBO
- **D-Wave PathIntegralAnnealingSampler** — quantum Monte Carlo annealing simulation
- **Multi-seed screening** — 6 outer iterations with 10 random seeds
- **Flexibility-enhanced mode** ($\phi=1.12$) — models throughput–cost trade-offs
- **7 scenarios** — Baseline, Supply_Surge, Demand_Crisis, Surplus, Tank_Light, Tank_Tight, Tank_Hard

## Requirements

```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
scipy>=1.7
dwave-ocean-sdk>=6.0
gurobipy>=9.0    (optional, for MILP baseline)
```

## Quick Start

```python
from qfeo_scheduler import init_data, run_experiment

# Initialize with real Brent price data
init_data(use_real_brent=True, T_days=360)

# Run main experiment (7 scenarios × 6 algorithms)
results = run_experiment(n_seeds=3)  # use 10 for publication
```

To reproduce the paper figures:

```python
from qfeo_scheduler import generate_paper_figures
generate_paper_figures(csv_path="results_main.csv", output_dir=".")
```

## Project Structure

```
QFEO_repo/
├── qfeo_scheduler.py      # Main experiment code
├── requirements.txt       # Python dependencies
├── LICENSE                # MIT License
└── README.md              # This file
```

## Paper

The accompanying paper *"A Block-QUBO Annealing Heuristic for Long-Horizon Refinery Scheduling with Near-Zero Mode Switching"* has been submitted to *Processes* (MDPI).

## Citation

```bibtex
@article{duan2026qfeo,
  title={A Block-QUBO Annealing Heuristic for Long-Horizon Refinery Scheduling with Near-Zero Mode Switching},
  author={Duan, Yingjun and He, Yuxin and Wang, Yukun},
  journal={Processes},
  year={2026}
}
```

## License

MIT
