# AeroNet Lite — Autonomous Drone Delivery Simulator

> A modular Python simulation of autonomous drone delivery featuring CSP layout validation, genetic algorithm fleet planning, A\* path routing, real-time disruption handling, demand forecasting, and anomaly detection.

**Team Members**
- Eman Ali — 23i-2564
- Mariam Shaiq — 23i-3250
- Fatima Siddiqa — 23i-2543

**Submitted to:** Ms. Mahnoor Tariq | **Date:** May 10, 2026

**GitHub:** https://github.com/eman-nu/AeroNet-Lite

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Module Reference](#module-reference)
- [Optional: Real Datasets](#optional-real-datasets)
- [Dashboard Visualization](#dashboard-visualization)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

AeroNet Lite simulates a 10×10 urban grid where autonomous drones are dispatched from hubs to complete deliveries. The system integrates six core modules:

| Module | File | Description |
|--------|------|-------------|
| 0 – City Grid | `grid_model.py` | 10×10 city with zones, hubs, chargers, medical pickups |
| 1 – CSP Validator | `layout_validator.py` | Four hard constraint checks on grid layout |
| 2 – Fleet Selector | `fleet_selector.py` | GA and brute-force drone fleet optimisation |
| 3 – A\* Planner | `astar_planner.py` | Shortest-path routing respecting no-fly zones |
| 4 – Simulator | `delivery_simulator.py` | 20-step real-time simulation with rerouting |
| 5 – ML Pipeline | `ml_pipeline.py` | Demand forecasting + anomaly classification |
| — Visualisation | `visualization.py` | Matplotlib dashboard (run via `main.py --vis`) |

---

## Repository Structure

```
AeroNet-Lite/
├── data/
│   └── train.csv                  ← Bike Sharing Demand dataset (Kaggle)
├── notebooks/
│   ├── demand_forecasting.ipynb   ← Module 5A: Demand Forecasting notebook
│   └── anomaly_classifier.ipynb   ← Module 5B: Anomaly Detection notebook
├── report/
│   └── final_report.docx          ← Project report
├── grid_model.py                  ← Shared data model
├── layout_validator.py            ← CSP rules R1–R4
├── fleet_selector.py              ← GA + brute-force fleet selection
├── astar_planner.py               ← A* path planning
├── delivery_simulator.py          ← 20-step simulation
├── ml_pipeline.py                 ← Demand forecasting + anomaly detection
├── visualization.py               ← Matplotlib dashboard
├── main.py                        ← Entry point
├── AeroNet_Lite_Simulation.ipynb  ← Colab-ready simulation notebook
├── aeronet_dashboard.png          ← Generated visualization output
└── README.md
```

---

## Architecture

```
main.py                  ← entry point; orchestrates all modules
│
├── grid_model.py        ← shared data model (Cell, grid builder, helpers)
├── layout_validator.py  ← CSP rules R1–R4
├── fleet_selector.py    ← Drone dataclass, GA, brute-force
├── astar_planner.py     ← A* search, Delivery dataclass, route chaining
├── delivery_simulator.py← 20-step simulation loop
├── ml_pipeline.py       ← sklearn forecasting + anomaly detection
└── visualization.py     ← matplotlib dashboard (optional)
```

---

## Prerequisites

- **Python 3.9 or later** (3.10+ recommended)
- **pip** (comes with Python)

### Required Python packages

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/eman-nu/AeroNet-Lite.git
cd AeroNet-Lite
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

Or using requirements file:

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import numpy, pandas, sklearn, matplotlib; print('All dependencies OK')"
```

---

## Running the Project

### Interactive Menu (Recommended)

```bash
python main.py
```

```
╔══════════════════════════════════════════════════════╗
║          AeroNet Lite  –  Main Menu                  ║
╠══════════════════════════════════════════════════════╣
║  1. Show city grid model                             ║
║  2. Run CSP layout validator                         ║
║  3. Run fleet selector  (GA)                         ║
║  4. Run fleet selector  (Brute-Force)                ║
║  5. Run A* path planner demo                         ║
║  6. Run 20-step simulation                           ║
║  7. Run ML pipeline (forecast + anomaly detection)   ║
║  8. Open visualization dashboard                     ║
║  9. Run ALL modules sequentially                     ║
║  0. Exit                                             ║
╚══════════════════════════════════════════════════════╝
```

### Command-Line Flags

| Flag | Description | Command |
|------|-------------|---------|
| *(none)* | Interactive menu | `python main.py` |
| `--full` | Run all modules sequentially | `python main.py --full` |
| `--sim` | 20-step simulation only | `python main.py --sim` |
| `--ml` | ML pipeline only | `python main.py --ml` |
| `--vis` | Open visualization dashboard | `python main.py --vis` |

### Running Modules Independently

```bash
python grid_model.py
python layout_validator.py
python fleet_selector.py
python astar_planner.py
python delivery_simulator.py
python ml_pipeline.py
```

### Running Notebooks

Open in VS Code or Jupyter:
```
notebooks/demand_forecasting.ipynb
notebooks/anomaly_classifier.ipynb
```

Or run in Google Colab using `AeroNet_Lite_Simulation.ipynb`.

---

## Module Reference

### Module 1 — CSP Validator

| Rule | Constraint |
|------|-----------|
| R1 | Industrial cells cannot be adjacent to Schools or Hospitals |
| R2 | Every Residential cell must be within 3 Manhattan cells of a Hub |
| R3 | Every Drone Hub must have a Charging Pad within 2 cells |
| R4 | At least one Hospital must have a Medical Pickup within 1 cell |

### Module 2 — Fleet Selector

| Drone | Cost | Payload | Range |
|-------|------|---------|-------|
| Light | $1,000 | 2.0 kg | 12 cells |
| Heavy | $1,800 | 5.0 kg | 20 cells |

### Module 3 — A\* Planner

| Zone | Move cost |
|------|----------|
| No-fly zone | ∞ (blocked) |
| Commercial | 0.8 (corridor discount) |
| All others | 1.0 |

### Module 5 — ML Pipeline

**Part A — Demand Forecasting** (Bike Sharing Demand dataset):
- Models: Linear Regression, Random Forest Regressor
- Best model MAE: **56.599** (Random Forest)

**Part B — Anomaly Detection:**
- Models: Decision Tree, Random Forest Classifier
- Best model Accuracy: **100%** (Random Forest)

---

## Optional: Real Datasets

### Bike Sharing Demand (Demand Forecasting)

1. Download `train.csv` from: https://www.kaggle.com/c/bike-sharing-demand
2. Place it at `data/train.csv`

The pipeline auto-detects and loads the real dataset. Without it, synthetic data is used as fallback.

---

## Dashboard Visualization

```bash
python main.py --vis
```

Produces a six-panel dashboard saved as `aeronet_dashboard.png`:
1. Zone Map
2. Route Map
3. Demand Heatmap
4. Anomaly Events
5. Fleet Status
6. Battery Levels

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `ModuleNotFoundError: matplotlib` | `pip install matplotlib` |
| `ModuleNotFoundError: seaborn` | `pip install seaborn` |
| `data/train.csv not found` | Expected — synthetic fallback activates automatically |
| Different results each run | Fixed seed 42 used — results are fully reproducible |

---

## Quick-Start Cheatsheet

```bash
# Clone
git clone https://github.com/eman-nu/AeroNet-Lite.git
cd AeroNet-Lite

# Install
pip install numpy pandas scikit-learn matplotlib seaborn

# Run everything
python main.py --full

# Run simulation only
python main.py --sim

# Run ML pipeline only
python main.py --ml
```
