import sys
import argparse
import random

from grid_model import (
    build_default_grid, print_grid_summary,
    hub_cells, charging_cells, medical_cells,
)
from layout_validator import validate_layout, print_validation_report
from fleet_selector import (
    ga_select, brute_force_select, create_fleet, print_fleet_report,
)
from astar_planner import (
    plan_delivery_route, generate_deliveries, print_route,
)
from delivery_simulator import run_simulation
from ml_pipeline import run_forecast, run_anomaly_detection

def demo_grid():
    """Module 0 – Build and display the 10×10 city grid."""
    print("\n" + "=" * 65)
    print("  MODULE 0 – CITY GRID MODEL")
    print("=" * 65)
    grid = build_default_grid()
    print_grid_summary(grid)

    hubs     = hub_cells(grid)
    chargers = charging_cells(grid)
    medicals = medical_cells(grid)
    print(f"  Hubs           : {len(hubs)}   at {[(h.row, h.col) for h in hubs]}")
    print(f"  Charging pads  : {len(chargers)}  at {[(c.row, c.col) for c in chargers]}")
    print(f"  Medical pickups: {len(medicals)} at {[(m.row, m.col) for m in medicals]}")
    return grid


def demo_validation(grid=None):
    """Module 1 – CSP layout validation."""
    if grid is None:
        grid = build_default_grid()
    print("\n" + "=" * 65)
    print("  MODULE 1 – CSP LAYOUT VALIDATOR")
    print("=" * 65)

    print("  [Test 1] Valid grid (all constraints satisfied by design)")
    report = validate_layout(grid)
    print_validation_report(report)

    print("  [Test 2] Invalid grid (forced R1 + R3 violations)")
    from grid_model import build_default_grid as _build
    broken_grid = _build()
    broken_grid[0][3].zone = "School"       # R1: School now adjacent to Industrial at (0,4)
    broken_grid[1][2].is_charging = False   # R3: removes charging pad near hub (1,0)
    broken_report = validate_layout(broken_grid)
    print_validation_report(broken_report)

    return report


def demo_fleet_selection(strategy: str = "ga", total_demand: float = 60.0):
    """Module 2 – Fleet selection (GA or brute-force)."""
    print("\n" + "=" * 65)
    print("  MODULE 2 – FLEET SELECTOR")
    print("=" * 65)
    if strategy == "brute":
        print("  Strategy: Brute-Force exhaustive search")
        n_l, n_h, score = brute_force_select(total_demand=total_demand)
        print_fleet_report(n_l, n_h, score, strategy="Brute-Force")
    else:
        print("  Strategy: Genetic Algorithm")
        n_l, n_h, score = ga_select(total_demand=total_demand)
        print_fleet_report(n_l, n_h, score, strategy="Genetic Algorithm")
    return n_l, n_h, score


def demo_astar(grid=None):
    """Module 3 – A* path planning on a sample route."""
    if grid is None:
        grid = build_default_grid()
    print("\n" + "=" * 65)
    print("  MODULE 3 – A* DELIVERY PATH PLANNER")
    print("=" * 65)

    hub     = (0, 0)
    pickup  = (3, 3)
    dropoff = (7, 7)
    print(f"  Route: Hub{hub} → Pickup{pickup} → Dropoff{dropoff} → Hub{hub}")

    result = plan_delivery_route(hub, pickup, dropoff, grid)
    print_route("Full delivery route", result)

    print("\n  -- Random delivery sample (3 deliveries) --")
    deliveries = generate_deliveries(grid, n=3)
    hubs = [(h.row, h.col) for h in hub_cells(grid)]
    drone_hub = hubs[0]
    for dv in deliveries:
        res = plan_delivery_route(drone_hub, dv.pickup, dv.dropoff, grid)
        label = f"Delivery {dv.delivery_id} | {dv.priority} | {dv.weight}kg"
        print_route(label, res)


def demo_ml_pipeline():
    """Module 5 – Demand forecasting + anomaly detection."""
    print("\n" + "=" * 65)
    print("  MODULE 5 – ML PIPELINE")
    print("=" * 65)
    print("\n  [Part A] Demand Forecasting")
    mae = run_forecast(verbose=True)
    print(f"\n  Best model MAE: {mae:.4f}")

    print("\n  [Part B] Anomaly Detection / Classification")
    acc = run_anomaly_detection(verbose=True)
    print(f"\n  Best model Accuracy: {acc*100:.2f}%")
    return mae, acc


def demo_visualization(grid=None):
    """Optional Module – matplotlib visual dashboard."""
    try:
        import visualization  # noqa: F401
        print("\n  [Visualization] Opening grid dashboard…")
        if grid is None:
            grid = build_default_grid()
        visualization.show_all(grid)
    except ImportError:
        print("\n  [Visualization] visualization.py not found – skipping.")
    except Exception as exc:
        print(f"\n  [Visualization] Error: {exc}")


# Full sequential run
def run_all():
    """Run every module in order, then launch the 20-step simulation."""
    random.seed(42)
    print("\n" + "█" * 65)
    print("  AERONET LITE  –  FULL SYSTEM DEMO")
    print("█" * 65)

    grid = demo_grid()
    demo_validation(grid)
    demo_fleet_selection(strategy="ga", total_demand=60.0)
    demo_astar(grid)

    print("\n" + "=" * 65)
    print("  MODULE 4  –  20-STEP SIMULATION  (integrates all modules)")
    print("=" * 65)
    run_simulation()

    demo_ml_pipeline()
    demo_visualization(grid)

    print("\n" + "█" * 65)
    print("  ALL MODULES COMPLETE")
    print("█" * 65 + "\n")


# Interactive menu

MENU = """
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
"""

def interactive_menu():
    grid = build_default_grid()   # shared grid for the session

    while True:
        print(MENU)
        choice = input("  Enter choice: ").strip()

        if choice == "1":
            demo_grid()
        elif choice == "2":
            demo_validation(grid)
        elif choice == "3":
            demo_fleet_selection(strategy="ga")
        elif choice == "4":
            demo_fleet_selection(strategy="brute")
        elif choice == "5":
            demo_astar(grid)
        elif choice == "6":
            run_simulation()
        elif choice == "7":
            demo_ml_pipeline()
        elif choice == "8":
            demo_visualization(grid)
        elif choice == "9":
            run_all()
        elif choice == "0":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice – please try again.")


# CLI entry point
def parse_args():
    parser = argparse.ArgumentParser(
        description="AeroNet Lite – Autonomous Drone Delivery Simulator"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run all modules sequentially (non-interactive)"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Run 20-step simulation only"
    )
    parser.add_argument(
        "--ml", action="store_true",
        help="Run ML pipeline only (forecast + anomaly detection)"
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Open visualization dashboard only"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.full:
        run_all()
    elif args.sim:
        run_simulation()
    elif args.ml:
        demo_ml_pipeline()
    elif args.vis:
        demo_visualization()
    else:
        # Default: interactive menu
        interactive_menu()
