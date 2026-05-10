"""
layout_validator.py  –  Module 1: CSP-based Grid Layout Validator
Checks four hard constraints and returns a structured validation report.
"""
from grid_model import (
    get_neighbors, manhattan, cells_of_type,
    hub_cells, charging_cells, medical_cells
)


def check_industrial_safety(grid):
    """R1: Industrial cells cannot be directly adjacent to Schools or Hospitals."""
    violations = []
    forbidden_neighbours = {"School", "Hospital"}
    for r in range(10):
        for c in range(10):
            if grid[r][c].zone == "Industrial":
                for nb in get_neighbors(grid, r, c):
                    if nb.zone in forbidden_neighbours:
                        violations.append(
                            f"R1 FAIL: Industrial ({r},{c}) is adjacent to "
                            f"{nb.zone} ({nb.row},{nb.col})"
                        )
    return violations


def check_residential_coverage(grid):
    """R2: Every Residential cell must be within 3 Manhattan cells of a Drone Hub."""
    violations = []
    hubs = [(h.row, h.col) for h in hub_cells(grid)]
    for cell in cells_of_type(grid, "Residential"):
        if not hubs:
            violations.append(f"R2 FAIL: No hubs defined at all.")
            break
        min_dist = min(manhattan((cell.row, cell.col), h) for h in hubs)
        if min_dist > 3:
            violations.append(
                f"R2 FAIL: Residential ({cell.row},{cell.col}) is {min_dist} "
                f"cells from nearest hub (max=3). "
                f"Suggested fix: add a hub near ({cell.row},{cell.col})."
            )
    return violations


def check_hub_charging(grid):
    """R3: Every Drone Hub must have a Charging Pad within 2 cells."""
    violations = []
    chargers = [(ch.row, ch.col) for ch in charging_cells(grid)]
    for hub in hub_cells(grid):
        if not chargers:
            violations.append(f"R3 FAIL: No charging pads defined.")
            break
        min_dist = min(manhattan((hub.row, hub.col), ch) for ch in chargers)
        if min_dist > 2:
            violations.append(
                f"R3 FAIL: Hub ({hub.row},{hub.col}) is {min_dist} cells from "
                f"nearest charging pad (max=2). "
                f"Suggested fix: add a charging pad near ({hub.row},{hub.col})."
            )
    return violations


def check_medical_access(grid):
    """R4: At least one Hospital must have a Medical Pickup point within 1 cell."""
    medicals = [(m.row, m.col) for m in medical_cells(grid)]
    hospitals = cells_of_type(grid, "Hospital")
    if not hospitals:
        return ["R4 FAIL: No Hospital cells defined."]
    if not medicals:
        return ["R4 FAIL: No Medical Pickup cells defined."]

    for hosp in hospitals:
        for med in medicals:
            if manhattan((hosp.row, hosp.col), med) <= 1:
                return []   # at least one hospital is served — constraint satisfied

    return [
        f"R4 FAIL: No Hospital has a Medical Pickup within 1 cell. "
        f"Hospitals: {[(h.row,h.col) for h in hospitals]}. "
        f"Medical pickups: {medicals}."
    ]


# ── Main entry point ──────────────────────────────────────────────────────────
def validate_layout(grid) -> dict:
    """
    Run all four CSP rules.
    Returns a dict with keys: passed, failed, violations, valid (bool).
    """
    rules = {
        "R1 - Industrial Safety":       check_industrial_safety(grid),
        "R2 - Residential Hub Coverage": check_residential_coverage(grid),
        "R3 - Hub Charging Access":      check_hub_charging(grid),
        "R4 - Medical Pickup Access":    check_medical_access(grid),
    }

    passed, failed, all_violations = [], [], []
    for rule, viols in rules.items():
        if viols:
            failed.append(rule)
            all_violations.extend(viols)
        else:
            passed.append(rule)

    return {
        "valid": len(failed) == 0,
        "passed": passed,
        "failed": failed,
        "violations": all_violations,
    }


def print_validation_report(report: dict):
    print("\n" + "="*60)
    print("     LAYOUT VALIDATION REPORT (CSP)")
    print("="*60)
    status = "VALID ✓" if report["valid"] else "INVALID ✗"
    print(f"  Overall status : {status}")
    print(f"  Rules passed   : {len(report['passed'])}/4")
    if report["passed"]:
        for r in report["passed"]:
            print(f"    ✓  {r}")
    if report["failed"]:
        print(f"  Rules failed   : {len(report['failed'])}/4")
        for r in report["failed"]:
            print(f"    ✗  {r}")
        print("\n  Violations:")
        for v in report["violations"]:
            print(f"    • {v}")
    print("="*60 + "\n")


if __name__ == "__main__":
    from grid_model import build_default_grid
    grid = build_default_grid()

    print("--- Test 1: Valid grid ---")
    print_validation_report(validate_layout(grid))

    print("--- Test 2: Force R1 failure ---")
    grid[0][3].zone = "School"   # put School next to Industrial at (0,4)
    print_validation_report(validate_layout(grid))
