from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random

# Zone types 
ZONES = ["Residential", "Commercial", "Industrial", "Hospital",
         "School", "Open Field"]

ZONE_COLORS = {
    "Residential": "#4A90D9",
    "Commercial":  "#F5A623",
    "Industrial":  "#7B7B7B",
    "Hospital":    "#D0021B",
    "School":      "#417505",
    "Open Field":  "#B8E986",
}

# Cell 
@dataclass
class Cell:
    row: int
    col: int
    zone: str = "Open Field"
    density: int = 0                 # people / km²  (rough proxy)
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0              # estimated delivery demand

    def __repr__(self):
        flags = []
        if self.is_hub:            flags.append("HUB")
        if self.is_charging:       flags.append("CHG")
        if self.is_medical_pickup: flags.append("MED")
        if self.no_fly:            flags.append("NFZ")
        tag = "|".join(flags)
        return f"Cell({self.row},{self.col} {self.zone[:3]} {tag})"


#Grid builder 
def build_default_grid() -> List[List[Cell]]:
    """
    Hard-coded 10×10 city that satisfies all four CSP constraints by design.
    All four rules pass on a clean (unmodified) grid:
      R1: No Industrial cell is adjacent to a School or Hospital.
      R2: Every Residential cell is within 3 Manhattan cells of a Drone Hub.
      R3: Every Drone Hub has a Charging Pad within 2 cells.
      R4: At least one Hospital has a Medical Pickup within 1 cell.
    Disruption tests (activating no-fly cells) may still be applied at runtime.

    Key layout decisions to satisfy the constraints:
      - Industrial zones are placed at (0,4), (4,8), (9,4) — all separated from
        Hospitals and Schools by at least one Open Field buffer cell.
      - Four hubs at (0,0), (0,6), (5,3), (8,8) cover every Residential cell
        within Manhattan distance 3.
      - Each hub has a Charging Pad within 2 cells.
      - Hospitals at (1,3) and (7,6) each have a Medical Pickup on an
        immediately adjacent cell.
    """
    zone_layout = [
        # col: 0             1             2             3             4             5             6             7             8             9
        ["Residential","Residential","Commercial","Commercial","Industrial","Open Field","Residential","Residential","School",     "Open Field"],  # row 0
        ["Residential","Residential","Commercial","Hospital", "Open Field","Open Field","Residential","Residential","Open Field","Open Field"],  # row 1
        ["Open Field", "Commercial", "Commercial","Commercial","Open Field","Open Field","Commercial","Commercial","Open Field","Open Field"],  # row 2
        ["Residential","Residential","Open Field","Open Field","Open Field","Residential","Residential","Residential","Open Field","Open Field"],  # row 3
        ["Residential","School",     "Open Field","Open Field","Open Field","Open Field","Residential","Residential","Industrial","Open Field"],  # row 4
        ["Open Field", "Open Field", "Open Field","Open Field","Open Field","Open Field","Open Field","Open Field","Open Field","Open Field"],  # row 5
        ["Residential","Residential","Residential","Open Field","Residential","Residential","Open Field","Commercial","Commercial","Open Field"],  # row 6
        ["Residential","Residential","Open Field","Open Field","Residential","Residential","Hospital", "Commercial","Open Field","Open Field"],  # row 7
        ["Open Field", "Open Field", "Open Field","Open Field","Open Field","Open Field","Open Field","Open Field","Residential","Residential"],  # row 8
        ["Open Field", "Open Field", "Open Field","Open Field","Industrial","Open Field","Open Field","Open Field","Residential","Residential"],  # row 9
    ]

    density_map = {
        "Residential": 5000, "Commercial": 3000, "Hospital": 500,
        "School": 800, "Industrial": 200, "Open Field": 50,
    }

    grid: List[List[Cell]] = []
    for r in range(10):
        row_cells = []
        for c in range(10):
            zone = zone_layout[r][c]
            cell = Cell(row=r, col=c, zone=zone,
                        density=density_map[zone],
                        demand=round(density_map[zone] / 1000 * random.uniform(0.8, 1.2), 2))
            row_cells.append(cell)
        grid.append(row_cells)

    # ── Hubs (satisfies R2: ALL residential cells within 3 Manhattan cells) ──
    # Verified coverage (every residential cell listed, max distance = 3):
    #   Hub (1,0): covers NW residentials (0,0),(0,1),(1,0),(1,1),(3,0),(3,1),(4,0)
    #   Hub (2,6): covers NE residentials (0,6),(0,7),(1,6),(1,7),(3,5),(3,6),(3,7),(4,6),(4,7)
    #   Hub (6,2): covers W/central residentials (6,0..2),(6,4),(7,0),(7,1),(7,4),(7,5),(4,0)
    #   Hub (7,8): covers SE residentials (6,5),(8,8),(8,9),(9,8),(9,9)
    for (r, c) in [(1,0), (2,6), (6,2), (7,8)]:
        grid[r][c].is_hub = True

    # ── Charging pads (satisfies R3: each hub has a charging pad within 2 cells)
    #   Hub (1,0) → Pad at (1,2)  distance=2 ✓
    #   Hub (2,6) → Pad at (1,6)  distance=1 ✓  (Residential cell — dual use)
    #   Hub (6,2) → Pad at (5,2)  distance=1 ✓
    #   Hub (7,8) → Pad at (6,8)  distance=1 ✓
    for (r, c) in [(1,2), (1,6), (5,2), (6,8)]:
        grid[r][c].is_charging = True

    # ── Medical pickup (satisfies R4: hospital must have pickup within 1 cell)
    #   Hospital at (1,3) → Pickup at (1,4)  distance=1 ✓  (Open Field cell)
    #   Hospital at (7,6) → Pickup at (7,5)  distance=1 ✓  (Residential cell)
    grid[1][4].is_medical_pickup = True
    grid[7][5].is_medical_pickup = True

    return grid


#  Utility helpers
def get_cell(grid, row: int, col: int) -> Optional[Cell]:
    if 0 <= row < 10 and 0 <= col < 10:
        return grid[row][col]
    return None


def get_neighbors(grid, row: int, col: int) -> List[Cell]:
    neighbors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        c = get_cell(grid, row+dr, col+dc)
        if c:
            neighbors.append(c)
    return neighbors


def manhattan(a: tuple, b: tuple) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def cells_of_type(grid, zone: str) -> List[Cell]:
    return [grid[r][c] for r in range(10) for c in range(10)
            if grid[r][c].zone == zone]


def hub_cells(grid) -> List[Cell]:
    return [grid[r][c] for r in range(10) for c in range(10)
            if grid[r][c].is_hub]


def charging_cells(grid) -> List[Cell]:
    return [grid[r][c] for r in range(10) for c in range(10)
            if grid[r][c].is_charging]


def medical_cells(grid) -> List[Cell]:
    return [grid[r][c] for r in range(10) for c in range(10)
            if grid[r][c].is_medical_pickup]


def print_grid_summary(grid):
    print("\n=== Grid Summary ===")
    for r in range(10):
        row_str = ""
        for c in range(10):
            cell = grid[r][c]
            z = cell.zone[0:2].upper()
            if cell.is_hub:            z = "HB"
            elif cell.is_charging:     z = "CH"
            elif cell.is_medical_pickup: z = "MP"
            elif cell.no_fly:          z = "NF"
            row_str += f"{z:>3}"
        print(f"Row {r}: {row_str}")
    print()
