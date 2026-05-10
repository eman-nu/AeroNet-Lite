import heapq
from typing import List, Optional, Tuple
from grid_model import manhattan


#Cost model
def move_cost(cell) -> float:
    """Cost to enter a cell."""
    if cell.no_fly:
        return float("inf")
    if cell.zone == "Commercial":
        return 0.8          # commercial corridor discount
    return 1.0


# A* core 
def astar(start: Tuple[int,int],
          goal: Tuple[int,int],
          grid) -> dict:
    """
    A* search from start to goal on a 10×10 grid.

    Returns:
        {
          "path":  [(r,c), …] or None,
          "cost":  float,
          "found": bool,
          "message": str,
        }
    """
    if start == goal:
        return {"path": [start], "cost": 0.0, "found": True,
                "message": "Already at destination."}

    # priority queue entries: (f, g, node)
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from = {start: None}
    g_cost    = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return {"path": path, "cost": g, "found": True,
                    "message": f"Path found ({len(path)} cells, cost {g:.2f})."}

        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < 10 and 0 <= nc < 10):
                continue
            neighbour = grid[nr][nc]
            step_cost = move_cost(neighbour)
            if step_cost == float("inf"):
                continue
            new_g = g + step_cost
            if new_g < g_cost.get((nr, nc), float("inf")):
                g_cost[(nr, nc)] = new_g
                came_from[(nr, nc)] = current
                h = manhattan((nr, nc), goal)
                heapq.heappush(open_heap, (new_g + h, new_g, (nr, nc)))

    return {"path": None, "cost": float("inf"), "found": False,
            "message": f"No safe route from {start} to {goal} (all paths blocked)."}


#  Full delivery route: hub -> pickup -> drop-off -> hub
def plan_delivery_route(hub: Tuple, pickup: Tuple, dropoff: Tuple, grid) -> dict:
    """Chains three A* calls to create a complete delivery route."""
    seg1 = astar(hub, pickup, grid)
    seg2 = astar(pickup, dropoff, grid) if seg1["found"] else None
    seg3 = astar(dropoff, hub, grid) if (seg2 and seg2["found"]) else None

    if seg3 and seg3["found"]:
        # Stitch segments 
        full_path = seg1["path"] + seg2["path"][1:] + seg3["path"][1:]
        total_cost = seg1["cost"] + seg2["cost"] + seg3["cost"]
        return {"path": full_path, "cost": total_cost, "found": True,
                "message": f"Full route planned. Total cost: {total_cost:.2f}."}
    else:
        failed_seg = "hub→pickup" if not seg1["found"] else \
                     "pickup→dropoff" if not (seg2 and seg2["found"]) else \
                     "dropoff→hub"
        return {"path": None, "cost": float("inf"), "found": False,
                "message": f"Route failed at segment {failed_seg}."}


# Delivery dataclass 
from dataclasses import dataclass, field

@dataclass
class Delivery:
    delivery_id: str
    pickup: Tuple[int,int]
    dropoff: Tuple[int,int]
    weight: float = 1.0       # kg
    priority: str = "normal"  # normal | medical | urgent
    status: str = "pending"   # pending | assigned | completed | delayed | failed
    assigned_drone: str = None
    route: list = field(default_factory=list)
    cost: float = 0.0


def generate_deliveries(grid, n: int = 8) -> List[Delivery]:
    """Generate random deliveries between non-no-fly cells."""
    import random
    walkable = [(r, c)
                for r in range(10) for c in range(10)
                if not grid[r][c].no_fly]
    deliveries = []
    for i in range(n):
        pickup, dropoff = random.sample(walkable, 2)
        weight = round(random.uniform(0.5, 4.5), 1)
        priority = "medical" if random.random() < 0.15 else "normal"
        deliveries.append(Delivery(f"D{i+1}", pickup, dropoff, weight, priority))
    return deliveries


def print_route(label: str, result: dict):
    print(f"\n  {label}")
    if result["found"]:
        path_str = " → ".join(str(p) for p in result["path"])
        print(f"    Status : ✓ Found")
        print(f"    Cost   : {result['cost']:.2f}")
        print(f"    Path   : {path_str}")
    else:
        print(f"    Status : ✗ Failed")
        print(f"    Reason : {result['message']}")


if __name__ == "__main__":
    from grid_model import build_default_grid
    grid = build_default_grid()
    result = plan_delivery_route((0,0), (3,3), (7,7), grid)
    print("\n=== A* Route Planner Test ===")
    print_route("Hub(0,0) → Pickup(3,3) → Dropoff(7,7) → Hub(0,0)", result)
