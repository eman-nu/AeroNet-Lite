"""
fleet_selector.py  –  Module 2: Fleet Selection
Selects a mix of Light and Heavy drones under a fixed budget.
Supports both brute-force exhaustive search and a Genetic Algorithm.
"""
import random
from dataclasses import dataclass
from typing import List, Tuple

# ── Drone specs ───────────────────────────────────────────────────────────────
@dataclass
class DroneType:
    name: str
    cost: int
    payload: float   # kg
    range_cells: int

LIGHT = DroneType("Light Drone",  cost=1000, payload=2.0, range_cells=12)
HEAVY = DroneType("Heavy Drone",  cost=1800, payload=5.0, range_cells=20)

BUDGET = 10_000      # total budget in currency units
MAX_DRONES = 10      # upper bound per type for search


# ── Fitness function ──────────────────────────────────────────────────────────
def coverage_score(n_light: int, n_heavy: int, total_demand: float) -> float:
    """
    Estimates how much demand the fleet can cover.
    Light drones handle small-parcel demand; heavy drones cover high-volume.
    """
    light_cap  = n_light * LIGHT.payload * (LIGHT.range_cells / 10)
    heavy_cap  = n_heavy * HEAVY.payload * (HEAVY.range_cells / 10)
    total_cap  = light_cap + heavy_cap
    coverage   = min(total_cap / max(total_demand, 1), 1.0)   # clamp [0,1]
    return coverage


def fitness(n_light: int, n_heavy: int, budget: int, total_demand: float) -> float:
    """
    score = 0.75 * coverage_pct  –  0.25 * budget_used_pct
    Higher is better.
    """
    cost = n_light * LIGHT.cost + n_heavy * HEAVY.cost
    if cost > budget:
        return -999.0          # infeasible
    coverage_pct  = coverage_score(n_light, n_heavy, total_demand)
    budget_pct    = cost / budget
    return 0.75 * coverage_pct - 0.25 * budget_pct


# ── Brute-force search ────────────────────────────────────────────────────────
def brute_force_select(budget: int = BUDGET,
                       total_demand: float = 50.0) -> Tuple[int, int, float]:
    best_score = -999.0
    best = (0, 0)
    for n_light in range(MAX_DRONES + 1):
        for n_heavy in range(MAX_DRONES + 1):
            sc = fitness(n_light, n_heavy, budget, total_demand)
            if sc > best_score:
                best_score = sc
                best = (n_light, n_heavy)
    return best[0], best[1], best_score


# ── Genetic Algorithm ─────────────────────────────────────────────────────────
def ga_select(budget: int = BUDGET,
              total_demand: float = 50.0,
              pop_size: int = 40,
              generations: int = 80,
              mutation_rate: float = 0.2) -> Tuple[int, int, float]:
    """
    Chromosome = [n_light, n_heavy]  (both ints 0..MAX_DRONES)
    """
    def rand_chrom():
        return [random.randint(0, MAX_DRONES), random.randint(0, MAX_DRONES)]

    def fit(chrom):
        return fitness(chrom[0], chrom[1], budget, total_demand)

    # Initialise population
    population = [rand_chrom() for _ in range(pop_size)]

    for _ in range(generations):
        # Evaluate
        scored = sorted(population, key=fit, reverse=True)
        # Elitism: keep top 20%
        elite = scored[:max(1, pop_size // 5)]
        new_pop = [list(e) for e in elite]

        # Crossover + mutation to fill rest of population
        while len(new_pop) < pop_size:
            p1 = random.choice(elite)
            p2 = random.choice(scored[:pop_size // 2])
            # Single-point crossover
            child = [p1[0], p2[1]]
            # Mutation
            if random.random() < mutation_rate:
                gene = random.randint(0, 1)
                child[gene] = max(0, min(MAX_DRONES,
                                         child[gene] + random.randint(-2, 2)))
            new_pop.append(child)
        population = new_pop

    best = max(population, key=fit)
    return best[0], best[1], fit(best)


# ── Drone instances ───────────────────────────────────────────────────────────
@dataclass
class Drone:
    drone_id: str
    dtype: DroneType
    position: tuple   # (row, col)
    hub: tuple
    battery: float = 100.0
    status: str = "idle"   # idle | en_route | anomaly | returning
    route: list = None
    route_index: int = 0
    current_delivery: object = None

    def __post_init__(self):
        if self.route is None:
            self.route = []


def create_fleet(n_light: int, n_heavy: int, hub_positions: list) -> List[Drone]:
    """Instantiate drone objects, distributing them across hubs."""
    drones = []
    hubs = hub_positions if hub_positions else [(0, 0)]
    idx = 0
    for i in range(n_light):
        hub = hubs[i % len(hubs)]
        drones.append(Drone(f"DL{i+1}", LIGHT, hub, hub))
        idx += 1
    for i in range(n_heavy):
        hub = hubs[(idx + i) % len(hubs)]
        drones.append(Drone(f"DH{i+1}", HEAVY, hub, hub))
    return drones


def print_fleet_report(n_light, n_heavy, score, budget=BUDGET, strategy="Genetic Algorithm"):
    cost = n_light * LIGHT.cost + n_heavy * HEAVY.cost
    print("\n" + "="*60)
    print("     FLEET SELECTION REPORT")
    print("="*60)
    print(f"  Strategy       : {strategy}")
    print(f"  Light drones   : {n_light}  (each ${LIGHT.cost:,}, {LIGHT.range_cells} cells, {LIGHT.payload}kg)")
    print(f"  Heavy drones   : {n_heavy}  (each ${HEAVY.cost:,}, {HEAVY.range_cells} cells, {HEAVY.payload}kg)")
    print(f"  Total cost     : ${cost:,} / ${budget:,} budget")
    print(f"  Remaining      : ${budget - cost:,}")
    print(f"  Fitness score  : {score:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    random.seed(42)
    n_l, n_h, sc = ga_select(total_demand=60.0)
    print_fleet_report(n_l, n_h, sc, strategy="Genetic Algorithm")
