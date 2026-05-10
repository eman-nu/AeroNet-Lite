import random
from typing import List, Dict
from grid_model import build_default_grid, hub_cells
from fleet_selector import Drone, create_fleet, ga_select
from astar_planner import (Delivery, generate_deliveries,
                            plan_delivery_route, astar)

STEPS_PER_TICK = 3


# Event logger 
event_log: List[str] = []

def log(step: int, msg: str):
    entry = f"Step {step:>2}: {msg}"
    event_log.append(entry)
    print(entry)


# Assign deliveries to drones 
def assign_deliveries(drones: List[Drone],
                      deliveries: List[Delivery],
                      grid,
                      step: int):
    idle_drones = [d for d in drones if d.status == "idle"]
    pending     = [dv for dv in deliveries if dv.status == "pending"]

    for drone, delivery in zip(idle_drones, pending):
        result = plan_delivery_route(drone.hub, delivery.pickup,
                                     delivery.dropoff, grid)
        if result["found"]:
            drone.route        = result["path"]
            drone.route_index  = 0
            drone.status       = "en_route"
            drone.current_delivery = delivery
            delivery.status    = "assigned"
            delivery.assigned_drone = drone.drone_id
            delivery.route     = result["path"]
            delivery.cost      = result["cost"]
            log(step, f"Delivery {delivery.delivery_id} assigned to {drone.drone_id} "
                      f"(cost={result['cost']:.1f}, {len(result['path'])} cells).")
        else:
            log(step, f"Delivery {delivery.delivery_id} FAILED routing: "
                      f"{result['message']}")
            delivery.status = "failed"


# Move drones one step along their route 
def step_drones(drones: List[Drone], deliveries: List[Delivery],
                grid, step: int):
    for drone in drones:
        if drone.status != "en_route" or not drone.route:
            continue

        # Advance STEPS_PER_TICK cells 
        for _ in range(STEPS_PER_TICK):
            if drone.route_index >= len(drone.route) - 1:
                break
            drone.route_index += 1
            drone.position = drone.route[drone.route_index]
            drone.battery -= random.uniform(0.5, 1.0)   # per-cell cost

        if drone.route_index >= len(drone.route) - 1:
            drone.status = "idle"
            if drone.current_delivery:
                drone.current_delivery.status = "completed"
                log(step, f"{drone.drone_id} completed delivery "
                          f"{drone.current_delivery.delivery_id} ✓")
                drone.current_delivery = None
            drone.route = []
            drone.route_index = 0


#Activate a no-fly cell and reroute affected drones 
def activate_no_fly(grid, row: int, col: int,
                    drones: List[Drone],
                    step: int):
    grid[row][col].no_fly = True
    log(step, f"[W] No-fly cell activated at ({row},{col}).")

    for drone in drones:
        if drone.status != "en_route":
            continue
        # Check if blocked cell is still ahead in route
        remaining = drone.route[drone.route_index:]
        if (row, col) in remaining:
            # Reroute from current position to delivery dropoff -> hub
            delivery = drone.current_delivery
            if delivery is None:
                continue
            new_result = plan_delivery_route(
                drone.position, delivery.pickup if delivery.status == "assigned"
                                else delivery.dropoff,
                drone.hub, grid)
            if new_result["found"]:
                drone.route       = [drone.position] + new_result["path"][1:]
                drone.route_index = 0
                log(step, f"{drone.drone_id} rerouted successfully "
                          f"(new cost={new_result['cost']:.1f}).")
            else:
                drone.status = "idle"
                if delivery:
                    delivery.status = "delayed"
                log(step, f"{drone.drone_id} CANNOT find safe route → "
                          f"delivery {delivery.delivery_id if delivery else '?'} delayed.")


#  Inject an anomaly 
def inject_anomaly(drones: List[Drone], step: int, anomaly_log: list):
    en_route = [d for d in drones if d.status == "en_route"]
    if not en_route:
        log(step, "No drones en route for anomaly injection.")
        return
    drone = random.choice(en_route)
    anomaly_type = random.choice(["Battery anomaly", "Route anomaly", "Sensor spike"])
    log(step, f"[!] {anomaly_type} detected for {drone.drone_id}!")
    anomaly_log.append({
        "drone": drone.drone_id, "type": anomaly_type,
        "step": step, "position": drone.position
    })
    if anomaly_type == "Battery anomaly":
        drone.battery -= random.uniform(25, 40)
    if drone.battery < 20 or anomaly_type == "Route anomaly":
        drone.status = "returning"
        drone.route  = []
        log(step, f"{drone.drone_id} forced to return to hub.")
        if drone.current_delivery:
            drone.current_delivery.status = "delayed"


def run_simulation():
    random.seed(42)
    global event_log
    event_log = []
    anomaly_log = []

    print("\n" + "="*65)
    print("          AERONET LITE  –  20-STEP SIMULATION")
    print("="*65)

    # Steps 1-3: Init, validate, select fleet
    grid = build_default_grid()
    log(1, "Grid initialized (10×10 city model).")

    from layout_validator import validate_layout
    report = validate_layout(grid)
    status = "passed" if report["valid"] else "FAILED"
    log(2, f"Layout validation {status}. "
         f"{len(report['passed'])}/4 rules passed.")
    if not report["valid"]:
        for v in report["violations"]:
            log(2, f"  Violation: {v}")

    n_l, n_h, score = ga_select(total_demand=60.0)
    hubs = [(h.row, h.col) for h in hub_cells(grid)]
    drones = create_fleet(n_l, n_h, hubs)
    log(3, f"Fleet selected: {n_l} light drones, {n_h} heavy drones "
         f"(fitness={score:.3f}).")

    # Steps 4-6: Generate deliveries and assign 
    deliveries = generate_deliveries(grid, n=8)
    log(4, f"Generated {len(deliveries)} deliveries.")

    assign_deliveries(drones, deliveries, grid, 5)

    log(6, "Initial routes computed via A* for all assigned deliveries.")

    # Steps 7-10: Move drones
    for s in range(7, 11):
        step_drones(drones, deliveries, grid, s)
        assigned = [d for d in drones if d.status == "en_route"]
        log(s, f"Drones moved. {len(assigned)} en route.")

    # Step 11: Activate a no-fly cell 
    activate_no_fly(grid, 4, 7, drones, 11)

    # Steps 12-14: Continue movement after rerouting 
    for s in range(12, 15):
        step_drones(drones, deliveries, grid, s)
        assign_deliveries(drones, deliveries, grid, s)   # pick up any new pending

    #  Steps 15-17: Demand forecast
    log(15, "Running demand forecast (ML pipeline)…")
    try:
        from ml_pipeline import run_forecast
        mae = run_forecast()
        log(15, f"Demand forecast complete. MAE = {mae:.3f}.")
    except Exception as e:
        log(15, f"Demand forecast skipped ({e}).")

    for s in range(16, 18):
        step_drones(drones, deliveries, grid, s)

    # Step 18: Inject anomaly
    inject_anomaly(drones, 18, anomaly_log)

    #  Steps 19-20: Final movement + summary 
    step_drones(drones, deliveries, grid, 19)
    for d in drones:
        if d.status == "returning":
            d.status = "idle"
            log(19, f"{d.drone_id} returned to hub safely.")

    step_drones(drones, deliveries, grid, 20)

    #  Final summary 
    completed = sum(1 for dv in deliveries if dv.status == "completed")
    delayed   = sum(1 for dv in deliveries if dv.status == "delayed")
    failed    = sum(1 for dv in deliveries if dv.status == "failed")
    pending   = sum(1 for dv in deliveries if dv.status in ("pending","assigned"))

    log(20, f"Simulation complete. "
           f"Completed={completed}, Delayed={delayed}, "
           f"Failed={failed}, In-progress={pending}.")

    print("\n" + "="*65)
    print(f"  FINAL SUMMARY")
    print("="*65)
    print(f"  Deliveries completed : {completed}")
    print(f"  Deliveries delayed   : {delayed}")
    print(f"  Deliveries failed    : {failed}")
    print(f"  In-progress          : {pending}")
    print(f"  Anomalies detected   : {len(anomaly_log)}")
    if anomaly_log:
        for a in anomaly_log:
            print(f"    • {a['type']} on {a['drone']} at step {a['step']}, pos {a['position']}")
    print("="*65 + "\n")

    return grid, drones, deliveries, event_log, anomaly_log


if __name__ == "__main__":
    run_simulation()
