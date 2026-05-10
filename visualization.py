"""
visualization.py  –  AeroNet Lite Visual Dashboard
=====================================================
Provides five views using matplotlib:
  1. Zone Map        – coloured 10×10 grid with overlays
  2. Route Map       – drone paths and no-fly cells
  3. Demand Heatmap  – delivery demand intensity per cell
  4. Anomaly View    – drone telemetry anomaly scatter
  5. Event Log Panel – simulation decisions step by step

Public API
----------
  show_zone_map(grid)
  show_route_map(grid, drones, deliveries)
  show_demand_heatmap(grid)
  show_anomaly_view(anomaly_log, drones)
  show_event_log(event_log)
  show_all(grid, drones=None, deliveries=None,
           anomaly_log=None, event_log=None)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec

# ── Try non-interactive backend if display is unavailable ─────────────────────
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

# ── Zone colour palette ───────────────────────────────────────────────────────
ZONE_COLORS = {
    "Residential": "#4A90D9",
    "Commercial":  "#F5A623",
    "Industrial":  "#7B7B7B",
    "Hospital":    "#D0021B",
    "School":      "#417505",
    "Open Field":  "#B8E986",
}

ZONE_ORDER = list(ZONE_COLORS.keys())   # for legend ordering

# ── Global style ──────────────────────────────────────────────────────────────
DARK_BG   = "#0D1117"
PANEL_BG  = "#161B22"
ACCENT    = "#58A6FF"
TEXT_COL  = "#E6EDF3"
MUTED     = "#8B949E"
GRID_LINE = "#21262D"
FONT_MONO = {"fontfamily": "monospace"}


def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_LINE,
        "axes.labelcolor":   TEXT_COL,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT_COL,
        "grid.color":        GRID_LINE,
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "figure.dpi":        110,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1. ZONE MAP
# ══════════════════════════════════════════════════════════════════════════════

def show_zone_map(grid, ax=None, title="City Zone Map"):
    """
    Draw the 10×10 coloured zone grid.
    Special cells (hub, charging, medical, no-fly) are marked with icons.
    """
    standalone = ax is None
    if standalone:
        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(title, color=ACCENT, fontsize=14, fontweight="bold", y=0.97)

    # Build colour matrix
    color_matrix = np.zeros((10, 10, 3))
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            hex_c = ZONE_COLORS.get(cell.zone, "#CCCCCC")
            rgb = matplotlib.colors.to_rgb(hex_c)
            color_matrix[r, c] = rgb

    ax.imshow(color_matrix, origin="upper", aspect="equal",
              extent=[-0.5, 9.5, 9.5, -0.5])

    # Grid lines
    for i in range(11):
        ax.axhline(i - 0.5, color=DARK_BG, linewidth=0.8)
        ax.axvline(i - 0.5, color=DARK_BG, linewidth=0.8)

    # Overlay icons
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            icons = []
            if cell.is_hub:            icons.append(("[H]", "#FFD700", 9))
            if cell.is_charging:       icons.append(("⚡", "#00FFFF", 10))
            if cell.is_medical_pickup: icons.append(("✚", "#FF4444", 11))
            if cell.no_fly:            icons.append(("✖", "#FF0000", 13))

            if icons:
                icon, col, sz = icons[0]
                ax.text(c, r, icon, ha="center", va="center",
                        color=col, fontsize=sz, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=2,
                                                    foreground=DARK_BG)])

    # Axis labels
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)], color=MUTED, fontsize=8)
    ax.set_yticklabels([str(i) for i in range(10)], color=MUTED, fontsize=8)
    ax.set_xlabel("Column", color=MUTED, fontsize=9)
    ax.set_ylabel("Row",    color=MUTED, fontsize=9)

    # Legend – zones
    patches = [mpatches.Patch(color=ZONE_COLORS[z], label=z)
               for z in ZONE_ORDER if z in ZONE_COLORS]
    # Icon legend
    icon_handles = [
        mpatches.Patch(color="#FFD700", label="[H] Drone Hub"),
        mpatches.Patch(color="#00FFFF", label="⚡ Charging Pad"),
        mpatches.Patch(color="#FF4444", label="✚ Medical Pickup"),
        mpatches.Patch(color="#FF0000", label="✖ No-Fly Zone"),
    ]
    legend = ax.legend(
        handles=patches + icon_handles,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        framealpha=0.15, fontsize=8,
        labelcolor=TEXT_COL, edgecolor=GRID_LINE,
    )

    if standalone:
        plt.tight_layout()
        plt.savefig("zone_map.png", bbox_inches="tight", facecolor=DARK_BG)
        print("  [Viz] Zone map saved → zone_map.png")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROUTE MAP
# ══════════════════════════════════════════════════════════════════════════════

ROUTE_COLORS = [
    "#58A6FF", "#3FB950", "#F78166", "#D2A8FF",
    "#FFA657", "#79C0FF", "#56D364", "#FF7B72",
]


def show_route_map(grid, drones=None, deliveries=None, ax=None,
                  title="Drone Route Map"):
    """
    Show drone paths overlaid on a greyscale zone grid.
    No-fly cells are highlighted in red.
    Each drone/delivery gets a distinct route colour.
    """
    standalone = ax is None
    if standalone:
        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(title, color=ACCENT, fontsize=14, fontweight="bold", y=0.97)

    # Greyscale zone background
    grey_matrix = np.zeros((10, 10, 3))
    for r in range(10):
        for c in range(10):
            cell = grid[r][c]
            hex_c = ZONE_COLORS.get(cell.zone, "#888888")
            rgb = matplotlib.colors.to_rgb(hex_c)
            grey = 0.25 * rgb[0] + 0.15 * rgb[1] + 0.1 * rgb[2] + 0.08
            if cell.no_fly:
                grey_matrix[r, c] = matplotlib.colors.to_rgb("#5C1A1A")
            else:
                grey_matrix[r, c] = (grey, grey, grey + 0.02)

    ax.imshow(grey_matrix, origin="upper", aspect="equal",
              extent=[-0.5, 9.5, 9.5, -0.5])

    # Grid lines
    for i in range(11):
        ax.axhline(i - 0.5, color=DARK_BG, linewidth=0.5)
        ax.axvline(i - 0.5, color=DARK_BG, linewidth=0.5)

    # No-fly cell markers
    for r in range(10):
        for c in range(10):
            if grid[r][c].no_fly:
                ax.text(c, r, "✖", ha="center", va="center",
                        color="#FF4444", fontsize=12, fontweight="bold")

    # Hub markers
    for r in range(10):
        for c in range(10):
            if grid[r][c].is_hub:
                ax.plot(c, r, "D", color="#FFD700", markersize=9,
                        markeredgecolor=DARK_BG, markeredgewidth=1.2)

    # Draw delivery routes
    if deliveries:
        for idx, dv in enumerate(deliveries):
            if not dv.route:
                continue
            color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
            path = dv.route
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, "-", color=color, linewidth=1.8,
                    alpha=0.85, solid_capstyle="round")
            # Pickup marker
            ax.plot(dv.pickup[1], dv.pickup[0], "o",
                    color=color, markersize=7,
                    markeredgecolor=DARK_BG, markeredgewidth=1,
                    label=f"{dv.delivery_id} ({dv.status})")
            # Dropoff marker
            ax.plot(dv.dropoff[1], dv.dropoff[0], "s",
                    color=color, markersize=7,
                    markeredgecolor=DARK_BG, markeredgewidth=1)

    # Draw drone current positions
    if drones:
        for drone in drones:
            pos = drone.position
            status_color = {
                "idle":      "#888888",
                "en_route":  "#3FB950",
                "returning": "#FFA657",
                "anomaly":   "#FF7B72",
            }.get(drone.status, "#FFFFFF")
            ax.plot(pos[1], pos[0], "^", color=status_color,
                    markersize=10, markeredgecolor=DARK_BG,
                    markeredgewidth=1.5,
                    label=f"{drone.drone_id} [{drone.status}]")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)], color=MUTED, fontsize=8)
    ax.set_yticklabels([str(i) for i in range(10)], color=MUTED, fontsize=8)
    ax.set_xlabel("Column", color=MUTED, fontsize=9)
    ax.set_ylabel("Row",    color=MUTED, fontsize=9)

    if deliveries or drones:
        legend = ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            framealpha=0.15, fontsize=7.5,
            labelcolor=TEXT_COL, edgecolor=GRID_LINE,
        )

    if standalone:
        plt.tight_layout()
        plt.savefig("route_map.png", bbox_inches="tight", facecolor=DARK_BG)
        print("  [Viz] Route map saved → route_map.png")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 3. DEMAND HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def show_demand_heatmap(grid, ax=None, title="Delivery Demand Heatmap"):
    """
    Colour-coded heatmap of per-cell delivery demand.
    """
    standalone = ax is None
    if standalone:
        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle(title, color=ACCENT, fontsize=14, fontweight="bold", y=0.97)

    demand_matrix = np.array(
        [[grid[r][c].demand for c in range(10)] for r in range(10)]
    )

    cmap = plt.cm.YlOrRd
    im = ax.imshow(demand_matrix, cmap=cmap, origin="upper",
                   aspect="equal", extent=[-0.5, 9.5, 9.5, -0.5])

    # Grid lines
    for i in range(11):
        ax.axhline(i - 0.5, color=DARK_BG, linewidth=0.6)
        ax.axvline(i - 0.5, color=DARK_BG, linewidth=0.6)

    # Value annotations
    vmin = demand_matrix.min()
    vmax = demand_matrix.max()
    mid  = (vmin + vmax) / 2
    for r in range(10):
        for c in range(10):
            val = demand_matrix[r, c]
            text_col = "white" if val < mid else "#222"
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    color=text_col, fontsize=6.5, fontweight="bold")

    # Hub markers on heatmap
    for r in range(10):
        for c in range(10):
            if grid[r][c].is_hub:
                ax.text(c, r - 0.32, "▲", ha="center", va="center",
                        color="#FFD700", fontsize=8)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Demand (deliveries/hr)", color=MUTED, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=7)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)], color=MUTED, fontsize=8)
    ax.set_yticklabels([str(i) for i in range(10)], color=MUTED, fontsize=8)
    ax.set_xlabel("Column", color=MUTED, fontsize=9)
    ax.set_ylabel("Row",    color=MUTED, fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.savefig("demand_heatmap.png", bbox_inches="tight", facecolor=DARK_BG)
        print("  [Viz] Demand heatmap saved → demand_heatmap.png")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANOMALY VIEW
# ══════════════════════════════════════════════════════════════════════════════

ANOMALY_COLORS = {
    "Battery anomaly": "#FFA657",
    "Route anomaly":   "#79C0FF",
    "Sensor spike":    "#F78166",
    "Normal":          "#3FB950",
}


def show_anomaly_view(anomaly_log=None, drones=None, ax=None,
                      title="Anomaly Detection View"):
    """
    Scatter plot of anomaly events by simulation step.
    Also shows a drone status summary bar chart.
    """
    standalone = ax is None
    if standalone:
        _apply_dark_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(title, color=ACCENT, fontsize=14, fontweight="bold", y=1.01)
        ax_scatter, ax_bar = axes
    else:
        # When embedded, use a single axes for the bar chart
        ax_scatter = ax
        ax_bar = None

    # ── Left: Anomaly scatter ─────────────────────────────────────────────────
    anomaly_log = anomaly_log or []
    if anomaly_log:
        for event in anomaly_log:
            atype = event.get("type", "Normal")
            step  = event.get("step", 0)
            drone = event.get("drone", "?")
            pos   = event.get("position", (0, 0))
            color = ANOMALY_COLORS.get(atype, "#FFFFFF")
            ax_scatter.scatter(step, pos[0] * 10 + pos[1],
                               color=color, s=120, zorder=5,
                               edgecolors=DARK_BG, linewidths=1)
            ax_scatter.annotate(
                f" {drone}\n {atype[:6]}",
                (step, pos[0] * 10 + pos[1]),
                fontsize=6.5, color=color,
                xytext=(4, 4), textcoords="offset points",
            )
        handles = [mpatches.Patch(color=c, label=t)
                   for t, c in ANOMALY_COLORS.items() if t != "Normal"]
        ax_scatter.legend(
            handles=handles, loc="upper right",
            framealpha=0.15, fontsize=7.5,
            labelcolor=TEXT_COL, edgecolor=GRID_LINE,
        )
    else:
        ax_scatter.text(0.5, 0.5, "No anomalies recorded",
                        ha="center", va="center",
                        color=MUTED, fontsize=11,
                        transform=ax_scatter.transAxes)

    ax_scatter.set_xlabel("Simulation Step", color=MUTED, fontsize=9)
    ax_scatter.set_ylabel("Cell Index (row×10 + col)", color=MUTED, fontsize=9)
    ax_scatter.set_title("Events by Step & Location", color=TEXT_COL, fontsize=10)
    ax_scatter.grid(True, alpha=0.2)

    # ── Right: Drone status bar chart ─────────────────────────────────────────
    if ax_bar is not None and drones:
        status_counts = {}
        for d in drones:
            status_counts[d.status] = status_counts.get(d.status, 0) + 1

        labels = list(status_counts.keys())
        values = [status_counts[l] for l in labels]
        colors = ["#3FB950" if l == "idle"
                  else "#FFA657" if l == "returning"
                  else "#58A6FF" if l == "en_route"
                  else "#FF7B72"
                  for l in labels]

        bars = ax_bar.bar(labels, values, color=colors, edgecolor=DARK_BG,
                          linewidth=1.2)
        for bar, val in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom",
                        color=TEXT_COL, fontsize=10, fontweight="bold")
        ax_bar.set_xlabel("Status", color=MUTED, fontsize=9)
        ax_bar.set_ylabel("Count",  color=MUTED, fontsize=9)
        ax_bar.set_title("Fleet Status Summary", color=TEXT_COL, fontsize=10)
        ax_bar.grid(axis="y", alpha=0.2)
    elif ax_bar is not None:
        ax_bar.text(0.5, 0.5, "No drone data",
                    ha="center", va="center",
                    color=MUTED, fontsize=11,
                    transform=ax_bar.transAxes)
        ax_bar.set_title("Fleet Status Summary", color=TEXT_COL, fontsize=10)

    if standalone:
        plt.tight_layout()
        plt.savefig("anomaly_view.png", bbox_inches="tight", facecolor=DARK_BG)
        print("  [Viz] Anomaly view saved → anomaly_view.png")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. EVENT LOG PANEL
# ══════════════════════════════════════════════════════════════════════════════

def show_event_log(event_log=None, ax=None, title="Simulation Event Log"):
    """
    Renders the simulation event log as a formatted text panel.
    """
    standalone = ax is None
    if standalone:
        _apply_dark_style()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle(title, color=ACCENT, fontsize=14, fontweight="bold", y=0.97)

    ax.axis("off")
    event_log = event_log or ["No events recorded."]
    # Colour-code by event keyword
    full_text = "\n".join(event_log[-40:])   # show last 40 lines max

    ax.text(
        0.01, 0.99, full_text,
        transform=ax.transAxes,
        fontsize=7.8,
        verticalalignment="top",
        fontfamily="monospace",
        color=TEXT_COL,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG,
                  edgecolor=GRID_LINE, alpha=0.9),
    )

    if standalone:
        plt.tight_layout()
        plt.savefig("event_log.png", bbox_inches="tight", facecolor=DARK_BG)
        print("  [Viz] Event log saved → event_log.png")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 6. COMBINED DASHBOARD  (show_all)
# ══════════════════════════════════════════════════════════════════════════════

def show_all(grid, drones=None, deliveries=None,
             anomaly_log=None, event_log=None):
    """
    Render a 3×2 dashboard combining all five views.
    Also saves individual PNGs and a combined dashboard PNG.

    If drones / deliveries / logs are not provided the simulation is
    run automatically to populate them.
    """
    _apply_dark_style()

    # ── Auto-run simulation if no live data provided ──────────────────────────
    if drones is None or deliveries is None:
        print("  [Viz] Running simulation to collect data for dashboard…")
        try:
            from delivery_simulator import run_simulation
            _, drones, deliveries, event_log_sim, anomaly_log_sim = run_simulation()
            event_log   = event_log   or event_log_sim
            anomaly_log = anomaly_log or anomaly_log_sim
        except Exception as exc:
            print(f"  [Viz] Simulation run failed: {exc}. Showing static views.")

    # ── Build dashboard figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(DARK_BG)

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.38, wspace=0.35,
                  left=0.05, right=0.97,
                  top=0.91, bottom=0.04)

    ax_zone    = fig.add_subplot(gs[0, 0])
    ax_route   = fig.add_subplot(gs[0, 1])
    ax_demand  = fig.add_subplot(gs[0, 2])
    ax_anomaly = fig.add_subplot(gs[1, 0])
    ax_fleet   = fig.add_subplot(gs[1, 1])
    ax_log     = fig.add_subplot(gs[1, 2])

    # ── Title bar ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.96,
             "AeroNet Lite  ·  Autonomous Drone Delivery Dashboard",
             ha="center", va="center",
             color=ACCENT, fontsize=16, fontweight="bold",
             fontfamily="monospace")

    # ── Draw each panel ───────────────────────────────────────────────────────
    show_zone_map(grid, ax=ax_zone)
    ax_zone.set_title("Zone Map", color=TEXT_COL, fontsize=10, pad=6)

    show_route_map(grid, drones=drones, deliveries=deliveries, ax=ax_route)
    ax_route.set_title("Route Map", color=TEXT_COL, fontsize=10, pad=6)

    show_demand_heatmap(grid, ax=ax_demand)
    ax_demand.set_title("Demand Heatmap", color=TEXT_COL, fontsize=10, pad=6)

    # Anomaly scatter (left bottom)
    show_anomaly_view(anomaly_log=anomaly_log, ax=ax_anomaly)
    ax_anomaly.set_title("Anomaly Events", color=TEXT_COL, fontsize=10, pad=6)

    # Fleet status bar (middle bottom) – standalone mini bar chart
    _draw_fleet_bar(ax_fleet, drones)
    ax_fleet.set_title("Fleet Status", color=TEXT_COL, fontsize=10, pad=6)

    # Event log (right bottom)
    show_event_log(event_log=event_log, ax=ax_log)
    ax_log.set_title("Event Log", color=TEXT_COL, fontsize=10, pad=6)

    # ── Save & show ───────────────────────────────────────────────────────────
    out_path = "aeronet_dashboard.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor=DARK_BG, dpi=130)
    print(f"  [Viz] Dashboard saved → {out_path}")

    try:
        plt.show()
    except Exception:
        pass   # headless / CI environment


def _draw_fleet_bar(ax, drones):
    """Mini fleet status bar chart used inside show_all."""
    if not drones:
        ax.text(0.5, 0.5, "No drone data",
                ha="center", va="center",
                color=MUTED, fontsize=11,
                transform=ax.transAxes)
        ax.axis("off")
        return

    status_counts: dict = {}
    battery_by_drone = {}
    for d in drones:
        status_counts[d.status] = status_counts.get(d.status, 0) + 1
        battery_by_drone[d.drone_id] = d.battery

    labels = list(status_counts.keys())
    values = [status_counts[l] for l in labels]
    palette = {
        "idle":      "#3FB950",
        "en_route":  "#58A6FF",
        "returning": "#FFA657",
        "anomaly":   "#FF7B72",
    }
    colors = [palette.get(l, "#8B949E") for l in labels]

    bars = ax.bar(labels, values, color=colors,
                  edgecolor=DARK_BG, linewidth=1.2, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                str(val),
                ha="center", va="bottom",
                color=TEXT_COL, fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(values) + 1.5)
    ax.set_ylabel("Count", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", alpha=0.2)

    # Mini battery strip
    drone_ids = list(battery_by_drone.keys())
    batteries = [battery_by_drone[d] for d in drone_ids]
    inset = ax.inset_axes([0.0, -0.42, 1.0, 0.28])
    inset.set_facecolor(PANEL_BG)
    bar_colors = ["#3FB950" if b > 50 else "#FFA657" if b > 20 else "#FF7B72"
                  for b in batteries]
    inset.bar(range(len(drone_ids)), batteries, color=bar_colors,
              edgecolor=DARK_BG, linewidth=0.8)
    inset.set_xticks(range(len(drone_ids)))
    inset.set_xticklabels(drone_ids, rotation=45, fontsize=6, color=MUTED)
    inset.set_ylabel("Battery %", color=MUTED, fontsize=6.5)
    inset.set_ylim(0, 110)
    inset.axhline(20, color="#FF7B72", linewidth=0.8, linestyle="--", alpha=0.7)
    inset.grid(axis="y", alpha=0.15)
    for spine in inset.spines.values():
        spine.set_edgecolor(GRID_LINE)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from grid_model import build_default_grid
    grid = build_default_grid()
    print("\n  Running full AeroNet Lite dashboard…")
    show_all(grid)
