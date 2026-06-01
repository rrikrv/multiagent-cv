# ============================================================
#  main.py — Entry point
#
#  USAGE:
#    python3 src/main.py
#    python3 src/main.py --case maze
#    python3 src/main.py --case bottleneck --fov 5
#    python3 src/main.py --case thin_walls --fov 3 --speed 50
#    python3 src/main.py --case my_new_map  (any folder in data/)
#
#  CONTROLS:
#    Space   — pause / resume
#    Q / Esc — quit
# ============================================================

import os, sys, argparse, cv2
sys.path.insert(0, os.path.dirname(__file__))

from map_parser import load_map
from simulator  import Simulator
from cv_module  import CVModule

DEFAULT_CASE     = "warehouse"
DEFAULT_FOV      = 0
DEFAULT_CELLSIZE = 18
DEFAULT_SPEED    = 100


def get_available_cases():
    """
    Scans the data/ folder and returns a list of all valid polygon cases.
    A valid case must contain map.png, scenario.json, and edges.csv.
    No code changes needed to add a new polygon — just drop the folder here.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    if not os.path.isdir(data_dir):
        return []
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.exists(os.path.join(data_dir, d, "scenario.json"))
        and os.path.exists(os.path.join(data_dir, d, "edges.csv"))
        and os.path.exists(os.path.join(data_dir, d, "map.png"))
    ])


def parse_args():
    """Reads command-line arguments: case name, FoV radius, simulation speed."""
    cases = get_available_cases()
    p = argparse.ArgumentParser(
        description="Multi-agent CV simulator. Works with any polygon in data/."
    )
    p.add_argument(
        "--case",
        default=DEFAULT_CASE,
        help=f"Polygon folder name inside data/. Available: {cases}. "
             f"Default: {DEFAULT_CASE}"
    )
    p.add_argument("--fov",   type=int, default=DEFAULT_FOV,
                   help="Field of view radius (0 = disabled, full map visible)")
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED,
                   help="Milliseconds per tick (lower = faster)")
    return p.parse_args()


def main():
    args     = parse_args()
    case     = args.case
    fov      = args.fov
    speed    = args.speed
    data_dir = os.path.join("data", case)

    # Check that the polygon folder exists and has all required files
    if not os.path.isdir(data_dir):
        available = get_available_cases()
        print(f"Error: folder '{data_dir}' not found.")
        print(f"Available cases: {available}")
        print(f"Place new polygon files into data/{case}/")
        return

    required = ["map.png", "scenario.json", "edges.csv", "metrics.json"]
    missing  = [f for f in required
                if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print(f"Error: missing files in '{data_dir}': {missing}")
        return

    print("=" * 45)
    print(f"  Case:   {case}")
    print(f"  FoV:    {fov if fov > 0 else 'disabled'}")
    print(f"  Speed:  {speed} ms/tick")
    print("=" * 45)

    # Step 1: load and binarize the map image
    grid = load_map(
        os.path.join(data_dir, "map.png"),
        metrics_path=os.path.join(data_dir, "metrics.json")
    )
    if grid is None:
        print("Error: failed to load map.")
        return
    h, w = grid.shape
    print(f"  Map: {w} x {h} cells")

    # Step 2: initialize the simulator (loads agents, graph, tasks)
    sim = Simulator(
        scenario_path=os.path.join(data_dir, "scenario.json"),
        edges_path=os.path.join(data_dir, "edges.csv"),
    )

    # Step 3: initialize the CV module (classification + visualization)
    cv_mod = CVModule(static_grid=grid,
                      cell_size=DEFAULT_CELLSIZE,
                      fov_radius=fov)

    print()
    print("  Space = pause,  Q/Esc = quit")
    print()

    paused = False
    tick   = 0
    window = f"CV Vision | {case}"

    while True:
        key = cv2.waitKey(speed) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            print(f"  {'[PAUSED]' if paused else '[RESUMED]'}")

        if not paused:
            # Move all agents one step forward
            sim.update()
            tick += 1

            # Check if all tasks are completed
            if sim.all_done():
                print(f"\n  All tasks completed in {tick} ticks!")
                positions = sim.get_agent_positions()
                markers   = sim.get_task_markers()
                cg        = cv_mod.classify(positions, markers)
                img       = cv_mod.render(cg)
                cv2.putText(img, f"DONE! Ticks: {tick}", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (60, 210, 100), 1, cv2.LINE_AA)
                cv2.imshow(window, img)
                cv2.waitKey(3000)
                break

        # Get current agent positions and task markers
        positions = sim.get_agent_positions()
        markers   = sim.get_task_markers()

        # Classify objects: background / static obstacles / dynamic agents
        cg  = cv_mod.classify(positions, markers)

        # Render the classified grid into an RGB image and display it
        img = cv_mod.render(cg)

        status = f"Tick: {tick}" + ("  [PAUSE]" if paused else "")
        cv2.putText(img, status, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # Scale down if the window is too wide for the screen
        max_w = 1200
        if img.shape[1] > max_w:
            scale = max_w / img.shape[1]
            img = cv2.resize(img, (max_w, int(img.shape[0] * scale)),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window, img)

    cv2.destroyAllWindows()
    print(f"  Finished at tick {tick}.")


if __name__ == "__main__":
    main()