"""
sim.py
------
MuJoCo simulation for the two-arm drawing robot.

Usage:
    python sim.py                          # manual mode (type t / s / h)
    python sim.py --pdf shapes/tri.pdf     # auto-draw shape from PDF
    python sim.py --pdf shapes/tri.pdf --mode dense   # smooth contour mode

Manual commands (typed in the terminal while simulation runs):
    t  → draw a test triangle
    s  → draw a test square
    h  → return arms to home position
    q  → quit
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import queue
import threading
import argparse

# ── Robot parameters (must match drawing_bot.xml) ────────────────────────────
L1, L2, D = 0.15, 0.27, 0.075


# ── Inverse Kinematics ────────────────────────────────────────────────────────
def get_ik_radians(x: float, y: float):
    """Return (ctrl_left, ctrl_right) motor angles in radians for tip position (x, y)."""
    # Left arm  (hip at x = -D)
    dist_l   = np.sqrt((x + D)**2 + y**2)
    phi_l    = np.arctan2(y, x + D)
    cos_l    = np.clip((L1**2 + dist_l**2 - L2**2) / (2 * L1 * dist_l), -1, 1)
    ctrl_l   = (phi_l + np.arccos(cos_l)) - (np.pi / 2)

    # Right arm (hip at x = +D)
    dist_r   = np.sqrt((x - D)**2 + y**2)
    phi_r    = np.arctan2(y, x - D)
    cos_r    = np.clip((L1**2 + dist_r**2 - L2**2) / (2 * L1 * dist_r), -1, 1)
    ctrl_r   = (np.pi / 2) - (phi_r - np.arccos(cos_r))

    return ctrl_l, ctrl_r


# ── Path generation ───────────────────────────────────────────────────────────
def _transition(start_ctrls, target_ctrls, steps: int = 1000):
    """Smooth linear interpolation in joint space from start → target."""
    path = []
    for t in np.linspace(0, 1, steps):
        l = start_ctrls[0] + t * (target_ctrls[0] - start_ctrls[0])
        r = start_ctrls[1] + t * (target_ctrls[1] - start_ctrls[1])
        path.append((l, r))
    return path


def generate_path_from_points(robot_pts, start_ctrls, steps_per_segment: int = 3000):
    """
    Build a full joint-angle path from an ordered list of (x, y) robot coords.

    Args:
        robot_pts:          List of (x, y) in metres (closed path).
        start_ctrls:        Current [ctrl_left, ctrl_right] joint angles.
        steps_per_segment:  Interpolation steps between consecutive waypoints.

    Returns:
        (path, transition_length)
        path              : List of (ctrl_l, ctrl_r) tuples.
        transition_length : Number of steps before actual drawing begins.
    """
    if not robot_pts:
        return [], 0

    # Move to the first point from wherever the arm currently is
    first_ik   = get_ik_radians(*robot_pts[0])
    transition = _transition(start_ctrls, first_ik, steps=1200)
    path       = list(transition)
    transition_length = len(path)

    # Draw each segment
    for i in range(len(robot_pts) - 1):
        x0, y0 = robot_pts[i]
        x1, y1 = robot_pts[i + 1]
        for t in np.linspace(0, 1, steps_per_segment):
            tx = x0 + t * (x1 - x0)
            ty = y0 + t * (y1 - y0)
            path.append(get_ik_radians(tx, ty))

    return path, transition_length


def generate_shape_path(shape_type: str, start_ctrls):
    """Built-in hardcoded shapes for quick manual testing."""
    shapes = {
        's': [[-0.03, 0.32], [0.03, 0.32], [0.03, 0.38], [-0.03, 0.38], [-0.03, 0.32]],
        't': [[0, 0.40], [-0.04, 0.32], [0.04, 0.32], [0, 0.40]],
    }
    corners = shapes.get(shape_type)
    if not corners:
        return [], 0
    return generate_path_from_points(corners, start_ctrls)


# ── Windows-safe keyboard listener ───────────────────────────────────────────
_cmd_queue: queue.Queue = queue.Queue()

def _stdin_listener():
    """Background thread: reads lines from stdin without blocking the render loop."""
    while True:
        try:
            line = input()
            _cmd_queue.put(line.strip().lower())
        except EOFError:
            break

threading.Thread(target=_stdin_listener, daemon=True).start()


# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RIA Drawing Bot Simulation")
parser.add_argument("--pdf",  type=str, default=None, help="Path to input PDF")
parser.add_argument("--mode", type=str, default="polygon",
                    choices=["polygon", "dense"], help="Contour extraction mode")
args = parser.parse_args()


# ── Load model ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
model = mujoco.MjModel.from_xml_path(os.path.join(_HERE, "drawing_bot.xml"))
data  = mujoco.MjData(model)

trace_history:     list = []
active_path:       list = []
p_idx:             int  = 0
draw_start_idx:    int  = 0   # trace only recorded after this index

print("\n" + "=" * 50)
print("  RIA Drawing Bot Simulation")
print("=" * 50)
print("  Commands: [t] Triangle  [s] Square  [h] Home  [q] Quit")
print("=" * 50 + "\n")


# ── Auto-load PDF if provided ─────────────────────────────────────────────────
if args.pdf:
    print(f"[PDF] Processing: {args.pdf}")
    try:
        from pipeline import run_pipeline
        robot_pts, _, _ = run_pipeline(args.pdf, mode=args.mode, save_debug=True)
        active_path, draw_start_idx = generate_path_from_points(
            robot_pts, [data.ctrl[0], data.ctrl[1]]
        )
        trace_history = []
        p_idx = 0
        print(f"[OK] Auto-drawing shape ({len(active_path)} steps, drawing starts at step {draw_start_idx})...\n")
    except Exception as e:
        print(f"[ERROR] Pipeline error: {e}")
        import traceback; traceback.print_exc()
        print("   Falling back to manual mode.\n")


# ── Main simulation loop ──────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # --- Check for keyboard commands ---
        try:
            cmd = _cmd_queue.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd in ("s", "t"):
            active_path, draw_start_idx = generate_shape_path(cmd, [data.ctrl[0], data.ctrl[1]])
            trace_history = []
            p_idx         = 0
            print(f"[>] Drawing {'square' if cmd == 's' else 'triangle'}...")

        elif cmd == "h":
            active_path   = _transition([data.ctrl[0], data.ctrl[1]], [0.0, 0.0])
            draw_start_idx = 0
            trace_history = []
            p_idx         = 0
            print("[>] Returning to home...")

        elif cmd == "q":
            print("Goodbye.")
            break

        # --- Advance robot along path ---
        if active_path:
            data.ctrl[0], data.ctrl[1] = active_path[p_idx]
            if p_idx < len(active_path) - 1:
                p_idx += 1
                # Only record trace AFTER the transition phase is complete
                if p_idx > draw_start_idx and p_idx % 15 == 0:
                    trace_history.append(data.site("pencil_point").xpos.copy())

        # --- Draw trace as green spheres ---
        viewer.user_scn.ngeom = 0
        for i, pos in enumerate(trace_history[:1000]):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.002, 0.002, 0.002],
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[0, 1, 0, 1],
            )
            viewer.user_scn.ngeom += 1

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)