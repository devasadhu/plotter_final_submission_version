"""
coord_mapper.py
---------------
Maps image pixel coordinates → robot workspace coordinates (meters).

Robot safe drawing zone (derived from drawing_bot.xml arm geometry):
  x ∈ [-0.04,  0.04]  (left–right)
  y ∈ [ 0.32,  0.40]  (forward distance from robot base)

The shape is scaled uniformly to fill ~85% of this zone,
centred in the workspace, and the Y-axis is flipped
(image Y increases downward; robot Y increases forward).
"""

import numpy as np
from typing import List, Tuple

# ── Robot workspace boundaries (meters) ──────────────────────────────────────
WS_X_MIN, WS_X_MAX = -0.04,  0.04
WS_Y_MIN, WS_Y_MAX =  0.32,  0.40
MARGIN = 0.85   # use 85% of the workspace to stay safely inside limits


def map_to_robot(
    pixel_points: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Scale and centre pixel (x, y) coordinates into the robot workspace.

    Args:
        pixel_points: Ordered list of (px, py) pixel coords (closed path).

    Returns:
        Ordered list of (rx, ry) in metres, same order as input.
    """
    if not pixel_points:
        return []

    pts  = np.array(pixel_points, dtype=float)
    px, py = pts[:, 0], pts[:, 1]

    # Bounding box of the shape in pixel space
    px_min, px_max = px.min(), px.max()
    py_min, py_max = py.min(), py.max()
    shape_w = px_max - px_min
    shape_h = py_max - py_min

    if shape_w < 1e-6 or shape_h < 1e-6:
        raise ValueError("Extracted shape has near-zero width or height.")

    # Available workspace with margin
    ws_w = (WS_X_MAX - WS_X_MIN) * MARGIN
    ws_h = (WS_Y_MAX - WS_Y_MIN) * MARGIN

    # Uniform scale (preserve aspect ratio)
    scale = min(ws_w / shape_w, ws_h / shape_h)

    # Centres
    cx = (WS_X_MIN + WS_X_MAX) / 2.0
    cy = (WS_Y_MIN + WS_Y_MAX) / 2.0
    shape_cx = (px_min + px_max) / 2.0
    shape_cy = (py_min + py_max) / 2.0

    robot_points = []
    for (px_i, py_i) in pixel_points:
        rx =  cx + (px_i - shape_cx) * scale          # X same direction
        ry =  cy - (py_i - shape_cy) * scale          # Y flipped (image ↓ = robot ↑)
        robot_points.append((float(rx), float(ry)))

    return robot_points


if __name__ == "__main__":
    # Quick sanity-check with a triangle
    test_px = [(300, 100), (100, 400), (500, 400), (300, 100)]
    result  = map_to_robot(test_px)
    print("✓ Mapped robot coords (metres):")
    for r in result:
        print(f"   x={r[0]:+.4f}  y={r[1]:.4f}")
