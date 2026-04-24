"""
pipeline.py
-----------
Orchestrates the full PDF → robot-path pipeline in one call.

Usage (from another script):
    from pipeline import run_pipeline
    robot_pts, pixel_pts, image = run_pipeline("shapes/triangle.pdf")

Usage (standalone test):
    python pipeline.py shapes/triangle.pdf
"""

import cv2
import numpy as np
from typing import Tuple, List

from pdf_parser      import pdf_to_image
from shape_extractor import extract_shape
from coord_mapper    import map_to_robot


def run_pipeline(
    pdf_path: str,
    mode: str = "polygon",
    dpi: int = 300,
    save_debug: bool = True,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]], np.ndarray]:
    """
    Full PDF → robot-path pipeline.

    Args:
        pdf_path:   Path to the input PDF.
        mode:       'polygon' (corners only) or 'dense' (smooth curves).
        dpi:        PDF rendering resolution.
        save_debug: If True, saves shape_input.png and shape_debug.png.

    Returns:
        robot_pts  : List of (x, y) coords in robot workspace (metres).
        pixel_pts  : List of (x, y) pixel coords (for preview / web UI).
        image      : The rendered PDF page as a BGR numpy array.
    """
    print(f"[1/3] Parsing PDF  ->  {pdf_path}")
    image = pdf_to_image(pdf_path, dpi=dpi)
    if save_debug:
        cv2.imwrite("shape_input.png", image)
        print("      Saved shape_input.png")

    print(f"[2/3] Extracting shape contour  (mode={mode})")
    pixel_pts = extract_shape(image, mode=mode)
    print(f"      Detected {len(pixel_pts)} point(s)")

    if save_debug:
        from shape_extractor import debug_draw
        debug_draw(image, pixel_pts, "shape_debug.png")

    print("[3/3] Mapping to robot workspace")
    robot_pts = map_to_robot(pixel_pts)
    print(f"      x in [{min(p[0] for p in robot_pts):.4f}, {max(p[0] for p in robot_pts):.4f}]")
    print(f"      y in [{min(p[1] for p in robot_pts):.4f}, {max(p[1] for p in robot_pts):.4f}]")
    print("[OK] Pipeline complete.")

    return robot_pts, pixel_pts, image


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <path_to.pdf>")
        sys.exit(1)

    robot_pts, pixel_pts, _ = run_pipeline(sys.argv[1])
    print("\nRobot path (metres):")
    for pt in robot_pts:
        print(f"  x={pt[0]:+.4f}  y={pt[1]:.4f}")
