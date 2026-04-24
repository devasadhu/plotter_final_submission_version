"""
shape_extractor.py
------------------
Detects the primary shape outline from a rendered PDF image.
Returns an ordered list of (x, y) pixel coordinates.

Supports two modes:
  - polygon   : fast, clean corner-only approximation (default)
  - dense     : smooth, many-point contour (good for circles/curves)
"""

import cv2
import numpy as np
from typing import List, Tuple


def extract_shape(
    image: np.ndarray,
    mode: str = "polygon",
    epsilon_factor: float = 0.02,
    max_dense_points: int = 500,
    min_area_fraction: float = 0.001,
) -> List[Tuple[int, int]]:
    """
    Extract the primary shape outline from a BGR image.

    Args:
        image:             Input BGR numpy array.
        mode:              'polygon' for corner approximation, 'dense' for full contour.
        epsilon_factor:    Polygon approx accuracy (fraction of arc length).
        max_dense_points:  Max points returned in dense mode.
        min_area_fraction: Ignore contours smaller than this fraction of image area.

    Returns:
        Closed list of (x, y) pixel coords, i.e. first point == last point.
    """
    h, w = image.shape[:2]
    min_area = h * w * min_area_fraction

    # --- Pre-process ---
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's threshold – auto-selects best value for black-on-white shapes
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological clean-up (removes speckle noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)

    # --- Find contours ---
    method = cv2.CHAIN_APPROX_SIMPLE if mode == "polygon" else cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, method)

    if not contours:
        raise RuntimeError(
            "No contours found. Make sure the PDF has a clear black shape on a white background."
        )

    # Keep only contours with meaningful area, take the largest
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        raise RuntimeError("No significant shape found (all contours too small).")

    largest = max(valid, key=cv2.contourArea)

    # --- Approximate / sample ---
    if mode == "polygon":
        epsilon = epsilon_factor * cv2.arcLength(largest, True)
        approx  = cv2.approxPolyDP(largest, epsilon, True)
        points  = [(int(p[0][0]), int(p[0][1])) for p in approx]
    else:  # dense
        step   = max(1, len(largest) // max_dense_points)
        points = [(int(largest[i][0][0]), int(largest[i][0][1]))
                  for i in range(0, len(largest), step)]

    # Ensure the path is closed
    if points[0] != points[-1]:
        points.append(points[0])

    return points


def debug_draw(image: np.ndarray, points: List[Tuple[int, int]], out_path: str = "shape_debug.png"):
    """Overlay detected contour on image and save for visual verification."""
    debug = image.copy()
    pts   = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug, [pts], True, (0, 255, 100), 3)
    for p in points:
        cv2.circle(debug, p, 7, (0, 100, 255), -1)
    cv2.imwrite(out_path, debug)
    print(f"[OK] Debug overlay saved -> {out_path}")


if __name__ == "__main__":
    import sys
    img = cv2.imread("shape_input.png")
    if img is None:
        print("Run  python pdf_parser.py <file.pdf>  first to generate shape_input.png")
        sys.exit(1)

    pts = extract_shape(img, mode="polygon")
    print(f"[OK] Detected {len(pts)} points: {pts}")
    debug_draw(img, pts)
