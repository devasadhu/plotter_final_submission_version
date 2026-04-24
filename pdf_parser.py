"""
pdf_parser.py
-------------
Converts a PDF page into a numpy image array using PyMuPDF (fitz).
No Poppler installation required — works out of the box on Windows.
"""

import fitz  # PyMuPDF
import numpy as np
import cv2


def pdf_to_image(pdf_path: str, page_number: int = 0, dpi: int = 300) -> np.ndarray:
    """
    Render a PDF page to a BGR numpy array (OpenCV-compatible).

    Args:
        pdf_path:    Path to the PDF file.
        page_number: Which page to render (0-indexed, default first page).
        dpi:         Rendering resolution (300 gives crisp images).

    Returns:
        numpy ndarray of shape (H, W, 3) in BGR format.
    """
    doc = fitz.open(pdf_path)

    if page_number >= len(doc):
        raise ValueError(f"PDF has {len(doc)} page(s), but page {page_number} was requested.")

    page = doc[page_number]
    zoom = dpi / 72.0          # 72 dpi is the PDF baseline
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    doc.close()

    # Convert to numpy
    img = np.frombuffer(pixmap.samples, dtype=np.uint8)
    img = img.reshape(pixmap.height, pixmap.width, pixmap.n)

    # Ensure BGR (OpenCV default)
    if pixmap.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to.pdf>")
        sys.exit(1)

    image = pdf_to_image(sys.argv[1])
    cv2.imwrite("shape_input.png", image)
    print(f"✓ Saved shape_input.png  [{image.shape[1]} x {image.shape[0]} px]")
