# RIA Drawing Bot

A MuJoCo-based two-arm robot simulation that reads a shape from a PDF and draws it automatically.

---

## Project Structure

```
RIA-PROJECT/
├── frontend/
│   └── index.html          ← Web UI (drag-drop PDF upload)
├── shapes/                 ← Put your test PDFs here
├── uploads/                ← Auto-created; stores uploaded PDFs
├── output/                 ← Auto-created; stores debug images
├── drawing_bot.xml         ← MuJoCo robot model
├── sim.py                  ← Simulation (auto-draws from PDF)
├── app.py                  ← Flask web server (main entry point)
├── pipeline.py             ← Orchestrates the full pipeline
├── pdf_parser.py           ← PDF → image (PyMuPDF)
├── shape_extractor.py      ← Image → shape contour (OpenCV)
├── coord_mapper.py         ← Pixel coords → robot workspace
└── requirements.txt        ← Python dependencies
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the web server
```bash
python app.py
```

### 3. Open the web UI
Navigate to: **http://localhost:5000**

### 4. Upload a PDF
- Drag-and-drop (or browse) a PDF containing a shape
- Click **Process PDF** — the pipeline runs automatically
- Preview the detected shape in the UI
- Click **Launch Simulation** — the MuJoCo window opens and the robot draws the shape

---

## CLI Usage (without web UI)

Draw from a PDF directly:
```bash
python sim.py --pdf shapes/triangle.pdf
```

Smooth contour mode (good for circles):
```bash
python sim.py --pdf shapes/circle.pdf --mode dense
```

Manual keyboard control (while sim is running):
| Key | Action |
|-----|--------|
| `t` | Draw test triangle |
| `s` | Draw test square |
| `h` | Return to home position |
| `q` | Quit |

---

## Pipeline Explained

```
PDF file
  └─▶ pdf_parser.py      → Renders PDF page to image (300 DPI, PyMuPDF)
        └─▶ shape_extractor.py → OpenCV contour detection → pixel (x,y) points
              └─▶ coord_mapper.py   → Scales to robot workspace (metres)
                    └─▶ sim.py          → IK → MuJoCo joint angles → robot draws
```

### Contour Modes
| Mode | Description | Use when |
|------|-------------|----------|
| `polygon` | Approximates contour to corner vertices (default) | Triangles, squares, rectangles |
| `dense` | Returns many evenly-spaced points | Circles, ellipses, curved shapes |

---

## PDF Guidelines for Best Results

- **Black outline** on a **white background**
- Shape should be **large** (fill most of the page)
- **Single shape** per page — no text, no borders
- Avoid very thin lines (< 2pt stroke)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mujoco` | Physics simulation & viewer |
| `numpy` | Numerical computing |
| `opencv-python` | Image processing & contour detection |
| `PyMuPDF` | PDF rendering (no Poppler needed on Windows) |
| `flask` | Web server |
| `flask-cors` | Cross-origin requests |

Install all at once:
```bash
pip install -r requirements.txt
```
