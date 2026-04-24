"""
app.py
------
Flask web server — the single entry point for the whole pipeline.

Run:  python app.py
Then open: http://localhost:5000
"""

import os
import sys
import base64
import subprocess

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from pipeline        import run_pipeline
from shape_extractor import debug_draw

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend")
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory state for last processed PDF
_state = {"pdf_path": None, "robot_pts": [], "mode": "polygon"}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the web UI."""
    return send_from_directory("frontend", "index.html")


@app.route("/process", methods=["POST"])
def process_pdf():
    """
    Accept a PDF upload, run the full pipeline, return:
      - pixel_points  : for canvas preview
      - robot_points  : final robot coordinates
      - preview_image : base64 debug overlay PNG
    """
    if "pdf" not in request.files:
        return jsonify({"error": "No file received. Please upload a PDF."}), 400

    f = request.files["pdf"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only .pdf files are supported."}), 400

    mode = request.form.get("mode", "polygon")   # 'polygon' or 'dense'

    # Save uploaded file
    pdf_path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(pdf_path)

    try:
        robot_pts, pixel_pts, image = run_pipeline(pdf_path, mode=mode, save_debug=False)

        # Save debug overlay
        debug_path = os.path.join(OUTPUT_DIR, "shape_debug.png")
        debug_draw(image, pixel_pts, debug_path)

        # Base64-encode the preview image for the frontend
        with open(debug_path, "rb") as fh:
            preview_b64 = base64.b64encode(fh.read()).decode()

        h, w = image.shape[:2]

        # Persist state for /simulate
        _state["pdf_path"]  = pdf_path
        _state["robot_pts"] = robot_pts
        _state["mode"]      = mode

        return jsonify({
            "success":       True,
            "pixel_points":  pixel_pts,
            "robot_points":  robot_pts,
            "image_width":   w,
            "image_height":  h,
            "num_points":    len(robot_pts),
            "preview_image": f"data:image/png;base64,{preview_b64}",
            "mode":          mode,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()   # print full trace to server console
        return jsonify({"error": str(exc)}), 500


@app.route("/simulate", methods=["POST"])
def simulate():
    """Launch sim.py in a new console window with the processed PDF."""
    if not _state["pdf_path"]:
        return jsonify({"error": "Upload and process a PDF first."}), 400

    pdf_path = _state["pdf_path"]
    mode     = _state["mode"]

    if not os.path.exists(pdf_path):
        return jsonify({"error": f"Processed PDF not found at: {pdf_path}"}), 400

    project_dir = os.path.dirname(os.path.abspath(__file__))
    sim_script  = os.path.join(project_dir, "sim.py")
    log_path    = os.path.join(project_dir, "output", "sim_error.log")
    cmd         = [sys.executable, sim_script, "--pdf", pdf_path, "--mode", mode]

    # Set UTF-8 encoding so Unicode prints don't crash the subprocess console
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        if sys.platform == "win32":
            # Open a visible console; stderr goes to log file for debugging
            with open(log_path, "w") as log_f:
                subprocess.Popen(
                    cmd,
                    cwd=project_dir,
                    env=env,
                    stderr=log_f,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
        else:
            subprocess.Popen(cmd, cwd=project_dir, env=env)

        return jsonify({"success": True,
                        "message": "Simulation launched! Check the MuJoCo window.",
                        "log": log_path})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/status")
def status():
    """Returns whether a PDF has been processed."""
    return jsonify({
        "ready":      _state["pdf_path"] is not None,
        "pdf_path":   _state["pdf_path"],
        "num_points": len(_state["robot_pts"]),
        "mode":       _state["mode"],
    })


# ── Global error handlers (always return JSON, never HTML) ────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "detail": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.errorhandler(Exception)
def unhandled(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  RIA Drawing Bot  -  Web Interface")
    print("=" * 52)
    print("  Open  http://localhost:5000  in your browser")
    print("=" * 52 + "\n")
    app.run(debug=False, port=5000)
