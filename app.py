"""VidReplica — Flask web application."""

import logging
import os
import threading
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

from pipeline import check_ffmpeg, jobs, run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    video = request.files.get("video")
    product_image = request.files.get("product_image")
    amendments = request.form.get("amendments", "").strip()

    if not video:
        return jsonify({"error": "No video file uploaded"}), 400
    if not product_image:
        return jsonify({"error": "No product image uploaded"}), 400

    # Validate file extensions
    video_ext = os.path.splitext(video.filename)[1].lower()
    image_ext = os.path.splitext(product_image.filename)[1].lower()
    if video_ext not in (".mp4", ".mov"):
        return jsonify({"error": "Video must be .mp4 or .mov"}), 400
    if image_ext not in (".jpg", ".jpeg", ".png"):
        return jsonify({"error": "Product image must be .jpg, .jpeg, or .png"}), 400

    # Save uploads
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_path = os.path.join(job_dir, f"source{video_ext}")
    product_path = os.path.join(job_dir, f"product{image_ext}")
    video.save(video_path)
    product_image.save(product_path)

    # Initialise job state
    jobs[job_id] = {
        "status": "queued",
        "current_segment": 0,
        "total_segments": 0,
        "download_url": None,
        "error": None,
    }

    # Run pipeline in background thread
    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, video_path, product_path, amendments),
        daemon=True,
    )
    thread.start()
    log.info("Job %s started", job_id)

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    path = os.path.join("outputs", f"{job_id}.mp4")
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name="vidreplica_output.mp4")


if __name__ == "__main__":
    check_ffmpeg()

    # Check required env vars
    missing = [k for k in ("ANTHROPIC_API_KEY", "BYTEPLUS_API_KEY") if not os.environ.get(k)]
    if missing:
        log.warning("Missing environment variables: %s — set them in .env", ", ".join(missing))

    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
