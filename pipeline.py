"""Core video replication pipeline."""

import glob
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import whisper

from prompts import generate_seedance_prompt
from seedance import download_video, generate_segment

log = logging.getLogger(__name__)

SEGMENT_DURATION = 8  # seconds
FRAMES_PER_SEGMENT = 9

# In-memory job store
jobs: dict[str, dict] = {}


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["ffmpeg", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("ffmpeg error: %s", result.stderr)
        # ffmpeg prints a long version banner to stderr first; take the tail to get the actual error
        tail = result.stderr[-800:].strip()
        raise RuntimeError(f"ffmpeg failed: {tail}")
    return result


def check_ffmpeg():
    """Raise if ffmpeg is not installed."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "ffmpeg is not installed or not on PATH. "
            "Install it: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )


def split_video(video_path: str, job_dir: str) -> list[str]:
    """Split a video into 8-second segments. Returns list of segment paths."""
    seg_pattern = os.path.join(job_dir, "seg_%03d.mp4")
    _run_ffmpeg([
        "-i", video_path,
        "-f", "segment",
        "-segment_time", str(SEGMENT_DURATION),
        "-reset_timestamps", "1",
        "-c", "copy",
        seg_pattern,
    ])
    segments = sorted(glob.glob(os.path.join(job_dir, "seg_*.mp4")))
    log.info("Split video into %d segments", len(segments))
    return segments


def extract_frames(segment_path: str, job_dir: str, seg_index: int) -> list[str]:
    """Extract 9 evenly-spaced frames from an 8-second segment."""
    frame_dir = os.path.join(job_dir, f"frames_{seg_index:03d}")
    os.makedirs(frame_dir, exist_ok=True)
    pattern = os.path.join(frame_dir, "frame_%02d.jpg")
    _run_ffmpeg([
        "-i", segment_path,
        "-vf", f"fps={FRAMES_PER_SEGMENT}/{SEGMENT_DURATION}",
        "-vframes", str(FRAMES_PER_SEGMENT),
        pattern,
    ])
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    log.info("Extracted %d frames for segment %d", len(frames), seg_index)
    return frames


def transcribe_segment(segment_path: str, whisper_model) -> str:
    """Transcribe audio from a video segment using Whisper."""
    try:
        result = whisper_model.transcribe(segment_path, language="en", fp16=False)
        text = result.get("text", "").strip()
        log.info("Transcript: %s", text[:100])
        return text
    except Exception as e:
        log.warning("Whisper failed on %s: %s — continuing without transcript", segment_path, e)
        return ""


def mux_audio(silent_path: str, audio_source_path: str, output_path: str) -> str:
    """Mux audio from the original source video into the silent generated video."""
    _run_ffmpeg([
        "-i", silent_path,
        "-i", audio_source_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y",
        output_path,
    ])
    log.info("Muxed audio from %s into %s", audio_source_path, output_path)
    return output_path


def stitch_segments(segment_paths: list[str], output_path: str) -> str:
    """Concatenate generated segment videos into one final MP4."""
    if not segment_paths:
        raise RuntimeError("No segments to stitch")

    list_file = output_path + ".txt"
    with open(list_file, "w") as f:
        for sp in segment_paths:
            f.write(f"file '{os.path.abspath(sp)}'\n")

    _run_ffmpeg([
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path,
    ])
    os.remove(list_file)
    log.info("Stitched %d segments → %s", len(segment_paths), output_path)
    return output_path


def run_pipeline(job_id: str, video_path: str, product_image_path: str, amendments: str):
    """Run the full replication pipeline (called in background thread)."""
    job = jobs[job_id]
    job_dir = os.path.join("segments", job_id)
    output_dir = os.path.join("outputs")

    try:
        os.makedirs(job_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        # Load Whisper model
        model_name = os.environ.get("WHISPER_MODEL", "base")
        log.info("[%s] Loading Whisper model: %s", job_id, model_name)
        job["status"] = "loading_whisper"
        whisper_model = whisper.load_model(model_name)

        # Step 1: Split video
        log.info("[%s] Splitting video into %ds segments...", job_id, SEGMENT_DURATION)
        job["status"] = "splitting"
        segments = split_video(video_path, job_dir)
        job["total_segments"] = len(segments)

        generated_segments = []

        for i, seg_path in enumerate(segments):
            seg_num = i + 1
            job["current_segment"] = seg_num
            log.info("[%s] Processing segment %d/%d", job_id, seg_num, len(segments))

            # Step 2: Extract frames
            job["status"] = f"extracting_frames"
            frames = extract_frames(seg_path, job_dir, i)

            # Step 3: Transcribe
            job["status"] = f"transcribing"
            transcript = transcribe_segment(seg_path, whisper_model)

            # Step 4: Generate Claude prompt
            job["status"] = f"generating_prompt"
            prompt = generate_seedance_prompt(frames, product_image_path, transcript, amendments)
            log.info("[%s] Segment %d prompt: %s", job_id, seg_num, prompt[:120])

            # Step 5: Generate via Seedance
            job["status"] = f"generating_video"
            video_url = None
            for attempt in range(3):
                video_url = generate_segment(frames, product_image_path, prompt)
                if video_url:
                    break
                log.warning("[%s] Segment %d attempt %d failed, retrying...", job_id, seg_num, attempt + 1)

            if video_url:
                dest = os.path.join(job_dir, f"gen_{i:03d}.mp4")
                download_video(video_url, dest)
                generated_segments.append(dest)
            else:
                log.error("[%s] Segment %d failed after 3 retries — skipping", job_id, seg_num)

        # Step 6: Stitch
        if not generated_segments:
            job["status"] = "failed"
            job["error"] = "All segments failed generation"
            return

        job["status"] = "stitching"
        silent_path = os.path.join(output_dir, f"{job_id}_silent.mp4")
        output_path = os.path.join(output_dir, f"{job_id}.mp4")
        stitch_segments(generated_segments, silent_path)

        job["status"] = "muxing_audio"
        mux_audio(silent_path, video_path, output_path)
        os.remove(silent_path)

        job["status"] = "complete"
        job["download_url"] = f"/download/{job_id}"
        log.info("[%s] Pipeline complete! Output: %s", job_id, output_path)

    except Exception as e:
        log.exception("[%s] Pipeline error: %s", job_id, e)
        job["status"] = "failed"
        job["error"] = str(e)

    finally:
        # Clean up temp files (keep outputs)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(product_image_path):
                os.remove(product_image_path)
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
        except Exception as e:
            log.warning("Cleanup error: %s", e)
