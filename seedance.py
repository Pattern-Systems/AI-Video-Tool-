"""Seedance 2.0 API client (BytePlus)."""

import base64
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
MAX_RETRIES = 3
POLL_INTERVAL = 10  # seconds


def _headers():
    return {
        "Authorization": f"Bearer {os.environ['BYTEPLUS_API_KEY']}",
        "Content-Type": "application/json",
    }


def _b64_image(path: str) -> str:
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def generate_segment(
    frame_paths: list[str],
    product_image_path: str,
    prompt: str,
    model: str | None = None,
) -> str | None:
    """Submit a segment to Seedance and poll until complete.

    Returns the URL of the generated video, or None on failure.
    """
    model = model or os.environ.get("SEEDANCE_MODEL", "seedance-1-0-pro")

    content = []
    for fp in frame_paths:
        content.append({"type": "image_url", "image_url": {"url": _b64_image(fp)}})
    content.append({"type": "image_url", "image_url": {"url": _b64_image(product_image_path)}})
    content.append({"type": "text", "text": prompt})

    body = {
        "model": model,
        "content": content,
        "duration": 8,
        "ratio": "9:16",
        "resolution": "1080p",
    }

    # Submit with retries + exponential backoff on 429
    task_id = None
    for attempt in range(MAX_RETRIES):
        resp = requests.post(f"{BASE_URL}/videos/generations", json=body, headers=_headers(), timeout=120)
        if resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            log.warning("Rate limited (429). Retrying in %ds...", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        task_id = resp.json().get("id") or resp.json().get("task_id")
        break

    if not task_id:
        log.error("Failed to create Seedance task after %d attempts", MAX_RETRIES)
        return None

    log.info("Seedance task created: %s", task_id)

    # Poll until done
    while True:
        time.sleep(POLL_INTERVAL)
        poll = requests.get(f"{BASE_URL}/videos/generations/{task_id}", headers=_headers(), timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status", "").lower()

        if status == "succeeded":
            video_url = data.get("video_url") or data.get("output", {}).get("video_url")
            log.info("Segment complete: %s", video_url)
            return video_url
        elif status == "failed":
            log.error("Seedance task %s failed: %s", task_id, data.get("error"))
            return None
        else:
            log.info("Seedance task %s status: %s", task_id, status)


def download_video(url: str, dest: str) -> str:
    """Download a video from URL to local path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info("Downloaded video to %s", dest)
    return dest
