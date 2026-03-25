"""Kling AI API client (Kuaishou)."""

import base64
import logging
import os
import time
from pathlib import Path

import jwt
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.klingai.com"
MAX_RETRIES = 3
POLL_INTERVAL = 10  # seconds


def _jwt_token() -> str:
    now = int(time.time())
    payload = {
        "iss": os.environ["KLING_ACCESS_KEY_ID"],
        "exp": now + 1800,
        "nbf": now - 5,
    }
    return jwt.encode(payload, os.environ["KLING_ACCESS_KEY_SECRET"], algorithm="HS256")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_jwt_token()}",
        "Content-Type": "application/json",
    }


def _b64_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def generate_segment(
    frame_paths: list[str],
    product_image_path: str,
    prompt: str,
    model: str | None = None,
) -> str | None:
    """Submit a segment to Kling and poll until complete.

    Returns the URL of the generated video, or None on failure.
    """
    model = model or os.environ.get("KLING_MODEL", "kling-v2-5-master")
    duration = os.environ.get("KLING_DURATION", "10")

    body = {
        "model_name": model,
        "image": _b64_image(frame_paths[0]),
        "prompt": prompt,
        "negative_prompt": "text, watermark, caption, subtitle, logo, blurry",
        "cfg_scale": 0.5,
        "mode": "pro",
        "duration": duration,
        "aspect_ratio": "9:16",
    }

    # Submit with retries + exponential backoff on 429
    task_id = None
    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            f"{BASE_URL}/v1/videos/image2video",
            json=body,
            headers=_headers(),
            timeout=120,
        )
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)  # 30s, 60s, 90s
            log.warning("Rate limited (429). Retrying in %ds...", wait)
            time.sleep(wait)
            continue
        if not resp.ok:
            log.error("Kling submit error %d: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            log.error("Kling API error: %s", result.get("message"))
            return None
        task_id = result["data"]["task_id"]
        break

    if not task_id:
        log.error("Failed to create Kling task after %d attempts", MAX_RETRIES)
        return None

    log.info("Kling task created: %s", task_id)

    # Poll until done
    while True:
        time.sleep(POLL_INTERVAL)
        poll = requests.get(
            f"{BASE_URL}/v1/videos/image2video/{task_id}",
            headers=_headers(),
            timeout=30,
        )
        poll.raise_for_status()
        data = poll.json()

        if data.get("code") != 0:
            log.error("Kling poll error: %s", data.get("message"))
            return None

        status = data["data"].get("task_status", "").lower()

        if status == "succeed":
            videos = data["data"].get("task_result", {}).get("videos", [])
            if videos:
                video_url = videos[0]["url"]
                log.info("Segment complete: %s", video_url)
                return video_url
            log.error("Kling task succeeded but no video URL in response")
            return None
        elif status == "failed":
            log.error("Kling task %s failed: %s", task_id, data["data"].get("task_status_msg"))
            return None
        else:
            log.info("Kling task %s status: %s", task_id, status)


def download_video(url: str, dest: str) -> str:
    """Download a video from URL to local path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    log.info("Downloaded video to %s", dest)
    return dest
