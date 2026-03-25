"""Claude prompt generation for Kling segments."""

import base64
import logging
import os
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)


def _encode_image(path: str) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    ext = Path(path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def generate_kling_prompt(
    frame_paths: list[str],
    product_image_path: str,
    transcript: str,
    amendments: str = "",
) -> str:
    """Use Claude to generate an optimised Kling prompt for one segment."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    n_frames = len(frame_paths)

    # Build content blocks: frames + product image + text instructions
    content = []
    for i, fp in enumerate(frame_paths, 1):
        content.append({"type": "text", "text": f"Frame {i} of {n_frames} from the source segment:"})
        content.append(_encode_image(fp))

    content.append({"type": "text", "text": "Product image to substitute into the video:"})
    content.append(_encode_image(product_image_path))

    instructions = (
        f"You are a video-generation prompt engineer. Analyse the {n_frames} sequential frames above — "
        "they represent a 10-second video segment. Also note the product image.\n\n"
        f"Audio transcript for this segment: \"{transcript or '(no speech detected)'}\"\n\n"
    )
    if amendments:
        instructions += f"User's additional instructions: \"{amendments}\"\n\n"

    instructions += (
        "Write a single Kling video-generation prompt. The prompt must:\n"
        f"1. Describe the scene, action, camera movement, energy, and pacing seen across the {n_frames} frames\n"
        "2. Describe the motion flow: how the action progresses from the first frame to the last\n"
        "3. Describe the product shown in the product image and instruct Kling to feature it prominently\n"
        "4. Incorporate the transcript (speech content, tone, energy)\n"
        "5. Describe replacing whatever product appears in the original with the provided product\n"
        "6. Apply any user amendments\n"
        "7. Do NOT include any on-screen text, captions, watermarks, or overlaid words in the scene — keep the scene clean of all text\n"
        "8. End with: Maintain 9:16 vertical format, high energy, TikTok Shop style.\n\n"
        "Output ONLY the prompt string, nothing else. No quotes, no explanation."
    )
    content.append({"type": "text", "text": instructions})

    log.info("Generating Kling prompt via Claude...")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    prompt_text = response.content[0].text.strip()
    log.info("Claude prompt generated (%d chars)", len(prompt_text))
    return prompt_text
