# VidReplica

Video replication tool that takes a source video, analyses it, and generates a replica with a substituted product using the Seedance 2.0 API (BytePlus).

## How it works

1. Upload a source video (MP4/MOV) + a product image (JPG/PNG)
2. Video is split into 8-second segments
3. 9 frames are extracted from each segment
4. Audio is transcribed locally via Whisper
5. Claude generates an optimised Seedance prompt per segment
6. Seedance 2.0 API generates a replica video for each segment
7. All segments are stitched into a final MP4

## Setup

### Prerequisites

- Python 3.11+
- ffmpeg installed and on PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`

### Install

```bash
cd AI-Video-Tool-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
BYTEPLUS_API_KEY=your-byteplus-key
WHISPER_MODEL=base          # optional: tiny, base, small, medium, large
SEEDANCE_MODEL=seedance-1-0-pro  # optional: change if your account has seedance-2-0
```

## Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## File structure

```
├── app.py              # Flask app, all routes
├── pipeline.py         # Core processing pipeline
├── seedance.py         # Seedance API client
├── prompts.py          # Claude prompt generation
├── requirements.txt
├── .env.example
├── templates/
│   └── index.html      # Frontend
├── uploads/            # Temp upload storage (auto-cleaned)
├── segments/           # Temp segment storage (auto-cleaned)
└── outputs/            # Final output videos
```
