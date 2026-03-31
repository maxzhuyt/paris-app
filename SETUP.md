# Police Report Generator — Setup & Architecture

## Overview

A Flask web app that generates police incident report narratives from bodycam video. Three pipeline pages:

- **Transcript** (`/transcript`) — Extracts audio via ffmpeg, transcribes with OpenAI Whisper, generates report with DeepSeek Reasoner
- **Video** (`/video`) — Sends full video to Gemini 2.5 Pro for multimodal report generation
- **Compare** (`/compare`) — Runs both pipelines side-by-side on the same video

## Environment Variables (`.env`)

All three keys are **required** for full functionality:

| Key | Service | Used by |
|-----|---------|---------|
| `OPENAI_API_KEY` | OpenAI (Whisper transcription) | Transcript, Compare |
| `DEEPSEEK_API_KEY` | DeepSeek (report generation via OpenAI-compatible API) | Transcript, Compare |
| `GEMINI_API_KEY` | Google Gemini (multimodal video analysis) | Video, Compare |

The `.env` file is loaded by `python-dotenv` at startup. Format:

```
OPENAI_API_KEY=sk-proj-...
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

## System Dependencies

These must be available on the host machine:

| Dependency | Purpose | Install |
|------------|---------|---------|
| **Python 3.12+** | Runtime | System package manager |
| **ffmpeg** | Audio extraction from video | `pip3 install imageio-ffmpeg` (app auto-detects bundled binary) or install system ffmpeg |
| **yt-dlp** | YouTube/video URL downloading | `pip3 install yt-dlp` |
| **Node.js** | Required by yt-dlp for YouTube JS challenge solving | System package manager |

The app resolves ffmpeg automatically: it first tries the `imageio-ffmpeg` Python package (bundled static binary), then falls back to `ffmpeg` on PATH.

## Python Dependencies

```bash
pip3 install -r requirements.txt
pip3 install imageio-ffmpeg  # bundled ffmpeg binary
```

`requirements.txt` contains: flask, gunicorn, python-dotenv, openai, google-genai, yt-dlp.

## Running the App

```bash
# Development
python app.py
# Runs on http://0.0.0.0:5000 with debug mode

# Production
gunicorn app:app --bind 0.0.0.0:5000
```

The `PORT` env var overrides the default port 5000.

## Project Structure

```
police-web-app/
├── app.py                  # Flask app — all routes, API endpoints, pipeline logic
├── .env                    # API keys (never commit)
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # Landing page with links to pipelines
│   ├── transcript.html     # Transcript pipeline UI
│   ├── video.html          # Gemini video pipeline UI
│   ├── compare.html        # Side-by-side comparison UI
│   └── history.html        # Experiment history viewer
├── video_cache/            # Cached video downloads (persists across requests)
├── test_app.py             # Pytest unit tests (18 tests)
├── static/                 # Static assets (currently empty)
├── Procfile                # Railway/Heroku deployment
└── railway.toml            # Railway deployment config
```

## Key Architecture Details

### Video Caching
- Videos downloaded from URLs are stored in `video_cache/` and reused across requests
- An in-memory dict (`video_cache`) maps URL to file path
- If the cached file is deleted from disk, the next request re-downloads it
- Uploaded files are cleaned up after processing; cached URL files are not

### localStorage Persistence
- All three pipeline pages save incident metadata to `localStorage` on every field change
- Keys: `meta_officerName`, `meta_officerRank`, `meta_incidentDateTime`, `meta_incidentLocation`, `meta_incidentType`, `meta_arrestInfo`, `meta_reportPerspective`, `video_url`
- On page load, fields are populated from localStorage and the user prompt is rebuilt
- If a video URL is cached, the URL tab auto-activates

### System Prompt
- The system prompt textarea contains a `{REPORT_PERSON}` template variable
- It is replaced live in the UI with the selected Report Perspective value ("third" or "first")
- The `SYSTEM_PROMPT_TEMPLATE` JS constant preserves the original template for re-substitution

### Experiment History
- Each API call creates an experiment record stored in `experiments` dict (in-memory, lost on restart)
- Viewable at `/history`, exportable as TXT
- Each experiment has an 8-char UUID, timestamp, params, and results

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/transcript` | Run transcript pipeline (Whisper + DeepSeek) |
| POST | `/api/video` | Run video pipeline (Gemini) |
| POST | `/api/compare` | Run both pipelines |
| GET | `/api/experiments` | List all experiments |
| GET | `/api/experiment/<id>` | Get single experiment |
| GET | `/api/export/<id>/txt` | Export experiment as TXT |
| GET | `/api/export/<id>/json` | Export experiment as JSON |

## Running Tests

```bash
python3 -m pytest test_app.py -v
```

Tests cover: video cache behavior, all API endpoints (URL caching, file cleanup, error cases), and page rendering.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `Authentication Fails (governor)` | Empty or invalid `DEEPSEEK_API_KEY` | Set key in `.env`, restart app |
| `GEMINI_API_KEY not set` | Missing Gemini key | Set key in `.env`, restart app |
| `yt-dlp failed` / `No supported JavaScript runtime` | yt-dlp can't solve YouTube challenges | Ensure Node.js is installed (`node --version`) |
| `audio file could not be decoded` | ffmpeg not found by Flask process | Install `imageio-ffmpeg` (`pip3 install imageio-ffmpeg`) — the app auto-detects it |
| `No video provided` (400) | Video URL field empty or wrong tab active | Check that URL tab is selected when using a URL |
