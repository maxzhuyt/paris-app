# PARIS — Police AI Report & Incident Simulator

A research tool for generating police incident reports from bodycam video using multiple AI pipelines.

## How It Works

Upload a bodycam video (or paste a URL) and PARIS generates a draft police report using one or both pipelines:

- **Pipeline 1 (Transcript):** Whisper transcription → DeepSeek report generation
- **Pipeline 2 (Video):** Direct video analysis via Gemini 2.5 Pro

You can compare outputs side-by-side, tune model parameters, and export results.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/maxzhuyt/paris-app.git
cd paris-app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys (see Environment Requirements below)

# Run
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Environment Requirements

### System Dependencies

| Dependency | Purpose | Install |
|------------|---------|---------|
| **Python 3.11+** | Runtime | `apt install python3.11` or [pyenv](https://github.com/pyenv/pyenv) |
| **ffmpeg** | Extract audio from video | `apt install ffmpeg` / `brew install ffmpeg` |
| **Node.js** | Used by yt-dlp for some extractors | `apt install nodejs` / `brew install node` |

### Python Packages

All listed in `requirements.txt`. Key dependencies:

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `gunicorn` | Production WSGI server |
| `openai` | Whisper transcription + DeepSeek API |
| `google-genai` | Gemini video analysis |
| `yt-dlp` | Download video from URLs |
| `torch`, `transformers` | Local Whisper inference (optional) |

> **Note on PyTorch:** For GPU support, install torch separately with CUDA before running `pip install -r requirements.txt`:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### API Keys

Create a `.env` file (see `.env.example`):

```
OPENAI_API_KEY=...    # OpenAI — Whisper transcription
DEEPSEEK_API_KEY=...  # DeepSeek — transcript-based report generation
GEMINI_API_KEY=...    # Google — Gemini video analysis
```

All three are required for full functionality. Pipeline 1 (transcript) needs OpenAI + DeepSeek. Pipeline 2 (video) needs Gemini.

## Pages

| Route | Description |
|-------|-------------|
| `/` | Home — run transcript pipeline |
| `/video` | Run video (Gemini) pipeline |
| `/compare` | Run both pipelines side-by-side |
| `/transcript` | Transcript-only view |
| `/history` | Browse past experiments |

## Deployment

Configured for Railway (`Procfile`, `railway.toml`). Set your API keys as environment variables in your deployment platform.
