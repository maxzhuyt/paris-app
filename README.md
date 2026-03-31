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
# Edit .env with your keys (see below)

# Run
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## API Keys Required

Create a `.env` file with:

```
OPENAI_API_KEY=your-openai-key
DEEPSEEK_API_KEY=your-deepseek-key
GEMINI_API_KEY=your-gemini-key
```

- **OpenAI** — for Whisper transcription
- **DeepSeek** — for transcript-based report generation
- **Gemini** — for direct video analysis

## Dependencies

- Python 3.11+
- `ffmpeg` installed on your system
- `yt-dlp` (installed via requirements, used for URL video downloads)

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
