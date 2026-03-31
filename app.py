#!/usr/bin/env python3
"""
PARIS — Police AI Report & Incident Simulator

Research-focused interface for generating and comparing police reports.
"""

import os
import json
import sqlite3
import tempfile
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

load_dotenv()

# Resolve ffmpeg binary — prefer imageio_ffmpeg bundled binary, fall back to PATH
def _find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return 'ffmpeg'

FFMPEG_BIN = _find_ffmpeg()
YTDLP_BIN = str(Path(__file__).parent / 'venv' / 'bin' / 'yt-dlp')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Persistent video cache for URL downloads
VIDEO_CACHE_DIR = Path('video_cache')
VIDEO_CACHE_DIR.mkdir(exist_ok=True)
video_cache = {}  # URL -> cached file path

# SQLite database for persistent experiment storage
DB_PATH = Path(__file__).parent / 'police_app.db'

def _get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT,
        timestamp TEXT,
        video_source TEXT,
        notes TEXT,
        params TEXT,
        status TEXT,
        transcript TEXT,
        report_transcript TEXT,
        report_video TEXT,
        error TEXT
    )''')
    # Migrate: add name column if missing (existing databases)
    try:
        conn.execute('ALTER TABLE experiments ADD COLUMN name TEXT')
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()

def save_experiment(exp):
    conn = _get_db()
    conn.execute('''INSERT OR REPLACE INTO experiments
        (id, name, timestamp, video_source, notes, params, status, transcript, report_transcript, report_video, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (exp['id'], exp.get('name', ''), exp['timestamp'], exp.get('video_source', ''),
         exp.get('notes', ''), json.dumps(exp.get('params', {})),
         exp.get('status', ''), exp.get('transcript'),
         exp.get('report_transcript'), exp.get('report_video'),
         exp.get('error')))
    conn.commit()
    conn.close()

def get_experiment_db(exp_id):
    conn = _get_db()
    row = conn.execute('SELECT * FROM experiments WHERE id = ?', (exp_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)

def list_experiments_db():
    conn = _get_db()
    rows = conn.execute('SELECT * FROM experiments ORDER BY timestamp DESC').fetchall()
    conn.close()
    return {row['id']: _row_to_dict(row) for row in rows}

def _row_to_dict(row):
    d = dict(row)
    d['params'] = json.loads(d['params']) if d['params'] else {}
    return d

init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare')
def compare_page():
    return render_template('compare.html')


@app.route('/transcript')
def transcript_page():
    return render_template('transcript.html')


@app.route('/video')
def video_page():
    return render_template('video.html')


@app.route('/history')
def history_page():
    return render_template('history.html', experiments=list_experiments_db())


@app.route('/api/experiments')
def list_experiments():
    return jsonify(list_experiments_db())


@app.route('/api/experiment/<exp_id>')
def get_experiment(exp_id):
    exp = get_experiment_db(exp_id)
    if exp:
        return jsonify(exp)
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/export/<exp_id>/<format>')
def export_experiment(exp_id, format):
    exp = get_experiment_db(exp_id)
    if not exp:
        return jsonify({'error': 'Not found'}), 404
    
    if format == 'json':
        return Response(
            json.dumps(exp, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment;filename=experiment_{exp_id}.json'}
        )
    elif format == 'txt':
        lines = [
            f"EXPERIMENT: {exp.get('name', '')} ({exp_id})",
            f"Timestamp: {exp['timestamp']}",
            f"Notes: {exp.get('notes', '')}",
            "",
            "=" * 60,
            "PARAMETERS",
            "=" * 60,
            json.dumps(exp['params'], indent=2),
            "",
        ]
        
        if exp.get('transcript'):
            lines.extend([
                "=" * 60,
                "TRANSCRIPT (Pipeline 1)",
                "=" * 60,
                exp['transcript'],
                "",
            ])
        
        if exp.get('report_transcript'):
            lines.extend([
                "=" * 60,
                "REPORT (Pipeline 1 - Transcript)",
                "=" * 60,
                exp['report_transcript'],
                "",
            ])
        
        if exp.get('report_video'):
            lines.extend([
                "=" * 60,
                "REPORT (Pipeline 2 - Video/Gemini)",
                "=" * 60,
                exp['report_video'],
                "",
            ])
        
        return Response(
            '\n'.join(lines),
            mimetype='text/plain',
            headers={'Content-Disposition': f'attachment;filename=experiment_{exp_id}.txt'}
        )
    
    return jsonify({'error': 'Invalid format'}), 400


@app.route('/api/transcript', methods=['POST'])
def run_transcript_pipeline():
    exp_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    whisper_temperature = float(request.form.get('whisper_temperature', 0.0))
    deepseek_temperature = float(request.form.get('deepseek_temperature', 0.0))
    deepseek_top_p = float(request.form.get('deepseek_top_p', 1.0))
    system_prompt = request.form.get('system_prompt', '')
    user_prompt = request.form.get('user_prompt', '')
    notes = request.form.get('notes', '')
    experiment_name = request.form.get('experiment_name', '').strip()
    video_source = request.form.get('video_source', '')

    video_path = None
    video_url = request.form.get('video_url', '').strip()
    from_cache = False

    if video_url:
        video_path = download_video_from_url(video_url)
        video_source = video_url
        from_cache = True
    elif 'video_file' in request.files:
        file = request.files['video_file']
        if file.filename:
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            video_path = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(video_path)
            video_source = file.filename

    if not video_path or not Path(video_path).exists():
        return jsonify({'error': 'No video provided'}), 400

    name = f"{experiment_name} ({exp_id})" if experiment_name else exp_id

    exp = {
        'id': exp_id,
        'name': name,
        'timestamp': timestamp,
        'video_source': video_source,
        'notes': notes,
        'params': {
            'pipeline': 'transcript',
            'models': 'whisper-1, deepseek-reasoner',
            'whisper_temperature': whisper_temperature,
            'deepseek_temperature': deepseek_temperature,
            'deepseek_top_p': deepseek_top_p,
            'system_prompt': system_prompt[:200] + '...' if len(system_prompt) > 200 else system_prompt,
            'user_prompt': user_prompt[:200] + '...' if len(user_prompt) > 200 else user_prompt,
        },
        'status': 'processing',
        'transcript': None,
        'report_transcript': None,
    }

    try:
        audio_path = Path(video_path).with_suffix('.wav')
        extract_audio(video_path, audio_path)

        transcript = transcribe_audio(audio_path, whisper_temperature)
        exp['transcript'] = transcript

        report = generate_report_transcript(
            transcript, system_prompt, user_prompt,
            deepseek_temperature, deepseek_top_p
        )
        exp['report_transcript'] = report
        exp['status'] = 'complete'

        save_experiment(exp)

        return jsonify({
            'success': True,
            'experiment_id': exp_id,
            'transcript': transcript,
            'report': report
        })

    except Exception as e:
        exp['status'] = 'error'
        exp['error'] = str(e)
        save_experiment(exp)
        return jsonify({'error': str(e), 'experiment_id': exp_id}), 500
    finally:
        if not from_cache and video_path and Path(video_path).exists():
            Path(video_path).unlink()


@app.route('/api/video', methods=['POST'])
def run_video_pipeline():
    exp_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    temperature = float(request.form.get('temperature', 0.0))
    top_p = float(request.form.get('top_p', 1.0))
    system_prompt = request.form.get('system_prompt', '')
    user_prompt = request.form.get('user_prompt', '')
    notes = request.form.get('notes', '')
    experiment_name = request.form.get('experiment_name', '').strip()
    video_source = ''

    video_path = None
    video_url = request.form.get('video_url', '').strip()
    from_cache = False

    if video_url:
        video_path = download_video_from_url(video_url)
        video_source = video_url
        from_cache = True
    elif 'video_file' in request.files:
        file = request.files['video_file']
        if file.filename:
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            video_path = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(video_path)
            video_source = file.filename

    if not video_path or not Path(video_path).exists():
        return jsonify({'error': 'No video provided'}), 400

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 500

    name = f"{experiment_name} ({exp_id})" if experiment_name else exp_id

    exp = {
        'id': exp_id,
        'name': name,
        'timestamp': timestamp,
        'video_source': video_source,
        'notes': notes,
        'params': {
            'pipeline': 'video',
            'models': 'gemini-2.5-pro',
            'temperature': temperature,
            'top_p': top_p,
            'system_prompt': system_prompt[:200] + '...' if len(system_prompt) > 200 else system_prompt,
            'user_prompt': user_prompt[:200] + '...' if len(user_prompt) > 200 else user_prompt,
        },
        'status': 'processing',
        'report_video': None,
    }

    from google import genai
    client = genai.Client(api_key=api_key)

    video_file = None
    try:
        video_file = upload_to_gemini(client, video_path)

        report = generate_report_gemini(
            client, video_file, system_prompt, user_prompt,
            temperature, top_p
        )
        exp['report_video'] = report
        exp['status'] = 'complete'

        save_experiment(exp)

        return jsonify({
            'success': True,
            'experiment_id': exp_id,
            'report': report
        })

    except Exception as e:
        exp['status'] = 'error'
        exp['error'] = str(e)
        save_experiment(exp)
        return jsonify({'error': str(e), 'experiment_id': exp_id}), 500
    finally:
        if video_file:
            try:
                client.files.delete(name=video_file.name)
            except:
                pass
        if not from_cache and video_path and Path(video_path).exists():
            Path(video_path).unlink()


@app.route('/api/compare', methods=['POST'])
def run_both_pipelines():
    """Run both pipelines and return results for comparison."""
    exp_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    # Get params
    whisper_temperature = float(request.form.get('whisper_temperature', 0.0))
    deepseek_temperature = float(request.form.get('deepseek_temperature', 0.0))
    deepseek_top_p = float(request.form.get('deepseek_top_p', 1.0))
    gemini_temperature = float(request.form.get('gemini_temperature', 0.0))
    gemini_top_p = float(request.form.get('gemini_top_p', 1.0))
    system_prompt = request.form.get('system_prompt', '')
    user_prompt_transcript = request.form.get('user_prompt_transcript', '')
    user_prompt_video = request.form.get('user_prompt_video', '')
    notes = request.form.get('notes', '')
    experiment_name = request.form.get('experiment_name', '').strip()
    video_source = ''

    video_path = None
    video_url = request.form.get('video_url', '').strip()
    from_cache = False

    if video_url:
        video_path = download_video_from_url(video_url)
        video_source = video_url
        from_cache = True
    elif 'video_file' in request.files:
        file = request.files['video_file']
        if file.filename:
            from werkzeug.utils import secure_filename
            filename = secure_filename(file.filename)
            video_path = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(video_path)
            video_source = file.filename

    if not video_path or not Path(video_path).exists():
        return jsonify({'error': 'No video provided'}), 400

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 500

    name = f"{experiment_name} ({exp_id})" if experiment_name else exp_id

    exp = {
        'id': exp_id,
        'name': name,
        'timestamp': timestamp,
        'video_source': video_source,
        'notes': notes,
        'params': {
            'pipeline': 'both',
            'models': 'whisper-1, deepseek-reasoner, gemini-2.5-pro',
            'whisper_temperature': whisper_temperature,
            'deepseek_temperature': deepseek_temperature,
            'deepseek_top_p': deepseek_top_p,
            'gemini_temperature': gemini_temperature,
            'gemini_top_p': gemini_top_p,
        },
        'status': 'processing',
        'transcript': None,
        'report_transcript': None,
        'report_video': None,
    }
    
    video_file = None
    try:
        # Pipeline 1: Transcript
        audio_path = Path(video_path).with_suffix('.wav')
        extract_audio(video_path, audio_path)
        
        transcript = transcribe_audio(audio_path, whisper_temperature)
        exp['transcript'] = transcript

        report_transcript = generate_report_transcript(
            transcript, system_prompt, user_prompt_transcript,
            deepseek_temperature, deepseek_top_p
        )
        exp['report_transcript'] = report_transcript

        # Pipeline 2: Gemini
        from google import genai
        client = genai.Client(api_key=api_key)
        video_file = upload_to_gemini(client, video_path)

        report_video = generate_report_gemini(
            client, video_file, system_prompt, user_prompt_video,
            gemini_temperature, gemini_top_p
        )
        exp['report_video'] = report_video
        exp['status'] = 'complete'
        
        save_experiment(exp)
        
        return jsonify({
            'success': True,
            'experiment_id': exp_id,
            'transcript': transcript,
            'report_transcript': report_transcript,
            'report_video': report_video
        })
        
    except Exception as e:
        exp['status'] = 'error'
        exp['error'] = str(e)
        save_experiment(exp)
        return jsonify({'error': str(e), 'experiment_id': exp_id}), 500
    finally:
        if video_file:
            try:
                client.files.delete(name=video_file.name)
            except:
                pass
        if not from_cache and video_path and Path(video_path).exists():
            Path(video_path).unlink()


def download_video_from_url(url: str) -> Path:
    # Return cached file if available
    if url in video_cache and video_cache[url].exists():
        return video_cache[url]

    output_template = str(VIDEO_CACHE_DIR / '%(id)s.%(ext)s')

    result = subprocess.run([
        YTDLP_BIN,
        '--js-runtimes', 'node',
        '--ffmpeg-location', FFMPEG_BIN,
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '--merge-output-format', 'mp4',
        '-o', output_template,
        '--no-playlist',
        '--print', 'after_move:filepath',
        url,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    cached_path = Path(result.stdout.strip())
    video_cache[url] = cached_path
    return cached_path


def extract_audio(video_path: Path, audio_path: Path):
    subprocess.run([
        FFMPEG_BIN, '-i', str(video_path),
        '-vn', '-ar', '16000', '-ac', '1',
        '-f', 'wav', '-y', str(audio_path),
    ], check=True, capture_output=True)


def transcribe_audio(audio_path: Path, temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    with open(audio_path, 'rb') as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
            temperature=temperature,
            response_format="text",
        )

    return result


def generate_report_transcript(
    transcript: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com",
    )

    if not system_prompt:
        system_prompt = "You are an AI assistant that drafts police incident report narratives for sworn officers. Using only the incident transcript and the structured metadata provided (such as officer name and rank, date and time, location, incident type, arrest information, and report style preferences), write a clear, objective, grammatically correct draft narrative in third-person that follows standard police-report conventions. Describe the call for service, the officer's arrival and observations, interactions with involved parties, the officer's actions, the behavior and statements of subjects, and the final disposition, in a logical chronological order. Do not speculate, infer motives, add facts that are not clearly supported by the transcript or metadata, or provide legal opinions beyond what is explicitly stated. When important information is missing or unclear but normally required in a report (for example exact DOB, injuries, or vehicle details), insert a bracketed placeholder such as [INSERT: missing detail] instead of guessing. Use neutral, professional language, convert spoken dialog into concise indirect speech except where a short direct quote is important, and output only the final narrative text with no headings, explanations, or commentary."

    if not user_prompt:
        user_prompt = "Based on the following transcript, write a detailed police incident report.\n\nTRANSCRIPT:\n{transcript}"

    if "{transcript}" in user_prompt:
        user_prompt = user_prompt.replace("{transcript}", transcript)
    else:
        user_prompt = f"{user_prompt}\n\nTRANSCRIPT:\n{transcript}"

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8192,
        top_p=top_p,
    )

    return response.choices[0].message.content


def upload_to_gemini(client, video_path: Path):
    import time

    video_file = client.files.upload(file=str(video_path))

    while video_file.state == "PROCESSING":
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state}")

    return video_file


def generate_report_gemini(
    client,
    video_file,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
) -> str:
    from google.genai import types

    if not system_prompt:
        system_prompt = "You are an AI assistant that drafts police incident report narratives for sworn officers. Using only the incident transcript and the structured metadata provided (such as officer name and rank, date and time, location, incident type, arrest information, and report style preferences), write a clear, objective, grammatically correct draft narrative in third-person that follows standard police-report conventions. Describe the call for service, the officer's arrival and observations, interactions with involved parties, the officer's actions, the behavior and statements of subjects, and the final disposition, in a logical chronological order. Do not speculate, infer motives, add facts that are not clearly supported by the transcript or metadata, or provide legal opinions beyond what is explicitly stated. When important information is missing or unclear but normally required in a report (for example exact DOB, injuries, or vehicle details), insert a bracketed placeholder such as [INSERT: missing detail] instead of guessing. Use neutral, professional language, convert spoken dialog into concise indirect speech except where a short direct quote is important, and output only the final narrative text with no headings, explanations, or commentary."

    if not user_prompt:
        user_prompt = "Based on the above metadata and the bodycam video footage, write a detailed police incident report narrative."

    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[video_file, user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=8192,
        ),
    )

    return response.text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
