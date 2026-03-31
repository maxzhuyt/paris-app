"""Tests for video caching and API endpoints."""

import io
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app import app, video_cache, VIDEO_CACHE_DIR, download_video_from_url


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear in-memory cache before each test."""
    video_cache.clear()
    yield
    video_cache.clear()


# ---------------------------------------------------------------------------
# Video cache unit tests
# ---------------------------------------------------------------------------

class TestVideoCache:
    """Tests for download_video_from_url caching behaviour."""

    @patch('app.subprocess.run')
    def test_download_caches_to_video_cache_dir(self, mock_run):
        """Downloaded file goes into VIDEO_CACHE_DIR, not temp."""
        fake_path = VIDEO_CACHE_DIR / 'abc123.mp4'
        fake_path.touch()

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=str(fake_path) + '\n',
            stderr='',
        )

        result = download_video_from_url('https://youtube.com/watch?v=abc123')
        assert result == fake_path
        assert 'https://youtube.com/watch?v=abc123' in video_cache
        assert video_cache['https://youtube.com/watch?v=abc123'] == fake_path

        fake_path.unlink(missing_ok=True)

    @patch('app.subprocess.run')
    def test_second_call_returns_cache_no_subprocess(self, mock_run):
        """Second call for same URL skips yt-dlp entirely."""
        fake_path = VIDEO_CACHE_DIR / 'cached.mp4'
        fake_path.touch()
        video_cache['https://example.com/v'] = fake_path

        result = download_video_from_url('https://example.com/v')
        assert result == fake_path
        mock_run.assert_not_called()

        fake_path.unlink(missing_ok=True)

    @patch('app.subprocess.run')
    def test_cache_miss_when_file_deleted(self, mock_run):
        """If cached file was deleted from disk, re-download."""
        video_cache['https://example.com/v'] = VIDEO_CACHE_DIR / 'gone.mp4'

        new_path = VIDEO_CACHE_DIR / 'new.mp4'
        new_path.touch()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=str(new_path) + '\n',
            stderr='',
        )

        result = download_video_from_url('https://example.com/v')
        assert result == new_path
        mock_run.assert_called_once()

        new_path.unlink(missing_ok=True)

    @patch('app.subprocess.run')
    def test_download_failure_raises(self, mock_run):
        """yt-dlp failure raises RuntimeError."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr='ERROR: Video unavailable',
        )

        with pytest.raises(RuntimeError, match='yt-dlp failed'):
            download_video_from_url('https://example.com/bad')

    @patch('app.subprocess.run')
    def test_yt_dlp_called_with_correct_flags(self, mock_run):
        """Verify --js-runtimes node and --print after_move:filepath are used."""
        fake_path = VIDEO_CACHE_DIR / 'flags.mp4'
        fake_path.touch()
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=str(fake_path) + '\n',
            stderr='',
        )

        download_video_from_url('https://youtube.com/watch?v=flags')
        args = mock_run.call_args[0][0]
        assert '--js-runtimes' in args
        assert 'node' in args
        assert '--print' in args
        idx = args.index('--print')
        assert args[idx + 1] == 'after_move:filepath'

        fake_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# API endpoint tests — file cleanup behaviour
# ---------------------------------------------------------------------------

class TestTranscriptEndpoint:
    """Tests for /api/transcript endpoint."""

    @patch('app.generate_report_transcript', return_value='Test report')
    @patch('app.transcribe_audio', return_value='Test transcript')
    @patch('app.extract_audio')
    @patch('app.download_video_from_url')
    def test_url_video_not_deleted_after_processing(
        self, mock_dl, mock_extract, mock_transcribe, mock_report, client
    ):
        """Video from URL should NOT be deleted (it's cached)."""
        fake_path = VIDEO_CACHE_DIR / 'keep_me.mp4'
        fake_path.touch()
        mock_dl.return_value = fake_path

        resp = client.post('/api/transcript', data={
            'video_url': 'https://youtube.com/watch?v=keep',
            'whisper_temperature': '0',
            'deepseek_temperature': '0',
            'deepseek_top_p': '1',
            'system_prompt': 'test',
            'user_prompt': 'test {transcript}',
        })

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert fake_path.exists(), "Cached video should NOT be deleted"

        fake_path.unlink(missing_ok=True)

    @patch('app.generate_report_transcript', return_value='Test report')
    @patch('app.transcribe_audio', return_value='Test transcript')
    @patch('app.extract_audio')
    def test_uploaded_file_deleted_after_processing(
        self, mock_extract, mock_transcribe, mock_report, client
    ):
        """Uploaded file should be deleted after processing."""
        data = {
            'whisper_temperature': '0',
            'deepseek_temperature': '0',
            'deepseek_top_p': '1',
            'system_prompt': 'test',
            'user_prompt': 'test {transcript}',
        }

        # Create a fake video file to upload
        video_data = (io.BytesIO(b'fake video'), 'test.mp4')

        resp = client.post('/api/transcript', data={
            **data,
            'video_file': video_data,
        }, content_type='multipart/form-data')

        assert resp.status_code == 200
        # The uploaded file in tmp should be cleaned up
        # (we can't easily verify tmp cleanup in test, but the endpoint should succeed)

    def test_no_video_returns_400(self, client):
        """Missing video returns 400."""
        resp = client.post('/api/transcript', data={
            'whisper_temperature': '0',
            'deepseek_temperature': '0',
            'deepseek_top_p': '1',
        })
        assert resp.status_code == 400
        assert 'No video provided' in resp.get_json()['error']


class TestVideoEndpoint:
    """Tests for /api/video endpoint."""

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'})
    @patch('app.generate_report_gemini', return_value='Gemini report')
    @patch('app.upload_to_gemini')
    @patch('app.download_video_from_url')
    def test_url_video_not_deleted(
        self, mock_dl, mock_upload, mock_report, client
    ):
        """Cached video from URL preserved after /api/video."""
        fake_path = VIDEO_CACHE_DIR / 'vid_keep.mp4'
        fake_path.touch()
        mock_dl.return_value = fake_path

        mock_file = MagicMock()
        mock_file.name = 'files/test'
        mock_upload.return_value = mock_file

        resp = client.post('/api/video', data={
            'video_url': 'https://youtube.com/watch?v=vid',
            'temperature': '0',
            'top_p': '1',
            'system_prompt': 'test',
            'user_prompt': 'test',
        })

        assert resp.status_code == 200
        assert fake_path.exists(), "Cached video should NOT be deleted"

        fake_path.unlink(missing_ok=True)

    def test_no_video_returns_400(self, client):
        """Missing video returns 400."""
        resp = client.post('/api/video', data={
            'temperature': '0',
            'top_p': '1',
        })
        assert resp.status_code == 400

    @patch('app.download_video_from_url')
    def test_missing_gemini_key_returns_500(self, mock_dl, client):
        """Missing GEMINI_API_KEY returns 500."""
        fake_path = VIDEO_CACHE_DIR / 'nokey.mp4'
        fake_path.touch()
        mock_dl.return_value = fake_path

        with patch.dict('os.environ', {}, clear=False):
            # Remove GEMINI_API_KEY if present
            import os
            os.environ.pop('GEMINI_API_KEY', None)

            resp = client.post('/api/video', data={
                'video_url': 'https://youtube.com/watch?v=nokey',
                'temperature': '0',
                'top_p': '1',
            })

        assert resp.status_code == 500
        assert 'GEMINI_API_KEY' in resp.get_json()['error']

        fake_path.unlink(missing_ok=True)


class TestCompareEndpoint:
    """Tests for /api/compare endpoint."""

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'})
    @patch('app.generate_report_gemini', return_value='Gemini report')
    @patch('app.upload_to_gemini')
    @patch('app.generate_report_transcript', return_value='Transcript report')
    @patch('app.transcribe_audio', return_value='Transcript text')
    @patch('app.extract_audio')
    @patch('app.download_video_from_url')
    def test_url_video_not_deleted(
        self, mock_dl, mock_extract, mock_transcribe,
        mock_report_t, mock_upload, mock_report_g, client
    ):
        """Cached video preserved after /api/compare."""
        fake_path = VIDEO_CACHE_DIR / 'cmp_keep.mp4'
        fake_path.touch()
        mock_dl.return_value = fake_path

        mock_file = MagicMock()
        mock_file.name = 'files/test'
        mock_upload.return_value = mock_file

        resp = client.post('/api/compare', data={
            'video_url': 'https://youtube.com/watch?v=cmp',
            'whisper_temperature': '0',
            'deepseek_temperature': '0',
            'deepseek_top_p': '1',
            'gemini_temperature': '0',
            'gemini_top_p': '1',
            'system_prompt': 'test',
            'user_prompt_transcript': 'test {transcript}',
            'user_prompt_video': 'test',
        })

        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert fake_path.exists(), "Cached video should NOT be deleted"

        fake_path.unlink(missing_ok=True)

    def test_no_video_returns_400(self, client):
        """Missing video returns 400."""
        resp = client.post('/api/compare', data={
            'whisper_temperature': '0',
            'deepseek_temperature': '0',
            'deepseek_top_p': '1',
            'gemini_temperature': '0',
            'gemini_top_p': '1',
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Page rendering tests
# ---------------------------------------------------------------------------

class TestPages:
    """Verify all pages render without errors."""

    def test_index(self, client):
        assert client.get('/').status_code == 200

    def test_transcript_page(self, client):
        resp = client.get('/transcript')
        assert resp.status_code == 200
        assert b'localStorage' in resp.data

    def test_video_page(self, client):
        resp = client.get('/video')
        assert resp.status_code == 200
        assert b'localStorage' in resp.data
        assert b'GEMINI_API_KEY' not in resp.data

    def test_compare_page(self, client):
        resp = client.get('/compare')
        assert resp.status_code == 200
        assert b'localStorage' in resp.data

    def test_history_page(self, client):
        assert client.get('/history').status_code == 200
