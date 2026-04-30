import pytest
import os
import tempfile
from pathlib import Path

# Patch SESSIONS_FILE to temp location before importing
import sessions

def test_save_and_load_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "sessions.json"
        sessions.SESSIONS_FILE = test_file

        user_id = "123456"
        source_path = Path(tmpdir) / "source.jpg"

        sessions.save_session(user_id, str(source_path), "IDLE")
        data = sessions.load_sessions()

        assert user_id in data
        assert data[user_id]["source_path"] == str(source_path)
        assert data[user_id]["state"] == "IDLE"

def test_get_session_new_user():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "sessions.json"
        sessions.SESSIONS_FILE = test_file

        user_id = "999999"
        result = sessions.get_session(user_id)

        assert result["state"] == "IDLE"
        assert result["source_path"] == ""

def test_set_awaiting_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "sessions.json"
        sessions.SESSIONS_FILE = test_file

        user_id = "111222"
        sessions.set_state(user_id, "AWAITING_SOURCE")
        result = sessions.get_session(user_id)

        assert result["state"] == "AWAITING_SOURCE"