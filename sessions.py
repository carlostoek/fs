#!/usr/bin/env python3
"""User session state management for Telegram bot."""

import json
import os
from pathlib import Path
from typing import Optional

SESSIONS_FILE = Path(__file__).parent / "sessions.json"


class SessionState:
    IDLE = "IDLE"
    AWAITING_SOURCE = "AWAITING_SOURCE"


def load_sessions() -> dict:
    """Load all sessions from disk."""
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_sessions(data: dict) -> None:
    """Save all sessions to disk."""
    SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_session(user_id: str) -> dict:
    """Get session for a user, creating default if needed."""
    sessions = load_sessions()
    if user_id not in sessions:
        sessions[user_id] = {
            "source_path": "",
            "state": SessionState.IDLE
        }
        save_sessions(sessions)
    return sessions[user_id]


def save_session(user_id: str, source_path: str, state: str) -> None:
    """Save session for a user."""
    sessions = load_sessions()
    sessions[user_id] = {
        "source_path": source_path,
        "state": state
    }
    save_sessions(sessions)


def set_state(user_id: str, state: str) -> None:
    """Update only the state for a user."""
    session = get_session(user_id)
    session["state"] = state
    save_sessions({user_id: session})


def set_source(user_id: str, source_path: str) -> None:
    """Set source image path for a user."""
    session = get_session(user_id)
    session["source_path"] = source_path
    session["state"] = SessionState.IDLE
    save_sessions({user_id: session})