# Telegram Bot Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Telegram bot interface to the face-swap batch system. Users set a source face via bot command, then send image albums for processing.

**Architecture:** Bot layer (`bot.py`) + session state (`sessions.json`) + existing face-swap engine. The bot downloads Telegram photos to temp directories, runs the existing `process_batch_replicate` or local pipeline, and sends results back.

**Tech Stack:** `python-telegram-bot>=20.0` (async), existing `src/` modules, JSON file for session persistence.

---

## File Map

| File | Responsibility |
|------|----------------|
| `bot.py` | Bot entry point, telegram handlers, main loop |
| `sessions.py` | Load/save user session state (source path, state) |
| `download.py` | Download Telegram file URLs → local temp files |
| `src/__init__.py` | Export `process_batch`, `process_batch_replicate`, `FaceDetector`, `FaceSwapper` |
| `src/replicate_swap.py` | Existing cloud processing (no changes) |
| `src/batch.py` | Existing local processing (no changes) |
| `requirements.txt` | Add `python-telegram-bot` |

---

## Task 1: Add python-telegram-bot to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Read current requirements.txt**

Run: `cat requirements.txt`
Expected output shows existing deps (replicate, opencv-python, etc.)

- [ ] **Step 2: Add python-telegram-bot**

```text
python-telegram-bot>=20.0
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt && git commit -m "deps: add python-telegram-bot"
```

---

## Task 2: Create sessions.py for user state persistence

**Files:**
- Create: `sessions.py`
- Test: `tests/test_sessions.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_sessions.py
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
```

Run: `python -m pytest tests/test_sessions.py -v`
Expected: ERROR — `sessions` module doesn't exist

- [ ] **Step 2: Create sessions.py**

```python
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
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_sessions.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add sessions.py tests/test_sessions.py && git commit -m "feat: add session state management"
```

---

## Task 3: Create download.py for Telegram file handling

**Files:**
- Create: `download.py`
- Test: `tests/test_download.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_download.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import download

@pytest.mark.asyncio
async def test_download_photo_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = MagicMock()
        mock_file.download_to_drive = AsyncMock(return_value=None)
        mock_file.file_path = Path(tmpdir) / "photo.jpg"

        with patch("telegram.Bot") as mock_bot:
            bot = mock_bot.return_value
            bot.get_file.return_value = mock_file

            result_path = await download.download_telegram_photo(
                bot, "file_id_123", tmpdir
            )

            assert result_path.exists()
            mock_file.download_to_drive.assert_called_once()
```

Run: `python -m pytest tests/test_download.py -v`
Expected: ERROR — `download` module doesn't exist

- [ ] **Step 2: Create download.py**

```python
#!/usr/bin/env python3
"""Download photos from Telegram and manage temp files."""

import uuid
from pathlib import Path
from typing import Optional

from telegram import Bot


async def download_telegram_photo(bot: Bot, file_id: str, temp_dir: Path) -> Path:
    """
    Download a Telegram photo to a local temp file.

    Args:
        bot: Telegram bot instance
        file_id: Telegram file ID
        temp_dir: Directory to save the file

    Returns:
        Path to downloaded file
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    file = await bot.get_file(file_id)
    file_path = temp_dir / f"{uuid.uuid4().hex}.jpg"
    await file.download_to_drive(file_path)

    return file_path


async def download_album(bot: Bot, photos: list, temp_dir: Path) -> list:
    """
    Download multiple Telegram photos.

    Args:
        bot: Telegram bot instance
        photos: List of (file_id, caption_or_none) tuples
        temp_dir: Directory to save files

    Returns:
        List of (Path, caption_or_none) tuples
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for item in photos:
        if isinstance(item, tuple):
            file_id, caption = item
        else:
            file_id = item
            caption = None

        path = await download_telegram_photo(bot, file_id, temp_dir)
        results.append((path, caption))

    return results


def cleanup_temp_files(paths: list) -> None:
    """Delete temp files after processing."""
    for path in paths:
        if path and path.exists():
            path.unlink()
```

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_download.py -v`
Expected: PASS (mocked async)

- [ ] **Step 4: Commit**

```bash
git add download.py tests/test_download.py && git commit -m "feat: add Telegram file download utilities"
```

---

## Task 4: Create sources directory structure

**Files:**
- Create: `sources/.gitkeep`

- [ ] **Step 1: Create directory and gitkeep**

```bash
mkdir -p sources && touch sources/.gitkeep && git add sources/.gitkeep && git commit -m "feat: add sources directory for user source images"
```

---

## Task 5: Create bot.py — main Telegram bot entry point

**Files:**
- Create: `bot.py`
- Modify: `src/__init__.py` (add `process_batch_replicate` export if missing)

**Bot flow:**

```
User sends /cambiar_source → state = AWAITING_SOURCE → Bot asks for photo
User sends photo          → if AWAITING_SOURCE: save source, state = IDLE
User sends album (1-10)   → if IDLE + source exists: process & return album
User sends album         → if IDLE + no source: error message
```

- [ ] **Step 1: Read src/__init__.py to check exports**

```bash
cat src/__init__.py
```

Expected: exports `FaceDetector`, `FaceSwapper`, `process_batch`, `process_single_image`
Note: `process_batch_replicate` may need to be added if not there.

- [ ] **Step 2: Add process_batch_replicate to src/__init__.py if missing**

```python
from .replicate_swap import process_batch_replicate

__all__ = ["FaceDetector", "FaceSwapper", "process_batch", "process_single_image", "process_batch_replicate"]
```

- [ ] **Step 3: Create bot.py**

```python
#!/usr/bin/env python3
"""
Telegram bot interface for face swap batch processing.
"""

import os
import asyncio
import tempfile
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from src import process_batch_replicate
from src.detector import FaceDetector
from src.face_swap import FaceSwapper
from src.batch import process_batch
import sessions
import download


# Bot token from environment
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Directory constants
BASE_DIR = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "sources"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Default model (same as main.py)
DEFAULT_MODEL = "ddvinh1/inswapper:25bdae46f2713138640b6e8c04dc4ca18625ce95b1863936b053eee42d9ba6db"


async def cmd_cambiar_source(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /cambiar_source command."""
    user_id = str(update.effective_user.id)
    sessions.set_state(user_id, sessions.SessionState.AWAITING_SOURCE)
    await update.message.reply_text(
        "Envía tu foto source (la cara que quieres usar para el swap)."
    )


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ayuda command."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    has_source = bool(session["source_path"])

    ayuda_text = (
        "🖼 *Face Swap Bot*\n\n"
        "Comandos:\n"
        "/cambiar_source — Cambiar la foto source\n"
        "/ayuda — Mostrar esta ayuda\n"
        "/estado — Ver estado actual\n\n"
    )

    if has_source:
        ayuda_text += "✅ Source configurado. Envía fotos para procesar."
    else:
        ayuda_text += "❌ Sin source. Usa /cambiar_source para configurar."

    await update.message.reply_text(ayuda_text, parse_mode="Markdown")


async def cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /estado command."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    has_source = bool(session["source_path"])
    state = session["state"]

    estado_text = (
        f"📋 *Estado*\n\n"
        f"Source: {'✅ Configurado' if has_source else '❌ No configurado'}\n"
        f"Estado: `{state}`"
    )

    await update.message.reply_text(estado_text, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photos."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    # Get the largest photo
    photo = update.message.photo[-1]
    file_id = photo.file_id

    if session["state"] == sessions.SessionState.AWAITING_SOURCE:
        # Save as source
        bot = context.bot
        source_path = SOURCES_DIR / f"{user_id}.jpg"
        SOURCES_DIR.mkdir(parents=True, exist_ok=True)

        file = await bot.get_file(file_id)
        await file.download_to_drive(source_path)

        sessions.set_source(user_id, str(source_path))

        await update.message.reply_text("✅ Source actualizado!")

    else:
        # Should not happen — album handler catches normal photos
        await update.message.reply_text(
            "Usa /cambiar_source para cambiar el source, o envía un álbum."
        )


async def handle_album(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming album (1-10 photos)."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    # Check if source is set
    if not session["source_path"]:
        await update.message.reply_text(
            "❌ Primero configura tu foto source con /cambiar_source"
        )
        return

    source_path = Path(session["source_path"])
    if not source_path.exists():
        await update.message.reply_text(
            "❌ Source no encontrado. Usa /cambiar_source para configurar de nuevo."
        )
        return

    # Get media from album
    media_group = update.message.media_group
    file_ids = [item.file_id for item in media_group]
    count = len(file_ids)

    # Send "processing" message
    status_msg = await update.message.reply_text(
        f"🔄 Procesando {count} imagen{'es' if count > 1 else ''}..."
    )

    # Download photos to temp dir
    bot = context.bot
    temp_input = Path(tempfile.mkdtemp(prefix="fs_input_"))
    temp_output = temp_input / "output"

    downloaded = []
    try:
        for file_id in file_ids:
            path = await download.download_telegram_photo(bot, file_id, temp_input)
            downloaded.append(path)

        # Process using existing pipeline
        if API_TOKEN:
            stats = process_batch_replicate(
                source_path=str(source_path),
                input_dir=temp_input,
                output_dir=temp_output,
                api_token=API_TOKEN,
                model=DEFAULT_MODEL,
                batch_size=len(file_ids)
            )
        else:
            providers = ["CPUExecutionProvider"]
            detector = FaceDetector(providers=providers)
            swapper = FaceSwapper(providers=providers)

            stats = process_batch(
                source_path=source_path,
                input_dir=temp_input,
                output_dir=temp_output,
                detector=detector,
                swapper=swapper,
                batch_size=len(file_ids)
            )

        processed = list(temp_output.glob("*"))
        processed.sort()

        if processed:
            # Send album back
            media = []
            for p in processed:
                with open(p, "rb") as f:
                    media.append(InputMediaPhoto(f.read()))

            await update.message.reply_media_group(media)

        else:
            await update.message.reply_text(
                "⚠️ No se pudieron procesar las imágenes."
            )

        # Update status
        await status_msg.edit_text(
            f"✅ Procesadas {stats['processed']}/{count} imagenes"
        )

    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)}")

    finally:
        # Cleanup
        download.cleanup_temp_files(downloaded)
        for p in temp_input.glob("*"):
            if p.is_file():
                p.unlink()
        temp_input.rmdir()


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle documents (if user sends as file instead of photo)."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    if session["state"] == sessions.SessionState.AWAITING_SOURCE:
        doc = update.message.document
        if doc.mime_type and doc.mime_type.startswith("image/"):
            file_id = doc.file_id
            source_path = SOURCES_DIR / f"{user_id}.jpg"
            SOURCES_DIR.mkdir(parents=True, exist_ok=True)

            bot = context.bot
            file = await bot.get_file(file_id)
            await file.download_to_drive(source_path)

            sessions.set_source(user_id, str(source_path))
            await update.message.reply_text("✅ Source actualizado!")
            return

    await update.message.reply_text(
        "Envía las fotos directamente (no como archivo)."
    )


def main():
    """Run the bot."""
    if not BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        print("Export it: export TELEGRAM_BOT_TOKEN='your_token'")
        return

    print("Starting Face Swap Bot...")

    app = Application.builder().token(BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("cambiar_source", cmd_cambiar_source))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    app.add_handler(CommandHandler("estado", cmd_estado))

    # Album handler (catches photo albums)
    app.add_handler(MessageHandler(
        filters.PHOTO,
        handle_album,
        block=False
    ))

    # Fallback for single photo without album context
    app.add_handler(MessageHandler(
        filters.Document.ALL,
        handle_document,
        block=False
    ))

    print("Bot ready. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run bot.py to verify syntax**

Run: `python bot.py --help 2>&1 | head -5` or check it imports without error
Expected: No import errors

Note: `BOT_TOKEN` won't be set so it will exit early with "not set" message — that's fine.

- [ ] **Step 5: Commit**

```bash
git add bot.py src/__init__.py && git commit -m "feat: add Telegram bot interface"
```

---

## Task 6: Verify end-to-end structure

- [ ] **Step 1: Verify all files exist**

```bash
ls -la bot.py sessions.py download.py sources/
```

- [ ] **Step 2: Verify imports work**

```bash
python -c "from src import process_batch_replicate; import sessions; import download"
```

Expected: No errors

- [ ] **Step 3: Verify bot starts (will exit for missing token)**

```bash
python bot.py 2>&1 | head -3
```

Expected: "Error: TELEGRAM_BOT_TOKEN not set"

---

## Running the Bot

```bash
# Install dependencies
pip install -r requirements.txt

# Set tokens
export TELEGRAM_BOT_TOKEN="your_bot_token"
export REPLICATE_API_TOKEN="your_replicate_token"  # optional, for cloud processing

# Run bot
python bot.py
```

---

## Spec Coverage Check

| Spec Item | Task |
|----------|------|
| `/cambiar_source` activates source mode | Task 5 (cmd_cambiar_source + handle_photo) |
| `/ayuda` shows instructions | Task 5 (cmd_ayuda) |
| `/estado` shows source status | Task 5 (cmd_estado) |
| Album 1-10 photos per message | Task 5 (handle_album) |
| "Procesando X imágenes..." single message | Task 5 (status_msg) |
| Save source per user | Task 5 (SOURCES_DIR / f"{user_id}.jpg") |
| Persist state per user | Task 2 (sessions.py) |
| Reuse existing processing pipeline | Task 5 (process_batch_replicate / process_batch) |
| Send album back to user | Task 5 (sendMediaGroup via reply_media_group) |

No gaps found.
