# Telegram Bot Interface for Face Swap — Design Spec

## Date
2026-04-30

## Overview

Integrate the existing face-swap batch system as a Telegram bot interface. Users interact with the bot to set a source face and send target image batches for processing.

---

## User Flow

1. **First time / No source set:**
   ```
   User sends images → Bot replies: "Primero configura /cambiar_source + tu foto"
   ```

2. **Set source:**
   ```
   User → /cambiar_source → Bot: "Envía tu foto source" → User sends photo → Bot: "Source actualizado ✓"
   ```

3. **Normal batch processing:**
   ```
   User sends album (1–10 photos) → Bot: "Procesando X imágenes..." → Bot sends processed album
   ```

4. **Change source:**
   - Any time via `/cambiar_source` command

---

## Commands

| Command     | Description                              |
|-------------|------------------------------------------|
| `/cambiar_source` | Activate source-awaiting mode       |
| `/ayuda`    | Show help / instructions                 |
| `/estado`   | Show current source status                |

---

## Bot States (per user)

```
IDLE ──(user sends /cambiar_source)──→ AWAITING_SOURCE
AWAITING_SOURCE ──(user sends photo)──→ IDLE (source saved)
IDLE ──(user sends album)──→ PROCESSING ──(done)──→ IDLE
```

### State Persistence

User state stored in `sessions.json`:

```json
{
  "123456": {
    "source_path": "/home/ubuntu/repos/fs/sources/123456.jpg",
    "state": "IDLE"
  }
}
```

Source images saved to `sources/{user_id}.jpg` (replaces previous if exists).

---

## Processing

- Album size: 1–10 photos per message (Telegram limit)
- User sends multiple albums sequentially, each processed independently
- **Single status message:** "Procesando X imágenes..." sent immediately; album sent when done
- No intermediate status updates during processing

### Processing Backend

Same as existing `main.py` logic:
- If `api_token` configured → use `process_batch_replicate`
- Otherwise → use `process_batch_local`
- Model selection preserved (default inswapper)

### Result Delivery

Processed images sent as a Telegram media album (same layout as received).

---

## File Structure

```
fs/
├── main.py                 # existing CLI entry
├── bot.py                  # new Telegram bot entry
├── src/
│   └── ...
├── sources/               # saved user source images
├── sessions.json          # user state persistence
├── requirements.txt       # add python-telegram-bot
└── docs/superpowers/specs/YYYY-MM-DD-telegram-design.md
```

---

## Dependencies

Add to `requirements.txt`:
```
python-telegram-bot>=20.0
```

---

## Configuration

Bot token via environment or config:
```python
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")  # reuse existing
```

---

## Implementation Notes

- Use `python-telegram-bot` async version
- Download Telegram file URLs → save to `input/{uuid}.jpg`
- Call existing `process_batch_replicate` / `process_batch` functions
- Output images moved to `output/` → sent via `sendMediaGroup`
- Cleanup temp files after sending
