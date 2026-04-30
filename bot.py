#!/usr/bin/env python3
"""
Telegram bot interface for face swap batch processing.
"""

import os
import asyncio
import tempfile
from pathlib import Path

from telegram import Update, InputMediaPhoto
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