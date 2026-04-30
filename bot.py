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

    # Detect album vs single photo via media_group_id
    is_album = update.message.media_group_id is not None

    # If it's an album and we're in normal mode, delegate to album handler
    if is_album and session["state"] != sessions.SessionState.AWAITING_SOURCE:
        await handle_album(update, context)
        return

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
        # Single photo with source already set - process it
        await process_single_photo(update, context)
        return


async def process_single_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process a single photo when source is already configured."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    photo = update.message.photo[-1]
    file_id = photo.file_id

    source_path = Path(session["source_path"])
    if not source_path.exists():
        await update.message.reply_text(
            "❌ Source no encontrado. Usa /cambiar_source para configurar de nuevo."
        )
        return

    status_msg = await update.message.reply_text("🔄 Procesando imagen...")

    temp_input = Path(tempfile.mkdtemp(prefix="fs_input_"))
    temp_output = temp_input / "output"

    try:
        path = await download.download_telegram_photo(context.bot, file_id, temp_input)

        if API_TOKEN:
            stats = process_batch_replicate(
                source_path=str(source_path),
                input_dir=temp_input,
                output_dir=temp_output,
                api_token=API_TOKEN,
                model=DEFAULT_MODEL,
                batch_size=1
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
                batch_size=1
            )

        processed = list(temp_output.glob("*"))
        if processed:
            with open(processed[0], "rb") as f:
                await update.message.reply_photo(photo=f.read())
        else:
            await update.message.reply_text("⚠️ No se pudo procesar la imagen.")

        await status_msg.edit_text(f"✅ Procesada {stats['processed']} imagen")

    except Exception as e:
        await status_msg.edit_text(f"❌ Error: {str(e)}")

    finally:
        download.cleanup_temp_files([path])
        for p in temp_input.glob("*"):
            if p.is_file():
                p.unlink()
        temp_input.rmdir()


async def handle_album(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming album (1-10 photos)."""
    user_id = str(update.effective_user.id)
    session = sessions.get_session(user_id)

    # If awaiting source, delegate to photo handler
    if session["state"] == sessions.SessionState.AWAITING_SOURCE:
        await handle_photo(update, context)
        return

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

    # Get media group ID and fetch all messages in the album
    media_group_id = update.message.media_group_id
    if not media_group_id:
        # Not actually an album, treat as single photo
        await handle_photo(update, context)
        return

    # Fetch all messages in this album via Bot API
    bot = context.bot
    chat_id = update.effective_chat.id
    messages = []
    async for msg in bot.getMediaGroup(chat_id, media_group_id):
        messages.append(msg)

    if not messages:
        await update.message.reply_text("⚠️ No se pudieron leer las fotos del álbum.")
        return

    file_ids = [msg.photo[-1].file_id for msg in messages]
    count = len(file_ids)

    # Send "processing" message
    status_msg = await update.message.reply_text(
        f"🔄 Procesando {count} imagen{'es' if count > 1 else ''}..."
    )

    # Download photos to temp dir
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

    # Photo handler - all photos (album vs single detected internally)
    app.add_handler(MessageHandler(
        filters.PHOTO,
        handle_photo,
        block=False
    ))

    # Fallback for single photo without album context
    app.add_handler(MessageHandler(
        filters.Document.ALL,
        handle_document,
        block=False
    ))

    print("Bot ready. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()