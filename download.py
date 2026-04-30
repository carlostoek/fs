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