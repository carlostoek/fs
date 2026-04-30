# tests/test_download.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import tempfile
from pathlib import Path

import download

@pytest.mark.asyncio
async def test_download_photo_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_file = MagicMock()
        mock_file.download_to_drive = AsyncMock(side_effect=lambda path: path.touch())
        mock_file.file_path = Path(tmpdir) / "photo.jpg"

        with patch("telegram.Bot") as mock_bot:
            bot = mock_bot.return_value
            bot.get_file = AsyncMock(return_value=mock_file)

            result_path = await download.download_telegram_photo(
                bot, "file_id_123", Path(tmpdir)
            )

            assert result_path.exists()
            mock_file.download_to_drive.assert_called_once()