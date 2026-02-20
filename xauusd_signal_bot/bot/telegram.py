from __future__ import annotations

import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)


TELEGRAM_MAX_LEN = 3900  # keep under 4096 to be safe


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)

    def _post(self, text: str, parse_mode: Optional[str]) -> requests.Response:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        # Telegram can be flaky; give small retry
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.post(url, data=payload, timeout=20)
                return r
            except Exception as e:
                last_exc = e
                time.sleep(0.6 * (attempt + 1))

        raise RuntimeError(f"Telegram send failed after retries: {last_exc}")

    def _split_chunks(self, text: str) -> list[str]:
        if len(text) <= TELEGRAM_MAX_LEN:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + TELEGRAM_MAX_LEN)

            # try not to cut mid-line
            cut = text.rfind("\n", start, end)
            if cut == -1 or cut <= start + 200:
                cut = end

            chunks.append(text[start:cut].strip())
            start = cut

        return [c for c in chunks if c]

    def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        """
        Sends Telegram message.
        - Splits long messages into multiple parts.
        - If Markdown fails, retries with plain text.
        """
        parts = self._split_chunks(text)

        for i, part in enumerate(parts, start=1):
            suffix = ""
            if len(parts) > 1:
                suffix = f"\n\n({i}/{len(parts)})"
                part_to_send = (part + suffix).strip()
            else:
                part_to_send = part

            # First attempt: requested parse_mode
            r = self._post(part_to_send, parse_mode=parse_mode)
            if r.status_code < 400:
                log.info("Telegram message sent (%s/%s)", i, len(parts))
                continue

            # If Markdown failed, retry without parse_mode
            log.warning("Telegram send failed with parse_mode=%s: %s", parse_mode, r.text)
            r2 = self._post(part_to_send, parse_mode=None)
            if r2.status_code >= 400:
                raise RuntimeError(f"Telegram API error {r2.status_code}: {r2.text}")

            log.info("Telegram message sent as plain text (%s/%s)", i, len(parts))