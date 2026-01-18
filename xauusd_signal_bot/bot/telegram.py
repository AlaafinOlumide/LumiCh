from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)

    def send_message(self, text: str, parse_mode: str = "Markdown") -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        r = requests.post(url, data=payload, timeout=20)
        if r.status_code >= 400:
            raise RuntimeError(f"Telegram API error {r.status_code}: {r.text}")
        log.info("Telegram message sent")
