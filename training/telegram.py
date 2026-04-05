"""Telegram notification helper for long-running jobs.

Reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from the environment.
If either is missing, notifications are silently skipped (no crash).

Usage::

    from training.telegram import TelegramNotifier

    tg = TelegramNotifier()
    tg.send("Hello from the solver!")
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import requests

# Load .env from project root (walks up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

log = logging.getLogger(__name__)


class TelegramNotifier:
    """Fire-and-forget Telegram message sender."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._last_send: float = 0.0

        if not self.enabled:
            log.warning(
                "Telegram notifications disabled — set TELEGRAM_BOT_TOKEN "
                "and TELEGRAM_CHAT_ID environment variables to enable."
            )

    @property
    def api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send(self, text: str) -> bool:
        """Send a message.  Returns True on success, False otherwise."""
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                self.api_url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            self._last_send = time.time()
            if resp.status_code != 200:
                log.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)
            return False

    def send_if_interval(self, text: str, interval_seconds: float = 600.0) -> bool:
        """Send only if *interval_seconds* have passed since the last send."""
        if time.time() - self._last_send >= interval_seconds:
            return self.send(text)
        return False

    # ── convenience formatters ───────────────────────────────────

    def format_progress(
        self,
        total: int,
        target: int,
        rate: float,
        elapsed_minutes: float,
        errors: int = 0,
    ) -> str:
        """Format a progress message string (does NOT send).

        Parameters
        ----------
        total    : situations generated so far.
        target   : target number of situations.
        rate     : situations per minute.
        elapsed_minutes : wall-clock minutes since start.
        errors   : number of errors encountered.

        Returns
        -------
        The formatted message string.
        """
        pct = total / target * 100 if target else 0
        remaining = (target - total) / rate if rate > 0 else float("inf")
        eta_h, eta_m = divmod(remaining, 60)

        lines = [
            "🃏 <b>GTO Dataset Generator</b>",
            f"✅ Generated: <b>{total:,}</b> / {target:,}  ({pct:.1f}%)",
            f"⚡ Rate: {rate:.1f} sit/min",
            f"⏱ Elapsed: {elapsed_minutes:.0f} min",
            f"🔮 ETA: {int(eta_h)}h {int(eta_m)}m",
        ]
        if errors:
            lines.append(f"⚠️ Errors: {errors}")
        return "\n".join(lines)

    def started(self, target: int) -> None:
        self.send(
            f"🚀 <b>GTO Dataset Generator started</b>\n"
            f"Target: {target:,} situations"
        )

    def stopped(self, total: int, reason: str = "signal") -> None:
        self.send(
            f"🛑 <b>Generator stopped</b> ({reason})\n"
            f"Total generated: {total:,}"
        )

    def error(self, msg: str) -> None:
        self.send(f"❌ <b>Generator error</b>\n{msg}")
