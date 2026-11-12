"""Telegram Bot API long polling, with requests (no new native Python packages).

MVP: /queue, /status, approve, reject, remove image, attest rights.
Every callback requires Telegram user ID allowlist AND a private chat.
MAX is deliberately never called here. No token is logged to stdout.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from database.db import get_session
from database.models import Post
from moderation.service import (
    create_pending_items, decide, get_item_preview, list_pending,
)

LOG = logging.getLogger("telegram_moderation")
ACTIONS = {
    "approve": "✅ Одобрить",
    "reject": "❌ Отклонить",
    "remove_image": "🖼 Убрать фото",
    "rights_confirmed": "📄 Права проверены",
}


def allowed_ids() -> set[int]:
    ids = os.getenv("TELEGRAM_ADMIN_IDS", "")
    return {int(part.strip()) for part in ids.split(",") if part.strip().isdigit()}


class Telegram:
    def __init__(self, token: str):
        if ":" not in token or "/" in token:
            raise ValueError("Не задан корректный TELEGRAM_BOT_TOKEN")
        self.endpoint = f"https://api.telegram.org/bot{token}/"
        self.http = requests.Session()

    def call(self, method: str, *, data: dict | None = None, files=None, timeout: int = 35):
        try:
            response = self.http.post(
                self.endpoint + method,
                data=data or {},
                files=files,
                timeout=(10, timeout),
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Requests exception messages may contain the bot token in the URL.
            # Never print or chain their raw content to terminal/log files.
            raise RuntimeError(
                f"Telegram {method} failed ({type(exc).__name__})"
            ) from None
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API {method}: {result.get('description', 'unknown error')}")
        return result["result"]

    def message(self, chat_id: int, text: str, buttons=None):
        payload = {"chat_id": chat_id, "text": text[:4096]}
        if buttons:
            import json
            payload["reply_markup"] = json.dumps({"inline_keyboard": buttons}, ensure_ascii=False)
        return self.call("sendMessage", data=payload)

    def answer(self, callback_id: str):
        try:
            self.call("answerCallbackQuery", data={"callback_query_id": callback_id}, timeout=12)
        except requests.RequestException as exc:
            LOG.warning("Unable to acknowledge callback: %s", type(exc).__name__)

    def photo(self, chat_id: int, path: Path):
        # JPEG/PNG photo; WebP is sent as document to preserve format.
        method = "sendDocument" if path.suffix.casefold() == ".webp" else "sendPhoto"
        field = "document" if method == "sendDocument" else "photo"
        with path.open("rb") as stream:
            return self.call(method, data={"chat_id": chat_id}, files={field: stream})


def _buttons(item_id: int, image_id: int | None):
    rows = [[
        {"text": ACTIONS["approve"], "callback_data": f"mod:{item_id}:approve"},
        {"text": ACTIONS["reject"], "callback_data": f"mod:{item_id}:reject"},
    ]]
    if image_id is not None:
        rows.append([
            {"text": ACTIONS["remove_image"], "callback_data": f"mod:{item_id}:remove_image"},
            {"text": ACTIONS["rights_confirmed"], "callback_data": f"mod:{item_id}:rights_confirmed"},
        ])
    return rows


def _deliver_pending(bot: Telegram, chat_id: int, limit: int = 5):
    sent = 0
    session = get_session()
    try:
        for item in list_pending(session, limit=limit):
            if item.telegram_message_id is not None:
                continue
            data = get_item_preview(session, item.id)
            if data is None:
                continue
            _, post, image = data
            if image is not None and image.local_path:
                path = Path(image.local_path)
                if path.is_file():
                    try:
                        bot.photo(chat_id, path)
                    except (requests.RequestException, OSError, RuntimeError) as exc:
                        LOG.warning("Photo preview failed for item %s: %s", item.id, type(exc).__name__)
            rights_line = (
                "Права на фото: требуют решения модератора."
                if image is not None
                else "Фото: отсутствует (текстовая публикация)."
            )
            if item.image_rights_decision == "remove_image":
                rights_line = "Фото исключено из публикации."
            elif item.image_rights_decision == "cleared":
                rights_line = "Модератор подтвердил разрешение на публикацию фото."
            header = f"МОДЕРАЦИЯ #{item.id} | Post #{post.id}\n{rights_line}\n\n"
            # Telegram text max 4096, send chunks first; buttons only on final.
            content = post.post_text or ""
            available = 3900 - len(header)
            chunks = [content[i:i + available] for i in range(0, len(content), available)] or [""]
            for chunk in chunks[:-1]:
                bot.message(chat_id, header + chunk)
            response = bot.message(chat_id, header + chunks[-1], buttons=_buttons(item.id, item.image_id))
            item.telegram_chat_id = chat_id
            item.telegram_message_id = response["message_id"]
            session.commit()
            sent += 1
        return sent
    finally:
        session.close()


def _on_update(bot: Telegram, update: dict):
    if "callback_query" in update:
        callback = update["callback_query"]
        bot.answer(callback["id"])
        sender = callback.get("from", {})
        message = callback.get("message") or {}
        chat = message.get("chat") or {}
        if (sender.get("id") not in allowed_ids() or chat.get("type") != "private"
                or chat.get("id") != sender.get("id")):
            return
        data = str(callback.get("data") or "").split(":")
        if len(data) != 3 or data[0] != "mod" or not data[1].isdigit() or data[2] not in ACTIONS:
            return
        item_id = int(data[1])
        with get_session() as session:
            result = decide(session, item_id, data[2], sender["id"])
        bot.message(chat["id"], f"#{item_id}: {result}")
        return

    message = update.get("message") or {}
    sender = message.get("from") or {}
    chat = message.get("chat") or {}
    if chat.get("type") != "private":
        return
    chat_id = chat.get("id")
    user_id = sender.get("id")
    command = (message.get("text") or "").strip().split(" ", 1)[0].split("@", 1)[0]
    if command == "/id":
        bot.message(chat_id, f"Ваш Telegram ID: {user_id}")
        return
    if user_id not in allowed_ids() or chat_id != user_id:
        bot.message(chat_id, "Нет доступа. Для подключения модератора внесите Telegram ID в локальный .env.")
        return
    if command in ("/start", "/help"):
        bot.message(chat_id, "Бот модерации запущен.\n"
                    "/queue — подготовить и отправить ожидающие новости\n"
                    "/status — посмотреть очередь\n"
                    "/id — ваш Telegram ID\n"
                    "MAX не публикует сообщения из этого бота.")
    elif command == "/status":
        with get_session() as session:
            from database.moderation_models import ModerationItem
            from sqlalchemy import func
            rows = session.query(ModerationItem.status, func.count()).group_by(ModerationItem.status).all()
        stats = ", ".join(f"{status}: {count}" for status, count in rows) or "пусто"
        bot.message(chat_id, f"Очередь: {stats}")
    elif command == "/queue":
        with get_session() as session:
            created = create_pending_items(session, limit=15)
        delivered = _deliver_pending(bot, chat_id, limit=10)
        bot.message(chat_id, f"Новых заданий: {created}; отправлено на модерацию: {delivered}.")
    else:
        bot.message(chat_id, "Команды: /queue, /status, /id, /help.")


def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    ids = allowed_ids()
    if not token or not ids:
        raise SystemExit("Укажите TELEGRAM_BOT_TOKEN и TELEGRAM_ADMIN_IDS в локальном .env")
    if os.getenv("MAX_PUBLISHING_ENABLED", "false").strip().casefold() in ("1", "true", "on", "yes"):
        raise SystemExit("Для тестирования бота установите MAX_PUBLISHING_ENABLED=false")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    bot = Telegram(token)
    me = bot.call("getMe")
    LOG.info("Telegram moderation bot @%s started; MAX is disabled.", me.get("username"))
    offset = None
    while True:
        try:
            params = {"timeout": 25, "allowed_updates": '["message","callback_query"]'}
            if offset is not None:
                params["offset"] = offset
            updates = bot.call("getUpdates", data=params, timeout=40)
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    _on_update(bot, update)
                except Exception:
                    LOG.exception("Error processing Telegram update %s", update.get("update_id"))
        except KeyboardInterrupt:
            LOG.info("Telegram bot stopped.")
            return
        except (requests.RequestException, RuntimeError) as exc:
            LOG.warning("Telegram API temporarily unavailable: %s", type(exc).__name__)
            time.sleep(5)


if __name__ == "__main__":
    main()
