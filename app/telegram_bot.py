import json
import os
import threading

import requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
REGISTERED_USERS_FILE = "registered_users.json"

_lock = threading.Lock()


def _load_users():
    if not os.path.exists(REGISTERED_USERS_FILE):
        return []
    with open(REGISTERED_USERS_FILE) as f:
        return json.load(f)


def _save_users(users):
    with open(REGISTERED_USERS_FILE, "w") as f:
        json.dump(users, f)


class RegistrationListener:
    """Слушает команду /start в фоновом потоке и регистрирует новых пользователей."""

    def __init__(self):
        self._running = True
        self._last_update_id = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                params = {"timeout": 30}
                if self._last_update_id is not None:
                    params["offset"] = self._last_update_id + 1
                resp = requests.get(f"{API_URL}/getUpdates", params=params, timeout=35)
                data = resp.json()
            except requests.RequestException:
                continue

            for update in data.get("result", []):
                self._last_update_id = update["update_id"]
                msg = update.get("message")
                if not msg or "text" not in msg:
                    continue
                if msg["text"].strip() != "/start":
                    continue

                chat_id = msg["chat"]["id"]
                name = msg["chat"].get("first_name", "")
                with _lock:
                    users = _load_users()
                    if chat_id not in users:
                        users.append(chat_id)
                        _save_users(users)
                        print(f"[telegram] новый подписчик: {name} ({chat_id})")

                requests.post(
                    f"{API_URL}/sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": f"Здравствуйте, {name}! Вы подписаны на уведомления о нарушениях на заводе.",
                    },
                    timeout=10,
                )

    def stop(self):
        self._running = False


def send_violation_alert(portrait_path: str, full_frame_path: str, when_str: str):
    """Отправляет всем зарегистрированным: дата/время, фото человека, потом полный кадр."""
    with _lock:
        users = _load_users()

    if not users:
        return

    caption = f"⚠️ Нарушение: телефон в цехе\n{when_str}"

    for chat_id in users:
        try:
            with open(portrait_path, "rb") as f:
                requests.post(
                    f"{API_URL}/sendPhoto",
                    data={"chat_id": chat_id, "caption": caption},
                    files={"photo": f},
                    timeout=20,
                )
            with open(full_frame_path, "rb") as f:
                requests.post(
                    f"{API_URL}/sendPhoto",
                    data={"chat_id": chat_id},
                    files={"photo": f},
                    timeout=20,
                )
        except requests.RequestException as e:
            print(f"[telegram] не удалось отправить {chat_id}: {e}")
