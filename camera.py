import os
import threading

import cv2

CAM_IP = os.environ.get("CAM_IP", "192.168.1.12")
CAM_USER = os.environ.get("CAM_USER", "admin")
CAM_PASS = os.environ.get("CAM_PASS")
CHANNEL = os.environ.get("CAM_CHANNEL", "101")

if not CAM_PASS:
    raise RuntimeError(
        "Не задан пароль камеры. Установите переменную окружения CAM_PASS "
        "(например: export CAM_PASS=... перед запуском)."
    )

RTSP_URL = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/{CHANNEL}"


def open_stream():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть поток: {RTSP_URL}")
    return cap


class LatestFrameReader:
    """Читает RTSP в отдельном потоке и всегда отдаёт самый свежий кадр.

    Нужен, когда обработка (YOLO и т.п.) медленнее, чем поток кадров с камеры:
    без этого cv2.VideoCapture.read() в основном цикле отстаёт от реального
    времени, буфер декодера копится, и в итоге на экране "рваные"/старые кадры.
    """

    def __init__(self):
        self._cap = open_stream()
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        return frame is not None, frame

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()
