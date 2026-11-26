import os
import sys

import cv2

CAM_IP = os.environ.get("CAM_IP", "192.168.1.12")
CAM_USER = os.environ.get("CAM_USER", "admin")
CAM_PASS = os.environ["CAM_PASS"]

CHANNELS_TO_TRY = ["101", "102"]

SNAPSHOT_PATH = "snapshot.jpg"


def try_stream(url: str):
    print(f"Пробую: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("  -> не удалось открыть поток")
        cap.release()
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("  -> поток открылся, но кадр прочитать не удалось")
        return None

    return frame


def main():
    for channel in CHANNELS_TO_TRY:
        url = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/{channel}"
        frame = try_stream(url)
        if frame is not None:
            h, w = frame.shape[:2]
            cv2.imwrite(SNAPSHOT_PATH, frame)
            print(f"\nУспех! Канал {channel}, разрешение {w}x{h}")
            print(f"Кадр сохранён в {SNAPSHOT_PATH}")
            return

    print("\nНе удалось подключиться ни по одному из каналов.")
    print("Проверьте:")
    print("  - логин/пароль (CAM_USER / CAM_PASS)")
    print("  - что RTSP включён на камере (Configuration -> Network -> Advanced Settings)")
    print("  - доступность IP: ping " + CAM_IP)
    sys.exit(1)


if __name__ == "__main__":
    main()
