import os

import cv2

CAM_IP = os.environ.get("CAM_IP", "192.168.1.12")
CAM_USER = os.environ.get("CAM_USER", "admin")
CAM_PASS = os.environ["CAM_PASS"]
CHANNEL = os.environ.get("CAM_CHANNEL", "101")

URL = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_IP}:554/Streaming/Channels/{CHANNEL}"


def main():
    print(f"Подключаюсь: {URL}")
    cap = cv2.VideoCapture(URL)
    if not cap.isOpened():
        print("Не удалось открыть поток")
        return

    print("Окно открыто. Нажмите 'q' в окне видео, чтобы выйти.")
    window = "HiWatch - Camera 01"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Кадр не получен, останавливаюсь")
            break

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
