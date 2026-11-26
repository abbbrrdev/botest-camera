import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from app import telegram_bot
from app.camera import LatestFrameReader
from app.text_overlay import put_text_ru

TELEGRAM_ENABLED = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))

# На headless-машине (без дисплея, например заводской сервер по SSH) окно
# показать нельзя — cv2.imshow там упадёт с ошибкой. SHOW_WINDOW=0 отключает
# окно, оставляя детекцию/алерты/Telegram работать как обычно.
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "1") != "0"

# Детект телефона: своя модель, дообученная на FPI-Det + реальных кадрах этой
# камеры. Работает надёжно, когда телефон физически виден на пикселях
# (хорошее освещение, не силуэт, не смаз движения) — на силуэтах против
# света объект не резолвится в принципе, это ограничение самой картинки,
# а не модели.
PHONE_DETECTION_ENABLED = True
PHONE_MODEL_PATH = "models/phone_head.pt"
PERSON_MODEL_PATH = "models/yolov8s.pt"

# --- Зона конвейера (калибровано по snapshot.jpg, см. roi_preview.jpg) ---
ROI = (1088, 320, 1472, 512)  # x1, y1, x2, y2

# --- Подсчёт поддонов: порог доли "переднего плана" внутри ROI ---
ENTER_THRESHOLD = 0.12
EXIT_THRESHOLD = 0.05

# --- Простой линии: если в ROI нет движения дольше этого времени, тревога ---
STALL_SECONDS = 5 * 60

# --- Детект "человек с телефоном" ---
# Каскад: найти человека (COCO) -> вырезать кроп вокруг него -> искать телефон
# уже в увеличенном кропе своей моделью (phone_head_v4).
PERSON_CLASS_ID = 0  # в COCO-модели (person_model)
PHONE_CLASS_ID = 0  # в своей модели (phone_model): 0=phone, 1=head
PERSON_CONF = 0.25
PHONE_CONF = 0.65  # ниже этого — сплошной шумовой пол модели, не реальные детекции
INFER_IMGSZ = 1280  # для полнокадрового прохода (поиск людей)
CROP_IMGSZ = 640  # для кропа человека
CROP_PAD_FRAC = 0.35  # запас вокруг рамки человека при кропе, доля от размера рамки
PHONE_ALERT_COOLDOWN = 15  # сек, чтобы не спамить одним и тем же нарушением

# Мягкий порог: отсекаем только совсем крошечные точки (компьютерный шум),
# НЕ фильтр "по расстоянию" — на реальных данных размер рамки почти не
# коррелирует с успехом детекции (решает освещение/смаз, а не дистанция);
# наш единственный уверенный кейс был на рамке высотой 83px.
MIN_PERSON_HEIGHT_PX = 50

LOG_PATH = "alerts.log"

# --- Фотоархив пойманных с телефоном: кроп именно человека в полный рост ---
PHONE_PHOTO_DIR = "phone"
PORTRAIT_PAD_FRAC = 0.08  # небольшой запас, чтобы не обрезать голову/ступни

WINDOW_NAME = "Factory Monitor - Camera 01"


def log_alert(message: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def run_loop(reader, person_model, phone_model):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)

    pallet_count = 0
    gate_occupied = False

    prev_gray_roi = None
    last_motion_time = time.time()
    line_stalled = False

    last_phone_alert = 0

    while True:
        ret, frame = reader.read()
        if not ret:
            time.sleep(0.05)
            continue
        clean_frame = frame.copy()  # без наших рамок/текста, для сохранения на диск

        x1, y1, x2, y2 = ROI
        roi = frame[y1:y2, x1:x2]

        # --- 1. Подсчёт поддонов (MOG2 gate) ---
        fg_mask = bg_subtractor.apply(roi)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_ratio = cv2.countNonZero(fg_mask) / fg_mask.size

        if not gate_occupied and fg_ratio > ENTER_THRESHOLD:
            gate_occupied = True
            pallet_count += 1
            log_alert(f"Поддон #{pallet_count} прошёл через зону конвейера")
        elif gate_occupied and fg_ratio < EXIT_THRESHOLD:
            gate_occupied = False

        # --- 4. Простой линии (frame diff) ---
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        if prev_gray_roi is not None:
            diff = cv2.absdiff(prev_gray_roi, gray_roi)
            motion_score = diff.mean()
            if motion_score > 2.0:
                last_motion_time = time.time()
                if line_stalled:
                    line_stalled = False
                    log_alert("Линия снова в движении")
        prev_gray_roi = gray_roi

        idle_for = time.time() - last_motion_time
        if idle_for > STALL_SECONDS and not line_stalled:
            line_stalled = True
            log_alert(f"ПРОСТОЙ: нет движения на конвейере {int(idle_for)} сек")

        # --- 3. Человек с телефоном: человек -> кроп -> специализированная модель телефона ---
        frame_h, frame_w = frame.shape[:2]
        now = time.time()

        person_results = person_model.predict(
            frame, classes=[PERSON_CLASS_ID], conf=PERSON_CONF, imgsz=INFER_IMGSZ, verbose=False
        )[0]
        person_boxes = [box.xyxy[0].tolist() for box in person_results.boxes]

        for xyxy in person_boxes:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 200, 0), 2)

        violation = False
        violators = []

        if PHONE_DETECTION_ENABLED:
            for person in person_boxes:
                px1, py1, px2, py2 = person
                pw, ph = px2 - px1, py2 - py1
                if ph < MIN_PERSON_HEIGHT_PX:
                    continue  # слишком мелкая точка, компьютерный шум

                pad_x, pad_y = pw * CROP_PAD_FRAC, ph * CROP_PAD_FRAC
                cx1, cy1, cx2, cy2 = clamp_box(
                    (px1 - pad_x, py1 - pad_y, px2 + pad_x, py2 + pad_y), frame_w, frame_h
                )
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                phone_results = phone_model.predict(
                    crop, classes=[PHONE_CLASS_ID], conf=PHONE_CONF, imgsz=CROP_IMGSZ, verbose=False
                )[0]
                if len(phone_results.boxes) == 0:
                    continue

                violation = True
                violators.append(person)
                best_conf = max(float(b.conf[0]) for b in phone_results.boxes)
                for box in phone_results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    fx1, fy1, fx2, fy2 = bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1
                    conf = float(box.conf[0])
                    cv2.rectangle(frame, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 0, 255), 2)
                    frame = put_text_ru(frame, f"телефон {conf:.2f}", (int(fx1), int(fy1) - 26), 22, (0, 0, 255))

            if violation and now - last_phone_alert > PHONE_ALERT_COOLDOWN:
                last_phone_alert = now
                log_alert(f"НАРУШЕНИЕ: обнаружен телефон, conf={best_conf:.2f}")

                for vp in violators:
                    vx1, vy1, vx2, vy2 = vp
                    vw, vh = vx2 - vx1, vy2 - vy1
                    ppad_x, ppad_y = vw * PORTRAIT_PAD_FRAC, vh * PORTRAIT_PAD_FRAC
                    sx1, sy1, sx2, sy2 = clamp_box(
                        (vx1 - ppad_x, vy1 - ppad_y, vx2 + ppad_x, vy2 + ppad_y), frame_w, frame_h
                    )
                    portrait = clean_frame[sy1:sy2, sx1:sx2]
                    if portrait.size == 0:
                        continue

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    portrait_path = os.path.join(PHONE_PHOTO_DIR, f"{stamp}_person.jpg")
                    full_path = os.path.join(PHONE_PHOTO_DIR, f"{stamp}_full.jpg")
                    cv2.imwrite(portrait_path, portrait)
                    cv2.imwrite(full_path, clean_frame)

                    if TELEGRAM_ENABLED:
                        when_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                        telegram_bot.send_violation_alert(portrait_path, full_path, when_str)

        # --- overlay ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        status_color = (0, 0, 255) if line_stalled else (0, 200, 0)
        status_text = "ПРОСТОЙ" if line_stalled else "РАБОТАЕТ"
        frame = put_text_ru(frame, f"Поддонов: {pallet_count}", (30, 20), 36, (0, 255, 0))
        frame = put_text_ru(frame, f"Линия: {status_text}", (30, 65), 36, status_color)
        if violation:
            frame = put_text_ru(frame, "ТЕЛЕФОН В ЦЕХЕ!", (30, 110), 40, (0, 0, 255))

        if SHOW_WINDOW:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    os.makedirs(PHONE_PHOTO_DIR, exist_ok=True)

    if TELEGRAM_ENABLED:
        telegram_bot.RegistrationListener()
        print("Telegram-бот запущен, слушает /start для регистрации подписчиков.")
    else:
        print("TELEGRAM_BOT_TOKEN не задан — уведомления в Telegram отключены.")

    reader = LatestFrameReader()
    person_model = YOLO(PERSON_MODEL_PATH)
    phone_model = YOLO(PHONE_MODEL_PATH) if PHONE_DETECTION_ENABLED else None

    if SHOW_WINDOW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("Запущено. 'q' в окне — выход.")
    else:
        print("Запущено без окна (SHOW_WINDOW=0). Ctrl+C — выход.")

    try:
        run_loop(reader, person_model, phone_model)
    except KeyboardInterrupt:
        pass
    finally:
        reader.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
