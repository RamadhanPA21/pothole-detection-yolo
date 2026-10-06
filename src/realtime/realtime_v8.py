from ultralytics import YOLO
import cv2

# =========================
# LOAD MODEL
# =========================
model = YOLO("yolov8_best.pt")

# Cek nama class (debug sekali saja)
print("Mapping Class:", model.names)

# =========================
# URL IP WEBCAM
# =========================
ip_camera_url = "http://10.222.62.138:8080/video"

# =========================
# BUKA STREAM
# =========================
cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

confidence_threshold = 0.15

# =========================
# LOOP REAL-TIME
# =========================
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("❌ Gagal mengambil frame dari IP Webcam")
        break

    # Resize (opsional)
    frame = cv2.resize(frame, (640, 480))

    # =========================
    # DETEKSI
    # =========================
    results = model(frame, conf=confidence_threshold)

    # =========================
    # VISUALISASI
    # =========================
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, scores):
                x1, y1, x2, y2 = map(int, box)

                class_id = int(cls)
                class_name = model.names[class_id]
                confidence = float(conf)

                # =========================
                # WARNA BERDASARKAN CLASS ID
                # =========================
                if class_id == 0:
                    color = (0, 0, 255)  # 🔴 Jalan Berlubang
                elif class_id == 1:
                    color = (0, 255, 0)  # 🟢 Jalan Retak
                else:
                    color = (255, 255, 0)  # 🟡 Default

                label = f"{class_name} {confidence:.2f}"

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Background label
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

                # Text
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Debug
                print(f"{class_name} | {confidence:.2f}")

    # =========================
    # TAMPILKAN
    # =========================
    cv2.imshow("Deteksi Jalan Berlubang & Retak (YOLO + IP Cam)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# RELEASE
# =========================
cap.release()
cv2.destroyAllWindows()