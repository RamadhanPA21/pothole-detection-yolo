from ultralytics import YOLO

def main():
    # Load model YOLOv8
    model = YOLO("yolov8n.pt")

    # Training
    results = model.train(
        data=r"C:\Rama\TUGAS AKHIR\PH and CR\V8\Potholes and Crack.v1i.yolov8\data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,       
        device=0,       
    )

    # Simpan model hasil training
    model.save("trained_model.pt")

if __name__ == "__main__":
    main()