from ultralytics import YOLO

def main():
    # Load model YOLOv11 (pretrained)
    model = YOLO("yolo11n.pt")

    # Training
    results = model.train(
        data=r"C:\Rama\TUGAS AKHIR\PH and CR\V11\Potholes and Crack.v2i.yolov11\data.yaml",
        epochs=150,
        imgsz=640,
        batch=4,        
        device=0,        
    )

    # Simpan model hasil training
    model.save("trained_model.pt")

if __name__ == "__main__":
    main()