from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # se télécharge automatiquement au premier lancement

results = model.track(
    source="data/videos/trafic.mp4",
    tracker="bytetrack.yaml",
    classes=[2, 7],  # 2 = car, 7 = truck (COCO)
    save=True,
    project="runs/detect",
    name="test",
)

print("Ça tourne. Vidéo annotée sauvegardée dans runs/detect/results/video_annotees/test/")