from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # se télécharge automatiquement au premier lancement

results = model.track(
    source="data/videos/bon_trafic.mp4",
    tracker="bytetrack.yaml",
    classes=[2, 7],  # 2 = car, 7 = truck (COCO)
    save=True,
    project="results/videos_annotees",
    name="test",
)

print("Ça tourne. Vidéo annotée sauvegardée dans results/videos_annotees/test/")