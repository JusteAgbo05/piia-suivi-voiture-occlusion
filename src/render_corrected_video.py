import argparse
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from reid_color_reconciliation import (
    CLASS_CAR,
    CLASS_TRUCK,
    apply_mapping,
    count_unique_ids,
    extract_track_records,
    find_and_reconcile,
)

CLASS_NAMES = {CLASS_CAR: "car", CLASS_TRUCK: "truck"}


def color_for_id(track_id: int) -> tuple:
    """Couleur BGR stable pour un ID donné (pour repérer visuellement la continuité)."""
    rng = random.Random(track_id)
    return (rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255))


def group_by_frame(records: list) -> dict:
    by_frame = defaultdict(list)
    for r in records:
        by_frame[r.frame].append(r)
    return by_frame


def draw_frame(frame_img, records_for_frame: list):
    out = frame_img.copy()
    for r in records_for_frame:
        x1, y1, x2, y2 = r.bbox
        color = color_for_id(r.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{r.track_id} {CLASS_NAMES.get(r.cls, '')}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return out


def _ensure_browser_playable(video_path: Path) -> Path:
    """Ré-encode la vidéo en H.264 + yuv420p pour qu'elle soit lisible dans un navigateur."""
    fixed_path = video_path.with_name(video_path.stem + "_web.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(fixed_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return fixed_path


def render_video(source_video: Path, records: list, out_path: Path, fps: float) -> int:
    cap = cv2.VideoCapture(str(source_video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    by_frame = group_by_frame(records)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = draw_frame(frame, by_frame.get(frame_idx, []))
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    # Ré-encodage en H.264 pour que la vidéo soit lisible dans un navigateur
    fixed = _ensure_browser_playable(raw_path)
    fixed.replace(out_path)
    raw_path.unlink(missing_ok=True)

    return frame_idx


def main():
    parser = argparse.ArgumentParser(description="Génère les vidéos avant/après réconciliation d'ID.")
    parser.add_argument("video", type=str, help="Chemin vers la vidéo source")
    parser.add_argument("--outdir", type=str, default="results/videos_annotees")
    parser.add_argument("--tracker", type=str, default="bytetrack_custom.yaml")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--debug", action="store_true", help="Affiche le détail des paires candidates évaluées")
    parser.add_argument("--save-crops", action="store_true", help="Sauvegarde les vignettes comparées pour chaque fusion retenue (vérification visuelle)")
    args = parser.parse_args()

    video_path = Path(args.video)
    outdir = Path(args.outdir)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0
    cap.release()

    print(f"Suivi de {video_path.name}...")
    results = model.track(
        source=str(video_path),
        tracker=args.tracker,
        classes=[CLASS_CAR, CLASS_TRUCK],
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
    )
    records = extract_track_records(results)

    print("Réconciliation des identifiants...")
    crops_dir = (outdir / f"{video_path.stem}_crops_comparaison") if args.save_crops else None
    mapping = find_and_reconcile(records, debug=args.debug, save_crops_dir=crops_dir)
    if crops_dir is not None:
        print(f"Vignettes comparées sauvegardées dans : {crops_dir}")
    corrected = apply_mapping(records, mapping)

    before_ids = count_unique_ids(records)
    after_ids = count_unique_ids(corrected)

    print(f"\nMapping appliqué : {mapping if mapping else '(aucune correction déclenchée)'}")
    print(f"IDs uniques (voitures) avant réconciliation : {before_ids}")
    print(f"IDs uniques (voitures) après réconciliation  : {after_ids}")

    avant_path = outdir / f"{video_path.stem}_avant.mp4"
    apres_path = outdir / f"{video_path.stem}_apres.mp4"

    print(f"\nRendu de la vidéo AVANT (IDs bruts) -> {avant_path}")
    render_video(video_path, records, avant_path, fps)

    print(f"Rendu de la vidéo APRES (IDs réconciliés) -> {apres_path}")
    render_video(video_path, corrected, apres_path, fps)

    print("\nTerminé. Compare les deux vidéos.")


if __name__ == "__main__":
    main()