import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from reid_color_reconciliation import (
    CLASS_CAR,
    CLASS_TRUCK,
    apply_mapping,
    count_unique_ids,
    extract_track_records,
    find_and_reconcile,
)

MODEL_PATH = "yolov8m.pt"
CONF = 0.1


def run_and_count(model, video_path: str, tracker_config: str) -> int:
    """Lance le suivi avec un tracker donné et renvoie le nombre d'IDs uniques (voitures)."""
    results = model.track(
        source=video_path,
        tracker=tracker_config,
        classes=[CLASS_CAR, CLASS_TRUCK],
        conf=CONF,
        persist=True,
        stream=True,
        verbose=False,
    )
    records = extract_track_records(results)
    return count_unique_ids(records, cls=CLASS_CAR)


def run_our_version(model, video_path: str) -> tuple:
    """Notre version : ByteTrack réglé + réconciliation par couleur/mouvement.
    Renvoie (ids_avant, ids_apres, mapping)."""
    results = model.track(
        source=video_path,
        tracker="bytetrack_custom.yaml",
        classes=[CLASS_CAR, CLASS_TRUCK],
        conf=CONF,
        persist=True,
        stream=True,
        verbose=False,
    )
    records = extract_track_records(results)
    before = count_unique_ids(records, cls=CLASS_CAR)
    mapping = find_and_reconcile(records)
    corrected = apply_mapping(records, mapping)
    after = count_unique_ids(corrected, cls=CLASS_CAR)
    return before, after, mapping


def main():
    parser = argparse.ArgumentParser(description="Évaluation comparative des 3 versions de suivi sur plusieurs vidéos.")
    parser.add_argument("videos", nargs="*", help="Chemins des vidéos à évaluer")
    parser.add_argument("--dir", type=str, help="Alternative : dossier contenant les vidéos (.mp4)")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--out-csv", type=str, default="results/metrics/evaluation_finale.csv")
    args = parser.parse_args()

    video_paths = list(args.videos)
    if args.dir:
        video_paths += sorted(str(p) for p in Path(args.dir).glob("*.mp4"))
    if not video_paths:
        raise SystemExit("Aucune vidéo fournie. Passe des chemins en argument, ou --dir un dossier.")

    model = YOLO(args.model)

    rows = []
    print(f"\n{'Vidéo':<20}{'ByteTrack':<12}{'BoT-SORT+ReID':<16}{'Notre version':<16}{'Mapping (notre version)'}")
    print("-" * 100)

    for video_path in video_paths:
        name = Path(video_path).name

        bytetrack_ids = run_and_count(model, video_path, "bytetrack.yaml")
        botsort_ids = run_and_count(model, video_path, "botsort_reid.yaml")
        our_before, our_after, mapping = run_our_version(model, video_path)

        print(f"{name:<20}{bytetrack_ids:<12}{botsort_ids:<16}{our_after:<16}{mapping if mapping else '(aucune correction)'}")

        rows.append({
            "video": name,
            "bytetrack_ids_uniques": bytetrack_ids,
            "botsort_reid_ids_uniques": botsort_ids,
            "notre_version_avant": our_before,
            "notre_version_apres": our_after,
            "nb_corrections": len(mapping),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRésultats sauvegardés dans : {out_path}")
    print("\nNote : le nombre de \"poids lourds/voitures réels\" (vérité terrain) n'est pas calculé")
    print("automatiquement — comptez-le à l'œil sur chaque vidéo pour compléter le tableau du rapport.")


if __name__ == "__main__":
    main()