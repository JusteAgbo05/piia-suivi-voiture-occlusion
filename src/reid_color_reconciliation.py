from dataclasses import dataclass, field

import cv2
import numpy as np
from pathlib import Path

CLASS_CAR = 2
CLASS_TRUCK = 7


@dataclass
class TrackRecord:
    """Une observation : un véhicule suivi à une frame donnée."""
    frame: int
    track_id: int
    cls: int
    bbox: tuple  # (x1, y1, x2, y2) en pixels
    crop: np.ndarray  # image BGR découpée 


# Extraction des enregistrements depuis les résultats Ultralytics


def extract_track_records(results) -> list:
    """
    Parcourt les résultats de model.track(..., stream=True) et extrait,
    pour chaque frame, un TrackRecord par véhicule suivi (avec sa vignette).
    """
    records = []
    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue
        frame_img = r.orig_img  # image BGR (numpy array) de cette frame
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        clss = r.boxes.cls.int().cpu().tolist()

        for box, tid, cls in zip(xyxy, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_img.shape[1], x2), min(frame_img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                crop = None
            else:
                crop = frame_img[y1:y2, x1:x2].copy()
            records.append(TrackRecord(frame_idx, tid, cls, (x1, y1, x2, y2), crop))
    return records



# Descripteur de couleur

def compute_hsv_histogram(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros(32 * 32, dtype=np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [1, 2], None, [32, 32], [0, 256, 0, 256])  # canaux a, b uniquement
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def histogram_similarity(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Similarité entre deux histogrammes (corrélation : 1 = identique, -1 = opposé)."""
    return float(cv2.compareHist(hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_CORREL))


# Détection des événements d'occlusion + réconciliation

def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _bbox_overlap_ratio(box_a, box_b) -> float:
    """Fraction de box_a recouverte par box_b (0 = aucun contact, 1 = totalement recouverte)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    return inter / area_a if area_a > 0 else 0.0


def _bbox_gap_distance(box_a, box_b) -> float:
    """Distance entre deux boîtes (0 si elles se touchent/chevauchent, sinon distance du bord le plus proche)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return (dx ** 2 + dy ** 2) ** 0.5


def _truck_overlapping(record, truck_records_at_frame: list, min_overlap_ratio: float) -> bool:
    if not truck_records_at_frame:
        return False
    return any(
        _bbox_overlap_ratio(record.bbox, truck.bbox) >= min_overlap_ratio
        for truck in truck_records_at_frame
    )


def _truck_adjacent(record, truck_records_at_frame: list, max_gap_px: float) -> bool:
    if not truck_records_at_frame:
        return False
    return any(
        _bbox_gap_distance(record.bbox, truck.bbox) <= max_gap_px
        for truck in truck_records_at_frame
    )


def _estimate_velocity(recs: list, lookback: int = 5):
    """Estime la vitesse (px/frame) du véhicule à partir de ses dernières observations."""
    if len(recs) < 2:
        return (0.0, 0.0)
    tail = recs[-lookback:] if len(recs) >= lookback else recs
    first, last = tail[0], tail[-1]
    dframes = last.frame - first.frame
    if dframes <= 0:
        return (0.0, 0.0)
    c0, c1 = _bbox_center(first.bbox), _bbox_center(last.bbox)
    return ((c1[0] - c0[0]) / dframes, (c1[1] - c0[1]) / dframes)


def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def find_and_reconcile(
    records: list,
    max_gap_frames: int = 45,
    max_dist_px: float = 150.0,
    similarity_thresh: float = 0.7,
    min_truck_overlap: float = 0.35,
    truck_adjacency_px: float = 100.0,
    motion_tolerance_px: float = 70.0,
    max_size_ratio: float = 2.0,
    debug: bool = False,
    save_crops_dir=None,
) -> dict:
    car_records = [r for r in records if r.cls == CLASS_CAR]
    truck_records = [r for r in records if r.cls == CLASS_TRUCK]

    trucks_by_frame = {}
    for r in truck_records:
        trucks_by_frame.setdefault(r.frame, []).append(r)

    by_id = {}
    for r in car_records:
        by_id.setdefault(r.track_id, []).append(r)
    for tid in by_id:
        by_id[tid].sort(key=lambda r: r.frame)

    last_seen = {tid: recs[-1] for tid, recs in by_id.items()}
    first_seen = {tid: recs[0] for tid, recs in by_id.items()}
    velocity = {tid: _estimate_velocity(recs) for tid, recs in by_id.items()}

    all_ids_sorted_by_first_frame = sorted(first_seen, key=lambda tid: first_seen[tid].frame)

    mapping = {}  # new_id -> canonical_id

    for lost_id, lost_rec in last_seen.items():
        if lost_id in mapping:
            continue
        if not _truck_overlapping(lost_rec, trucks_by_frame.get(lost_rec.frame, []), min_truck_overlap):
            continue

        vx, vy = velocity[lost_id]
        lost_center = _bbox_center(lost_rec.bbox)
        lost_area = _bbox_area(lost_rec.bbox)

        best_candidate, best_score = None, similarity_thresh
        for new_id in all_ids_sorted_by_first_frame:
            if new_id == lost_id or new_id in mapping:
                continue
            new_first = first_seen[new_id]
            gap = new_first.frame - lost_rec.frame
            if gap <= 0 or gap > max_gap_frames:
                continue

            dist = _distance(lost_center, _bbox_center(new_first.bbox))
            if dist > max_dist_px:
                continue

            # Position prédite à partir de la trajectoire connue avant la disparition
            predicted = (lost_center[0] + vx * gap, lost_center[1] + vy * gap)
            motion_error = _distance(predicted, _bbox_center(new_first.bbox))
            if motion_error > motion_tolerance_px:
                continue

            new_area = _bbox_area(new_first.bbox)
            if lost_area > 0 and new_area > 0:
                ratio = max(lost_area, new_area) / min(lost_area, new_area)
                if ratio > max_size_ratio:
                    continue

            if not _truck_adjacent(new_first, trucks_by_frame.get(new_first.frame, []), truck_adjacency_px):
                continue

            sim = histogram_similarity(
                compute_hsv_histogram(lost_rec.crop),
                compute_hsv_histogram(new_first.crop),
            )
            if debug:
                print(f"  candidat: {lost_id}(f{lost_rec.frame}) -> {new_id}(f{new_first.frame}) "
                      f"| gap={gap} dist={dist:.0f} motion_err={motion_error:.0f} sim={sim:.2f}")
            if sim > best_score:
                best_candidate, best_score = new_id, sim

        if best_candidate is not None:
            mapping[best_candidate] = lost_id
            if save_crops_dir is not None:
                new_rec = first_seen[best_candidate]
                save_crops_dir.mkdir(parents=True, exist_ok=True)
                pair_name = f"{lost_id}_f{lost_rec.frame}_to_{best_candidate}_f{new_rec.frame}"
                if lost_rec.crop is not None:
                    cv2.imwrite(str(save_crops_dir / f"{pair_name}__avant.jpg"), lost_rec.crop)
                if new_rec.crop is not None:
                    cv2.imwrite(str(save_crops_dir / f"{pair_name}__apres.jpg"), new_rec.crop)

    return mapping


def apply_mapping(records: list, mapping: dict) -> list:
    """Renvoie une nouvelle liste de records avec les IDs réconciliés."""
    corrected = []
    for r in records:
        new_id = mapping.get(r.track_id, r.track_id)
        corrected.append(TrackRecord(r.frame, new_id, r.cls, r.bbox, r.crop))
    return corrected


def count_unique_ids(records: list, cls: int = CLASS_CAR) -> int:
    return len({r.track_id for r in records if r.cls == cls})