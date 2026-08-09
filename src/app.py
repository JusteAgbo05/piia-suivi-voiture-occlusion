from collections import defaultdict
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

try:
    import spaces  # disponible uniquement sur l'infrastructure Hugging Face
    gpu_decorator = spaces.GPU(duration=120)  # 120s de budget GPU par appel (ZeroGPU)
except ImportError:
    def gpu_decorator(fn):  # no-op en local, où le package `spaces` n'existe pas
        return fn

from reid_color_reconciliation import (
    CLASS_CAR,
    CLASS_TRUCK,
    apply_mapping,
    count_unique_ids,
    extract_track_records,
    find_and_reconcile,
)
from render_corrected_video import _ensure_browser_playable, render_video

MODEL_PATH = "yolov8m.pt"

STANDARD_TRACKERS = {
    "ByteTrack (réglages par défaut)": "bytetrack.yaml",
    "BoT-SORT + ReID générique": "botsort_reid.yaml",
}
OUR_VERSION_LABEL = "Notre version (ByteTrack + réconciliation couleur/mouvement)"
ALL_METHODS = list(STANDARD_TRACKERS.keys()) + [OUR_VERSION_LABEL]

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def _summarize(nb_frames: int, unique_ids: set, frames_per_id: dict, label: str, extra: str = "") -> str:
    duree_moyenne = (sum(frames_per_id.values()) / len(frames_per_id)) if frames_per_id else 0
    return (
        f"**Méthode utilisée** : {label}\n\n"
        f"**Images traitées** : {nb_frames}\n\n"
        f"**Identifiants uniques attribués** : {len(unique_ids)}\n\n"
        f"**Durée moyenne de suivi par identifiant** : {duree_moyenne:.1f} images\n\n"
        f"{extra}"
        f"_Un nombre élevé d'identifiants uniques par rapport au nombre réel de "
        f"véhicules visibles suggère des changements d'identité (ID Switch)._"
    )


def _run_standard(video_path: str, tracker_config: str, conf: float, label: str):
    """ByteTrack ou BoT-SORT+ReID : on utilise directement le rendu vidéo natif d'Ultralytics."""
    model = get_model()
    out_dir = Path("gradio_runs")
    run_name = f"run_{Path(video_path).stem}_{tracker_config.replace('.yaml', '')}"

    results = model.track(
        source=video_path,
        tracker=tracker_config,
        classes=[CLASS_CAR, CLASS_TRUCK],
        conf=conf,
        persist=True,
        save=True,
        project=str(out_dir),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    unique_ids, frames_per_id, nb_frames, save_dir = set(), defaultdict(int), 0, None
    for r in results:
        nb_frames += 1
        save_dir = r.save_dir
        if r.boxes.id is not None:
            for tid in r.boxes.id.int().tolist():
                unique_ids.add(tid)
                frames_per_id[tid] += 1

    annotated_path = None
    if save_dir is not None:
        candidates = list(Path(save_dir).glob("*.avi")) + list(Path(save_dir).glob("*.mp4"))
        annotated_path = candidates[0] if candidates else None
        if annotated_path is not None:
            annotated_path = _ensure_browser_playable(annotated_path)

    summary = _summarize(nb_frames, unique_ids, frames_per_id, label)
    return (str(annotated_path) if annotated_path else None), summary


def _run_our_version(video_path: str, conf: float):
    """Notre version : suivi + réconciliation par couleur/mouvement, puis rendu vidéo maison."""
    model = get_model()
    results = model.track(
        source=video_path,
        tracker="bytetrack_custom.yaml",
        classes=[CLASS_CAR, CLASS_TRUCK],
        conf=conf,
        persist=True,
        stream=True,
        verbose=False,
    )
    records = extract_track_records(results)
    before = count_unique_ids(records, cls=CLASS_CAR)
    mapping = find_and_reconcile(records)
    corrected = apply_mapping(records, mapping)

    unique_ids = {r.track_id for r in corrected if r.cls == CLASS_CAR}
    frames_per_id = defaultdict(int)
    for r in corrected:
        if r.cls == CLASS_CAR:
            frames_per_id[r.track_id] += 1
    nb_frames = len({r.frame for r in corrected})

    out_dir = Path("gradio_runs") / f"{Path(video_path).stem}_notre_version"
    annotated_path = out_dir / f"{Path(video_path).stem}_corrige.mp4"
    render_video(Path(video_path), corrected, annotated_path, fps=25.0)

    extra = f"**IDs uniques avant réconciliation** : {before}  ·  **corrections appliquées** : {len(mapping)}\n\n"
    summary = _summarize(nb_frames, unique_ids, frames_per_id, OUR_VERSION_LABEL, extra=extra)
    return str(annotated_path), summary


@gpu_decorator
def run_tracking(video_path: str, method_label: str, conf: float):
    if video_path is None:
        return None, "⚠ Merci d'uploader une vidéo."

    if method_label == OUR_VERSION_LABEL:
        return _run_our_version(video_path, conf)

    tracker_config = STANDARD_TRACKERS.get(method_label, "bytetrack.yaml")
    return _run_standard(video_path, tracker_config, conf, method_label)


demo = gr.Interface(
    fn=run_tracking,
    inputs=[
        gr.Video(label="Vidéo de trafic à analyser"),
        gr.Dropdown(choices=ALL_METHODS, value=OUR_VERSION_LABEL, label="Méthode de suivi"),
        gr.Slider(minimum=0.05, maximum=0.5, value=0.1, step=0.05, label="Seuil de confiance de détection"),
    ],
    outputs=[
        gr.Video(label="Vidéo annotée (détections + ID)"),
        gr.Markdown(label="Résumé du suivi"),
    ],
    title="Suivi d'une voiture à travers une occlusion — Groupe 4",
    description=(
        "Projet Intégrateur 1 — AMA PIIA, Cohorte 2. Upload une vidéo de trafic : le système détecte "
        "les voitures et poids lourds, les suit avec la méthode choisie, et affiche le nombre "
        "d'identifiants uniques attribués (indicateur d'ID Switch)."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)