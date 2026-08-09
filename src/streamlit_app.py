import tempfile
from pathlib import Path

import streamlit as st

from app import ALL_METHODS, OUR_VERSION_LABEL, run_tracking

st.set_page_config(page_title="Suivi d'une voiture à travers une occlusion — Groupe 4", layout="wide")

st.title("Suivi d'une voiture à travers une occlusion — Groupe 4")
st.markdown(
    "Projet Intégrateur 1 — AMA PIIA, Cohorte 2. Upload une vidéo de trafic : le système détecte "
    "les voitures et poids lourds, les suit avec la méthode choisie, et affiche le nombre "
    "d'identifiants uniques attribués (indicateur d'ID Switch)."
)

col_left, col_right = st.columns(2)

with col_left:
    uploaded_file = st.file_uploader("Vidéo de trafic à analyser", type=["mp4", "avi", "mov"])
    method_label = st.selectbox("Méthode de suivi", ALL_METHODS, index=ALL_METHODS.index(OUR_VERSION_LABEL))
    conf = st.slider("Seuil de confiance de détection", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    run_button = st.button("Lancer l'analyse", type="primary")

    if uploaded_file is not None:
        st.video(uploaded_file)

with col_right:
    if run_button:
        if uploaded_file is None:
            st.warning("Merci d'uploader une vidéo.")
        else:
            with st.spinner("Traitement en cours (peut prendre 1 à quelques minutes)..."):
                # Streamlit fournit un buffer en mémoire : on l'écrit dans un fichier
                # temporaire, car notre pipeline (Ultralytics, OpenCV) attend un chemin.
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                annotated_video, summary = run_tracking(tmp_path, method_label, conf)

            if annotated_video is not None:
                st.video(annotated_video)
            else:
                st.error("Aucune vidéo annotée générée.")
            st.markdown(summary)
    else:
        st.info("Upload une vidéo et clique sur *Lancer l'analyse*.")