"""
Application de démonstration — version Streamlit Community Cloud
====================================================================
Interface produit (thème sombre moderne), même logique métier que app.py.

Groupe 4 — Projet Intégrateur 1, AMA PIIA, Cohorte 2

Lancement local :
    streamlit run streamlit_app.py
"""

import tempfile
import time
from pathlib import Path

import streamlit as st

from app import ALL_METHODS, OUR_VERSION_LABEL, run_tracking

# ---------------------------------------------------------------------------
# Configuration de page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Tracking · Occlusion Recovery",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Thème — dashboard sombre moderne, accent violet/cyan
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0B0D12;
        --surface: #12151D;
        --surface-2: #191D28;
        --border: rgba(255,255,255,0.08);
        --border-strong: rgba(255,255,255,0.14);
        --accent: #7C5CFF;
        --accent-2: #22D3EE;
        --text: #E7E9EE;
        --muted: #8B92A5;
        --success: #34D399;
        --warning: #FBBF24;
        --danger: #F87171;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    code, .stCaption, .stCode { font-family: 'JetBrains Mono', monospace; }

    #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}

    .stApp { background: var(--bg); }

    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] label { color: var(--muted) !important; font-weight: 500; font-size: 0.82rem; }
    section[data-testid="stSidebar"] h3 { color: var(--text) !important; font-weight: 700; }

    /* En-tête produit */
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 16px;
        background:
            radial-gradient(circle at 15% 20%, rgba(124,92,255,0.18), transparent 45%),
            radial-gradient(circle at 85% 0%, rgba(34,211,238,0.14), transparent 40%),
            var(--surface);
        border: 1px solid var(--border);
        margin-bottom: 1.6rem;
    }
    .hero .badge {
        display: inline-block;
        color: var(--accent-2);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        background: rgba(34,211,238,0.1);
        border: 1px solid rgba(34,211,238,0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        margin-bottom: 0.7rem;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #C9CEDD 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { color: var(--muted); font-size: 0.95rem; margin: 0; max-width: 62ch; line-height: 1.5; }

    /* Cartes */
    .card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1.1rem;
    }
    .card-title {
        display: flex; align-items: center; gap: 0.5rem;
        color: var(--text);
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .card-title .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--accent-2);
        box-shadow: 0 0 8px var(--accent-2);
    }
    .empty-state { color: var(--muted); font-size: 0.88rem; padding: 1.5rem 0; text-align: center; }

    [data-testid="stMetric"] {
        background-color: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; }

    .stButton > button {
        background: linear-gradient(90deg, var(--accent) 0%, #6947F5 100%);
        color: #FFFFFF;
        font-weight: 700;
        border-radius: 9px;
        border: none;
        padding: 0.65rem 1.2rem;
        width: 100%;
        box-shadow: 0 4px 14px rgba(124,92,255,0.35);
    }
    .stButton > button:hover { filter: brightness(1.08); color: #FFFFFF; }

    .stDownloadButton > button {
        background-color: var(--surface-2);
        color: var(--text) !important;
        font-weight: 600;
        border-radius: 9px;
        border: 1px solid var(--border-strong);
        width: 100%;
    }
    .stDownloadButton > button:hover { border-color: var(--accent-2); color: var(--accent-2) !important; }

    h1, h2, h3, .stMarkdown p, .stMarkdown li { color: var(--text); }
    .stAlert { border-radius: 10px; }

    .status-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.78rem; font-weight: 600;
        padding: 0.25rem 0.7rem; border-radius: 999px;
        background: rgba(52,211,153,0.1); color: var(--success);
        border: 1px solid rgba(52,211,153,0.3);
    }

    .footer-note {
        color: var(--muted); font-size: 0.78rem; text-align: center;
        margin-top: 2rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge">Multi-Object Tracking · Occlusion Recovery</div>
        <h1>Suivi de véhicules à travers une occlusion</h1>
        <p>Détecte et suit les véhicules d'une vidéo de trafic, puis corrige les changements
        d'identifiant provoqués par l'occlusion d'un poids lourd. Comparez trois méthodes :
        ByteTrack seul, BoT-SORT + ReID générique, et notre approche
        (couleur · trajectoire · détection d'occlusion réelle).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — contrôles
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Configuration")
    uploaded_file = st.file_uploader("Vidéo de trafic", type=["mp4", "avi", "mov"])
    method_label = st.selectbox("Méthode de suivi", ALL_METHODS, index=ALL_METHODS.index(OUR_VERSION_LABEL))
    conf = st.slider("Seuil de confiance", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    run_button = st.button("Lancer l'analyse", type="primary")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem; color:#8B92A5; line-height:1.6;'>"
        "<b style='color:#E7E9EE;'>Groupe 4</b><br>Projet Intégrateur 1<br>AMA · PIIA Cohorte 2"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Corps principal
# ---------------------------------------------------------------------------
col_video, col_results = st.columns([1, 1], gap="large")

with col_video:
    st.markdown('<div class="card"><div class="card-title"><span class="dot"></span>Vidéo source</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        video_bytes = uploaded_file.getvalue()
        st.video(video_bytes)
        st.download_button(
            "⭳ Télécharger la vidéo source",
            data=video_bytes,
            file_name=uploaded_file.name,
            mime="video/mp4",
            use_container_width=True,
        )
    else:
        st.markdown('<div class="empty-state">Upload une vidéo dans la barre latérale pour commencer.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    if run_button:
        if uploaded_file is None:
            st.warning("Merci d'uploader une vidéo avant de lancer l'analyse.")
        else:
            t0 = time.time()
            with st.spinner("Détection, suivi et réconciliation en cours..."):
                suffix = Path(uploaded_file.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                annotated_video, summary = run_tracking(tmp_path, method_label, conf)
            elapsed = time.time() - t0

            st.markdown('<div class="card"><div class="card-title"><span class="dot"></span>Vidéo annotée</div>', unsafe_allow_html=True)
            if annotated_video is not None:
                annotated_bytes = Path(annotated_video).read_bytes()
                st.video(annotated_bytes)
                st.download_button(
                    "⭳ Télécharger la vidéo annotée",
                    data=annotated_bytes,
                    file_name=f"annotee_{Path(uploaded_file.name).stem}.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
            else:
                st.error("Aucune vidéo annotée générée.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title"><span class="dot"></span>Résumé du suivi</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">● Traitement terminé en {elapsed:.1f}s</span>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown(summary)
            st.caption(f"Méthode : {method_label}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card"><div class="card-title"><span class="dot"></span>Résultats</div>'
            '<div class="empty-state">Les résultats (vidéo annotée + statistiques de suivi) '
            "s'afficheront ici une fois l'analyse lancée.</div></div>",
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='footer-note'>Projet Intégrateur 1 — Vidéoprotection intelligente & suivi multi-objets · "
    "Groupe 4 · AMA PIIA Cohorte 2</div>",
    unsafe_allow_html=True,
)