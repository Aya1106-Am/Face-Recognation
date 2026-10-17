import pickle
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

st.set_page_config(page_title="Reconnaissance faciale", layout="centered")
st.title("Application de reconnaissance faciale")


@st.cache_resource
def charger_base():
    chemin = Path(__file__).parent / "base_visages.pkl"
    if not chemin.exists():
        return None
    with open(chemin, "rb") as f:
        return pickle.load(f)


base = charger_base()
if base is None or not base["encodages"]:
    st.error("base_visages.pkl est introuvable ou vide. Crée-la d'abord depuis le notebook.")
    st.stop()

seuil = st.sidebar.slider("Seuil de reconnaissance (plus petit = plus strict)", 0.30, 0.80, 0.60, 0.01)


def identifier(encodage, seuil_):
    distances = face_recognition.face_distance(base["encodages"], encodage)
    idx = int(np.argmin(distances))
    nom = base["noms"][idx] if distances[idx] < seuil_ else "Inconnue"
    return nom, float(distances[idx])


def analyser_rgb(image_rgb, seuil_):
    positions = face_recognition.face_locations(image_rgb)
    encodages = face_recognition.face_encodings(image_rgb, positions)
    return [(pos, *identifier(enc, seuil_)) for pos, enc in zip(positions, encodages)]


def dessiner(img_bgr, resultats, echelle=1):
    for (top, right, bottom, left), nom, dist in resultats:
        top, right, bottom, left = [int(c * echelle) for c in (top, right, bottom, left)]
        couleur = (0, 200, 0) if nom != "Inconnue" else (0, 0, 255)
        cv2.rectangle(img_bgr, (left, top), (right, bottom), couleur, 2)
        cv2.putText(img_bgr, f"{nom} ({dist:.2f})", (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)
    return img_bgr


def reduire(img, max_cote=1000):
    h, w = img.shape[:2]
    f = max_cote / max(h, w)
    return cv2.resize(img, (int(w * f), int(h * f))) if f < 1 else img


def afficher_resultats(image_rgb):
    image_rgb = reduire(image_rgb)
    resultats = analyser_rgb(image_rgb, seuil)
    annotee = dessiner(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), resultats)
    st.image(cv2.cvtColor(annotee, cv2.COLOR_BGR2RGB), caption=f"{len(resultats)} visage(s) détecté(s)")
    if not resultats:
        st.info("Aucun visage détecté.")
    for i, (_, nom, dist) in enumerate(resultats, start=1):
        if nom != "Inconnue":
            st.success(f"Visage {i} identifié : {nom} (distance : {dist:.2f})")
        else:
            st.warning(f"Visage {i} : personne non reconnue (distance : {dist:.2f})")


if WEBRTC_OK:
    class Processeur(VideoProcessorBase):
        def __init__(self):
            self.seuil = 0.6
            self.compteur = 0
            self.resultats = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.compteur += 1
            if self.compteur % 3 == 0:          # une image sur 3 pour rester fluide
                petite = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                rgb = cv2.cvtColor(petite, cv2.COLOR_BGR2RGB)
                self.resultats = analyser_rgb(rgb, self.seuil)
            img = dessiner(img, self.resultats, echelle=4)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


onglet_live, onglet_photo, onglet_fichier = st.tabs(["Caméra en direct", "Photo caméra", "Fichier"])

with onglet_live:
    if not WEBRTC_OK:
        st.warning("Installe d'abord le module : uv pip install streamlit-webrtc")
    else:
        st.write("Clique sur **START**, puis autorise la caméra dans le navigateur.")
        ctx = webrtc_streamer(
            key="reco-live",
            video_processor_factory=Processeur,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.seuil = seuil

with onglet_photo:
    photo = st.camera_input("Prends une photo")
    if photo is not None:
        afficher_resultats(np.array(Image.open(photo).convert("RGB")))

with onglet_fichier:
    fichier = st.file_uploader("Uploader une photo", type=["jpg", "jpeg", "png"])
    if fichier is not None:
        afficher_resultats(np.array(ImageOps.exif_transpose(Image.open(fichier)).convert("RGB")))
