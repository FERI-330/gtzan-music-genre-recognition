import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import tempfile, os

# ── Konfiguráció ──────────────────────────────────────────────────────────────
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠", "disco": "🕺",
    "hiphop": "🎤", "jazz": "🎷", "metal": "🤘", "pop": "🎶",
    "reggae": "🌴", "rock": "⚡",
}

SR               = 22050
SEGMENT_DURATION = 3.0
N_MELS           = 128
HOP_LENGTH       = 512
IMG_SIZE         = 224
IMAGENET_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_VOTE_SEGMENTS  = 7        # hány szegmensre átlagolunk

MODEL_PATH = Path(__file__).parent.parent / "models" / "cnn_gtzan.onnx"


def stop_app() -> None:
    st.stop()
    raise SystemExit(0)


# ── Modell betöltés (egyszer, cache-elve) ─────────────────────────────────────
@st.cache_resource(show_spinner="Modell betöltése...")
def load_session():
    if not MODEL_PATH.exists():
        return None
    available = ort.get_available_providers()
    providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
    if not providers:
        providers = [available[0]]
    return ort.InferenceSession(str(MODEL_PATH), providers=providers)


# ── Audio előfeldolgozás ──────────────────────────────────────────────────────
def load_audio_to_tmp(audio_bytes: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def mel_segment(path: str, start: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, offset=start, duration=SEGMENT_DURATION, mono=True)
    if len(y) < SR * 0.5:
        raise ValueError("Túl rövid szegmens")
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


def mel_to_tensor(mel_db: np.ndarray) -> np.ndarray:
    norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img  = Image.fromarray((norm * 255).astype(np.uint8))
    img  = img.transpose(Image.FLIP_TOP_BOTTOM)
    img  = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGB")
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)   # (1,3,224,224)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_file(path: str, sess: ort.InferenceSession):
    duration = librosa.get_duration(path=path)
    max_start = max(0.0, duration - SEGMENT_DURATION)
    starts = np.linspace(0, max_start, min(N_VOTE_SEGMENTS, max(1, int(max_start / 1.5) + 1)))

    all_probs, first_mel = [], None
    for start in starts:
        try:
            mel_db = mel_segment(path, float(start))
            tensor = mel_to_tensor(mel_db)
            logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
            all_probs.append(softmax(logits))
            if first_mel is None:
                first_mel = mel_db
        except Exception:
            continue

    if not all_probs:
        return None, None
    return np.mean(all_probs, axis=0), first_mel


# ── Spektrogram ábra ──────────────────────────────────────────────────────────
def plot_mel(mel_db: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3))
    img = librosa.display.specshow(
        mel_db, sr=SR, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-spektrogram (első szegmens)", fontsize=11)
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GTZAN Műfajfelismerő",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 GTZAN Zenei Műfajfelismerő")
st.markdown(
    "**VGG-stílusú CNN · Mel-spektrogram · ONNX Runtime**  "
    "| 10 műfaj | 85.66% validációs pontosság"
)
st.divider()

# Modell betöltés
sess = load_session()
if sess is None:
    st.error(
        f"❌ Az ONNX modell nem található: `{MODEL_PATH}`\n\n"
        "Futtasd le a `03_CNN_Model_v2.ipynb` ONNX export celláját!"
    )
    stop_app()

provider = sess.get_providers()[0].replace("ExecutionProvider", "")
st.caption(f"✅ Modell betöltve · Eszköz: **{provider}**")

# Fájl feltöltés
uploaded = st.file_uploader(
    "Tölts fel egy zenei fájlt (.wav, .mp3, .ogg, .flac)",
    type=["wav", "mp3", "ogg", "flac"],
    label_visibility="visible",
)

if not uploaded:
    st.info("⬆️ Tölts fel egy hangfájlt a műfaj előrejelzéséhez!")
    stop_app()

# Audio bytes + tmp fájl
audio_bytes = uploaded.read()
suffix = "." + uploaded.name.rsplit(".", 1)[-1]
tmp_path = load_audio_to_tmp(audio_bytes, suffix)

try:
    # Lejátszó
    st.audio(audio_bytes)

    # Inference
    with st.spinner(f"Elemzés ({N_VOTE_SEGMENTS} szegmens átlaga)..."):
        probs, mel_db = predict_file(tmp_path, sess)

    if probs is None:
        st.error("❌ Nem sikerült feldolgozni a fájlt. Próbálj egy másik .wav-ot!")
        stop_app()

    top_idx   = np.argsort(probs)[::-1]
    top_genre = GENRES[top_idx[0]]

    st.success(
        f"{GENRE_EMOJI[top_genre]} Felismert műfaj: **{top_genre.upper()}** "
        f"({probs[top_idx[0]]*100:.1f}%)"
    )

    # ── Két oszlop: eredmények + spektrogram ─────────────────────────────────
    col_res, col_spec = st.columns([1, 1], gap="large")

    with col_res:
        st.subheader("🏆 Top-3 előrejelzés")
        medals = ["🥇", "🥈", "🥉"]
        for rank, idx in enumerate(top_idx[:3]):
            g, p = GENRES[idx], probs[idx]
            st.markdown(f"{medals[rank]} **{GENRE_EMOJI[g]} {g.capitalize()}**")
            st.progress(float(p), text=f"{p*100:.1f}%")

        st.subheader("📊 Teljes eloszlás")
        for idx in top_idx:
            g, p = GENRES[idx], probs[idx]
            c1, c2 = st.columns([4, 1])
            with c1:
                st.progress(float(p), text=f"{GENRE_EMOJI[g]} {g}")
            with c2:
                st.markdown(f"`{p*100:.1f}%`")

    with col_spec:
        st.subheader("🖼️ Mel-spektrogram")
        if mel_db is not None:
            fig = plot_mel(mel_db)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

finally:
    os.unlink(tmp_path)

st.divider()
st.caption(
    "Projekt: GTZAN Music Genre Recognition · "
    "GTZANCNNModel (VGG-stílusú, 4×Conv blokk) · "
    "PyTorch → ONNX Runtime"
)
