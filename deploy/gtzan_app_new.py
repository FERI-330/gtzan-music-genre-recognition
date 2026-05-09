import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import onnxruntime as ort
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tempfile, os, io, base64

# ── Config ────────────────────────────────────────────────────────────────────
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
SEGMENT_DURATION = 3.0    # each window length in seconds
SEGMENT_STEP     = 3.0    # hop between windows — same as duration = no overlap, no gap
N_MELS           = 128
HOP_LENGTH       = 512
IMG_SIZE         = 224
IMAGENET_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pixels per second in the strip (controls visual width)
PX_PER_SEC  = 40          # 3s segment → 120px wide
STRIP_H     = 180         # spectrogram height px
LABEL_H     = 28          # bottom label bar height px

MODEL_PATH = Path(__file__).parent.parent / "models" / "cnn_gtzan.onnx"

def stop_app() -> None:
    st.stop()
    raise SystemExit(0)

@st.cache_resource(show_spinner="Loading model...")
def load_session():
    if not MODEL_PATH.exists():
        return None
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not providers:
        providers = [available[0]]
    return ort.InferenceSession(str(MODEL_PATH), providers=providers)

def load_audio_to_tmp(audio_bytes: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name

def mel_segment(path: str, start: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, offset=start, duration=SEGMENT_DURATION, mono=True)
    if len(y) < SR * 0.5:
        raise ValueError("Too short")
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)

def mel_to_tensor(mel_db: np.ndarray) -> np.ndarray:
    norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img  = Image.fromarray((norm * 255).astype(np.uint8))
    img  = img.transpose(Image.FLIP_TOP_BOTTOM)
    img  = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGB")
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

def predict_file_full(path: str, sess: ort.InferenceSession):
    """
    Covers the ENTIRE song with non-overlapping 3s windows.
    Returns:
      - overall_probs: (10,) averaged over all segments
      - seg_data: list of (start_sec, mel_db, probs) — every 3s, no gaps
    """
    duration = librosa.get_duration(path=path)
    starts   = np.arange(0.0, duration - SEGMENT_DURATION * 0.5, SEGMENT_STEP)

    all_probs, seg_data = [], []
    progress = st.progress(0, text="Analysing segments...")
    n_total  = len(starts)

    for i, start in enumerate(starts):
        try:
            mel_db = mel_segment(path, float(start))
            tensor = mel_to_tensor(mel_db)
            logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
            p      = softmax(logits)
            all_probs.append(p)
            seg_data.append((float(start), mel_db, p))
        except Exception:
            pass
        progress.progress((i + 1) / n_total, text=f"Segment {i+1}/{n_total}…")

    progress.empty()

    if not all_probs:
        return None, []
    return np.mean(all_probs, axis=0), seg_data

# ── Build seamless horizontal PNG ─────────────────────────────────────────────
_MAGMA = cm.get_cmap("magma")

def mel_to_rgb_strip(mel_db: np.ndarray, w: int, h: int) -> Image.Image:
    norm         = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    norm_flipped = np.flipud(norm)
    small        = Image.fromarray((norm_flipped * 255).astype(np.uint8), mode="L")
    small        = small.resize((w, h), Image.LANCZOS)
    arr          = np.array(small, dtype=np.float32) / 255.0
    rgba         = (_MAGMA(arr) * 255).astype(np.uint8)
    return Image.fromarray(rgba[:, :, :3], mode="RGB")

def build_seamless_png(seg_data: list, top_genre: str, conf: float) -> bytes:
    """
    Each segment occupies exactly PX_PER_SEC * SEGMENT_DURATION pixels wide.
    No gaps, no separators → truly seamless spectrogram timeline.
    One genre label bar across the full bottom.
    Time ticks every 30 seconds.
    """
    seg_w    = int(PX_PER_SEC * SEGMENT_DURATION)  # px per segment
    n        = len(seg_data)
    total_w  = seg_w * n
    total_h  = STRIP_H + LABEL_H

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 24))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_tick = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        font_lbl  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_tick = ImageFont.load_default()
        font_lbl  = font_tick

    # Paste strips seamlessly
    for i, (start, mel_db, _) in enumerate(seg_data):
        strip = mel_to_rgb_strip(mel_db, seg_w, STRIP_H)
        canvas.paste(strip, (i * seg_w, 0))

    # Time tick marks every 30s (white vertical line + label)
    tick_interval = 30  # seconds
    for t in range(0, int(n * SEGMENT_DURATION), tick_interval):
        x = int(t * PX_PER_SEC)
        if x >= total_w:
            break
        # semi-transparent white tick line
        draw.line([(x, STRIP_H - 18), (x, STRIP_H - 1)], fill=(255, 255, 255), width=1)
        draw.text((x + 2, STRIP_H - 16), f"{t}s", fill=(220, 220, 220), font=font_tick)

    # Single genre label bar — full width
    if conf >= 0.60:
        bar_col = (28, 120, 60)
    elif conf >= 0.40:
        bar_col = (160, 120, 10)
    else:
        bar_col = (150, 40, 40)

    draw.rectangle([0, STRIP_H, total_w - 1, total_h - 1], fill=bar_col)
    duration_str = f"{int(n * SEGMENT_DURATION)}s"
    label = (
        f"  {top_genre.upper()}   {conf*100:.1f}%"
        f"  ·  {n} segments  ·  {duration_str} total  ·  {SEGMENT_DURATION:.0f}s / segment"
    )
    draw.text((8, STRIP_H + 7), label, fill=(240, 240, 240), font=font_lbl)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GTZAN Music Genre", page_icon="🎵", layout="wide")

st.title("🎵 GTZAN Music Genre Recognizer")
st.markdown(
    "**VGG-style CNN · Mel-spectrogram · ONNX Runtime** "
    "| 10 genres | 85.66% validation accuracy"
)
st.divider()

sess = load_session()
if sess is None:
    st.error(f"❌ ONNX model not found: `{MODEL_PATH}`")
    stop_app()

provider = sess.get_providers()[0].replace("ExecutionProvider", "")
st.caption(f"✅ Model loaded · Device: **{provider}**")

uploaded = st.file_uploader(
    "Upload a music file (.wav, .mp3, .ogg, .flac)",
    type=["wav", "mp3", "ogg", "flac"],
)
if not uploaded:
    st.info("⬆️ Upload an audio file to start.")
    stop_app()

audio_bytes = uploaded.read()
suffix      = "." + uploaded.name.rsplit(".", 1)[-1]
tmp_path    = load_audio_to_tmp(audio_bytes, suffix)

try:
    st.audio(audio_bytes)

    probs, seg_data = predict_file_full(tmp_path, sess)

    if probs is None:
        st.error("❌ Could not process the file.")
        stop_app()

    top_idx   = np.argsort(probs)[::-1]
    top_genre = GENRES[top_idx[0]]
    top_conf  = float(probs[top_idx[0]])

    # Big genre card
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#1a1a2e,#16213e);
            border:1px solid #333;border-radius:12px;
            padding:1.2rem 2rem;margin:0.5rem 0 1rem 0;text-align:center;
        ">
            <div style="font-size:3rem;line-height:1.1;">{GENRE_EMOJI[top_genre]}</div>
            <div style="font-size:1.8rem;font-weight:700;color:#f0f0f0;letter-spacing:3px;">
                {top_genre.upper()}
            </div>
            <div style="font-size:1rem;color:#aaa;margin-top:0.3rem;">
                {top_conf*100:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Seamless scrollable spectrogram
    with st.spinner("Rendering spectrogram..."):
        png_bytes = build_seamless_png(seg_data, top_genre, top_conf)
    png_b64 = base64.b64encode(png_bytes).decode()

    st.markdown(
        f'''<div style="
            overflow-x:auto;overflow-y:hidden;white-space:nowrap;
            background:#121218;border:1px solid #2a2a3a;
            border-radius:8px;padding:6px;
        ">
            <img src="data:image/png;base64,{png_b64}"
                 style="height:{STRIP_H + LABEL_H}px;width:auto;
                        display:inline-block;vertical-align:top;" />
        </div>''',
        unsafe_allow_html=True,
    )

    # Top-3 + distribution
    st.divider()
    col_res, col_dist = st.columns([1, 1], gap="large")

    with col_res:
        st.subheader("🏆 Top-3 prediction")
        for rank, idx in enumerate(top_idx[:3]):
            g, p = GENRES[idx], probs[idx]
            st.markdown(f"{'🥇🥈🥉'[rank]} **{GENRE_EMOJI[g]} {g.capitalize()}**")
            st.progress(float(p), text=f"{p*100:.1f}%")

    with col_dist:
        st.subheader("📊 Full distribution")
        for idx in top_idx:
            g, p = GENRES[idx], probs[idx]
            c1, c2 = st.columns([4, 1])
            with c1:
                st.progress(float(p), text=f"{GENRE_EMOJI[g]} {g}")
            with c2:
                st.markdown(f"`{p*100:.1f}%`")

finally:
    os.unlink(tmp_path)

st.divider()
st.caption("GTZAN Music Genre Recognition · VGG-style CNN · PyTorch → ONNX Runtime")
