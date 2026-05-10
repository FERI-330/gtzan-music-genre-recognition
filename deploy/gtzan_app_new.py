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

# Config 
GENRES = [
 "blues", "classical", "country", "disco", "hiphop",
 "jazz", "metal", "pop", "reggae", "rock",
]
GENRE_EMOJI = {
 "blues": "", "classical": "", "country": "", "disco": "",
 "hiphop": "", "jazz": "", "metal": "", "pop": "",
 "reggae": "", "rock": "",
}

SR = 22050
SEGMENT_DURATION = 3.0
SEGMENT_STEP = 3.0
N_MELS = 128
HOP_LENGTH = 512
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Classification uses only N_VOTE evenly-spaced segments → fast
N_VOTE = 7

PX_PER_SEC = 40
STRIP_H = 180
LABEL_H = 28

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
 img = Image.fromarray((norm * 255).astype(np.uint8))
 img = img.transpose(Image.FLIP_TOP_BOTTOM)
 img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS).convert("RGB")
 arr = np.array(img, dtype=np.float32) / 255.0
 arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
 return arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def softmax(x: np.ndarray) -> np.ndarray:
 e = np.exp(x - x.max())
 return e / e.sum()

# PHASE 1: Quick classification (N_VOTE segments) 
def classify_quick(path: str, sess: ort.InferenceSession):
 """Sample N_VOTE evenly-spaced segments for fast majority vote."""
 duration = librosa.get_duration(path=path)
 max_start = max(0.0, duration - SEGMENT_DURATION)
 starts = np.linspace(0.0, max_start, min(N_VOTE, max(1, int(max_start / 1.5) + 1)))

 all_probs = []
 for start in starts:
 try:
 mel_db = mel_segment(path, float(start))
 tensor = mel_to_tensor(mel_db)
 logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
 all_probs.append(softmax(logits))
 except Exception:
 pass

 if not all_probs:
 return None
 return np.mean(all_probs, axis=0)

# PHASE 2: Full spectrogram (every segment, no CNN) 
def build_full_seg_data(path: str) -> list:
 """Load every SEGMENT_STEP segment for visualisation — no CNN inference."""
 duration = librosa.get_duration(path=path)
 starts = np.arange(0.0, duration - SEGMENT_DURATION * 0.5, SEGMENT_STEP)
 seg_data = []
 prog = st.progress(0, text="Generating spectrogram…")
 n = len(starts)
 for i, start in enumerate(starts):
 try:
 mel_db = mel_segment(path, float(start))
 seg_data.append((float(start), mel_db))
 except Exception:
 pass
 prog.progress((i + 1) / n, text=f"Spectrogram: segment {i+1}/{n}…")
 prog.empty()
 return seg_data

# Build seamless PNG 
_MAGMA = cm.get_cmap("magma")

def mel_to_rgb_strip(mel_db: np.ndarray, w: int, h: int) -> Image.Image:
 norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
 norm_flipped = np.flipud(norm)
 small = Image.fromarray((norm_flipped * 255).astype(np.uint8), mode="L")
 small = small.resize((w, h), Image.LANCZOS)
 arr = np.array(small, dtype=np.float32) / 255.0
 rgba = (_MAGMA(arr) * 255).astype(np.uint8)
 return Image.fromarray(rgba[:, :, :3], mode="RGB")

def build_seamless_png(seg_data: list, top_genre: str, conf: float) -> bytes:
 seg_w = int(PX_PER_SEC * SEGMENT_DURATION)
 n = len(seg_data)
 total_w = seg_w * n
 total_h = STRIP_H + LABEL_H

 canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 24))
 draw = ImageDraw.Draw(canvas)

 try:
 font_tick = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
 font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
 except Exception:
 font_tick = ImageFont.load_default()
 font_lbl = font_tick

 for i, (start, mel_db) in enumerate(seg_data):
 strip = mel_to_rgb_strip(mel_db, seg_w, STRIP_H)
 canvas.paste(strip, (i * seg_w, 0))

 for t in range(0, int(n * SEGMENT_DURATION), 30):
 x = int(t * PX_PER_SEC)
 if x >= total_w:
 break
 draw.line([(x, STRIP_H - 18), (x, STRIP_H - 1)], fill=(255, 255, 255), width=1)
 draw.text((x + 2, STRIP_H - 16), f"{t}s", fill=(220, 220, 220), font=font_tick)

 bar_col = (28, 120, 60) if conf >= 0.60 else (160, 120, 10) if conf >= 0.40 else (150, 40, 40)
 draw.rectangle([0, STRIP_H, total_w - 1, total_h - 1], fill=bar_col)
 label = (
 f" {top_genre.upper()} {conf*100:.1f}%"
 f" · {n} segments · {int(n * SEGMENT_DURATION)}s total · {SEGMENT_DURATION:.0f}s / segment"
 )
 draw.text((8, STRIP_H + 7), label, fill=(240, 240, 240), font=font_lbl)

 buf = io.BytesIO()
 canvas.save(buf, format="PNG", optimize=True)
 buf.seek(0)
 return buf.getvalue()

# Synced spectrogram HTML component 
def spectrogram_player_html(png_b64: str, strip_h: int, label_h: int,
 px_per_sec: int, audio_duration: float) -> str:
 total_h = strip_h + label_h
 return f"""
<div style="font-family:sans-serif;">

 <!-- Auto-scroll toggle -->
 <div style="margin-bottom:6px; display:flex; align-items:center; gap:10px;">
 <label style="display:flex; align-items:center; gap:8px;
 font-size:0.82rem; color:#ccc; cursor:pointer; user-select:none;">
 <div id="toggle-wrap" onclick="toggleAutoScroll()" style="
 width:38px; height:20px; border-radius:10px;
 background:#2a7a4f; cursor:pointer; position:relative;
 transition:background 0.2s;">
 <div id="toggle-knob" style="
 position:absolute; top:2px; left:18px;
 width:16px; height:16px; border-radius:50%;
 background:#fff; transition:left 0.2s;
 box-shadow:0 1px 3px rgba(0,0,0,0.4);"></div>
 </div>
 Auto-scroll
 </label>
 <span id="autoscroll-status" style="font-size:0.75rem; color:#888;">ON</span>
 </div>

 <!-- Scrollable spectrogram -->
 <div id="spec-scroll" style="
 overflow-x: auto; overflow-y: hidden;
 white-space: nowrap; background: #121218;
 border: 1px solid #2a2a3a; border-radius: 8px;
 padding: 6px; position: relative;">
 <div id="spec-wrap" style="position:relative; display:inline-block; vertical-align:top;">
 <img id="spec-img"
 src="data:image/png;base64,{png_b64}"
 style="height:{total_h}px; width:auto; display:block;" />
 <div id="playhead" style="
 position:absolute; top:0; left:0;
 width:3px; height:{strip_h}px;
 background:rgba(255,255,255,0.92);
 box-shadow:0 0 8px rgba(255,255,255,0.6), 0 0 2px rgba(255,255,255,1);
 pointer-events:none; display:none;"></div>
 </div>
 </div>
</div>

<script>
(function() {{
 const PX_PER_SEC = {px_per_sec};
 const scroll = document.getElementById('spec-scroll');
 const playhead = document.getElementById('playhead');
 const wrap = document.getElementById('spec-wrap');
 const toggleWrap = document.getElementById('toggle-wrap');
 const toggleKnob = document.getElementById('toggle-knob');
 const statusLabel = document.getElementById('autoscroll-status');

 let audio = null;
 let autoScroll = true;
 let userScroll = false;
 let scrollTimer = null;
 let caughtUp = false; // one-time catch-up after spectrogram loads

 window.toggleAutoScroll = function() {{
 autoScroll = !autoScroll;
 toggleWrap.style.background = autoScroll ? '#2a7a4f' : '#555';
 toggleKnob.style.left = autoScroll ? '18px' : '2px';
 statusLabel.textContent = autoScroll ? 'ON' : 'OFF';
 }};

 scroll.addEventListener('scroll', () => {{
 userScroll = true;
 clearTimeout(scrollTimer);
 scrollTimer = setTimeout(() => {{ userScroll = false; }}, 1500);
 }});

 function findAudio() {{
 try {{
 const frames = window.parent.document.querySelectorAll('audio');
 if (frames.length > 0) return frames[frames.length - 1];
 }} catch(e) {{}}
 return document.querySelector('audio');
 }}

 function update() {{
 if (!audio) {{
 audio = findAudio();
 if (!audio) return;
 }}
 const t = audio.currentTime;
 const x = t * PX_PER_SEC;

 playhead.style.left = x + 'px';
 playhead.style.display = 'block';

 // One-time catch-up: jump view to current playback position when spec loads
 if (!caughtUp) {{
 caughtUp = true;
 const visW = scroll.clientWidth;
 scroll.scrollLeft = Math.max(0, x - visW / 2);
 }}

 if (autoScroll && !userScroll && !audio.paused) {{
 const visW = scroll.clientWidth;
 scroll.scrollLeft = Math.max(0, x - visW / 2);
 }}
 }}

 setInterval(update, 100);

 // Click to seek
 wrap.style.cursor = 'pointer';
 wrap.addEventListener('click', (e) => {{
 if (!audio) return;
 const rect = wrap.getBoundingClientRect();
 const clickX = e.clientX - rect.left + scroll.scrollLeft;
 audio.currentTime = clickX / PX_PER_SEC;
 }});
}})();
</script>
"""

# UI 
st.set_page_config(page_title="GTZAN Music Genre", page_icon="", layout="wide")

st.title(" GTZAN Music Genre Recognizer")
st.markdown(
 "**VGG-style CNN · Mel-spectrogram · ONNX Runtime** "
 "| 10 genres | 85.66% validation accuracy"
)
st.divider()

sess = load_session()
if sess is None:
 st.error(f" ONNX model not found: `{MODEL_PATH}`")
 stop_app()

provider = sess.get_providers()[0].replace("ExecutionProvider", "")
st.caption(f" Model loaded · Device: **{provider}**")

uploaded = st.file_uploader(
 "Upload a music file (.wav, .mp3, .ogg, .flac)",
 type=["wav", "mp3", "ogg", "flac"],
)
if not uploaded:
 st.info(" Upload an audio file to start.")
 stop_app()

audio_bytes = uploaded.read()
suffix = "." + uploaded.name.rsplit(".", 1)[-1]
tmp_path = load_audio_to_tmp(audio_bytes, suffix)

try:
 # PHASE 1: Fast classification 
 with st.spinner(f" Classifying… (sampling {N_VOTE} segments)"):
 probs = classify_quick(tmp_path, sess)

 if probs is None:
 st.error(" Could not process the file.")
 stop_app()

 top_idx = np.argsort(probs)[::-1]
 top_genre = GENRES[top_idx[0]]
 top_conf = float(probs[top_idx[0]])

 # Genre result card
 st.markdown(
 f"""
 <div style="
 background:linear-gradient(135deg,#1a1a2e,#16213e);
 border:1px solid #333; border-radius:12px;
 padding:1.2rem 2rem; margin:0.5rem 0 1rem 0; text-align:center;">
 <div style="font-size:3rem;line-height:1.1;">{GENRE_EMOJI[top_genre]}</div>
 <div style="font-size:1.8rem;font-weight:700;color:#f0f0f0;letter-spacing:3px;">
 {top_genre.upper()}
 </div>
 <div style="font-size:1rem;color:#aaa;margin-top:0.3rem;">
 {top_conf*100:.1f}% confidence · {N_VOTE} sample segments
 </div>
 </div>
 """,
 unsafe_allow_html=True,
 )

 # Spectrogram placeholder — will be filled by Phase 2
 spec_placeholder = st.empty()
 audio_placeholder = st.empty()

 st.divider()

 # Top-3 + distribution — rendered immediately, before Phase 2 starts
 col_res, col_dist = st.columns([1, 1], gap="large")
 with col_res:
 st.subheader(" Top-3 prediction")
 for rank, idx in enumerate(top_idx[:3]):
 g, p = GENRES[idx], probs[idx]
 st.markdown(f"{''[rank]} **{GENRE_EMOJI[g]} {g.capitalize()}**")
 st.progress(float(p), text=f"{p*100:.1f}%")

 with col_dist:
 st.subheader(" Full distribution")
 for idx in top_idx:
 g, p = GENRES[idx], probs[idx]
 c1, c2 = st.columns([4, 1])
 with c1:
 st.progress(float(p), text=f"{GENRE_EMOJI[g]} {g}")
 with c2:
 st.markdown(f"`{p*100:.1f}%`")

 # PHASE 2: Full spectrogram (all segments, mel only, no CNN) 
 with spec_placeholder.container():
 st.caption(" Generating full spectrogram strip — will catch up to current playback position when ready…")
 seg_data = build_full_seg_data(tmp_path)
 audio_duration = librosa.get_duration(path=tmp_path)

 png_bytes = build_seamless_png(seg_data, top_genre, top_conf)
 png_b64 = base64.b64encode(png_bytes).decode()

 st.caption(
 f" {len(seg_data)} segments · {SEGMENT_DURATION:.0f}s each · "
 "Playhead follows audio · Click strip to seek"
 )

 import streamlit.components.v1 as components
 components.html(
 spectrogram_player_html(png_b64, STRIP_H, LABEL_H, PX_PER_SEC, audio_duration),
 height=STRIP_H + LABEL_H + 55,
 scrolling=False,
 )

 with audio_placeholder.container():
 st.audio(audio_bytes)

finally:
 os.unlink(tmp_path)

st.divider()
st.caption("GTZAN Music Genre Recognition · VGG-style CNN · PyTorch → ONNX Runtime")
