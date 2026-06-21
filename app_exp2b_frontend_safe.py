from pathlib import Path
import base64, io
import numpy as np
import streamlit as st
from PIL import Image

import torch, torchvision as tv, joblib

# ---------- Paths ----------
VISION_CKPT   = Path("results/vision/best.pth")
SENSORS_MODEL = Path("results/exp2_sensors/model_rf.joblib")
BG_IMAGE_PATH = Path("/home/debby/Documents/pngs/livestock.jpg")  # your background

BRAND_GREEN = "#166534"
TEXT_DARK   = "#0B1220"
TEXT_MUTED  = "#334155"

# ---------- Background (no z-index issues) ----------
def set_readable_bg(image_path: Path, overlay=0.55):
    b64 = ""
    if image_path.exists():
        b64 = base64.b64encode(image_path.read_bytes()).decode()
    # use layered background-image: gradient on top of image → no overlay element
    css = f"""
    <style>
    .stApp {{
        background-image:
          linear-gradient(rgba(0,0,0,{overlay}), rgba(0,0,0,{overlay})),
          url("data:image/jpg;base64,{b64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,.18);
        padding: 1.6rem 1.8rem;
    }}
    h1.title-green {{
        color: {BRAND_GREEN}; text-transform: uppercase; font-weight: 900; letter-spacing: .03em;
    }}
    h2,h3,h4,h5 {{ color: {TEXT_DARK}; }}
    .muted {{ color:{TEXT_MUTED}; font-size:.95rem; }}
    .metric-row {{ display:flex; flex-wrap:wrap; gap:12px; }}
    .metric-card {{
        flex:1 1 200px; background:white; border:1px solid rgba(0,0,0,.08);
        border-radius:12px; padding:12px 14px; box-shadow:0 4px 14px rgba(0,0,0,.10);
    }}
    .metric-title {{ color:{TEXT_MUTED}; font-size:.9rem; }}
    .metric-value {{ color:{TEXT_DARK}; font-size:1.5rem; font-weight:800; }}
    .ok  {{ color:{BRAND_GREEN}; font-weight:900; }}
    .bad {{ color:#B91C1C; font-weight:900; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def metric_card(title, value):
    st.markdown(
        f"""<div class="metric-card">
               <div class="metric-title">{title}</div>
               <div class="metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

# ---------- Cached models ----------
@st.cache_resource
def load_vision():
    if not VISION_CKPT.exists():
        raise FileNotFoundError(f"Missing vision checkpoint: {VISION_CKPT}")
    m = tv.models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 2)
    m.load_state_dict(torch.load(VISION_CKPT, map_location="cpu"))
    m.eval()
    tfm = tv.transforms.Compose([
        tv.transforms.Resize((224,224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return m, tfm

@st.cache_resource
def load_sensor():
    if not SENSORS_MODEL.exists():
        raise FileNotFoundError(f"Missing sensor model: {SENSORS_MODEL}")
    return joblib.load(SENSORS_MODEL)

@torch.no_grad()
def p_vis(m, tfm, pil: Image.Image) -> float:
    x = tfm(pil.convert("RGB")).unsqueeze(0)
    return float(torch.softmax(m(x), dim=1)[0,1].item())

def p_sens(pipe, v: dict) -> float:
    import numpy as np
    x = np.array([[
        v["body_temp_c"], v["heart_rate_bpm"], v["resp_rate_bpm"], v["activity_idx"],
        v["cough_count"], v["rumination_min"], v["ambient_temp_c"], v["ambient_humidity_pct"]
    ]], dtype=float)
    return float(pipe.predict_proba(x)[:,1][0])

def fuse(alpha, ps, pv): return alpha*ps + (1-alpha)*pv
def decide(p, tau): return "UNHEALTHY" if p >= tau else "HEALTHY"

PRESETS = {
  "Cattle": dict(bt=38.3, hr=70, rr=22, act=0.6, cough=0, rum=40.0, ambt=27.0, ambh=65.0),
  "Dairy Cow": dict(bt=38.5, hr=75, rr=24, act=0.6, cough=0, rum=42.0, ambt=27.0, ambh=65.0),
  "Beef Cattle": dict(bt=38.2, hr=68, rr=20, act=0.6, cough=0, rum=38.0, ambt=27.0, ambh=65.0),
  "Goat": dict(bt=39.4, hr=88, rr=24, act=0.6, cough=0, rum=35.0, ambt=27.0, ambh=65.0),
  "Sheep": dict(bt=39.1, hr=75, rr=20, act=0.6, cough=0, rum=38.0, ambt=27.0, ambh=65.0),
  "Pig": dict(bt=39.2, hr=85, rr=30, act=0.6, cough=0, rum=0.0, ambt=27.0, ambh=65.0),
  "Chicken": dict(bt=41.4, hr=300, rr=30, act=0.6, cough=0, rum=0.0, ambt=27.0, ambh=65.0),
  "Turkey": dict(bt=41.1, hr=290, rr=28, act=0.6, cough=0, rum=0.0, ambt=27.0, ambh=65.0),
  "Camel": dict(bt=36.8, hr=50, rr=16, act=0.6, cough=0, rum=40.0, ambt=27.0, ambh=45.0),
  "Donkey": dict(bt=37.5, hr=44, rr=18, act=0.6, cough=0, rum=0.0, ambt=27.0, ambh=55.0),
}

st.set_page_config(page_title="LIVESTOCK HEALTH — FUSION", page_icon="🐄", layout="centered")
set_readable_bg(BG_IMAGE_PATH, overlay=0.6)

st.markdown("<h1 class='title-green'>LIVESTOCK HEALTH — VISION + SENSORS (FUSION)</h1>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Simple inputs for farmers • Tunable weights for experts • Works offline on this device.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Fusion & Policy")
    alpha = st.slider("Sensor weight α", 0.0, 1.0, 0.45, 0.05)
    tau   = st.slider("Decision threshold τ", 0.0, 1.0, 0.50, 0.05)
    st.divider()
    species = st.selectbox("Species preset", list(PRESETS.keys()))
    base = PRESETS[species]

colA, colB = st.columns([1,1], gap="large")

with colA:
    st.subheader("1) Photo (optional)")
    up = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png","bmp","webp"])
    pil = Image.open(io.BytesIO(up.read())) if up else None
    if pil: st.image(pil, use_container_width=True)

with colB:
    st.subheader("2) Quick symptoms")
    feels_hot      = st.toggle("Feels hot / warm ears", False)
    breathing_fast = st.toggle("Breathing faster than usual", False)
    coughing       = st.toggle("Coughing heard", False)
    less_active    = st.toggle("Less active / moving less", False)
    eating_less    = st.toggle("Eating/chewing cud less", False)
    heat_stress    = st.toggle("Hot & humid / midday heat", False)

    with st.expander("Advanced numbers (optional)"):
        body_temp_c = st.number_input("Body temp (°C)", value=base["bt"], step=0.1, format="%.1f")
        heart_rate_bpm = st.number_input("Heart rate (bpm)", value=base["hr"], step=1)
        resp_rate_bpm = st.number_input("Resp rate (bpm)", value=base["rr"], step=1)
        activity_idx = st.slider("Activity index (0–1)", 0.0, 1.0, float(base["act"]), 0.01)
        cough_count = st.number_input("Coughs per hour", value=base["cough"], step=1, min_value=0)
        rumination_min = st.number_input("Rumination min/hour", value=base["rum"], step=1.0, min_value=0.0, format="%.1f")
        ambient_temp_c = st.number_input("Ambient temp (°C)", value=base["ambt"], step=0.5, format="%.1f")
        ambient_humidity_pct = st.number_input("Ambient humidity (%)", value=base["ambh"], step=1.0, format="%.1f")

    # start from baseline, apply toggles
    vals = dict(
        body_temp_c=base["bt"], heart_rate_bpm=base["hr"], resp_rate_bpm=base["rr"],
        activity_idx=float(base["act"]), cough_count=base["cough"], rumination_min=base["rum"],
        ambient_temp_c=base["ambt"], ambient_humidity_pct=base["ambh"],
    )
    if feels_hot:        vals["body_temp_c"] += 1.0
    if breathing_fast:   vals["resp_rate_bpm"] += 8; vals["heart_rate_bpm"] += 10
    if coughing:         vals["cough_count"] = max(vals["cough_count"], 3)
    if less_active:      vals["activity_idx"] = max(0.0, vals["activity_idx"] - 0.25)
    if eating_less:      vals["rumination_min"] = max(0.0, vals["rumination_min"] - 15.0)
    if heat_stress:      vals["ambient_temp_c"] += 4.0; vals["ambient_humidity_pct"] += 10.0

    # prefer advanced fields if provided (Streamlit always defines them inside expander)
    vals.update(dict(
        body_temp_c=body_temp_c, heart_rate_bpm=heart_rate_bpm, resp_rate_bpm=resp_rate_bpm,
        activity_idx=activity_idx, cough_count=cough_count, rumination_min=rumination_min,
        ambient_temp_c=ambient_temp_c, ambient_humidity_pct=ambient_humidity_pct,
    ))

st.markdown("---")
if st.button("🔮 Predict", type="primary", use_container_width=True):
    try:
        m, tfm = load_vision()
        pipe = load_sensor()
        pv = p_vis(m, tfm, pil) if pil else None
        ps = p_sens(pipe, vals)
        pf = ps if pv is None else (alpha*ps + (1-alpha)*pv)
        decision = "UNHEALTHY" if pf >= tau else "HEALTHY"

        st.subheader("Results")
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        metric_card("p_vis (image)", f"{(pv if pv is not None else np.nan):.3f}")
        metric_card("p_sens (sensors)", f"{ps:.3f}")
        metric_card("p_fused", f"{pf:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        if decision == "UNHEALTHY":
            st.markdown(f"<div class='bad'>Decision → <b>UNHEALTHY</b> (τ={tau:.2f}, α={alpha:.2f})</div>", unsafe_allow_html=True)
            st.warning("Action: recheck temperature, isolate if necessary, consult a vet.")
        else:
            st.markdown(f"<div class='ok'>Decision → <b>HEALTHY</b> (τ={tau:.2f}, α={alpha:.2f})</div>", unsafe_allow_html=True)
            st.info("No strong warning signs right now. Keep observing.")

        st.markdown(
            "<div class='muted'>Tip: increase α to trust sensors more (early fevers). "
            "Raise τ for fewer false alarms (action mode); lower τ for screening.</div>",
            unsafe_allow_html=True,
        )
    except FileNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
