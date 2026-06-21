from pathlib import Path
import base64, io, csv, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch, torchvision as tv, joblib

# -------- Paths (same as your project) --------
VISION_CKPT   = Path("results/vision/best.pth")
SENSORS_MODEL = Path("results/exp2_sensors/model_rf.joblib")
HERO_IMAGE    = Path("/home/debby/Documents/pngs/cows.jpg")
HISTORY_FILE  = Path("results/ui_history.csv")

# -------- Theme tokens --------
GREEN_BRAND  = "#166534"
GREEN_SOFT_2 = "#f3fbf7"
DARK         = "#0B1220"
SLATE        = "#475569"
WHITE        = "#FFFFFF"

CSS = f"""
<style>
body, .stApp {{
  background: linear-gradient(180deg, #f7fcfa 0%, #f9fffb 100%) !important;
}}
.block-container {{
  background: {WHITE};
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 18px;
  box-shadow: 0 14px 30px rgba(0,0,0,.10);
  padding: 1.2rem 1.4rem;
}}
.block-container * {{ color: {DARK} !important; }}

.hero {{
  width: 100%; height: 360px; border-radius: 18px; margin-bottom: 16px;
  position: relative; overflow: hidden;
  box-shadow: 0 18px 36px rgba(0,0,0,.18);
  border: 1px solid rgba(0,0,0,0.08);
}}
.hero-overlay {{
  position: absolute; inset: 0;
  background: linear-gradient(180deg, rgba(0,0,0,0.15) 0%, rgba(0,0,0,0.58) 100%);
}}
.hero-title {{
  position: absolute; left: 28px; bottom: 22px;
  color: #EAF6EE; font-weight: 1000; font-size: 2.35rem; letter-spacing: .05em;
  text-transform: uppercase; text-shadow: 0 3px 10px rgba(0,0,0,.25);
}}
.hero-sub {{
  position: absolute; left: 28px; bottom: 70px;
  color: #EAF6EE; font-size: 1.08rem; font-weight: 700; opacity: .98;
}}

.section {{
  background: {GREEN_SOFT_2}; border: 1px solid rgba(0,0,0,.06);
  border-left: 6px solid {GREEN_BRAND};
  border-radius: 12px; padding: 12px 14px; margin: 8px 0 12px 0;
}}
.section h3 {{ margin: 0; color: {GREEN_BRAND}; font-weight: 900; letter-spacing: .02em; }}

.group {{
  background: #FFFFFF; border: 1px solid rgba(0,0,0,.08);
  border-radius: 14px; padding: 14px; margin-bottom: 12px;
  box-shadow: 0 6px 16px rgba(0,0,0,.06);
}}

.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 10px 0; }}
.chip {{
  display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700;
  background:#e9f6ef; color:{GREEN_BRAND}; border:1px solid rgba(0,0,0,.06);
}}

.metric-row {{ display:flex; flex-wrap:wrap; gap:12px; margin-top: 8px; }}
.metric-card {{
  flex:1 1 220px; background:#fff; border:1px solid rgba(0,0,0,.08);
  border-radius:12px; padding:12px 14px; box-shadow:0 4px 14px rgba(0,0,0,.08);
}}
.metric-title {{ color:{SLATE}!important; font-size:.9rem; }}
.metric-value {{ color:{DARK}!important; font-size:1.45rem; font-weight:800; }}

.badge {{
  display:inline-block; padding:8px 12px; border-radius:999px; font-weight:900;
  border:1px solid rgba(0,0,0,.08); margin-top:10px; font-size:1.0rem;
}}
.badge-ok {{ background:#E8F5EC; color:{GREEN_BRAND}; }}
.badge-bad {{ background:#FDEBEC; color:#B91C1C; }}
</style>
"""

def hero_banner(image_path: Path):
    if image_path.exists():
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        st.markdown(
            f"""
            <div class='hero' style='background-image:
                url("data:image/jpg;base64,{b64}");
                background-size: cover; background-position: center;'>
              <div class='hero-overlay'></div>
              <div class='hero-sub'>🌿 Vision + Sensors • 🔌 Offline • 👨🏾‍🌾 Farmer-friendly</div>
              <div class='hero-title'>LIVESTOCK HEALTH MONITORING — EXP 2B FUSION</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='section'><h3>LIVESTOCK HEALTH MONITORING — EXP 2B FUSION</h3></div>", unsafe_allow_html=True)

def metric_card(title, value):
    st.markdown(
        f"""<div class="metric-card">
               <div class="metric-title">{title}</div>
               <div class="metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

def append_history(row: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    newfile = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if newfile: w.writeheader()
        w.writerow(row)

# -------- Models (cached) --------
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
    x = np.array([[
        v["body_temp_c"], v["heart_rate_bpm"], v["resp_rate_bpm"], v["activity_idx"],
        v["cough_count"], v["rumination_min"], v["ambient_temp_c"], v["ambient_humidity_pct"]
    ]], dtype=float)
    return float(pipe.predict_proba(x)[:,1][0])

def fuse(alpha, ps, pv): return alpha*ps + (1-alpha)*pv
def decide(p, tau): return "UNHEALTHY" if p >= tau else "HEALTHY"

# -------- Presets --------
PRESETS = {
  "Cattle": dict(bt=38.3, hr=70, rr=22, act=0.6, cough=0, rum=40.0, ambt=27.0, ambh=65.0),
  "Goat":   dict(bt=39.4, hr=88, rr=24, act=0.6, cough=0, rum=35.0, ambt=27.0, ambh=65.0),
  "Sheep":  dict(bt=39.1, hr=75, rr=20, act=0.6, cough=0, rum=38.0, ambt=27.0, ambh=65.0),
  "Pig":    dict(bt=39.2, hr=85, rr=30, act=0.6, cough=0, rum=0.0,  ambt=27.0, ambh=65.0),
  "Chicken":dict(bt=41.4, hr=300,rr=30, act=0.6, cough=0, rum=0.0,  ambt=27.0, ambh=65.0),
  "Camel":  dict(bt=36.8, hr=50, rr=16, act=0.6, cough=0, rum=40.0, ambt=27.0, ambh=45.0),
  "Donkey": dict(bt=37.5, hr=44, rr=18, act=0.6, cough=0, rum=0.0,  ambt=27.0, ambh=55.0),
}

# -------- Page --------
st.set_page_config(page_title="Livestock Health — Fusion (Exp 2b)", page_icon="🐄", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
hero_banner(HERO_IMAGE)

# Sidebar: policy + presets
with st.sidebar:
    st.markdown("<div class='section'><h3>⚙️ Fusion & Policy</h3></div>", unsafe_allow_html=True)
    # Quick modes (optional)
    cA, cB = st.columns(2)
    if cA.button("🟢 Screening"):
        st.session_state['alpha'] = 0.45
        st.session_state['tau']   = 0.40
    if cB.button("🛑 Action"):
        st.session_state['alpha'] = 0.45
        st.session_state['tau']   = 0.60

    alpha_default = st.session_state.get('alpha', 0.45)
    tau_default   = st.session_state.get('tau', 0.50)
    alpha = st.slider("Sensor weight (α)", 0.0, 1.0, float(alpha_default), 0.05, help="Higher → trust sensors more")
    tau   = st.slider("Decision threshold (τ)", 0.0, 1.0, float(tau_default), 0.05, help="Higher → fewer alerts")

    species = st.selectbox("🐾 Species", list(PRESETS.keys()))
    base = PRESETS[species]
    st.markdown(f"<div class='chips'><span class='chip'>α={alpha:.2f}</span><span class='chip'>τ={tau:.2f}</span><span class='chip'>{species}</span></div>", unsafe_allow_html=True)

# ---------- TABS ----------
tab_check, tab_hist, tab_about, tab_settings = st.tabs(["🩺 Checkup", "🕘 History", "ℹ️ About", "⚙️ Settings"])

# TAB: Checkup
with tab_check:
    st.markdown("<div class='section'><h3>🩺 Checkup</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1], gap="large")

    # Vision
    with col1:
        st.markdown("<div class='group'>", unsafe_allow_html=True)
        st.markdown("#### 🖼️ Photo (optional)")
        up = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png","bmp","webp"])
        pil = Image.open(io.BytesIO(up.read())) if up else None
        if pil: st.image(pil, use_container_width=True)
        if st.button("ℹ️ What the image model looks for"):
            st.info("Round/raised nodules, clustered spots, patchy texture. Clear side photo in good light helps.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Sensors
    with col2:
        st.markdown("<div class='group'>", unsafe_allow_html=True)
        st.markdown("#### 🧪 Quick sensor inputs")
        temp_c     = st.number_input("🌡️ Temperature (°C)", value=base["bt"], step=0.1, format="%.1f")
        body_cond  = st.selectbox("⚖️ Body condition", ["Normal", "Thin", "Overweight"])
        activity   = st.selectbox("🏃 Activity", ["Normal", "Low"])
        cough      = st.selectbox("🤧 Coughing", ["No", "Yes"])
        discharge  = st.selectbox("👁️ Eye/Nasal discharge", ["None", "Mild", "Heavy"])
        with st.expander("More details (optional)"):
            hr   = st.number_input("❤️ Heart rate (bpm)", value=base["hr"], step=1)
            rr   = st.number_input("🫁 Resp rate (bpm)", value=base["rr"], step=1)
            ambt = st.number_input("🌤️ Ambient temp (°C)", value=base["ambt"], step=0.5, format="%.1f")
            ambh = st.number_input("💧 Ambient humidity (%)", value=base["ambh"], step=1.0, format="%.1f")
            rum  = st.number_input("🐮 Rumination (min/hr)", value=base["rum"], step=1.0, format="%.1f")
            acti = st.slider("📈 Activity index (0–1)", 0.0, 1.0, float(base["act"]), 0.01)
        if st.button("ℹ️ What these sensor fields mean"):
            st.markdown("- Temp: fever often > 39.5°C (species vary)\n- Activity: low can signal pain/fever\n- Cough/Discharge: respiratory concerns\n- Body condition gives context")
        st.markdown("</div>", unsafe_allow_html=True)

    # Map inputs → features
    vals = dict(
        body_temp_c=temp_c, heart_rate_bpm=base["hr"], resp_rate_bpm=base["rr"],
        activity_idx=float(base["act"]), cough_count=base["cough"], rumination_min=base["rum"],
        ambient_temp_c=base["ambt"], ambient_humidity_pct=base["ambh"],
    )
    if body_cond == "Thin":
        vals["activity_idx"] = max(0.0, vals["activity_idx"] - 0.10); vals["rumination_min"] = max(0.0, vals["rumination_min"] - 5.0)
    elif body_cond == "Overweight":
        vals["activity_idx"] = max(0.0, vals["activity_idx"] - 0.05)
    if activity == "Low": vals["activity_idx"] = max(0.0, vals["activity_idx"] - 0.25)
    if cough == "Yes": vals["cough_count"] = max(vals["cough_count"], 3)
    if discharge == "Mild":
        vals["resp_rate_bpm"] += 4; vals["heart_rate_bpm"] += 5
    elif discharge == "Heavy":
        vals["resp_rate_bpm"] += 8; vals["heart_rate_bpm"] += 10
    if 'hr' in locals():   vals["heart_rate_bpm"] = hr
    if 'rr' in locals():   vals["resp_rate_bpm"] = rr
    if 'ambt' in locals(): vals["ambient_temp_c"] = ambt
    if 'ambh' in locals(): vals["ambient_humidity_pct"] = ambh
    if 'rum' in locals():  vals["rumination_min"] = rum
    if 'acti' in locals(): vals["activity_idx"] = acti

    st.markdown("---")
    cA, cB, cC = st.columns([1,1,1])
    go           = cA.button("🔮 Predict", type="primary", use_container_width=True)
    explain_fuse = cB.button("🧠 Explain Fusion")
    farmer_tips  = cC.button("🆘 Farmer Tips")

    if explain_fuse:
        st.info("Fusion: p_fused = α·p_sensors + (1−α)·p_vision. Increase α to trust sensors (e.g., early fever). Raise τ to reduce false alerts.")
    if farmer_tips:
        st.markdown("• If temp ≥ 39.5°C and activity low → isolate & give water; call a vet. • Use daylight photos; wipe mud before a picture.")

    if go:
        try:
            m, tfm = load_vision()
            pipe = load_sensor()
            pv = p_vis(m, tfm, pil) if pil else None
            ps = p_sens(pipe, vals)
            pf = ps if pv is None else fuse(alpha, ps, pv)
            decision = "UNHEALTHY" if pf >= tau else "HEALTHY"

            st.markdown("<div class='section'><h3>✅ Results</h3></div>", unsafe_allow_html=True)
            st.markdown('<div class="metric-row">', unsafe_allow_html=True)
            metric_card("p_vis (image)",  f"{(pv if pv is not None else np.nan):.3f}")
            metric_card("p_sens (sensors)", f"{ps:.3f}")
            metric_card("p_fused",         f"{pf:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)

            if decision == "UNHEALTHY":
                st.markdown(f"<span class='badge badge-bad'>UNHEALTHY</span>", unsafe_allow_html=True)
                st.warning("Action: recheck temperature, isolate if necessary, consult a vet.")
            else:
                st.markdown(f"<span class='badge badge-ok'>HEALTHY</span>", unsafe_allow_html=True)
                st.info("No strong warning signs right now. Keep observing.")

            why = []
            if vals['body_temp_c'] >= 39.5:   why.append("High temperature")
            if activity == "Low":             why.append("Low activity")
            if cough == "Yes":                why.append("Coughing")
            if discharge in ("Mild","Heavy"): why.append("Eye/nasal discharge")
            if pv is not None and pv >= 0.6:  why.append("Image suggests lesions")
            if why: st.markdown("**Why:** " + ", ".join(why))

            # Save to history
            row = {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "species": species, "alpha": round(alpha,2), "tau": round(tau,2),
                "temp_c": vals["body_temp_c"], "activity_idx": vals["activity_idx"],
                "cough_count": vals["cough_count"], "rr": vals["resp_rate_bpm"],
                "decision": decision, "p_vis": None if pv is None else round(pv,3),
                "p_sens": round(ps,3), "p_fused": round(pf,3),
            }
            append_history(row)

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# TAB: History
with tab_hist:
    st.markdown("<div class='section'><h3>🕘 History</h3></div>", unsafe_allow_html=True)
    if HISTORY_FILE.exists():
        df = pd.read_csv(HISTORY_FILE)
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
        st.caption("Tip: delete results/ui_history.csv to clear.")
    else:
        st.info("No history yet. Make a prediction in Checkup.")

# TAB: About
with tab_about:
    st.markdown("<div class='section'><h3>ℹ️ About</h3></div>", unsafe_allow_html=True)
    st.write(
        "- **Goal:** early & reliable alerts by fusing **vision** + **sensors**.\n"
        "- **Vision:** MobileNetV2 transfer learning.\n"
        "- **Sensors:** RandomForest on temperature, HR/RR, activity, coughs, rumination, ambient.\n"
        "- **Fusion:** **p_fused = α·p_sens + (1−α)·p_vis**; tune **α** and **τ** in the sidebar."
    )

# TAB: Settings
with tab_settings:
    st.markdown("<div class='section'><h3>⚙️ Settings</h3></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("🟢 Screening (α=0.45, τ=0.40)"):
        st.session_state['alpha'] = 0.45; st.session_state['tau'] = 0.40
        st.success("Screening mode set")
    if c2.button("🛑 Action (α=0.45, τ=0.60)"):
        st.session_state['alpha'] = 0.45; st.session_state['tau'] = 0.60
        st.success("Action mode set")
