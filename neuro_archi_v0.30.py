
"""
neuro_archi_v0.30.py
"""

import io
import os
import math
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops
import streamlit as st

# Optional dependency for personas
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# ---------------------------
# Version & base scaling
# ---------------------------
VERSION = "0.30"
BASE_CANVAS = 640
BASE_LABEL_PAD = 84
BASE_FONT = 24
LABEL_FONT_MULTIPLIER = 3.0  # labels are 3× larger (and scale with render_scale)

# ---------------------------
# Rendering baseline (reference look) for Baseline visual (fixed look)
# ---------------------------
REF_HR  = 95.0
REF_HRV = 70.0
REF_SC  = 8.0

# ---------------------------
# Neutral defaults for entered inputs (starting values in UI only)
# ---------------------------
NEUTRAL_HR   = 75.0   # bpm – moderate adult resting HR
NEUTRAL_HRV  = 60.0   # ms – mid of typical resting range
NEUTRAL_SC   = 6.0    # µS – moderate tonic level

# ---------------------------
# Full physiological ranges (normalization domains)
# ---------------------------
HR_MIN,  HR_MAX  = 30.0, 220.0
HRV_MIN, HRV_MAX = 5.0, 250.0
SC_MIN,  SC_MAX  = 0.1, 40.0

# ---------------------------
# Tunable rendering constants (base, will be scaled by render_scale)
# ---------------------------
R_MIN_BASE, R_MAX_BASE = 120, 220
THICK_MIN_BASE, THICK_MAX_BASE = 16, 80
ALPHA_MIN, ALPHA_MAX = 200, 255
DEFAULT_LAYERS = 5
DEFAULT_RANDOMNESS = 0.10
DEFAULT_GLOW_BLUR_BASE = 12

# Normalization domains (expanded)
HR_RANGE  = (HR_MIN, HR_MAX)
HRV_RANGE = (HRV_MIN, HRV_MAX)
SC_RANGE  = (SC_MIN, SC_MAX)

# ---------------------------
# Utilities
# ---------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def normalize(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def interpolate_color(c1: Tuple[int,int,int], c2: Tuple[int,int,int], t: float) -> Tuple[int,int,int]:
    return tuple(int(round(lerp(c1[i], c2[i], t))) for i in range(3))

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join([ch*2 for ch in h])
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ---------------------------
# Physio -> shape/color mappings (globals set later based on render_scale)
# ---------------------------
R_MIN = R_MIN_BASE
R_MAX = R_MAX_BASE
THICK_MIN = THICK_MIN_BASE
THICK_MAX = THICK_MAX_BASE
CANVAS_SIZE = BASE_CANVAS
LABEL_PAD_H = BASE_LABEL_PAD
GLOW_BLUR_BASE = DEFAULT_GLOW_BLUR_BASE

def map_hr_to_size_and_jaggedness(hr: float) -> Tuple[float, float]:
    # Lower HR -> bigger & smoother; Higher HR -> smaller & more jagged
    t = 1.0 - normalize(hr, *HR_RANGE)
    base_r = lerp(R_MIN, R_MAX, t)
    jag = lerp(0.01, 0.12, 1.0 - t)
    return base_r, jag

def map_hrv_to_thickness(hrv: float) -> float:
    t = normalize(hrv, *HRV_RANGE)
    return lerp(THICK_MIN, THICK_MAX, t)

def make_sc_gradient(sc_value: float,
                     low_pair: Tuple[Tuple[int,int,int], Tuple[int,int,int]],
                     high_pair: Tuple[Tuple[int,int,int], Tuple[int,int,int]]):
    sc_t = normalize(sc_value, *SC_RANGE)
    start = interpolate_color(low_pair[0], high_pair[0], sc_t)
    end   = interpolate_color(low_pair[1], high_pair[1], sc_t)
    def g(s: float) -> Tuple[int,int,int]:
        s = max(0.0, min(1.0, s))
        return interpolate_color(start, end, s)
    return g

# ---------------------------
# Rendering internals
# ---------------------------
def _make_layer(center,
                base_r,
                thickness,
                jag_amp,
                randomness,
                gradient_fn,
                alpha,
                rng,
                seeds,
                freq_mult: float = 1.0,
                gradient_gamma: float = 0.9):
    W = CANVAS_SIZE
    H = CANVAS_SIZE
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    X = xx - cx
    Y = yy - cy
    r = np.sqrt(X*X + Y*Y) + 1e-6
    theta = np.arctan2(Y, X)

    # Multi-sine waviness
    k1, k2, k3 = int(6*freq_mult), int(11*freq_mult), int(17*freq_mult)
    p1, p2, p3 = seeds
    wav = (np.sin(k1*theta + p1) +
           0.6*np.sin(k2*theta + p2) +
           0.4*np.sin(k3*theta + p3)) / 2.0

    # Coarse angular noise
    bins = 360
    ang = ((theta + math.pi) / (2*math.pi) * (bins-1)).astype(np.int32)
    noise_table = rng.normal(0.0, 1.0, size=bins)
    ang_noise = noise_table[ang]

    # Radius & thickness fields
    R_theta = base_r * (1.0 + jag_amp*wav + randomness*0.05*ang_noise)
    T_theta = thickness * (1.0 + randomness*0.1*ang_noise)

    inner = R_theta
    outer = R_theta + T_theta

    # Soft band mask
    def smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
        return t*t*(3 - 2*t)

    band = smoothstep(inner - 1.8, inner + 1.8, r) * (1.0 - smoothstep(outer - 1.8, outer + 1.8, r))
    band = np.clip(band, 0.0, 1.0)

    # Band-relative gradient with gamma
    steps = 256
    lut = np.array([gradient_fn(float(v)) for v in np.linspace(0,1,steps)], dtype=np.uint8)
    den = np.maximum(outer - inner, 1e-6)
    s_band = np.clip((r - inner) / den, 0.0, 1.0)
    if abs(gradient_gamma - 1.0) > 1e-6:
        s_band = np.power(s_band, gradient_gamma)
    idx = (s_band * (steps-1)).astype(np.int32)

    color = np.zeros((H, W, 4), dtype=np.uint8)
    color[..., :3] = lut[idx]

    # Alpha with mild texture
    texture = 0.85 + 0.3 * rng.random(size=band.shape)
    a = np.clip(band * texture * alpha, 0, 255).astype(np.uint8)
    color[..., 3] = a

    img = Image.fromarray(color, mode="RGBA")
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))

def draw_aura_ring(size, hr, hrv, sc, num_layers, randomness, glow_blur_scaled,
                   gradient_low, gradient_high, label, gradient_gamma: float,
                   edge_intensity: float, label_font_size: int) -> Image.Image:
    base_r, jag = map_hr_to_size_and_jaggedness(hr)
    thick = map_hrv_to_thickness(hrv)
    grad_fn = make_sc_gradient(sc, gradient_low, gradient_high)

    # Edge intensity mapping (v0.17 feel at 1.0)
    jag_scale_base, freq_base, rand_base = 1.6, 1.6, 1.25
    jag_scale = 1.0 + (jag_scale_base - 1.0) * edge_intensity
    freq_mult = 1.0 + (freq_base - 1.0) * edge_intensity
    rand_boost = 1.0 + (rand_base - 1.0) * edge_intensity

    jag *= jag_scale

    W = size
    H = size + LABEL_PAD_H
    top_h = size
    cx = cy = size // 2

    full_canvas = Image.new("RGBA", (W, H), (255,255,255,0))

    rng = np.random.default_rng()

    ring_canvas = Image.new("RGBA", (size, size), (255,255,255,0))
    for i in range(num_layers):
        layer_r = base_r * (1.0 + rng.normal(0.0, randomness*0.03*rand_boost))
        layer_th = thick * (1.0 + rng.normal(0.0, randomness*0.05*rand_boost))
        layer_jag = jag * (1.0 + rng.normal(0.0, randomness*0.2*rand_boost))
        alpha = int(lerp(ALPHA_MIN, ALPHA_MAX, i / max(1, num_layers-1)))
        seeds = (rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi))

        layer_img = _make_layer(
            (cx, cy),
            layer_r, layer_th, layer_jag,
            randomness*rand_boost,
            grad_fn,
            alpha,
            rng,
            seeds,
            freq_mult=freq_mult,
            gradient_gamma=gradient_gamma,
        )
        ring_canvas = Image.alpha_composite(ring_canvas, layer_img)

    if glow_blur_scaled > 0:
        r1 = float(glow_blur_scaled) * 1.25
        r2 = float(glow_blur_scaled) * 2.0
        glow1 = ring_canvas.filter(ImageFilter.GaussianBlur(radius=r1))
        glow2 = glow1.filter(ImageFilter.GaussianBlur(radius=r2))
        ring_canvas = ImageChops.screen(glow2, ring_canvas)

    full_canvas.paste(ring_canvas, (0, 0), ring_canvas)

    draw = ImageDraw.Draw(full_canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", label_font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (W - tw) // 2
    y = top_h + (LABEL_PAD_H - th) // 2
    draw.text((x+1, y+1), label, fill=(0,0,0,128), font=font)
    draw.text((x, y), label, fill=(0,0,0,230), font=font)

    return full_canvas

def composite_triptych(imgs: List[Image.Image], pad: int = 24) -> Image.Image:
    W, H = imgs[0].size
    out = Image.new("RGBA", (W*3 + pad*2, H), (255,255,255,0))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0), im)
        x += W + pad
    bg = Image.new("RGB", out.size, (255,255,255))
    bg.paste(out, mask=out.split()[-1])
    return bg

# ---------------------------
# Delta -> normalized n -> effective absolutes (vs REF)
# ---------------------------
def deltas_to_effective_abs_vs_ref(anchor_hr: float, anchor_hrv: float, anchor_sc: float,
                                   hr_in: float, hrv_in: float, sc_in: float,
                                   cap_dhr: float, cap_dhrvp: float, cap_dsc: float) -> Dict:
    # Raw deltas vs participant anchor
    dHR   = hr_in  - anchor_hr
    dHRVp = (hrv_in / max(anchor_hrv, 1e-6) - 1.0) * 100.0
    dSC   = sc_in  - anchor_sc

    # Clamp using adjustable caps
    dHR_eff   = clamp(dHR,   -cap_dhr,   +cap_dhr)
    dHRV_effp = clamp(dHRVp, -cap_dhrvp, +cap_dhrvp)
    dSC_eff   = clamp(dSC,   -cap_dsc,   +cap_dsc)

    # Normalize to [-1,1] with current caps
    n_hr  = dHR_eff   / max(cap_dhr, 1e-6)
    n_hrv = dHRV_effp / max(cap_dhrvp, 1e-6)
    n_sc  = dSC_eff   / max(cap_dsc, 1e-6)

    # Map to full range around REF
    if n_hr >= 0:
        hr_eff = REF_HR + n_hr * (HR_MAX - REF_HR)
    else:
        hr_eff = REF_HR + n_hr * (REF_HR - HR_MIN)
    hr_eff = clamp(hr_eff, HR_MIN, HR_MAX)

    if n_sc >= 0:
        sc_eff = REF_SC + n_sc * (SC_MAX - REF_SC)
    else:
        sc_eff = REF_SC + n_sc * (REF_SC - SC_MIN)
    sc_eff = clamp(sc_eff, SC_MIN, SC_MAX)

    if n_hrv >= 0:
        hrv_eff = math.exp((1.0 - n_hrv) * math.log(REF_HRV) + n_hrv * math.log(HRV_MAX))
    else:
        n = abs(n_hrv)
        hrv_eff = math.exp((1.0 - n) * math.log(REF_HRV) + n * math.log(HRV_MIN))
    hrv_eff = clamp(hrv_eff, HRV_MIN, HRV_MAX)

    clipped = {
        "heart rate":             abs(dHR)   > cap_dhr   + 1e-9,
        "heart rate variability": abs(dHRVp) > cap_dhrvp + 1e-9,
        "skin conductance":       abs(dSC)   > cap_dsc   + 1e-9,
    }

    return {
        "anchor": {"HR": anchor_hr, "HRV": anchor_hrv, "SC": anchor_sc},
        "entered_abs": {"HR": hr_in, "HRV": hrv_in, "SC": sc_in},
        "delta": {"ΔHR": dHR, "ΔHRV%": dHRVp, "ΔSC": dSC},
        "delta_clamped": {"ΔHR": dHR_eff, "ΔHRV%": dHRV_effp, "ΔSC": dSC_eff},
        "n_normalized": {"n_HR": n_hr, "n_HRV": n_hrv, "n_SC": n_sc},
        "effective_abs_vs_ref": {"HR": hr_eff, "HRV": hrv_eff, "SC": sc_eff},
        "any_clipped": any(clipped.values()),
        "clipped_flags": clipped,
    }

# ---------------------------
# Personas loader & renderer
# ---------------------------
EXPECTED_COLS = [
    "persona",
    "baseline_hr", "baseline_hrv", "baseline_sc",
    "supportive_hr", "supportive_hrv", "supportive_sc",
    "unsupportive_hr", "unsupportive_hrv", "unsupportive_sc",
]

def _norm_cols(cols):
    return [c.strip().lower().replace(" ", "_") for c in cols]

def load_personas_df(uploaded_file_or_path):
    if not HAS_PANDAS:
        st.error("Pandas not available. Install pandas to use personas rendering.")
        return None
    try:
        if uploaded_file_or_path is None:
            return None

        # detect type
        is_buffer = hasattr(uploaded_file_or_path, "read")
        name = getattr(uploaded_file_or_path, "name", None) if is_buffer else str(uploaded_file_or_path)
        ext = (os.path.splitext(name)[1] or "").lower() if name else ""

        if ext == ".csv":
            df = pd.read_csv(uploaded_file_or_path)
        elif ext in (".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"):
            try:
                df = pd.read_excel(uploaded_file_or_path, engine="openpyxl")
            except ImportError:
                st.warning("Excel engine 'openpyxl' missing. Install it or upload a CSV instead.")
                return None
        else:
            # Try Excel then CSV
            try:
                df = pd.read_excel(uploaded_file_or_path, engine="openpyxl")
            except Exception:
                try:
                    df = pd.read_csv(uploaded_file_or_path)
                except Exception as e:
                    st.error(f"Unsupported file or missing engine. Details: {e}")
                    return None

        df.columns = _norm_cols(df.columns)
        return df
    except Exception as e:
        st.error(f"Failed to read data: {e}")
        return None

def personas_have_expected_cols(df):
    cols = set(df.columns.tolist())
    missing = [c for c in EXPECTED_COLS if c not in cols]
    return missing

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=f"Aura Rings v{VERSION}", page_icon="💠", layout="centered")
st.title(f"Aura Rings v{VERSION}")

with st.sidebar:
    # 1) Sensitivity first
    with st.expander("🧭 Sensitivity (visual caps)", expanded=True):
        cap_dhr   = st.slider("Heart Rate cap (± bpm)", 5.0, 30.0, 15.0, 0.5)
        cap_dhrvp = st.slider("HRV cap (± %)", 5.0, 50.0, 25.0, 1.0)
        cap_dsc   = st.slider("Skin Conductance cap (± µS)", 0.2, 3.0, 1.0, 0.1)

    # 2) Personas second (upload only; no default checkbox)
    with st.expander("🧪 Personas (Excel/CSV)", expanded=False):
        st.caption("Upload a .xlsx (requires openpyxl) or .csv with columns: persona, baseline_[hr|hrv|sc], supportive_[hr|hrv|sc], unsupportive_[hr|hrv|sc].")
        uploaded = st.file_uploader("Upload personas file", type=["xlsx","csv"])
        personas_btn = st.button("Render personas")

    # 3) Render Controls third
    with st.expander("🎛️ Render Controls", expanded=False):
        render_scale = st.select_slider("Render scale", options=[1, 2, 3], value=1,
                                        help="1 = base res, 3 = ~9× pixels for print")
        num_layers = st.slider("Number of layers", 1, 10, DEFAULT_LAYERS)
        randomness = st.slider("Randomness", 0.0, 0.5, DEFAULT_RANDOMNESS, 0.01)
        glow_blur_base = st.slider("Glow blur radius (strong bloom)", 0, 40, DEFAULT_GLOW_BLUR_BASE)
        edge_intensity = st.slider("Edge intensity", 0.0, 1.5, 1.0, 0.05)

    # 4) Gradients fourth
    with st.expander("🎨 Gradients", expanded=False):
        gradient_gamma = st.slider("Gradient contrast (gamma)", 0.6, 1.4, 0.9, 0.05)
        low_start = st.color_picker("Low SC – inner color", "#2ec4b6")
        low_end   = st.color_picker("Low SC – outer color", "#9be15d")
        high_start= st.color_picker("High SC – inner color", "#c81d77")
        high_end  = st.color_picker("High SC – outer color", "#ff6a00")

        gradient_low = (hex_to_rgb(low_start), hex_to_rgb(low_end))
        gradient_high= (hex_to_rgb(high_start), hex_to_rgb(high_end))

    # 5) Behavior last
    with st.expander("⚙️ Behavior", expanded=False):
        auto_render = st.checkbox("Auto-render on change", value=True)

# Apply render scale to globals
CANVAS_SIZE = int(BASE_CANVAS * render_scale)
LABEL_PAD_H = int(BASE_LABEL_PAD * render_scale)
R_MIN = int(R_MIN_BASE * render_scale)
R_MAX = int(R_MAX_BASE * render_scale)
THICK_MIN = int(THICK_MIN_BASE * render_scale)
THICK_MAX = int(THICK_MAX_BASE * render_scale)
GLOW_BLUR_BASE = glow_blur_base * render_scale
LABEL_FONT_SIZE = int(BASE_FONT * LABEL_FONT_MULTIPLIER * render_scale)

# --- Inputs (no subheader) ---
colA, colB, colC = st.columns(3)

def inputs_abs(col, title, defaults):
    with col:
        st.markdown(f"**{title}**")
        hr  = st.number_input("HR (bpm)", 30.0, 220.0, defaults[0], step=1.0, key=f"hr_{title}")
        hrv = st.number_input("HRV (ms)", 5.0, 250.0, defaults[1], step=1.0, key=f"hrv_{title}")
        sc  = st.number_input("SC (uS)", 0.1, 40.0, defaults[2], step=0.1, key=f"sc_{title}")
    return hr, hrv, sc

# Use neutral defaults for all three input panels
baseline_defaults     = (NEUTRAL_HR, NEUTRAL_HRV, NEUTRAL_SC)
supportive_defaults   = (NEUTRAL_HR, NEUTRAL_HRV, NEUTRAL_SC)
unsupportive_defaults = (NEUTRAL_HR, NEUTRAL_HRV, NEUTRAL_SC)

hrA_in, hrvA_in, scA_in = inputs_abs(colA, "Baseline",     baseline_defaults)
hrB_in, hrvB_in, scB_in = inputs_abs(colB, "Supportive",   supportive_defaults)
hrC_in, hrvC_in, scC_in = inputs_abs(colC, "Unsupportive", unsupportive_defaults)

should_render = auto_render or st.button("🎨 Generate Visual")

def render_triptych(anchor_hr, anchor_hrv, anchor_sc,
                    A_vals, B_vals, C_vals,
                    num_layers, randomness, glow_blur_scaled,
                    gradient_low, gradient_high, gradient_gamma, edge_intensity):
    (hrA_eff, hrvA_eff, scA_eff) = (REF_HR, REF_HRV, REF_SC)
    (hrB_in, hrvB_in, scB_in) = B_vals
    (hrC_in, hrvC_in, scC_in) = C_vals

    diagB = deltas_to_effective_abs_vs_ref(anchor_hr, anchor_hrv, anchor_sc, hrB_in, hrvB_in, scB_in,
                                           cap_dhr, cap_dhrvp, cap_dsc)
    diagC = deltas_to_effective_abs_vs_ref(anchor_hr, anchor_hrv, anchor_sc, hrC_in, hrvC_in, scC_in,
                                           cap_dhr, cap_dhrvp, cap_dsc)

    hrB_eff, hrvB_eff, scB_eff = diagB["effective_abs_vs_ref"]["HR"], diagB["effective_abs_vs_ref"]["HRV"], diagB["effective_abs_vs_ref"]["SC"]
    hrC_eff, hrvC_eff, scC_eff = diagC["effective_abs_vs_ref"]["HR"], diagC["effective_abs_vs_ref"]["HRV"], diagC["effective_abs_vs_ref"]["SC"]

    ringA = draw_aura_ring(
        CANVAS_SIZE, hrA_eff, hrvA_eff, scA_eff, num_layers, randomness, glow_blur_scaled,
        gradient_low, gradient_high, "Baseline",
        gradient_gamma=gradient_gamma, edge_intensity=edge_intensity, label_font_size=LABEL_FONT_SIZE
    )
    ringB = draw_aura_ring(
        CANVAS_SIZE, hrB_eff, hrvB_eff, scB_eff, num_layers, randomness, glow_blur_scaled,
        gradient_low, gradient_high, "Supportive",
        gradient_gamma=gradient_gamma, edge_intensity=edge_intensity, label_font_size=LABEL_FONT_SIZE
    )
    ringC = draw_aura_ring(
        CANVAS_SIZE, hrC_eff, hrvC_eff, scC_eff, num_layers, randomness, glow_blur_scaled,
        gradient_low, gradient_high, "Unsupportive",
        gradient_gamma=gradient_gamma, edge_intensity=edge_intensity, label_font_size=LABEL_FONT_SIZE
    )
    triptych = composite_triptych([ringA, ringB, ringC])
    return triptych, diagB, diagC

if should_render:
    with st.spinner("Rendering rings..."):
        # Anchor is participant Baseline (the entered baseline inputs)
        anchor_hr, anchor_hrv, anchor_sc = hrA_in, hrvA_in, scA_in
        triptych, diagB, diagC = render_triptych(
            anchor_hr, anchor_hrv, anchor_sc,
            (hrA_in, hrvA_in, scA_in),
            (hrB_in, hrvB_in, scB_in),
            (hrC_in, hrvC_in, scC_in),
            num_layers, randomness, GLOW_BLUR_BASE,
            gradient_low, gradient_high, gradient_gamma, edge_intensity
        )

    st.image(triptych, caption="Composite Triptych", use_container_width=True)

    with st.expander("Supportive – details"):
        st.write("**Anchor (participant Baseline):**", {"HR": anchor_hr, "HRV": anchor_hrv, "SC": anchor_sc})
        st.write("**Effective absolutes used for rendering (vs reference):**", diagB["effective_abs_vs_ref"])
        if diagB["any_clipped"]:
            clipped_metrics = [k for k, v in diagB["clipped_flags"].items() if v]
            which = ", ".join(clipped_metrics)
            st.warning(f"Note: the visual may under represent the large change in {which}.")

    with st.expander("Unsupportive – details"):
        st.write("**Anchor (participant Baseline):**", {"HR": anchor_hr, "HRV": anchor_hrv, "SC": anchor_sc})
        st.write("**Effective absolutes used for rendering (vs reference):**", diagC["effective_abs_vs_ref"])
        if diagC["any_clipped"]:
            clipped_metrics = [k for k, v in diagC["clipped_flags"].items() if v]
            which = ", ".join(clipped_metrics)
            st.warning(f"Note: the visual may under represent the large change in {which}.")

    buf = io.BytesIO()
    triptych.save(buf, format="PNG", dpi=(300,300))
    st.download_button("💾 Download PNG", data=buf.getvalue(), file_name=f"aura_triptych_v{VERSION.replace('.','_')}.png", mime="image/png")

# Personas rendering
if 'personas_btn' in globals() and personas_btn:
    with st.spinner("Loading personas..."):
        df = load_personas_df(uploaded) if uploaded is not None else None
        if df is None:
            st.warning("Please upload a personas file to render.")
    if df is not None:
        missing = personas_have_expected_cols(df)
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.dataframe(df.head())
            st.info("Ensure your data matches the expected schema or upload a CSV with the right column names.")
        else:
            st.success(f"Loaded {len(df)} personas.")
            max_rows = min(len(df), 12)  # keep UI responsive
            for idx in range(max_rows):
                row = df.iloc[idx]
                name = str(row.get('persona', f'Persona {idx+1}'))
                st.subheader(f"👤 {name}")
                # Show parameters under persona name
                base_hr, base_hrv, base_sc = float(row['baseline_hr']), float(row['baseline_hrv']), float(row['baseline_sc'])
                sup_hr, sup_hrv, sup_sc    = float(row['supportive_hr']), float(row['supportive_hrv']), float(row['supportive_sc'])
                uns_hr, uns_hrv, uns_sc    = float(row['unsupportive_hr']), float(row['unsupportive_hrv']), float(row['unsupportive_sc'])
                st.markdown(
                    f"**Baseline:** HR {base_hr:.1f} bpm, HRV {base_hrv:.1f} ms, SC {base_sc:.2f} µS  \n"
                    f"**Supportive:** HR {sup_hr:.1f} bpm, HRV {sup_hrv:.1f} ms, SC {sup_sc:.2f} µS  \n"
                    f"**Unsupportive:** HR {uns_hr:.1f} bpm, HRV {uns_hrv:.1f} ms, SC {uns_sc:.2f} µS"
                )
                triptych, _, _ = render_triptych(
                    base_hr, base_hrv, base_sc,
                    (base_hr, base_hrv, base_sc),
                    (sup_hr, sup_hrv, sup_sc),
                    (uns_hr, uns_hrv, uns_sc),
                    num_layers, randomness, GLOW_BLUR_BASE,
                    gradient_low, gradient_high, gradient_gamma, edge_intensity
                )
                st.image(triptych, use_container_width=True)
