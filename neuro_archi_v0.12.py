
"""
neuro_archi_v0.11.py
Streamlit app that renders a triptych of watercolor-like "aura rings"
using physiological inputs: HR (bpm), HRV (ms), SC (µS).

v0.11 changes:
- Labels now show ONLY the environment description (Baseline, Supportive, Inhibiting) – metrics removed.
- Labels are rendered OUTSIDE the circle (below the ring) on extra canvas padding.
- Increased label font size for readability.
- Keeps v0.10 band-relative gradient and environment-specific waviness/jaggedness.
"""

import io
import math
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops
import streamlit as st

# ---------------------------
# Tunable constants
# ---------------------------
CANVAS_SIZE = 640
LABEL_PAD_H = 84               # extra vertical space reserved for label under the ring
BG_COLOR = (255, 255, 255, 0)  # transparent for compositing
R_MIN, R_MAX = 120, 220        # radius range driven by HR (lower HR = larger ring)
THICK_MIN, THICK_MAX = 16, 80  # thickness driven by HRV (higher HRV = thicker)
ALPHA_MIN, ALPHA_MAX = 200, 255
DEFAULT_LAYERS = 5
DEFAULT_RANDOMNESS = 0.10
DEFAULT_GLOW_BLUR = 12  # strong by default

# Normalization domains
HR_RANGE = (50.0, 110.0)     # bpm
HRV_RANGE = (20.0, 120.0)    # ms
SC_RANGE = (1.0, 15.0)       # µS

# ---------------------------
# Utility functions
# ---------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def normalize(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def interpolate_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return tuple(int(round(lerp(c1[i], c2[i], t))) for i in range(3))

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join([ch*2 for ch in h])
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# ---------------------------
# Mapping physiological data
# ---------------------------
def map_hr_to_size_and_jaggedness(hr: float) -> Tuple[float, float]:
    """Lower HR -> larger & smoother ring. Higher HR -> smaller & more jagged."""
    t = 1.0 - normalize(hr, *HR_RANGE)  # invert so lower HR => bigger
    base_r = lerp(R_MIN, R_MAX, t)
    jag = lerp(0.01, 0.12, 1.0 - t)  # jag increases with HR
    return base_r, jag

def map_hrv_to_thickness(hrv: float) -> float:
    """Higher HRV -> thicker ring."""
    t = normalize(hrv, *HRV_RANGE)
    return lerp(THICK_MIN, THICK_MAX, t)

def make_sc_gradient(sc_value: float,
                     low_pair: Tuple[Tuple[int,int,int], Tuple[int,int,int]],
                     high_pair: Tuple[Tuple[int,int,int], Tuple[int,int,int]]) -> Callable[[float], Tuple[int,int,int]]:
    """Return a gradient function influenced by SC value."""
    sc_t = normalize(sc_value, *SC_RANGE)
    start = interpolate_color(low_pair[0], high_pair[0], sc_t)
    end   = interpolate_color(low_pair[1], high_pair[1], sc_t)

    def g(s: float) -> Tuple[int,int,int]:
        s = clamp(s, 0.0, 1.0)
        return interpolate_color(start, end, s)
    return g

# ---------------------------
# Rendering
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
                use_band_gradient: bool = True,
                gradient_gamma: float = 0.9):
    """Render one semi-transparent layer of the ring to an RGBA image."""
    W = CANVAS_SIZE
    H = CANVAS_SIZE  # ring draws only in the top square (label area is separate below)
    cx, cy = center
    yy, xx = np.mgrid[0:H, 0:W]
    X = xx - cx
    Y = yy - cy
    r = np.sqrt(X*X + Y*Y) + 1e-6
    theta = np.arctan2(Y, X)

    # Multi-sine waviness (allow frequency multiplier)
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

    # ---- Radial gradient lookup (band-relative) ----
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
    # Small per-layer soften
    return img.filter(ImageFilter.GaussianBlur(radius=0.8))

def draw_aura_ring(size, hr, hrv, sc, num_layers, randomness, glow_blur,
                   gradient_low, gradient_high, label, env: str,
                   use_band_gradient: bool, gradient_gamma: float) -> Image.Image:
    base_r, jag = map_hr_to_size_and_jaggedness(hr)
    thick = map_hrv_to_thickness(hrv)
    grad_fn = make_sc_gradient(sc, gradient_low, gradient_high)

    # Environment-specific shaping
    if env.lower() == "supportive":
        jag = max(jag, 0.05)        # ensure some waviness
        freq_mult = 1.3             # slightly more ripples
        rand_boost = 1.1
    elif env.lower() == "inhibiting":
        jag *= 1.6                  # stronger jag amplitude
        freq_mult = 1.6             # more frequent waves
        rand_boost = 1.25
    else:
        freq_mult = 1.0
        rand_boost = 1.0

    # Full canvas includes extra label padding below
    W = size
    H = size + LABEL_PAD_H
    top_area_h = size
    cx = cy = size // 2

    # Start with transparent canvas (RGBA)
    full_canvas = Image.new("RGBA", (W, H), BG_COLOR)

    # Draw ring layers into a top-area canvas then paste
    ring_canvas = Image.new("RGBA", (size, size), BG_COLOR)
    rng = np.random.default_rng(42)

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
            use_band_gradient=use_band_gradient,
            gradient_gamma=gradient_gamma,
        )
        ring_canvas = Image.alpha_composite(ring_canvas, layer_img)

    # Strong glow bloom: double blur + screen
    if glow_blur > 0:
        r1 = float(glow_blur) * 1.25
        r2 = float(glow_blur) * 2.0
        glow1 = ring_canvas.filter(ImageFilter.GaussianBlur(radius=r1))
        glow2 = glow1.filter(ImageFilter.GaussianBlur(radius=r2))
        ring_canvas = ImageChops.screen(glow2, ring_canvas)

    # Paste ring into the top part of full canvas
    full_canvas.paste(ring_canvas, (0, 0), ring_canvas)

    # Label in the bottom padding area (outside the ring)
    draw = ImageDraw.Draw(full_canvas)
    # Try to load a scalable font; fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = (W - tw) // 2
    # Position text centered within the label pad area
    y = top_area_h + (LABEL_PAD_H - th) // 2
    # Slight shadow for readability
    draw.text((x+1, y+1), label, fill=(0,0,0,128), font=font)
    draw.text((x, y), label, fill=(0,0,0,230), font=font)

    return full_canvas

def composite_triptych(imgs: List[Image.Image], pad: int = 24) -> Image.Image:
    # All images share the same size (CANVAS_SIZE x (CANVAS_SIZE+LABEL_PAD_H))
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Aura Rings – neuro_archi_v0.11", page_icon="💠", layout="centered")
st.title("💠 Aura Ring Visualizer – neuro_archi_v0.11")
st.caption("Band-relative gradients (inner→outer), labels outside the circle, and clearer environment styling.")

with st.sidebar:
    st.header("Render Controls")
    num_layers = st.slider("Number of layers", 1, 10, DEFAULT_LAYERS)
    randomness = st.slider("Randomness", 0.0, 0.5, DEFAULT_RANDOMNESS, 0.01)
    glow_blur = st.slider("Glow blur radius (strong bloom)", 0, 40, DEFAULT_GLOW_BLUR)

    st.subheader("Gradient Controls")
    use_band_gradient = st.checkbox("Use band-relative gradient (inner→outer)", value=True)
    gradient_gamma = st.slider("Gradient contrast (gamma)", 0.6, 1.4, 0.9, 0.05, help="Lower = higher contrast across ring thickness")

    st.subheader("Gradients (SC color mapping)")
    low_start = st.color_picker("Low SC – inner color", "#2ec4b6")
    low_end   = st.color_picker("Low SC – outer color", "#9be15d")
    high_start= st.color_picker("High SC – inner color", "#c81d77")
    high_end  = st.color_picker("High SC – outer color", "#ff6a00")

    gradient_low = (hex_to_rgb(low_start), hex_to_rgb(low_end))
    gradient_high= (hex_to_rgb(high_start), hex_to_rgb(high_end))

st.subheader("Inputs")
colA, colB, colC = st.columns(3)

def inputs(col, title, defaults):
    with col:
        st.markdown(f"**{title}**")
        hr  = st.number_input("HR (bpm)", 30.0, 220.0, defaults[0], step=1.0, key=f"hr_{title}")
        hrv = st.number_input("HRV (ms)", 5.0, 250.0, defaults[1], step=1.0, key=f"hrv_{title}")
        sc  = st.number_input("SC (µS)", 0.1, 40.0, defaults[2], step=0.1, key=f"sc_{title}")
    return hr, hrv, sc

# Defaults:
# - Supportive = best (low HR, high HRV, low SC)
# - Inhibiting = worst (high HR, low HRV, high SC)
# - Baseline = mid-range
baseline_defaults   = ( (HR_RANGE[0]+HR_RANGE[1])/2.0, (HRV_RANGE[0]+HRV_RANGE[1])/2.0, (SC_RANGE[0]+SC_RANGE[1])/2.0 )  # (~80, ~70, ~8)
supportive_defaults = ( HR_RANGE[0], HRV_RANGE[1], SC_RANGE[0] )   # (50, 120, 1)
inhibiting_defaults = ( HR_RANGE[1], HRV_RANGE[0], SC_RANGE[1] )   # (110, 20, 15)

hrA, hrvA, scA = inputs(colA, "Baseline",   baseline_defaults)
hrB, hrvB, scB = inputs(colB, "Supportive", supportive_defaults)
hrC, hrvC, scC = inputs(colC, "Inhibiting", inhibiting_defaults)

if st.button("🎨 Generate Visual"):
    with st.spinner("Rendering rings..."):
        ringA = draw_aura_ring(
            CANVAS_SIZE, hrA, hrvA, scA, num_layers, randomness, glow_blur,
            gradient_low, gradient_high,
            "Baseline", env="baseline",
            use_band_gradient=use_band_gradient, gradient_gamma=gradient_gamma
        )
        ringB = draw_aura_ring(
            CANVAS_SIZE, hrB, hrvB, scB, num_layers, randomness, glow_blur,
            gradient_low, gradient_high,
            "Supportive", env="supportive",
            use_band_gradient=use_band_gradient, gradient_gamma=gradient_gamma
        )
        ringC = draw_aura_ring(
            CANVAS_SIZE, hrC, hrvC, scC, num_layers, randomness, glow_blur,
            gradient_low, gradient_high,
            "Inhibiting", env="inhibiting",
            use_band_gradient=use_band_gradient, gradient_gamma=gradient_gamma
        )
        triptych = composite_triptych([ringA, ringB, ringC])

    st.image(triptych, caption="Composite Triptych", use_column_width=True)

    buf = io.BytesIO()
    triptych.save(buf, format="PNG")
    st.download_button("💾 Download PNG", data=buf.getvalue(), file_name="aura_triptych_v0_11.png", mime="image/png")
