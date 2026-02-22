"""
Solar Rooftop Analyzer — AI-powered solar potential assessment.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import base64
import csv
import io
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from rasterio.transform import from_bounds

from src.data import (
    generate_synthetic_tile,
    load_image,
    load_or_create_mask,
    prepare_display_rgb,
)
from src.estimate import (
    SolarConfig,
    estimate_all_roofs,
    format_report,
    load_yield_baselines,
    apply_yield_baseline,
)
from src.utils import (
    alignment_debug_info,
    check_crs_units,
    ensure_output_dir,
    print_raster_info,
    validate_polygon_raster_alignment,
)
from src.vectorize import mask_to_polygons, polygons_to_geojson
from src.viz import (
    alignment_debug_figure,
    annotate_polygons,
    overlay_polygons,
    side_by_side,
)

# New modules (Phases 3–6)
from src.policy import load_policy_records, format_policy_summary, policy_to_dict
from src.market import load_market_intelligence, format_market_summary, market_to_dict, default_market_record
from src.confidence import compute_tile_confidence, confidence_to_dict
from src.feasibility import enrich_polygons_with_feasibility


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Solar Rooftop Analyzer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ──────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #F0F4F8 !important;
}
.block-container { max-width: 1200px; padding-top: 1.5rem !important; }
header[data-testid="stHeader"] { background: #fff !important; border-bottom: 1px solid #E2E8F0; }

/* Hide default Streamlit chrome */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── FIX: Sidebar toggle button visibility ───────── */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="collapsedControl"],
[data-testid="collapsedControl"] > button,
section[data-testid="stSidebar"] button[kind="header"] {
    background: #1E293B !important;
    color: #fff !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}
/* Also style the > arrow icon for the collapsed sidebar button */
[data-testid="collapsedControl"] {
    color: #1E293B !important;
}
[data-testid="collapsedControl"] svg {
    fill: #1E293B !important;
    stroke: #1E293B !important;
}
/* Make sure the sidebar collapse button area itself is visible */
button[data-testid="baseButton-header"] {
    background: #334155 !important;
    color: #fff !important;
    border-radius: 8px !important;
}

/* ── Top navbar ──────────────────────────────────── */
.top-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 0; margin-bottom: 0.5rem;
}
.top-nav-brand {
    display: flex; align-items: center; gap: 12px;
}
.top-nav-brand .logo {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #2563EB, #1E40AF);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-size: 20px; font-weight: 700;
}
.top-nav-brand h1 {
    font-size: 1.25rem; font-weight: 700; color: #0F172A;
    margin: 0; line-height: 1.2;
}
.top-nav-brand p {
    font-size: 0.78rem; color: #64748B; margin: 0;
}
.top-nav-actions {
    display: flex; gap: 10px; align-items: center;
}
.btn-settings {
    padding: 8px 16px; border-radius: 8px; border: 1px solid #E2E8F0;
    background: #fff; color: #334155; font-size: 0.82rem; font-weight: 500;
    cursor: pointer; display: flex; align-items: center; gap: 6px;
}
.btn-primary-custom {
    padding: 8px 20px; border-radius: 8px; border: none;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: #fff; font-size: 0.82rem; font-weight: 600;
    cursor: pointer; display: flex; align-items: center; gap: 6px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.25);
}

/* ── Tab bar ─────────────────────────────────────── */
.tab-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.2rem;
}
.tab-pills {
    display: flex; gap: 4px; background: #E2E8F0; border-radius: 10px; padding: 3px;
}
.tab-pill {
    padding: 8px 18px; border-radius: 8px; font-size: 0.82rem; font-weight: 500;
    color: #475569; cursor: pointer; transition: all 0.2s;
}
.tab-pill.active {
    background: #2563EB; color: #fff; box-shadow: 0 2px 6px rgba(37,99,235,0.25);
}

/* ── Buildings badge (hover reveals overlay) ─────── */
.buildings-badge-wrapper {
    position: relative; display: inline-block;
}
.buildings-badge {
    padding: 8px 18px; border-radius: 10px; border: 1px solid #E2E8F0;
    background: #fff; color: #334155; font-size: 0.82rem; font-weight: 600;
    display: flex; align-items: center; gap: 8px; cursor: pointer;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06); transition: all 0.2s;
}
.buildings-badge:hover {
    border-color: #2563EB; box-shadow: 0 2px 12px rgba(37,99,235,0.15);
}
.buildings-badge .dot {
    width: 8px; height: 8px; border-radius: 50%; background: #22C55E;
}
.hover-overlay {
    display: none; position: absolute; top: 110%; right: 0; z-index: 1000;
    background: #fff; border-radius: 16px; padding: 8px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.18); border: 1px solid #E2E8F0;
    min-width: 420px;
}
.hover-overlay img {
    border-radius: 12px; width: 100%; height: auto;
}
.buildings-badge-wrapper:hover .hover-overlay {
    display: block;
}

/* ── Image gallery — BIGGER images ───────────────── */
.img-gallery {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-bottom: 1.5rem;
}
.img-card {
    border-radius: 16px; overflow: hidden; position: relative;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); border: 1px solid #E2E8F0;
    background: #111827;
}
.img-card img {
    width: 100%; height: auto;
    display: block;
}
.img-card .img-label {
    position: absolute; bottom: 12px; left: 12px;
    background: rgba(0,0,0,0.55); backdrop-filter: blur(8px);
    color: #fff; padding: 6px 14px; border-radius: 8px;
    font-size: 0.75rem; font-weight: 500;
}

/* ── Warning banner (visible on light bg) ────────── */
.warning-banner {
    background: #FEF3C7; border: 1px solid #F59E0B;
    border-radius: 10px; padding: 10px 16px;
    font-size: 0.8rem; color: #92400E; font-weight: 500;
    margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}

/* ── Metric cards ────────────────────────────────── */
.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #fff; border-radius: 16px; padding: 20px 18px;
    border: 1px solid #E2E8F0; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
.metric-card .icon-box {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; margin-bottom: 14px;
}
.icon-blue   { background: #EFF6FF; color: #2563EB; }
.icon-indigo { background: #EEF2FF; color: #4F46E5; }
.icon-amber  { background: #FFFBEB; color: #D97706; }
.icon-green  { background: #F0FDF4; color: #16A34A; }
.metric-card .value {
    font-size: 1.75rem; font-weight: 800; color: #0F172A; line-height: 1.1;
}
.metric-card .label {
    font-size: 0.78rem; color: #64748B; margin-top: 4px; font-weight: 400;
}
.metric-card .sublabel {
    font-size: 0.68rem; color: #94A3B8; margin-top: 2px;
}

/* ── Annual banner (split: blue + green) ─────────── */
.annual-split {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0;
    border-radius: 18px; overflow: hidden;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3);
}
.annual-left {
    background: linear-gradient(135deg, #2563EB 0%, #1E40AF 60%, #1E3A8A 100%);
    padding: 28px 32px; color: #fff;
    display: flex; align-items: center; gap: 18px;
}
.annual-right {
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    padding: 28px 32px; color: #fff;
    display: flex; align-items: center; gap: 18px;
}
.annual-left .icon-circle, .annual-right .icon-circle {
    width: 52px; height: 52px; border-radius: 50%;
    background: rgba(255,255,255,0.18); display: flex;
    align-items: center; justify-content: center; font-size: 24px;
    flex-shrink: 0;
}
.annual-left .banner-label, .annual-right .banner-label {
    font-size: 0.82rem; color: rgba(255,255,255,0.85); font-weight: 400;
}
.annual-left .banner-value, .annual-right .banner-value {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
}
.annual-left .banner-sub, .annual-right .banner-sub {
    font-size: 0.78rem; color: rgba(255,255,255,0.65); margin-top: 2px;
}

/* ── Insight cards ───────────────────────────────── */
.insight-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
    margin-bottom: 1.5rem;
}
.insight-card {
    background: #fff; border-radius: 14px; padding: 18px;
    border: 1px solid #E2E8F0; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.insight-card .insight-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; margin-bottom: 10px;
}
.insight-card .insight-title {
    font-size: 0.85rem; font-weight: 600; color: #0F172A; margin-bottom: 4px;
}
.insight-card .insight-text {
    font-size: 0.75rem; color: #64748B; line-height: 1.45;
}

/* ── Section headers ─────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.05rem; font-weight: 700; color: #0F172A;
    margin: 1.5rem 0 1rem 0; padding-bottom: 8px;
    border-bottom: 2px solid #E2E8F0;
}
.section-header .sh-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: #EFF6FF; color: #2563EB;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}

/* ── Info grid ───────────────────────────────────── */
.info-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
    background: #fff; border-radius: 14px; padding: 20px;
    border: 1px solid #E2E8F0; margin-bottom: 1rem;
}
.info-item .info-label {
    font-size: 0.72rem; color: #94A3B8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;
}
.info-item .info-value {
    font-size: 0.92rem; color: #0F172A; font-weight: 600;
}

/* ── Confidence grid ─────────────────────────────── */
.conf-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
    background: #fff; border-radius: 14px; padding: 20px;
    border: 1px solid #E2E8F0; margin-bottom: 0.5rem;
}
.conf-item .conf-label {
    font-size: 0.72rem; color: #94A3B8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 4px;
}
.conf-item .conf-value {
    font-size: 1.35rem; color: #0F172A; font-weight: 700;
}

/* ── Market grid ─────────────────────────────────── */
.market-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px;
    margin-bottom: 0.5rem;
}
.market-card {
    background: #fff; border-radius: 14px; padding: 18px;
    border: 1px solid #E2E8F0;
}
.market-card .mk-label {
    font-size: 0.72rem; color: #94A3B8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 6px;
}
.market-card .mk-value {
    font-size: 1.4rem; color: #0F172A; font-weight: 700;
}

/* ── Confidence notes ────────────────────────────── */
.conf-notes {
    background: #F8FAFC; border-radius: 10px; padding: 14px 18px;
    border: 1px solid #E2E8F0; margin-top: 8px;
}
.conf-notes li {
    font-size: 0.78rem; color: #64748B; margin-bottom: 4px;
}

/* ── Responsive ──────────────────────────────────── */
@media (max-width: 768px) {
    .metric-grid, .insight-grid, .market-grid { grid-template-columns: 1fr 1fr; }
    .conf-grid { grid-template-columns: 1fr 1fr; }
    .img-gallery { grid-template-columns: 1fr; }
}

/* ── Streamlit overrides ─────────────────────────── */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetric"] { display: none !important; }
div.stTabs [data-baseweb="tab-list"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_to_base64(image_array: np.ndarray) -> str:
    """Convert (C,H,W) or (H,W,3) uint8 image to base64 PNG."""
    from PIL import Image
    rgb = prepare_display_rgb(image_array)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _mask_to_base64(mask: np.ndarray) -> str:
    """Convert (H,W) binary mask to a colored base64 PNG."""
    from PIL import Image
    # Create a colored mask: buildings = bright cyan, background = dark
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask > 0] = [0, 200, 255]   # cyan for buildings
    rgb[mask == 0] = [15, 23, 42]   # dark blue-grey background
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#111827")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _format_smart(value: float, unit: str) -> tuple:
    """Auto-scale values for display."""
    if unit == "kW" and value >= 1000:
        return f"{value / 1000:,.1f}", "MW"
    if unit == "kWh" and value >= 1_000_000:
        return f"{value / 1_000_000:,.1f}", "GWh"
    if unit == "kWh" and value >= 1000:
        return f"{value / 1000:,.1f}", "MWh"
    return f"{value:,.0f}", unit


def _compute_solar_savings(system_kw: float, annual_kwh: float, tariff: float) -> dict:
    """Comprehensive savings calculator for Indian residential rooftop solar.

    Uses real-world data for Pune / Maharashtra:
      - Installation cost : ~₹67,000 per kW (market average incl. installation)
      - PM Surya Ghar subsidy (central + state)
      - Annual electricity savings = generation × tariff
      - Payback period = net cost / annual savings
      - Lifetime benefit over 25 years

    Subsidy chart (PM Surya Ghar Muft Bijli Yojana):
    ┌──────────┬──────────────┬──────────────┬─────────────┐
    │ Capacity │ Central Govt │ State Govt   │ Total       │
    ├──────────┼──────────────┼──────────────┼─────────────┤
    │ 1 kW     │ ₹30,000      │ ₹15,000      │ ₹45,000     │
    │ 2 kW     │ ₹60,000      │ ₹30,000      │ ₹90,000     │
    │ 3 kW     │ ₹78,000      │ ₹30,000      │ ₹1,08,000   │
    │ 4–5 kW   │ ₹78,000      │ ₹30,000      │ ₹1,08,000   │
    └──────────┴──────────────┴──────────────┴─────────────┘
    """
    INSTALLATION_COST_PER_KW = 67_000   # ₹/kW (avg. market rate)
    SYSTEM_LIFESPAN_YEARS = 25
    # Typical monthly consumption for a middle-class Indian household
    AVG_MONTHLY_BILL_KWH = 250          # ~250 units/month baseline

    if system_kw <= 0 or annual_kwh <= 0:
        return {
            "central_subsidy": 0, "state_subsidy": 0, "total_subsidy": 0,
            "gross_cost": 0, "net_cost": 0,
            "annual_saving": 0, "payback_years": 0,
            "lifetime_saving": 0, "bill_reduction_pct": 0,
        }

    # ── Subsidy calculation ──
    capped_kw = min(system_kw, 5.0)     # subsidy caps at 5 kW
    if capped_kw <= 2:
        central = capped_kw * 30_000
    else:
        central = 78_000                # capped
    state = 15_000 if capped_kw < 1.5 else 30_000
    total_subsidy = central + state

    # ── Installation cost ──
    gross_cost = system_kw * INSTALLATION_COST_PER_KW
    net_cost = max(gross_cost - total_subsidy, 0)

    # ── Annual electricity savings ──
    annual_saving = annual_kwh * tariff

    # ── Payback period ──
    payback_years = net_cost / annual_saving if annual_saving > 0 else 0

    # ── Lifetime savings (over 25 years) ──
    lifetime_saving = (annual_saving * SYSTEM_LIFESPAN_YEARS) - net_cost

    # ── Bill reduction estimate ──
    annual_consumption = AVG_MONTHLY_BILL_KWH * 12
    bill_reduction_pct = min((annual_kwh / annual_consumption) * 100, 100) if annual_consumption > 0 else 0

    return {
        "central_subsidy": round(central),
        "state_subsidy": round(state),
        "total_subsidy": round(total_subsidy),
        "gross_cost": round(gross_cost),
        "net_cost": round(net_cost),
        "annual_saving": round(annual_saving, 2),
        "payback_years": round(payback_years, 1),
        "lifetime_saving": round(lifetime_saving),
        "bill_reduction_pct": round(bill_reduction_pct, 0),
    }


# ---------------------------------------------------------------------------
# Load config data (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_baselines():
    return load_yield_baselines()

@st.cache_data
def _load_policies():
    return load_policy_records()

@st.cache_data
def _load_market():
    return load_market_intelligence()


baselines = _load_baselines()
policy_records = _load_policies()
market_records = _load_market()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Settings")

data_mode = st.sidebar.radio(
    "Data Mode",
    ["Synthetic demo", "SpaceNet tile"],
    help="Synthetic mode generates fake buildings for a zero-data demo.",
)

st.sidebar.subheader("📍 Location")
location_names = sorted(set(
    list(baselines.keys()) + list(policy_records.keys()) + list(market_records.keys())
))
display_names = []
for k in location_names:
    rec = baselines.get(k) or policy_records.get(k) or market_records.get(k)
    display_names.append(rec.location_name if rec else k.title())
if not display_names:
    display_names = ["Default"]

selected_location_display = st.sidebar.selectbox(
    "City / Region", display_names, index=0,
)
selected_location_key = selected_location_display.lower()
use_baseline = st.sidebar.checkbox("Use location yield baseline", value=True)

st.sidebar.subheader("Solar Assumptions")
usability = st.sidebar.slider("Roof usability factor", 0.3, 0.9, 0.65)
panel_density = st.sidebar.slider("Panel power density (kW/m²)", 0.10, 0.25, 0.18)
perf_ratio = st.sidebar.slider("Performance ratio", 0.60, 0.95, 0.78)
monthly_gen = st.sidebar.slider("Monthly gen (kWh/kW)", 70.0, 150.0, 110.0)

config = SolarConfig(
    roof_usability_factor=usability,
    panel_power_density_kw_per_m2=panel_density,
    performance_ratio=perf_ratio,
    monthly_generation_kwh_per_kw=monthly_gen,
)

active_baseline = None
if use_baseline and selected_location_key in baselines:
    active_baseline = baselines[selected_location_key]
    config = apply_yield_baseline(config, active_baseline)

st.sidebar.subheader("Vectorization")
min_area = st.sidebar.number_input("Min polygon area (filter)", value=25.0, min_value=1.0)
simplify_tol = st.sidebar.slider("Simplify tolerance", 0.0, 5.0, 1.0)

st.sidebar.subheader("🔧 Debug & Overlays")
show_alignment_debug = st.sidebar.checkbox("Show alignment debug view", value=False)
load_policy_overlays = st.sidebar.checkbox("Load policy/market overlays", value=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

image = mask = transform = crs = None
warnings_list: list[str] = []

if data_mode == "Synthetic demo":
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0)
    n_buildings = st.sidebar.slider("Num buildings", 3, 20, 8)
    image, mask, transform, crs = generate_synthetic_tile(
        num_buildings=n_buildings, seed=int(seed)
    )
    warnings_list.append("Using synthetic data — areas computed with 0.3 m/pixel GSD in projected CRS (EPSG:32643).")
else:
    st.sidebar.subheader("Data Paths")
    data_dir = Path("data/raw")
    img_dir = data_dir / "images"
    if img_dir.exists():
        available = sorted(img_dir.glob("*.tif"))
        if available:
            selected = st.sidebar.selectbox(
                "Select tile", available, format_func=lambda p: p.name,
            )
            if selected:
                image, transform, crs = load_image(selected)
                mask_path = data_dir / "masks" / selected.name
                label_stem = selected.stem
                label_path = data_dir / "labels" / f"{label_stem}.geojson"
                if mask_path.exists():
                    mask, _, _ = load_or_create_mask(selected, mask_path)
                elif label_path.exists():
                    mask, _, _ = load_or_create_mask(selected, label_path)
        else:
            st.info("No .tif files found in `data/raw/images/`.")
    else:
        st.info("Create `data/raw/images/` and `data/raw/masks/` directories.")

    uploaded = st.sidebar.file_uploader("Or upload a GeoTIFF", type=["tif", "tiff"])
    if uploaded is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        image, transform, crs = load_image(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# TOP NAVBAR
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="top-nav">
    <div class="top-nav-brand">
        <div class="logo">☀️</div>
        <div>
            <h1>Solar Rooftop Analyzer</h1>
            <p>AI-powered solar potential assessment</p>
        </div>
    </div>
    <div class="top-nav-actions">
        <span class="btn-settings">⚙️ Settings</span>
        <span class="btn-primary-custom">📤 Upload Satellite Image</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE — store results in session_state so downloads don't reset
# ═══════════════════════════════════════════════════════════════════════════

if image is not None and mask is not None:
    run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run:
        use_pixel_coords = (crs is None)
        crs_units = check_crs_units(crs)
        if crs is None:
            warnings_list.append("No CRS detected — all coordinates are in pixel space.")

        with st.spinner("Analyzing rooftops..."):
            polygons = mask_to_polygons(
                mask, transform=transform, crs=crs,
                min_area=min_area, simplify_tolerance=simplify_tol,
                use_pixel_coords=use_pixel_coords,
            )

        if not polygons:
            st.error("No polygons found! Check mask threshold or min_area filter.")
        else:
            if use_pixel_coords or transform is None:
                effective_transform = from_bounds(
                    0, 0, mask.shape[1], mask.shape[0],
                    mask.shape[1], mask.shape[0],
                )
            else:
                effective_transform = transform

            if crs is not None:
                aligned, align_warnings = validate_polygon_raster_alignment(
                    (transform.c, transform.f + transform.e * mask.shape[0],
                     transform.c + transform.a * mask.shape[1], transform.f),
                    [p["geometry"] for p in polygons],
                )
                warnings_list.extend(align_warnings)

            align_debug = alignment_debug_info(
                effective_transform, mask.shape, polygons, crs
            )

            with st.spinner("Computing feasibility..."):
                polygons = enrich_polygons_with_feasibility(polygons, image=image, mask=mask)

            with st.spinner("Computing solar estimates..."):
                per_roof, aggregate = estimate_all_roofs(polygons, config)

            tile_conf = compute_tile_confidence(
                imagery_date=None, num_polygons=len(polygons),
                min_area_used=min_area,
                overlap_ratio=align_debug.get("overlap_ratio") if align_debug else None,
                alignment_warnings=[w for w in warnings_list if "overlap" in w.lower()],
                has_crs=(crs is not None),
            )

            with st.spinner("Rendering..."):
                # Overlay figure
                fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(8, 8))
                overlay_polygons(
                    image, polygons, ax=ax_overlay,
                    transform=effective_transform, title="",
                )
                overlay_b64 = _fig_to_base64(fig_overlay)

                # Raw image + mask as base64
                raw_b64 = _img_to_base64(image)
                mask_b64 = _mask_to_base64(mask)

                # Side-by-side for export
                fig_sbs = side_by_side(image, mask, polygons, transform=effective_transform)
                sbs_buf = io.BytesIO()
                fig_sbs.savefig(sbs_buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig_sbs)
                sbs_buf.seek(0)

                # GeoJSON
                from shapely.geometry import mapping
                features = []
                for i, poly in enumerate(polygons):
                    features.append({
                        "type": "Feature", "id": i,
                        "properties": {"id": i, "area_value": round(poly["area_value"], 2), "area_unit": poly["area_unit"]},
                        "geometry": mapping(poly["geometry"]),
                    })
                geojson_str = json.dumps({"type": "FeatureCollection", "features": features}, indent=2)

                # Report
                report = format_report(per_roof, aggregate, config)

                # Meta
                meta_sidecar = json.dumps({
                    "source_raster": None, "crs_epsg": crs.to_epsg() if crs else None,
                    "coordinates": "georeferenced" if crs else "pixel",
                    "num_polygons": len(polygons),
                }, indent=2)

                # Enriched CSV
                csv_buf = io.StringIO()
                fieldnames = [
                    "polygon_id", "roof_area", "roof_area_unit",
                    "usable_area", "estimated_system_kw",
                    "estimated_monthly_kwh", "estimated_annual_kwh",
                    "customer_type_proxy", "feasibility_confidence_score",
                    "shading_risk_label", "roof_plane_orientation",
                    "roof_plane_tilt", "heritage_flag",
                ]
                writer = csv.DictWriter(csv_buf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for i, r in enumerate(per_roof):
                    poly = polygons[i] if i < len(polygons) else {}
                    row = {**r}
                    row["customer_type_proxy"] = poly.get("customer_type_proxy", "")
                    row["feasibility_confidence_score"] = poly.get("feasibility_confidence_score", "")
                    row["shading_risk_label"] = poly.get("shading_risk_label", "")
                    row["roof_plane_orientation"] = poly.get("roof_plane_orientation", "")
                    row["roof_plane_tilt"] = poly.get("roof_plane_tilt", "")
                    row["heritage_flag"] = poly.get("heritage_or_restricted_zone_flag", "")
                    writer.writerow(row)
                csv_str = csv_buf.getvalue()

                # Tile summary
                tile_summary = json.dumps({
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "total_roofs": aggregate["num_roofs"],
                    "total_roof_area": aggregate["total_roof_area"],
                    "total_roof_area_unit": aggregate["total_roof_area_unit"],
                    "total_system_kw": aggregate["total_system_kw"],
                    "total_monthly_kwh": aggregate["total_monthly_kwh"],
                    "total_annual_kwh": aggregate["total_annual_kwh"],
                    "yield_baseline": {
                        "location": active_baseline.location_name if active_baseline else "Default",
                        "source": active_baseline.source_name if active_baseline else "Conservative estimate",
                        "annual_kwh_per_kw": active_baseline.annual_yield_kwh_per_kw if active_baseline else 1320,
                        "confidence": active_baseline.confidence if active_baseline else "Low",
                    },
                    "policy_summary": policy_to_dict(policy_records.get(selected_location_key)) if policy_records.get(selected_location_key) else None,
                    "market_summary": market_to_dict(market_records.get(selected_location_key)) if market_records.get(selected_location_key) else None,
                    "confidence": confidence_to_dict(tile_conf),
                    "disclaimers": [config.pre_assessment_disclaimer] + config.limitations,
                }, indent=2)

                # Snapshot
                snapshot = json.dumps({
                    "selected_location": selected_location_display,
                    "policy": policy_to_dict(policy_records.get(selected_location_key)) if policy_records.get(selected_location_key) else None,
                    "market": market_to_dict(market_records.get(selected_location_key)) if market_records.get(selected_location_key) else None,
                    "yield_baseline": {
                        "location": active_baseline.location_name,
                        "annual_kwh_per_kw": active_baseline.annual_yield_kwh_per_kw,
                        "monthly_kwh_per_kw": active_baseline.monthly_yield_kwh_per_kw,
                        "source": active_baseline.source_name,
                        "confidence": active_baseline.confidence,
                    } if active_baseline else None,
                }, indent=2)

            # ── Compute per-building savings using subsidy chart ──
            _pol = policy_records.get(selected_location_key)
            tariff = _pol.example_tariff_residential_per_kwh if (_pol and _pol.example_tariff_residential_per_kwh) else 0.10
            tariff_unit = _pol.example_tariff_unit if _pol else "USD"

            per_building_savings = []
            total_savings_electricity = 0
            total_subsidy = 0
            total_net_cost = 0
            total_gross_cost = 0
            total_lifetime = 0
            for i, r in enumerate(per_roof):
                sys_kw = r["estimated_system_kw"]
                annual_kwh = r["estimated_annual_kwh"]
                sv = _compute_solar_savings(sys_kw, annual_kwh, tariff)
                per_building_savings.append({
                    "building_id": i + 1,
                    "system_kw": round(sys_kw, 2),
                    "annual_gen_kwh": round(annual_kwh, 1),
                    "gross_cost": sv["gross_cost"],
                    "total_subsidy": sv["total_subsidy"],
                    "net_cost": sv["net_cost"],
                    "annual_saving": sv["annual_saving"],
                    "payback_years": sv["payback_years"],
                    "lifetime_saving": sv["lifetime_saving"],
                    "bill_reduction_pct": sv["bill_reduction_pct"],
                })
                total_savings_electricity += sv["annual_saving"]
                total_subsidy += sv["total_subsidy"]
                total_net_cost += sv["net_cost"]
                total_gross_cost += sv["gross_cost"]
                total_lifetime += sv["lifetime_saving"]

            # ── Store everything in session_state ──
            st.session_state["results"] = {
                "overlay_b64": overlay_b64,
                "raw_b64": raw_b64,
                "mask_b64": mask_b64,
                "num_roofs": aggregate["num_roofs"],
                "total_area": aggregate["total_roof_area"],
                "area_unit": aggregate["total_roof_area_unit"],
                "total_kw": aggregate["total_system_kw"],
                "total_monthly": aggregate["total_monthly_kwh"],
                "total_annual": aggregate["total_annual_kwh"],
                "total_savings_electricity": round(total_savings_electricity, 2),
                "total_subsidy": round(total_subsidy),
                "total_net_cost": round(total_net_cost),
                "total_gross_cost": round(total_gross_cost),
                "total_lifetime_saving": round(total_lifetime),
                "per_building_savings": per_building_savings,
                "tariff": tariff,
                "tariff_unit": tariff_unit,
                "sfx": aggregate.get("label_suffix", ""),
                "per_roof": per_roof,
                "tile_conf": tile_conf,
                "align_debug": align_debug,
                "warnings_list": warnings_list,
                "polygons": polygons,
                "effective_transform": effective_transform,
                # Pre-computed export data (bytes/strings)
                "sbs_png": sbs_buf.getvalue(),
                "geojson_str": geojson_str,
                "meta_sidecar": meta_sidecar,
                "report": report,
                "csv_str": csv_str,
                "tile_summary": tile_summary,
                "snapshot": snapshot,
            }


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY — reads from session_state so downloads don't reset
# ═══════════════════════════════════════════════════════════════════════════

if "results" in st.session_state:
    R = st.session_state["results"]

    # Guard against stale session state from older app version
    if "per_building_savings" not in R:
        st.warning("Session data is outdated. Please click **Run Pipeline** again.")
        del st.session_state["results"]
        st.stop()

    num_roofs = R["num_roofs"]
    total_area = R["total_area"]
    area_unit = R["area_unit"]
    total_kw = R["total_kw"]
    total_monthly = R["total_monthly"]
    total_annual = R["total_annual"]
    overlay_b64 = R["overlay_b64"]
    raw_b64 = R["raw_b64"]
    mask_b64 = R["mask_b64"]
    tile_conf = R["tile_conf"]
    align_debug = R["align_debug"]
    warn_list = R["warnings_list"]

    cap_val, cap_unit = _format_smart(total_kw, "kW")
    month_val, month_unit = _format_smart(total_monthly, "kWh")
    annual_val, annual_unit = _format_smart(total_annual, "kWh")

    # ── Warnings — styled, right below navbar ──
    for w_msg in warn_list:
        st.markdown(f'<div class="warning-banner">⚠️ {w_msg}</div>', unsafe_allow_html=True)

    # ── TAB BAR + BUILDINGS BADGE ──
    st.markdown(f"""
    <div class="tab-bar">
        <div class="tab-pills">
            <div class="tab-pill">Satellite</div>
            <div class="tab-pill">Building Detection</div>
            <div class="tab-pill active">Solar Analysis</div>
        </div>
        <div class="buildings-badge-wrapper">
            <div class="buildings-badge">
                <span class="dot"></span>
                🏢 {num_roofs} Buildings Detected
            </div>
            <div class="hover-overlay">
                <img src="data:image/png;base64,{overlay_b64}" alt="Roof Polygon Overlay"/>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── IMAGE GALLERY: Left = Original, Right = Mask ──
    st.markdown(f"""
    <div class="img-gallery">
        <div class="img-card">
            <img src="data:image/png;base64,{raw_b64}" alt="Satellite Image"/>
            <div class="img-label">🛰️ Original Satellite Image</div>
        </div>
        <div class="img-card">
            <img src="data:image/png;base64,{mask_b64}" alt="Segmentation Mask"/>
            <div class="img-label">🎯 Building Segmentation Mask</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 METRIC CARDS ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="icon-box icon-blue">🏢</div>
            <div class="value">{num_roofs}</div>
            <div class="label">Total Rooftops</div>
            <div class="sublabel">buildings detected</div>
        </div>
        <div class="metric-card">
            <div class="icon-box icon-indigo">📐</div>
            <div class="value">{total_area:,.0f}</div>
            <div class="label">Roof Area</div>
            <div class="sublabel">{area_unit} analyzed</div>
        </div>
        <div class="metric-card">
            <div class="icon-box icon-amber">⚡</div>
            <div class="value">{cap_val} {cap_unit}</div>
            <div class="label">System Capacity</div>
            <div class="sublabel">total potential</div>
        </div>
        <div class="metric-card">
            <div class="icon-box icon-green">📊</div>
            <div class="value">{month_val} {month_unit}</div>
            <div class="label">Monthly Output</div>
            <div class="sublabel">estimated generation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ANNUAL BANNER (blue) + MONEY SAVED (green) ──
    baseline_note = ""
    if active_baseline:
        baseline_note = f"Based on {active_baseline.location_name} yield data ({active_baseline.source_name})"
    else:
        baseline_note = "Potential yearly solar energy output"

    total_savings = R["total_savings_electricity"]
    total_sub = R["total_subsidy"]
    tariff_unit = R["tariff_unit"]
    tariff_val = R["tariff"]

    # Average per-resident values
    avg_savings = total_savings / num_roofs if num_roofs else 0
    avg_net_cost = R["total_net_cost"] / num_roofs if num_roofs else 0
    avg_payback = avg_net_cost / avg_savings if avg_savings > 0 else 0
    per_bldg = R["per_building_savings"]
    avg_bill_reduction = sum(b["bill_reduction_pct"] for b in per_bldg) / len(per_bldg) if per_bldg else 0
    avg_lifetime = R["total_lifetime_saving"] / num_roofs if num_roofs else 0

    if avg_savings >= 1_00_000:
        avg_sav_str = f"{avg_savings / 1_00_000:,.2f}L"
    elif avg_savings >= 1_000:
        avg_sav_str = f"{avg_savings / 1_000:,.1f}K"
    else:
        avg_sav_str = f"{avg_savings:,.0f}"

    st.markdown(f"""
    <div class="annual-split">
        <div class="annual-left">
            <div class="icon-circle">⚡</div>
            <div>
                <div class="banner-label">Estimated Annual Generation</div>
                <div class="banner-value">{annual_val} {annual_unit}</div>
                <div class="banner-sub">{baseline_note}</div>
            </div>
        </div>
        <div class="annual-right">
            <div class="icon-circle">💰</div>
            <div>
                <div class="banner-label">Avg. Money Saved per annum per resident</div>
                <div class="banner-value">₹{avg_sav_str}</div>
                <div class="banner-sub">Payback: ~{avg_payback:.1f} yrs · Bill reduction: ~{avg_bill_reduction:.0f}% · {num_roofs} buildings</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ── KEY INSIGHTS ──
    avg_kw = total_kw / num_roofs if num_roofs else 0
    avg_monthly = total_monthly / num_roofs if num_roofs else 0
    avg_gross = R["total_gross_cost"] / num_roofs if num_roofs else 0
    avg_sub = total_sub / num_roofs if num_roofs else 0

    st.markdown(f"""
    <div style="font-size:1.05rem;font-weight:700;color:#0F172A;margin-bottom:10px;">Key Insights</div>
    <div class="insight-grid">
        <div class="insight-card">
            <div class="insight-icon icon-green">🟢</div>
            <div class="insight-title">High Potential Area</div>
            <div class="insight-text">This area shows strong solar potential with {num_roofs} suitable rooftops identified.</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon icon-blue">⚡</div>
            <div class="insight-title">Average System Size</div>
            <div class="insight-text">Typical rooftop could fit a {avg_kw:,.1f} kW system, generating ~{avg_monthly:,.0f} kWh monthly.</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon icon-amber">💰</div>
            <div class="insight-title">Investment &amp; Payback</div>
            <div class="insight-text">Avg. cost ₹{avg_gross:,.0f} → ₹{avg_net_cost:,.0f} after subsidy (₹{avg_sub:,.0f}). Payback in ~{avg_payback:.1f} years.</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon icon-green">🌿</div>
            <div class="insight-title">25-Year Lifetime Savings</div>
            <div class="insight-text">Each household saves avg. ₹{avg_lifetime:,.0f} over system lifetime (25 yrs), with ~{avg_bill_reduction:.0f}% bill reduction.</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon icon-amber">⚠️</div>
            <div class="insight-title">Site Survey Required</div>
            <div class="insight-text">Professional assessment needed for roof condition, shading, and structural capacity.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── POLICY & INCENTIVES ──
    st.markdown("""<div class="section-header"><div class="sh-icon">📋</div>Policy & Incentives</div>""", unsafe_allow_html=True)
    policy_rec = policy_records.get(selected_location_key) if load_policy_overlays else None
    if policy_rec:
        tariff_str = f"{policy_rec.example_tariff_residential_per_kwh} {policy_rec.example_tariff_unit}/kWh" if policy_rec.example_tariff_residential_per_kwh is not None else "N/A"
        subsidy_str = "✅ Yes" if policy_rec.subsidy_available else "❌ No"
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-item"><div class="info-label">Compensation</div><div class="info-value">{policy_rec.compensation_regime}</div></div>
            <div class="info-item"><div class="info-label">Example Tariff</div><div class="info-value">{tariff_str}</div></div>
            <div class="info-item"><div class="info-label">Subsidy</div><div class="info-value">{subsidy_str}</div></div>
            <div class="info-item"><div class="info-label">Source</div><div class="info-value">{policy_rec.policy_source} ({policy_rec.policy_date})</div></div>
        </div>
        """, unsafe_allow_html=True)
        if policy_rec.subsidy_notes:
            st.markdown(f'<div style="font-size:0.82rem;color:#475569;margin:-0.5rem 0 0.5rem 0;">{policy_rec.subsidy_notes}</div>', unsafe_allow_html=True)
        if policy_rec.notes:
            st.markdown(f'<div style="font-size:0.78rem;color:#64748B;margin-bottom:0.5rem;">{policy_rec.notes}</div>', unsafe_allow_html=True)
    else:
        st.caption("No policy data loaded for this location.")

    # ── MARKET INTELLIGENCE ──
    st.markdown("""<div class="section-header"><div class="sh-icon">📊</div>Market Intelligence</div>""", unsafe_allow_html=True)
    market_rec = market_records.get(selected_location_key) if load_policy_overlays else None
    if market_rec:
        ad_score = f"{market_rec.adoption_density_score:.0%}" if market_rec.adoption_density_score is not None else "N/A"
        am_score = f"{market_rec.adoption_momentum_score:.0%}" if market_rec.adoption_momentum_score is not None else "N/A"
        pv_dens = f"{market_rec.installed_pv_density_per_km2:.1f} kW/km²" if market_rec.installed_pv_density_per_km2 is not None else "N/A"
        st.markdown(f"""
        <div class="market-grid">
            <div class="market-card"><div class="mk-label">Adoption Density</div><div class="mk-value">{ad_score}</div></div>
            <div class="market-card"><div class="mk-label">Market Maturity</div><div class="mk-value">{market_rec.market_maturity_segment}</div></div>
            <div class="market-card"><div class="mk-label">PV Density</div><div class="mk-value">{pv_dens}</div></div>
        </div>
        <div class="market-grid" style="margin-top:-6px;">
            <div class="market-card"><div class="mk-label">Adoption Momentum</div><div class="mk-value">{am_score}</div></div>
            <div class="market-card"><div class="mk-label">Customer Mix</div><div class="mk-value">{market_rec.customer_mix_proxy}</div></div>
            <div class="market-card"><div class="mk-label">Programme</div><div class="mk-value" style="font-size:0.9rem;">{market_rec.program_eligibility_tag or 'N/A'}</div></div>
        </div>
        """, unsafe_allow_html=True)
        if market_rec.notes:
            st.markdown(f'<div style="font-size:0.78rem;color:#64748B;margin-top:4px;">{market_rec.notes}</div>', unsafe_allow_html=True)
    else:
        st.warning("No market intelligence layer loaded for this tile/city.")

    # ── DATA CONFIDENCE / QUALITY ──
    st.markdown("""<div class="section-header"><div class="sh-icon">🛡️</div>Data Confidence / Quality</div>""", unsafe_allow_html=True)
    seg_label = f"{tile_conf.segmentation_confidence:.1%}" if tile_conf.segmentation_confidence else "N/A"
    vec_label = f"{tile_conf.vectorization_confidence:.1%}"
    st.markdown(f"""
    <div class="conf-grid">
        <div class="conf-item"><div class="conf-label">Overall Confidence</div><div class="conf-value">{tile_conf.overall_confidence_score:.0f}/100</div></div>
        <div class="conf-item"><div class="conf-label">Segmentation Confidence</div><div class="conf-value">{seg_label}</div></div>
        <div class="conf-item"><div class="conf-label">Data Recency</div><div class="conf-value">{tile_conf.data_recency_label}</div></div>
        <div class="conf-item"><div class="conf-label">Vectorization Confidence</div><div class="conf-value">{vec_label}</div></div>
    </div>
    """, unsafe_allow_html=True)
    if tile_conf.uncertainty_notes:
        notes_html = "".join(f"<li>{n}</li>" for n in tile_conf.uncertainty_notes)
        st.markdown(f'<div class="conf-notes"><ul style="margin:0;padding-left:18px;">{notes_html}</ul></div>', unsafe_allow_html=True)

    # ── ALIGNMENT DEBUG ──
    if show_alignment_debug and align_debug:
        with st.expander("🔍 Alignment Debug View", expanded=True):
            col_dbg1, col_dbg2 = st.columns(2)
            with col_dbg1:
                st.markdown("**Raster bounds:** `{}`".format(
                    tuple(round(v, 4) for v in align_debug["raster_bounds"])
                ))
                st.markdown("**Polygon bounds:** `{}`".format(
                    tuple(round(v, 4) for v in align_debug["polygon_bounds"])
                    if align_debug["polygon_bounds"] else "N/A"
                ))
                st.markdown(f"**Overlap ratio:** {align_debug['overlap_ratio']:.2%}")
                st.markdown(f"**Coord space:** {align_debug['coordinate_space']}")
            with col_dbg2:
                for note in align_debug.get("notes", []):
                    st.info(note)
            dbg_fig = alignment_debug_figure(
                image, R["polygons"], transform=R["effective_transform"]
            )
            st.pyplot(dbg_fig)
            plt.close(dbg_fig)

    # ── DOWNLOADS (all use pre-computed data — no recomputation on click) ──
    st.markdown("""<div class="section-header"><div class="sh-icon">📥</div>Downloads</div>""", unsafe_allow_html=True)

    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
    with col_dl1:
        st.download_button("📸 Overlay PNG", data=R["sbs_png"], file_name="solar_overlay.png", mime="image/png")
    with col_dl2:
        st.download_button("📍 GeoJSON", data=R["geojson_str"], file_name="roof_polygons.geojson", mime="application/json")
    with col_dl3:
        st.download_button("📄 Meta JSON", data=R["meta_sidecar"], file_name="footprints.meta.json", mime="application/json")
    with col_dl4:
        st.download_button("📝 Report (TXT)", data=R["report"], file_name="solar_report.txt", mime="text/plain")

    st.caption("**Enriched exports:**")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.download_button("📊 Enriched CSV", data=R["csv_str"], file_name="roofs_enriched.csv", mime="text/csv")
    with col_e2:
        st.download_button("📋 Tile Summary", data=R["tile_summary"], file_name="tile_summary.json", mime="application/json")
    with col_e3:
        st.download_button("📸 Policy/Market", data=R["snapshot"], file_name="policy_market_snapshot.json", mime="application/json")
    with col_e4:
        if show_alignment_debug and align_debug:
            debug_log = json.dumps({
                "alignment_debug": {k: (list(v) if isinstance(v, tuple) else v) for k, v in align_debug.items()},
                "confidence": confidence_to_dict(tile_conf),
                "warnings": warn_list,
            }, indent=2)
            st.download_button("🔍 Debug Log", data=debug_log, file_name="debug_log.json", mime="application/json")
        else:
            st.caption("Enable debug for log.")

    # ── Disclaimer ──
    st.markdown(f'<div style="margin-top:1rem;padding:14px 18px;background:#FEF3C7;border:1px solid #F59E0B;border-radius:10px;font-size:0.78rem;color:#92400E;">{config.pre_assessment_disclaimer}</div>', unsafe_allow_html=True)

elif image is not None and mask is None:
    st.info("Image loaded but no mask available. Provide a mask/label file or switch to Synthetic mode.")
else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:48px;margin-bottom:16px;">☀️</div>
        <h2 style="color:#0F172A;font-weight:700;">Ready to Analyze</h2>
        <p style="color:#64748B;max-width:480px;margin:0 auto;">
            Open the sidebar (☰) to configure settings, then click <strong>Run Pipeline</strong>
            to start analyzing rooftop solar potential.
        </p>
    </div>
    """, unsafe_allow_html=True)
