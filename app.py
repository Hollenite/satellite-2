"""
Streamlit demo app — AI-Assisted Rooftop Solar Pre-Assessment.

Run with:
    cd d:\\PROJECTS\\AIML-Hackathon-21feb\\satellite
    streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

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
    page_title="☀️ Solar Rooftop Pre-Assessment",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ AI-Assisted Rooftop Solar Pre-Assessment")
st.caption(
    "Upload a satellite tile or use synthetic data to estimate rooftop solar potential. "
    "This is a **pre-assessment tool** — not a substitute for professional site survey."
)


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
# Sidebar — configuration
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Configuration")

data_mode = st.sidebar.radio(
    "Data Mode",
    ["Synthetic demo", "SpaceNet tile"],
    help="Synthetic mode generates fake buildings for a zero-data demo.",
)

# --- Location selector ---
st.sidebar.subheader("📍 Location")
location_names = sorted(set(
    list(baselines.keys()) + list(policy_records.keys()) + list(market_records.keys())
))
# Prettify for display
display_names = []
for k in location_names:
    rec = baselines.get(k) or policy_records.get(k) or market_records.get(k)
    display_names.append(rec.location_name if rec else k.title())

if not display_names:
    display_names = ["Default"]

selected_location_display = st.sidebar.selectbox(
    "City / Region",
    display_names,
    index=0,
    help="Select a location for yield baselines and policy context.",
)
selected_location_key = selected_location_display.lower()

use_baseline = st.sidebar.checkbox(
    "Use location yield baseline",
    value=True,
    help="Override monthly/annual generation with location-specific data.",
)

# --- Solar assumptions ---
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

# Apply yield baseline if selected
active_baseline = None
if use_baseline and selected_location_key in baselines:
    active_baseline = baselines[selected_location_key]
    config = apply_yield_baseline(config, active_baseline)

st.sidebar.subheader("Vectorization")
min_area = st.sidebar.number_input("Min polygon area (filter)", value=25.0, min_value=1.0)
simplify_tol = st.sidebar.slider("Simplify tolerance", 0.0, 5.0, 1.0)

# --- Debug / overlay toggles ---
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
    warnings_list.append("⚠️ Using synthetic data — no real CRS. Area estimates are pixel-based.")

else:
    # SpaceNet tile mode
    st.sidebar.subheader("Data Paths")
    data_dir = Path("data/raw")

    # Auto-discover tiles
    img_dir = data_dir / "images"
    if img_dir.exists():
        available = sorted(img_dir.glob("*.tif"))
        if available:
            selected = st.sidebar.selectbox(
                "Select tile",
                available,
                format_func=lambda p: p.name,
            )
            if selected:
                image, transform, crs = load_image(selected)

                # Look for matching mask/label
                mask_path = data_dir / "masks" / selected.name
                label_stem = selected.stem
                label_path = data_dir / "labels" / f"{label_stem}.geojson"

                if mask_path.exists():
                    mask, _, _ = load_or_create_mask(selected, mask_path)
                elif label_path.exists():
                    mask, _, _ = load_or_create_mask(selected, label_path)
                else:
                    st.warning(
                        f"No mask/label found for {selected.name}. "
                        f"Looking in masks/ or labels/ directories."
                    )
        else:
            st.info("No .tif files found in `data/raw/images/`. Place SpaceNet tiles there.")
    else:
        st.info(
            "Create `data/raw/images/` and `data/raw/masks/` (or `labels/`) directories, "
            "then add SpaceNet tiles."
        )

    # Upload fallback
    uploaded = st.sidebar.file_uploader("Or upload a GeoTIFF", type=["tif", "tiff"])
    if uploaded is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        image, transform, crs = load_image(tmp_path)
        st.sidebar.info("Uploaded image loaded. Mask will need to be provided separately or use GT mode.")


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if image is not None and mask is not None:
    run = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run:
        # Determine if polygons will be in pixel-coord mode
        use_pixel_coords = (crs is None)

        # CRS check
        crs_units = check_crs_units(crs)
        if crs is None:
            warnings_list.append("⚠️ No CRS — all coordinates are in pixel space.")

        # Vectorize
        with st.spinner("Vectorizing mask → polygons..."):
            polygons = mask_to_polygons(
                mask,
                transform=transform,
                crs=crs,
                min_area=min_area,
                simplify_tolerance=simplify_tol,
                use_pixel_coords=use_pixel_coords,
            )

        if not polygons:
            st.error("No polygons found! Check mask threshold or min_area filter.")
        else:
            # ---------------------------------------------------------
            # CRITICAL: Compute the EFFECTIVE transform that was actually
            # used inside mask_to_polygons to create the polygon coords.
            # This must match what we pass to geo_to_pixel_coords() in viz.
            # ---------------------------------------------------------
            if use_pixel_coords or transform is None:
                effective_transform = from_bounds(
                    0, 0,
                    mask.shape[1], mask.shape[0],
                    mask.shape[1], mask.shape[0],
                )
            else:
                effective_transform = transform

            # Alignment check
            if crs is not None:
                aligned, align_warnings = validate_polygon_raster_alignment(
                    (transform.c, transform.f + transform.e * mask.shape[0],
                     transform.c + transform.a * mask.shape[1], transform.f),
                    [p["geometry"] for p in polygons],
                )
                warnings_list.extend(align_warnings)

            # Alignment debug info
            align_debug = alignment_debug_info(
                effective_transform, mask.shape, polygons, crs
            )

            # Enrich with feasibility (Phase 4)
            with st.spinner("Computing feasibility..."):
                polygons = enrich_polygons_with_feasibility(polygons, image=image, mask=mask)

            # Solar estimation
            with st.spinner("Computing solar estimates..."):
                per_roof, aggregate = estimate_all_roofs(polygons, config)

            # Compute confidence (Phase 6)
            tile_conf = compute_tile_confidence(
                imagery_date=None,
                num_polygons=len(polygons),
                min_area_used=min_area,
                overlap_ratio=align_debug.get("overlap_ratio") if align_debug else None,
                alignment_warnings=[w for w in warnings_list if "overlap" in w.lower()],
                has_crs=(crs is not None),
            )

            # ============================================================
            # Visualization — pass EFFECTIVE transform for correct overlay
            # ============================================================
            with st.spinner("Generating visualizations..."):
                fig = side_by_side(
                    image, mask, polygons,
                    transform=effective_transform,
                )

            # === Display ===
            st.subheader("📊 Results")

            # Warnings panel
            if warnings_list:
                for w_msg in warnings_list:
                    st.warning(w_msg)

            # Main figure
            st.pyplot(fig)
            plt.close(fig)

            # Alignment debug panel (Phase 1)
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
                        if align_debug["crs_epsg"]:
                            st.markdown(f"**CRS EPSG:** {align_debug['crs_epsg']}")
                    with col_dbg2:
                        for note in align_debug.get("notes", []):
                            st.info(note)

                    # Debug figure
                    dbg_fig = alignment_debug_figure(
                        image, polygons, transform=effective_transform
                    )
                    st.pyplot(dbg_fig)
                    plt.close(dbg_fig)

            # Metrics card
            st.subheader("☀️ Solar Estimate Summary")
            sfx = aggregate.get("label_suffix", "")
            cols = st.columns(4)
            cols[0].metric("Roofs Detected", aggregate["num_roofs"])
            cols[1].metric(
                f"Total Roof Area ({aggregate['total_roof_area_unit']})",
                f"{aggregate['total_roof_area']:,.1f}{sfx}",
            )
            cols[2].metric("System Capacity", f"{aggregate['total_system_kw']:,.2f} kW{sfx}")
            cols[3].metric("Monthly Generation", f"{aggregate['total_monthly_kwh']:,.0f} kWh{sfx}")

            st.metric("Estimated Annual Generation", f"{aggregate['total_annual_kwh']:,.0f} kWh{sfx}")

            # Yield source info
            if active_baseline:
                st.info(
                    f"📍 **Yield baseline:** {active_baseline.location_name} "
                    f"({active_baseline.region}) — "
                    f"{active_baseline.annual_yield_kwh_per_kw} kWh/kW/yr | "
                    f"Source: {active_baseline.source_name} | "
                    f"Confidence: {active_baseline.confidence}\n\n"
                    f"_{active_baseline.source_note}_"
                )
            else:
                st.caption("Using default yield assumptions (no location baseline selected).")

            # Per-roof table with feasibility
            with st.expander("📋 Per-Roof Details"):
                for r in per_roof:
                    pid = r['polygon_id']
                    # Find matching polygon for feasibility data
                    poly = polygons[pid] if pid < len(polygons) else {}
                    ctype = poly.get("customer_type_proxy", "—")
                    fconf = poly.get("feasibility_confidence_score", "—")

                    st.text(
                        f"Roof #{pid:>3}: "
                        f"area={r['roof_area']:>8.1f} {r['roof_area_unit']}, "
                        f"usable={r['usable_area']:>8.1f}, "
                        f"kW={r['estimated_system_kw']:>6.2f}, "
                        f"kWh/mo={r['estimated_monthly_kwh']:>8.1f}  "
                        f"[{ctype} | feasibility:{fconf}%]"
                    )

            # Assumptions
            with st.expander("🔧 Assumptions Used"):
                st.markdown(f"**Label:** {config.assumptions_label}")
                st.markdown(f"- Roof usability: {config.roof_usability_factor}")
                st.markdown(f"- Panel density: {config.panel_power_density_kw_per_m2} kW/m²")
                st.markdown(f"- Performance ratio: {config.performance_ratio}")
                st.markdown(f"- Monthly gen factor: {config.monthly_generation_kwh_per_kw} kWh/kW")

            # Limitations
            with st.expander("⚠️ Confidence / Limitations"):
                for lim in config.limitations:
                    st.markdown(f"- {lim}")

            st.info(config.pre_assessment_disclaimer)

            # ============================================
            # NEW PANELS (Phases 3, 5, 6)
            # ============================================

            # --- Policy & Incentives (Phase 3) ---
            st.subheader("📋 Policy & Incentives")
            policy_rec = policy_records.get(selected_location_key) if load_policy_overlays else None
            if policy_rec:
                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown(f"**Compensation:** {policy_rec.compensation_regime}")
                    st.markdown(f"**Subsidy:** {'✅ Yes' if policy_rec.subsidy_available else '❌ No'}")
                    if policy_rec.subsidy_notes:
                        st.markdown(f"_{policy_rec.subsidy_notes}_")
                with pc2:
                    if policy_rec.example_tariff_residential_per_kwh is not None:
                        st.markdown(
                            f"**Example tariff:** "
                            f"{policy_rec.example_tariff_residential_per_kwh} "
                            f"{policy_rec.example_tariff_unit}/kWh"
                        )
                    st.markdown(f"**Source:** {policy_rec.policy_source} ({policy_rec.policy_date})")
                    if policy_rec.notes:
                        st.caption(policy_rec.notes)
            else:
                st.caption("No policy data loaded for this location.")

            # --- Market Intelligence (Phase 5) ---
            st.subheader("📊 Market Intelligence")
            market_rec = market_records.get(selected_location_key) if load_policy_overlays else None
            if market_rec:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    if market_rec.adoption_density_score is not None:
                        st.metric("Adoption Density", f"{market_rec.adoption_density_score:.0%}")
                    if market_rec.adoption_momentum_score is not None:
                        st.metric("Adoption Momentum", f"{market_rec.adoption_momentum_score:.0%}")
                with mc2:
                    st.metric("Market Maturity", market_rec.market_maturity_segment)
                    st.metric("Customer Mix", market_rec.customer_mix_proxy)
                with mc3:
                    if market_rec.installed_pv_density_per_km2 is not None:
                        st.metric("PV Density", f"{market_rec.installed_pv_density_per_km2:.1f} kW/km²")
                    if market_rec.program_eligibility_tag:
                        st.markdown(f"**Programme:** {market_rec.program_eligibility_tag}")
                if market_rec.notes:
                    st.caption(market_rec.notes)
            else:
                st.warning("No market intelligence layer loaded for this tile/city.")

            # --- Data Confidence / Quality (Phase 6) ---
            st.subheader("🛡️ Data Confidence / Quality")
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                st.metric("Overall Confidence", f"{tile_conf.overall_confidence_score:.0f}/100")
                st.metric("Data Recency", tile_conf.data_recency_label)
            with qc2:
                seg_label = f"{tile_conf.segmentation_confidence:.1%}" if tile_conf.segmentation_confidence else "N/A"
                st.metric("Segmentation Confidence", seg_label)
                st.metric("Vectorization Confidence", f"{tile_conf.vectorization_confidence:.1%}")
            with qc3:
                if tile_conf.uncertainty_notes:
                    for note in tile_conf.uncertainty_notes:
                        st.caption(f"• {note}")

            # === Exports ===
            st.subheader("📥 Downloads")

            # --- Row 1: Original exports (unchanged) ---
            col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)

            # Overlay PNG
            with col_dl1:
                buf = io.BytesIO()
                fig2 = side_by_side(
                    image, mask, polygons,
                    transform=effective_transform,
                )
                fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig2)
                buf.seek(0)
                st.download_button(
                    "⬇️ Overlay PNG",
                    data=buf,
                    file_name="solar_overlay.png",
                    mime="image/png",
                )

            # GeoJSON
            with col_dl2:
                features = []
                for i, poly in enumerate(polygons):
                    from shapely.geometry import mapping
                    features.append({
                        "type": "Feature",
                        "id": i,
                        "properties": {
                            "id": i,
                            "area_value": round(poly["area_value"], 2),
                            "area_unit": poly["area_unit"],
                        },
                        "geometry": mapping(poly["geometry"]),
                    })
                geojson_str = json.dumps({
                    "type": "FeatureCollection",
                    "features": features,
                }, indent=2)
                st.download_button(
                    "⬇️ GeoJSON",
                    data=geojson_str,
                    file_name="roof_polygons.geojson",
                    mime="application/json",
                )

            # Meta JSON sidecar
            with col_dl3:
                meta_sidecar = json.dumps({
                    "source_raster": None,
                    "crs_epsg": crs.to_epsg() if crs else None,
                    "coordinates": "georeferenced" if crs else "pixel",
                    "num_polygons": len(polygons),
                    "warning": (
                        "GeoJSON consumers may assume WGS84. "
                        "Use this metadata sidecar to interpret coordinates."
                    ),
                }, indent=2)
                st.download_button(
                    "⬇️ Meta JSON",
                    data=meta_sidecar,
                    file_name="footprints.meta.json",
                    mime="application/json",
                )

            # Report
            with col_dl4:
                report = format_report(per_roof, aggregate, config)
                st.download_button(
                    "⬇️ Report (TXT)",
                    data=report,
                    file_name="solar_report.txt",
                    mime="text/plain",
                )

            # --- Row 2: New enriched exports (Phase 7) ---
            st.caption("**Enriched exports:**")
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)

            # Enriched roof CSV
            with col_e1:
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
                st.download_button(
                    "⬇️ Enriched CSV",
                    data=csv_buf.getvalue(),
                    file_name="roofs_enriched.csv",
                    mime="text/csv",
                )

            # Tile summary JSON
            with col_e2:
                tile_summary = {
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
                    "policy_summary": policy_to_dict(policy_rec) if policy_rec else None,
                    "market_summary": market_to_dict(market_rec) if market_rec else None,
                    "confidence": confidence_to_dict(tile_conf),
                    "disclaimers": [config.pre_assessment_disclaimer] + config.limitations,
                }
                st.download_button(
                    "⬇️ Tile Summary JSON",
                    data=json.dumps(tile_summary, indent=2),
                    file_name="tile_summary.json",
                    mime="application/json",
                )

            # Policy/Market snapshot JSON
            with col_e3:
                snapshot = {
                    "selected_location": selected_location_display,
                    "policy": policy_to_dict(policy_rec) if policy_rec else None,
                    "market": market_to_dict(market_rec) if market_rec else None,
                    "yield_baseline": {
                        "location": active_baseline.location_name,
                        "annual_kwh_per_kw": active_baseline.annual_yield_kwh_per_kw,
                        "monthly_kwh_per_kw": active_baseline.monthly_yield_kwh_per_kw,
                        "source": active_baseline.source_name,
                        "confidence": active_baseline.confidence,
                    } if active_baseline else None,
                }
                st.download_button(
                    "⬇️ Policy/Market Snapshot",
                    data=json.dumps(snapshot, indent=2),
                    file_name="policy_market_snapshot.json",
                    mime="application/json",
                )

            # Debug log (if debug mode)
            with col_e4:
                if show_alignment_debug and align_debug:
                    debug_log = json.dumps({
                        "alignment_debug": {
                            k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in align_debug.items()
                        },
                        "confidence": confidence_to_dict(tile_conf),
                        "warnings": warnings_list,
                    }, indent=2)
                    st.download_button(
                        "⬇️ Debug Log",
                        data=debug_log,
                        file_name="alignment_debug_log.json",
                        mime="application/json",
                    )
                else:
                    st.caption("Enable debug view for debug log export.")

elif image is not None and mask is None:
    st.info("Image loaded but no mask available. Provide a mask/label file or switch to Synthetic mode.")

else:
    st.info("Select a data mode and tile to begin.")
