# ☀️ RoofIntel

> **Pre-assessment only** — this tool provides rough estimates for initial screening.
> It is NOT a substitute for professional site survey or engineering design.

---

## Quick Start

### Prerequisites
- **Python 3.10+** (pyproj requires 3.10+)
- Windows / Linux / macOS

### Setup

```powershell
# 1. Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install core deps
pip install numpy matplotlib Pillow shapely pyproj rasterio streamlit

# 3. Install PyTorch (CPU — fastest for tonight)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install segmentation model
pip install segmentation-models-pytorch

# 5. Optional (skip if any fail)
pip install geopandas pandas scikit-image
```

#### ⚠️ Windows GDAL / rasterio issues

If `pip install rasterio` fails:

```powershell
# Option A: Use conda (recommended)
conda install -c conda-forge rasterio

# Option B: Pre-built wheel
# Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# pip install rasterio‑<version>‑cp310‑cp310‑win_amd64.whl
```

#### Fallback minimal install (if geopandas fails)
The pipeline works without geopandas. Core stack: `rasterio + shapely + pyproj + json`

---

## Demo Modes

### 🟢 Demo Mode A — Ground-Truth Pipeline (recommended first)

Uses existing labels/masks as "fake predictions" to prove the full product pipeline:

```
satellite image → GT mask → vectorize → solar estimate → overlay
```

**How to run:**
```powershell
streamlit run app.py
# Select "Synthetic demo" or load a SpaceNet tile with matching labels
```

**What this proves:** Geospatial pipeline correctness, UX flow, export functionality.

### 🔵 Demo Mode B — Model Inference Pipeline

Trains a U-Net on SpaceNet data, then runs inference:

```
satellite image → U-Net → predicted mask → vectorize → solar estimate → overlay
```

**How to run:**
```powershell
# Train
python -m src.train --data_dir data/raw --epochs 10

# Infer
python -m src.infer --image data/raw/images/tile.tif --checkpoint checkpoints/best.pth

# Then run Streamlit with the predicted mask
```

---

## Data Setup (SpaceNet)

### Minimal setup — one tile

```
satellite/data/raw/
├── images/
│   └── tile_001.tif          # RGB or multispectral GeoTIFF
├── masks/                     # Option A: raster masks
│   └── tile_001.tif           # Binary building mask (0/1)
└── labels/                    # Option B: vector labels
    └── tile_001.geojson       # Building footprint polygons
```

Download from: [spacenet.ai/datasets](https://spacenet.ai/datasets/)

### Synthetic mode (zero data needed)

The app includes a built-in synthetic tile generator. Select **"Synthetic demo"** in the sidebar — no downloads required.

---

## Project Structure

```
satellite/
├── app.py                  # Streamlit demo app
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── data.py             # Image/mask loading, RGB prep, synthetic gen, dataset
│   ├── vectorize.py        # Mask → polygons → GeoJSON + sidecar metadata
│   ├── viz.py              # Overlay visualization, side-by-side figures
│   ├── estimate.py         # Solar estimation with configurable assumptions
│   ├── train.py            # U-Net training pipeline
│   ├── infer.py            # Model inference + mask export
│   └── utils.py            # CRS/reprojection/alignment/area helpers
├── data/raw/               # Place SpaceNet tiles here
├── outputs/                # Generated masks, GeoJSONs, PNGs
└── checkpoints/            # Model weights
```

---

## GeoJSON & CRS Handling

Output GeoJSON files are written as **standard FeatureCollections** (no embedded CRS).
A **sidecar metadata file** is saved alongside:

```
outputs/footprints.geojson       ← polygon geometries
outputs/footprints.meta.json     ← CRS, transform, coordinate type
```

> ⚠️ GeoJSON consumers (geojson.io, QGIS) may assume WGS84.
> Use the sidecar `.meta.json` to interpret coordinates correctly.

---

## Solar Estimation Assumptions

| Parameter | Default | Notes |
|-----------|---------|-------|
| Roof usability | 0.65 | Excludes tanks, stairs, shadows |
| Panel density | 0.18 kW/m² | Conservative for Indian market |
| Performance ratio | 0.78 | Inverter + wiring + soiling losses |
| Monthly gen factor | 110 kWh/kW | India avg varies 100–130 by region |

All parameters are **configurable** in the Streamlit sidebar.

> [!IMPORTANT]
> **Formula clarity:** `monthly_generation_kwh_per_kw` is a **delivered** value — it already accounts for `performance_ratio`, inverter losses, soiling, and temperature derating. Do NOT multiply by `performance_ratio` again. The formula is:
> ```
> system_kw = usable_area × panel_power_density
> monthly_kwh = system_kw × monthly_generation_kwh_per_kw
> annual_kwh = monthly_kwh × 12  (or override via annual_generation_kwh_per_kw)
> ```

### Limitations
- No shading analysis
- No roof tilt/azimuth modelling
- No structural load assessment
- No net metering or grid interconnection logic
- Irradiance values are regional averages

---

## Adaptation Notes for Different Regions

### Domain Gap
Building styles vary significantly across the 10 dataset cities (Wuhan, Taiwan, Los Angeles, Ottawa, Cairo, Milan, Santiago, Cordoba, Venice, New York):
- Different building styles, materials, and roof shapes
- Varying density and urban layout patterns
- Climate-specific rooftop features (snow loads, heat mitigation, water tanks)

### Fine-Tuning Path
1. Collect 50–100 annotated tiles for your target region
2. Fine-tune the pretrained U-Net on this data
3. Validate on held-out tiles from the target region
4. Adjust `SolarConfig` defaults per city/region

### Product Positioning
- Frame as "AI-assisted pre-assessment" — not final engineering
- Adapt policy config files for local incentive programmes
- Region-specific customizations: irradiance maps, tariff rates, subsidy amounts

---

## Debugging Checklist

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `rasterio` won't install | Missing GDAL | Use conda or pre-built wheel |
| Polygons don't overlap image | CRS mismatch | Check `print_raster_info()`, ensure transform is passed |
| Area in degrees² | Geographic CRS | pyproj auto-reprojects; check `pyproj` is installed |
| Model produces blank mask | Insufficient training / wrong threshold | Lower threshold to 0.3, train more epochs |
| Streamlit crashes on plot | Matplotlib threading | `matplotlib.use("Agg")` is already set |
| GeoJSON looks wrong in QGIS | CRS assumption mismatch | Read `.meta.json` sidecar for true CRS |

---

## CLI Quick Reference

```powershell
# Streamlit app
streamlit run app.py

# Train model
python -m src.train --data_dir data/raw --epochs 10 --batch_size 4

# Run inference
python -m src.infer --image data/raw/images/tile.tif --checkpoint checkpoints/best.pth

# Quick data sanity check
python -c "from src.data import load_image; img,t,c = load_image('data/raw/images/tile.tif'); print(img.shape, c, t)"

# Quick synthetic test
python -c "from src.data import generate_synthetic_tile; img,m,t,c = generate_synthetic_tile(); print(img.shape, m.shape, t, c)"
```

---

## Decision-Support Layers (v2)

The tool now includes additional data layers that transform it from a simple detection demo into a decision-support product for solar companies and government users.

### 📍 Location Yield Baselines

Location-specific PV yield data (`data/config/location_yield_baselines.json`) overrides the default India-average generation assumptions for more accurate estimates.

| Field | Description |
|-------|-------------|
| `annual_yield_kwh_per_kw` | Location-specific yield (kWh/kW/year) |
| `monthly_yield_kwh_per_kw` | Derived monthly value |
| `source_name` | e.g. "NREL PVWatts / PVGIS" |
| `confidence` | Low / Medium / High |

Pre-loaded: Wuhan, Taiwan, Los Angeles, Ottawa, Cairo, Milan, Santiago, Cordoba, Venice, New York.

### 📋 Policy & Economics Metadata

Per-city policy context (`data/config/policy_metadata.json`):
- **Compensation regime**: Net Metering / Net Billing / FIT / etc.
- **Subsidy info**: programme details, region-specific notes
- **Example tariffs**: residential tariff/kWh
- **Source/date**: policy provenance

> These are contextual metadata — not a financial calculator.

### 📊 Market Intelligence

City-level market data (`data/config/market_intelligence_sample.json`):
- **Adoption density/momentum scores** (0–1)
- **Market maturity**: Emerging / Growth / Mature
- **Customer mix**: Residential-heavy / C&I-heavy / Mixed
- **Programme eligibility tags**

### 🏠 Roof Feasibility Features

Each roof polygon includes feasibility metadata:

| Field | Status in MVP |
|-------|--------------|
| `customer_type_proxy` | ✅ Computed from area threshold |
| `heritage_flag` | 🔶 Placeholder (default: false) |
| `shading_risk_score` | ❌ Needs DSM/LiDAR data |
| `roof_plane_orientation` | ❌ Needs stereo imagery |
| `roof_plane_tilt` | ❌ Needs LiDAR |
| `interconnection_risk` | ❌ Needs utility data |

Fields that cannot be computed return `null` with a `metric_status` note explaining what data is needed. This ensures the schema is extensible.

### 🛡️ Confidence & Uncertainty Scoring

Every export includes:
- `imagery_date` and `data_recency_label` (Fresh/Moderate/Stale/Unknown)
- `segmentation_confidence` (model probability, if available)
- `vectorization_confidence` (heuristic, 0–1)
- `overall_confidence_score` (0–100)
- `uncertainty_notes` (aggregated warnings)

---

## Alignment Debugging

If roof polygons appear rotated or misaligned on the overlay:

1. Enable **"Show alignment debug view"** in the sidebar
2. Run the pipeline — the debug panel shows:
   - Raster bounds vs polygon bounds
   - Overlap ratio
   - Coordinate space (pixel vs geo)
   - CRS info
3. Download the **Debug Log** for detailed diagnostics
4. Common causes:
   - Polygons in geo-space but image displayed in pixel-space (fixed in v2)
   - CRS mismatch between image and mask
   - Inverted Y-axis (top-origin vs bottom-origin)

---

## Loading Real Data Later

The config files use simple JSON arrays. To add your own data:

1. **Yield baselines**: Add entries to `data/config/location_yield_baselines.json`
2. **Policy records**: Add entries to `data/config/policy_metadata.json`
3. **Market intel**: Add entries to `data/config/market_intelligence_sample.json`

All loaders fall back gracefully if files are missing — no hard dependency.

---

## Enriched Exports

In addition to the original exports (PNG, GeoJSON, Meta JSON, Report TXT), the app now exports:

| Export | Format | Contents |
|--------|--------|----------|
| Enriched CSV | `.csv` | One row per roof polygon with all fields |
| Tile Summary | `.json` | Aggregate stats + policy + market + confidence |
| Policy/Market Snapshot | `.json` | Config data used for this run |
| Debug Log | `.json` | Alignment + confidence diagnostics |

---

## Updated Project Structure

```
satellite/
├── app.py                          # Streamlit demo app
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data.py                     # Image/mask loading, RGB prep, synthetic gen
│   ├── vectorize.py                # Mask → polygons → GeoJSON
│   ├── viz.py                      # Overlay viz + alignment debug
│   ├── estimate.py                 # Solar estimation + yield baselines
│   ├── train.py                    # U-Net training pipeline
│   ├── infer.py                    # Model inference + mask export
│   ├── utils.py                    # CRS/alignment/area helpers
│   ├── policy.py                   # Policy/economics metadata     [NEW]
│   ├── market.py                   # Market intelligence layer     [NEW]
│   ├── confidence.py               # Confidence/uncertainty scoring [NEW]
│   └── feasibility.py              # Roof feasibility features     [NEW]
├── data/
│   ├── raw/                        # Satellite tiles + masks
│   └── config/                     # Configuration files            [NEW]
│       ├── location_yield_baselines.json
│       ├── policy_metadata.json
│       └── market_intelligence_sample.json
├── outputs/
├── checkpoints/
└── scripts/
```

