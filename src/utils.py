"""
Geospatial utility helpers — CRS, reprojection, sanity checks.

This module handles the fiddly geospatial plumbing so that other modules
can stay focused on their own logic.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

try:
    from pyproj import CRS, Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    warnings.warn("pyproj not installed — area calculations will be pixel-based only.")


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def is_projected_crs(crs) -> bool:
    """Return True if the CRS uses a projected coordinate system (metres)."""
    if crs is None:
        return False
    if HAS_PYPROJ:
        return CRS(crs).is_projected
    # Rough heuristic: EPSG codes < 32000 are often geographic
    try:
        epsg = crs.to_epsg()
        return epsg is not None and epsg >= 32600
    except Exception:
        return False


def pick_local_projected_crs(geom, source_crs) -> Optional[CRS]:
    """
    Pick a UTM zone CRS that covers the centroid of *geom*.

    Falls back to EPSG:3857 (Web Mercator) if UTM zone lookup fails.
    Returns None if pyproj is missing.
    """
    if not HAS_PYPROJ:
        return None
    try:
        source = CRS(source_crs)
        if source.is_projected:
            return source  # already in metres

        # Get centroid in lon/lat
        centroid = geom.centroid
        lon, lat = centroid.x, centroid.y

        # UTM zone number
        zone_number = int((lon + 180) / 6) + 1
        hemisphere = 326 if lat >= 0 else 327  # 326xx = North, 327xx = South
        epsg_code = hemisphere * 100 + zone_number
        return CRS.from_epsg(epsg_code)
    except Exception:
        return CRS.from_epsg(3857)  # Web Mercator fallback


def reproject_geometry(geom, crs_from, crs_to):
    """
    Reproject a Shapely geometry from *crs_from* to *crs_to*.

    Uses pyproj with ``always_xy=True`` so (lon, lat) order is preserved.
    Returns the original geometry unchanged if pyproj is missing.
    """
    if not HAS_PYPROJ:
        warnings.warn("pyproj missing — returning geometry without reprojection.")
        return geom

    transformer = Transformer.from_crs(
        CRS(crs_from), CRS(crs_to), always_xy=True
    )
    return shapely_transform(transformer.transform, geom)


# ---------------------------------------------------------------------------
# Area helpers
# ---------------------------------------------------------------------------

def compute_area_m2(geom, crs) -> Tuple[float, str]:
    """
    Compute area of *geom* in square metres if possible.

    Returns (area_value, unit_label).
    - Projected CRS → uses polygon.area directly
    - Geographic CRS → reprojects to local UTM first
    - Missing CRS or pyproj → pixel area with warning
    """
    if crs is None or not HAS_PYPROJ:
        return float(geom.area), "pixels²"

    if is_projected_crs(crs):
        return float(geom.area), "m²"

    # Geographic → reproject to local UTM
    target_crs = pick_local_projected_crs(geom, crs)
    if target_crs is None:
        return float(geom.area), "degrees² (WARNING: not metric)"

    reprojected = reproject_geometry(geom, crs, target_crs)
    return float(reprojected.area), "m²"


# ---------------------------------------------------------------------------
# Alignment / sanity checks
# ---------------------------------------------------------------------------

def polygon_bounds_overlap_ratio(
    raster_bounds: Tuple[float, float, float, float],
    polygon_bounds: Tuple[float, float, float, float],
) -> float:
    """
    Compute the ratio of intersection area to union area for two bounding boxes.

    Returns 0.0 if no overlap, 1.0 if identical.
    """
    rbox = box(*raster_bounds)
    pbox = box(*polygon_bounds)
    if rbox.is_empty or pbox.is_empty:
        return 0.0
    intersection = rbox.intersection(pbox).area
    union = rbox.union(pbox).area
    return intersection / union if union > 0 else 0.0


def validate_polygon_raster_alignment(
    raster_bounds: Tuple[float, float, float, float],
    polygons: list,
    threshold: float = 0.2,
) -> Tuple[bool, list]:
    """
    Check that the union bounds of *polygons* overlaps with *raster_bounds*.

    Returns:
        (is_aligned, warning_messages) — bool + list of warning strings.
    """
    warn_msgs: list = []

    if not polygons:
        warn_msgs.append("No polygons to validate alignment for.")
        return True, warn_msgs

    # Compute union bounds of all polygons
    all_bounds = [p.bounds for p in polygons]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    poly_union_bounds = (minx, miny, maxx, maxy)

    ratio = polygon_bounds_overlap_ratio(raster_bounds, poly_union_bounds)
    if ratio < threshold:
        msg = (
            f"⚠️  Low polygon/raster overlap ({ratio:.2%}). "
            f"Raster bounds: {raster_bounds}, Polygon union bounds: {poly_union_bounds}. "
            f"This may indicate a CRS mismatch or pixel-vs-geo coordinate confusion."
        )
        warnings.warn(msg)
        warn_msgs.append(msg)
        return False, warn_msgs
    return True, warn_msgs


# ---------------------------------------------------------------------------
# Raster metadata logging
# ---------------------------------------------------------------------------

def print_raster_info(path: str | Path) -> dict:
    """
    Print and return key metadata for a raster file.

    Returns dict with keys: shape, crs, transform, bounds, dtype.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        info = {
            "path": str(path),
            "shape": (src.count, src.height, src.width),
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }
    print(f"── Raster info: {path.name} ──")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


def check_crs_units(crs) -> str:
    """Return 'meters', 'degrees', or 'unknown', with a warning for degrees."""
    if crs is None:
        warnings.warn("No CRS found — coordinates are pixel-based.")
        return "unknown"
    if is_projected_crs(crs):
        return "meters"
    warnings.warn(
        "CRS is geographic (lat/lon). Area calculations will need reprojection."
    )
    return "degrees"


def ensure_output_dir(path: str | Path) -> Path:
    """Create output directory if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Alignment debug (Phase 1)
# ---------------------------------------------------------------------------

def alignment_debug_info(
    transform,
    mask_shape: Tuple[int, int],
    polygons: list,
    crs=None,
) -> dict:
    """
    Compute alignment diagnostic info for the debug panel.

    Returns a dict with raster_bounds, polygon_bounds, overlap_ratio,
    coordinate_space, and a list of diagnostic notes.
    """
    h, w = mask_shape
    if transform is not None:
        # Raster bounds from affine transform
        left = transform.c
        top = transform.f
        right = left + transform.a * w
        bottom = top + transform.e * h
        raster_bounds = (min(left, right), min(top, bottom),
                         max(left, right), max(top, bottom))
    else:
        raster_bounds = (0, 0, w, h)

    notes = []

    if not polygons:
        return {
            "raster_bounds": raster_bounds,
            "polygon_bounds": None,
            "overlap_ratio": 0.0,
            "coordinate_space": "pixel" if crs is None else "geo",
            "crs_epsg": crs.to_epsg() if crs else None,
            "notes": ["No polygons to check."],
        }

    # Extract geometries
    geoms = [p["geometry"] if isinstance(p, dict) else p for p in polygons]
    all_bounds = [g.bounds for g in geoms]
    poly_bounds = (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )

    ratio = polygon_bounds_overlap_ratio(raster_bounds, poly_bounds)

    # Detect coordinate space
    coord_space = "pixel" if crs is None else "geo"
    if crs is None and transform is None:
        coord_space = "pixel (synthetic)"

    # Diagnostic notes
    if ratio < 0.1:
        notes.append(f"CRITICAL: Very low overlap ({ratio:.1%}) — likely CRS mismatch or pixel/geo confusion.")
    elif ratio < 0.5:
        notes.append(f"WARNING: Moderate overlap ({ratio:.1%}) — check transform/CRS.")
    else:
        notes.append(f"OK: Good overlap ({ratio:.1%}).")

    if crs is not None and not is_projected_crs(crs):
        notes.append("CRS is geographic (degrees) — polygon coords are in lat/lon.")
    elif crs is not None:
        notes.append(f"CRS is projected (metres) — EPSG:{crs.to_epsg() if crs else '?'}.")

    return {
        "raster_bounds": raster_bounds,
        "polygon_bounds": poly_bounds,
        "overlap_ratio": ratio,
        "coordinate_space": coord_space,
        "crs_epsg": crs.to_epsg() if crs else None,
        "notes": notes,
    }
