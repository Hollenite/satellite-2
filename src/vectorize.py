"""
Raster mask → vector polygon pipeline.

Converts binary building masks to georeferenced (or pixel-coordinate)
polygons, writes GeoJSON + metadata sidecar, and computes polygon areas
with proper CRS handling via pyproj.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import mapping, shape as shapely_shape

from src.utils import (
    check_crs_units,
    compute_area_m2,
    ensure_output_dir,
    is_projected_crs,
    validate_polygon_raster_alignment,
)


def clean_polygon(geom):
    """Fix invalid geometries with buffer(0). Returns None if unfixable."""
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return None
    return geom


# ---------------------------------------------------------------------------
# Core: mask → polygons
# ---------------------------------------------------------------------------

def mask_to_polygons(
    mask: np.ndarray,
    transform: rasterio.Affine = None,
    crs=None,
    min_area: float = 25.0,
    simplify_tolerance: float = 1.0,
    use_pixel_coords: bool = False,
) -> List[Dict]:
    """
    Convert a binary mask to a list of polygon dicts.

    Each dict has keys: 'geometry' (Shapely), 'area_value', 'area_unit',
    'pixel_area'.

    Args:
        mask: (H, W) uint8 binary mask
        transform: rasterio Affine; pass None for pixel coords
        crs: rasterio CRS object
        min_area: discard polygons below this area (in native units)
        simplify_tolerance: Douglas-Peucker simplification tolerance
        use_pixel_coords: if True, ignore transform (output in pixel coords)

    Returns:
        List of polygon dicts, sorted by area descending.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    binary = (mask > 0).astype(np.uint8)

    # Choose transform — rasterio >= 1.5 requires a valid Affine, so we use
    # an identity transform for pixel-coordinate mode instead of None.
    if use_pixel_coords or transform is None:
        from rasterio.transform import from_bounds
        effective_transform = from_bounds(0, 0, mask.shape[1], mask.shape[0],
                                          mask.shape[1], mask.shape[0])
    else:
        effective_transform = transform

    results = []
    for geom_dict, value in rio_shapes(binary, transform=effective_transform):
        if value == 0:
            continue  # background

        geom = shapely_shape(geom_dict)
        geom = clean_polygon(geom)
        if geom is None:
            continue

        # Simplify
        if simplify_tolerance > 0:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)
            if geom.is_empty:
                continue

        # Pixel area (always available)
        pixel_area = float(geom.area)

        # Metric area
        effective_crs = None if use_pixel_coords else crs
        area_value, area_unit = compute_area_m2(geom, effective_crs)

        # Filter by min area (use the best available area)
        filter_area = area_value if area_unit == "m²" else pixel_area
        if filter_area < min_area:
            continue

        results.append({
            "geometry": geom,
            "area_value": area_value,
            "area_unit": area_unit,
            "pixel_area": pixel_area,
        })

    # Sort largest first
    results.sort(key=lambda r: r["area_value"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------

def polygons_to_geojson(
    polygons: List[Dict],
    output_path: str | Path,
    crs=None,
    raster_path: Optional[str] = None,
    transform: Optional[rasterio.Affine] = None,
    is_georeferenced: bool = True,
) -> Path:
    """
    Write polygons to GeoJSON + a sidecar metadata file.

    Writes:
        <output_path>.geojson — standard GeoJSON FeatureCollection
        <output_path>.meta.json — CRS, transform, coordinate type, timestamp

    Args:
        polygons: list of dicts from mask_to_polygons
        output_path: base path (without extension)
        crs: rasterio CRS for metadata sidecar
        raster_path: source raster path for provenance
        transform: source raster transform for provenance
        is_georeferenced: whether coordinates are georeferenced or pixel

    Returns:
        Path to the saved GeoJSON file.
    """
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)

    geojson_path = output_path.with_suffix(".geojson")
    meta_path = output_path.with_name(output_path.stem + ".meta.json")

    # Build GeoJSON FeatureCollection
    features = []
    for i, poly in enumerate(polygons):
        feature = {
            "type": "Feature",
            "id": i,
            "properties": {
                "id": i,
                "area_value": round(poly["area_value"], 2),
                "area_unit": poly["area_unit"],
                "pixel_area": round(poly["pixel_area"], 2),
            },
            "geometry": mapping(poly["geometry"]),
        }
        features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(geojson_path, "w") as f:
        json.dump(collection, f, indent=2)

    # Sidecar metadata
    crs_wkt = None
    crs_epsg = None
    if crs is not None:
        try:
            crs_wkt = crs.to_wkt()
        except Exception:
            pass
        try:
            crs_epsg = crs.to_epsg()
        except Exception:
            pass

    transform_list = None
    if transform is not None:
        transform_list = list(transform)[:6]

    meta = {
        "source_raster": str(raster_path) if raster_path else None,
        "crs_epsg": crs_epsg,
        "crs_wkt": crs_wkt,
        "raster_transform": transform_list,
        "coordinates": "georeferenced" if is_georeferenced else "pixel",
        "num_polygons": len(polygons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warning": (
            "GeoJSON spec assumes WGS84 for geographic coordinates. "
            "Use this sidecar file to interpret the coordinate system correctly."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved {len(polygons)} polygons → {geojson_path}")
    print(f"   Metadata sidecar → {meta_path}")
    return geojson_path
