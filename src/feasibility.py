"""
Roof feasibility features.

Adds per-roof feasibility metadata (orientation, tilt, shading risk,
customer type, etc.) to the polygon records.  Uses a progressive
design: compute what we can, return null + status note for the rest.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Area threshold (m²) for customer type classification
_RESIDENTIAL_MAX_AREA_M2 = 200.0


# ---------------------------------------------------------------------------
# Core feasibility function
# ---------------------------------------------------------------------------

def compute_feasibility(
    polygon_dict: dict,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute or placeholder feasibility features for a single roof polygon.

    Args:
        polygon_dict: dict from mask_to_polygons (must have 'geometry',
                      'area_value', 'area_unit')
        image: optional full tile image (bands, H, W) for heuristics
        mask: optional full tile mask (H, W) for heuristics

    Returns:
        Dict of feasibility fields to merge into the polygon record.
    """
    area = polygon_dict.get("area_value", 0)
    area_unit = polygon_dict.get("area_unit", "pixels²")

    feats: dict = {}

    # --- Roof orientation (not computable from nadir satellite) ---
    feats["roof_plane_orientation"] = None
    feats["roof_plane_orientation_status"] = (
        "Not available in MVP — requires stereo imagery or LiDAR"
    )

    # --- Roof tilt (not computable from nadir satellite) ---
    feats["roof_plane_tilt"] = None
    feats["roof_plane_tilt_status"] = (
        "Not available in MVP — requires LiDAR or DSM data"
    )

    # --- Shading risk (rough heuristic) ---
    shading = _estimate_shading_risk(polygon_dict, image)
    feats["shading_risk_score"] = shading["score"]
    feats["shading_risk_label"] = shading["label"]
    feats["shading_risk_status"] = shading["status"]

    # --- Customer type proxy ---
    ct = _classify_customer_type(area, area_unit)
    feats["customer_type_proxy"] = ct

    # --- Interconnection risk ---
    feats["interconnection_risk_proxy"] = None
    feats["interconnection_risk_status"] = (
        "Not available in MVP — requires utility feeder data and grid topology"
    )

    # --- Heritage / restricted zone flag ---
    feats["heritage_or_restricted_zone_flag"] = False
    feats["heritage_flag_status"] = (
        "Placeholder — no heritage zone GIS layer loaded"
    )

    # --- Feasibility confidence ---
    feats["feasibility_confidence_score"] = _feasibility_confidence(feats)

    return feats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_shading_risk(
    polygon_dict: dict,
    image: Optional[np.ndarray] = None,
) -> dict:
    """
    Rough shading risk heuristic.

    If image is available, check mean brightness in the polygon region.
    Otherwise, return placeholder.
    """
    if image is None:
        return {
            "score": None,
            "label": "Unknown",
            "status": "Not computed — needs image pixel analysis or DSM data",
        }

    # Very rough proxy: we can't easily clip the polygon from the image
    # without rasterizing it, so return placeholder with a note.
    return {
        "score": None,
        "label": "Unknown",
        "status": (
            "Heuristic not applied in MVP — requires per-polygon raster "
            "clipping which is computationally expensive for demo"
        ),
    }


def _classify_customer_type(area: float, area_unit: str) -> str:
    """Classify roof as Residential or C&I based on area threshold."""
    if area_unit != "m²":
        # Can't reliably classify if area is in pixels
        return "Unknown (pixel-based area)"

    if area <= _RESIDENTIAL_MAX_AREA_M2:
        return "Residential"
    else:
        return "C&I (Commercial/Industrial)"


def _feasibility_confidence(feats: dict) -> int:
    """
    Compute a 0–100 feasibility confidence score based on how many
    fields are actually computed vs placeholders.
    """
    total_fields = 6
    computed = 0

    # Count actually computed fields
    if feats.get("shading_risk_score") is not None:
        computed += 1
    if feats.get("customer_type_proxy") and "Unknown" not in feats["customer_type_proxy"]:
        computed += 1
    if feats.get("roof_plane_orientation") is not None:
        computed += 1
    if feats.get("roof_plane_tilt") is not None:
        computed += 1
    if feats.get("interconnection_risk_proxy") is not None:
        computed += 1
    # Heritage flag is always "computed" (default False)
    computed += 1

    return int((computed / total_fields) * 100)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def enrich_polygons_with_feasibility(
    polygons: List[Dict],
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Add feasibility fields to each polygon dict (non-destructive).

    Returns a NEW list of polygon dicts — original dicts are not mutated.
    """
    enriched = []
    for poly in polygons:
        new_poly = dict(poly)  # shallow copy
        feats = compute_feasibility(new_poly, image=image, mask=mask)
        new_poly.update(feats)
        enriched.append(new_poly)
    return enriched
