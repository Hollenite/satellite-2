"""
Confidence & uncertainty scoring.

Assembles a tile/run-level confidence record from available signals:
imagery metadata, segmentation model output, vectorization quality,
and alignment warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TileConfidence:
    """Run-level confidence and data quality metadata."""
    imagery_date: Optional[str] = None
    data_recency_label: str = "Unknown"     # Fresh / Moderate / Stale / Unknown
    segmentation_confidence: Optional[float] = None   # 0–1 (mean prob)
    vectorization_confidence: Optional[float] = None  # 0–1 (rule-based)
    overall_confidence_score: Optional[float] = None   # 0–100
    uncertainty_notes: List[str] = field(default_factory=list)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_recency_label(imagery_date_str: Optional[str]) -> str:
    """
    Classify imagery date as Fresh / Moderate / Stale / Unknown.

    Fresh:    < 1 year old
    Moderate: 1–3 years old
    Stale:    > 3 years old
    Unknown:  no date available
    """
    if not imagery_date_str:
        return "Unknown"
    try:
        img_date = datetime.fromisoformat(imagery_date_str.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - img_date).days
        if age_days < 365:
            return "Fresh"
        elif age_days < 365 * 3:
            return "Moderate"
        else:
            return "Stale"
    except (ValueError, TypeError):
        return "Unknown"


def compute_vectorization_confidence(
    num_polygons: int,
    min_area_used: float,
    overlap_ratio: Optional[float] = None,
) -> float:
    """
    Heuristic vectorization confidence (0–1).

    Higher when:
    - polygon count is reasonable (5–50)
    - overlap ratio is good
    """
    score = 0.5  # base

    # Polygon count
    if 3 <= num_polygons <= 100:
        score += 0.2
    elif num_polygons > 0:
        score += 0.1

    # Overlap ratio (from alignment check)
    if overlap_ratio is not None:
        if overlap_ratio > 0.5:
            score += 0.3
        elif overlap_ratio > 0.2:
            score += 0.15

    return min(score, 1.0)


def compute_tile_confidence(
    imagery_date: Optional[str] = None,
    num_polygons: int = 0,
    min_area_used: float = 25.0,
    overlap_ratio: Optional[float] = None,
    alignment_warnings: Optional[List[str]] = None,
    segmentation_mean_prob: Optional[float] = None,
    has_crs: bool = False,
) -> TileConfidence:
    """
    Assemble a confidence record from all available signals.

    Returns a TileConfidence dataclass.
    """
    notes: List[str] = []

    # Recency
    recency = data_recency_label(imagery_date)
    if recency == "Unknown":
        notes.append("Imagery date is unknown — recency cannot be assessed.")
    elif recency == "Stale":
        notes.append("Imagery is more than 3 years old — buildings may have changed.")

    # Segmentation confidence
    seg_conf = segmentation_mean_prob
    if seg_conf is None:
        notes.append("No segmentation probability available — using binary threshold only.")

    # Vectorization confidence
    vec_conf = compute_vectorization_confidence(num_polygons, min_area_used, overlap_ratio)

    # CRS check
    if not has_crs:
        notes.append("No CRS metadata — areas are in pixel units, not real-world metres.")

    # Alignment warnings
    if alignment_warnings:
        notes.extend(alignment_warnings)

    # Overall score (weighted average of what's available)
    components = []
    if seg_conf is not None:
        components.append(seg_conf * 100)
    components.append(vec_conf * 100)
    if recency == "Fresh":
        components.append(90)
    elif recency == "Moderate":
        components.append(60)
    elif recency == "Stale":
        components.append(30)
    else:
        components.append(40)

    overall = sum(components) / len(components) if components else 0

    return TileConfidence(
        imagery_date=imagery_date,
        data_recency_label=recency,
        segmentation_confidence=round(seg_conf, 3) if seg_conf is not None else None,
        vectorization_confidence=round(vec_conf, 3),
        overall_confidence_score=round(overall, 1),
        uncertainty_notes=notes,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def confidence_to_dict(conf: TileConfidence) -> dict:
    """Convert TileConfidence to a plain dict for JSON serialization."""
    return {
        "imagery_date": conf.imagery_date,
        "data_recency_label": conf.data_recency_label,
        "segmentation_confidence": conf.segmentation_confidence,
        "vectorization_confidence": conf.vectorization_confidence,
        "overall_confidence_score": conf.overall_confidence_score,
        "uncertainty_notes": conf.uncertainty_notes,
        "timestamp": conf.timestamp,
    }
