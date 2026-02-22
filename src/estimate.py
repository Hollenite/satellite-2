"""
Deterministic solar estimation for rooftop pre-assessment.

All assumptions are configurable via SolarConfig. Outputs are clearly
labelled as pre-assessment estimates, not final engineering values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SolarConfig:
    """
    All solar estimation assumptions in one place.

    Defaults are conservative and appropriate for residential
    rooftop solar pre-assessment.
    """
    # -- Roof parameters --
    roof_usability_factor: float = 0.65
    """Fraction of roof area usable for panels (excludes tanks, stairs, shadows)."""

    # -- Panel parameters --
    panel_power_density_kw_per_m2: float = 0.18
    """kW per m² of panel area. Conservative: 0.18, Moderate: 0.20, Premium: 0.22."""

    # -- System performance --
    performance_ratio: float = 0.78
    """Overall system derate (inverter, wiring, soiling, temperature losses)."""

    # -- Generation assumptions --
    peak_sun_hours_per_day: float = 4.5
    """Average daily peak sun hours. Typical range: 3–6 depending on region."""

    monthly_generation_kwh_per_kw: float = 110.0
    """
    Estimated monthly energy DELIVERED per kW installed (kWh/kW/month).

    This is a DELIVERED value — it already accounts for performance_ratio,
    inverter losses, soiling, temperature derating, etc.
    Do NOT multiply by performance_ratio again.
    Typical range: 90–150 depending on region and season.
    """

    annual_generation_kwh_per_kw: Optional[float] = None
    """
    Optional override for annual generation (kWh/kW/year).
    When set, this is used instead of monthly_generation_kwh_per_kw * 12.
    """

    # -- Labels --
    assumptions_label: str = "Residential rooftop (conservative defaults)"

    pre_assessment_disclaimer: str = (
        "⚠️ PRE-ASSESSMENT ONLY — These estimates are for initial screening "
        "purposes. They do NOT account for roof tilt/azimuth, structural "
        "assessment, shading analysis, or local regulatory approvals. "
        "A qualified solar installer must perform a detailed site survey "
        "before system design and installation."
    )

    limitations: List[str] = field(default_factory=lambda: [
        "No shading analysis performed",
        "No roof tilt or azimuth considered (assumes flat/near-flat)",
        "No structural load assessment",
        "No net metering or grid interconnection logic",
        "Irradiance values are regional averages, not site-specific",
        "Roof usability is a rough estimate (tanks, parapets, stairheads ignored)",
    ])


# ---------------------------------------------------------------------------
# Location yield baseline (Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class LocationYieldBaseline:
    """Location-specific PV yield data for more accurate estimates."""
    location_name: str
    country: str = ""
    region: str = ""
    annual_yield_kwh_per_kw: float = 1320.0
    monthly_yield_kwh_per_kw: float = 110.0
    source_name: str = ""
    source_note: str = ""
    confidence: str = "Low"  # Low / Medium / High


def load_yield_baselines(path: str | Path = None) -> Dict[str, LocationYieldBaseline]:
    """
    Load location yield baselines from a JSON file.

    Returns a dict keyed by location_name (lower-cased).
    Returns empty dict if file is missing.
    """
    import json
    from pathlib import Path as _Path
    _path = _Path(path) if path else _Path("data/config/location_yield_baselines.json")
    if not _path.exists():
        return {}

    with open(_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines: Dict[str, LocationYieldBaseline] = {}
    for entry in data:
        bl = LocationYieldBaseline(
            location_name=entry.get("location_name", "Unknown"),
            country=entry.get("country", ""),
            region=entry.get("region", ""),
            annual_yield_kwh_per_kw=entry.get("annual_yield_kwh_per_kw", 1320),
            monthly_yield_kwh_per_kw=entry.get("monthly_yield_kwh_per_kw", 110),
            source_name=entry.get("source_name", ""),
            source_note=entry.get("source_note", ""),
            confidence=entry.get("confidence", "Low"),
        )
        baselines[bl.location_name.lower()] = bl
    return baselines


def apply_yield_baseline(config: SolarConfig, baseline: LocationYieldBaseline) -> SolarConfig:
    """
    Return a NEW SolarConfig with yield values overridden by the baseline.

    The original config is NOT mutated.
    """
    from dataclasses import replace
    return replace(
        config,
        monthly_generation_kwh_per_kw=baseline.monthly_yield_kwh_per_kw,
        annual_generation_kwh_per_kw=baseline.annual_yield_kwh_per_kw,
        assumptions_label=f"{baseline.location_name} baseline ({baseline.source_name})",
    )


# ---------------------------------------------------------------------------
# Estimation functions
# ---------------------------------------------------------------------------

def estimate_single_roof(
    area_m2: float,
    config: SolarConfig = None,
    area_unit: str = "m²",
) -> Dict:
    """
    Compute solar estimate for a single roof polygon.

    Args:
        area_m2: roof area (ideally in m²)
        config: SolarConfig with assumptions
        area_unit: "m²" or "pixels²" — affects labelling

    Returns:
        Dict with usable_area, system_kw, monthly_kwh, annual_kwh, unit info.
    """
    if config is None:
        config = SolarConfig()

    is_metric = (area_unit == "m²")
    label_suffix = "" if is_metric else " (demo estimate, pixel-based)"

    usable_area = area_m2 * config.roof_usability_factor
    system_kw = usable_area * config.panel_power_density_kw_per_m2
    monthly_kwh = system_kw * config.monthly_generation_kwh_per_kw
    if config.annual_generation_kwh_per_kw is not None:
        annual_kwh = system_kw * config.annual_generation_kwh_per_kw
    else:
        annual_kwh = monthly_kwh * 12

    return {
        "roof_area": round(area_m2, 2),
        "roof_area_unit": area_unit,
        "usable_area": round(usable_area, 2),
        "estimated_system_kw": round(system_kw, 2),
        "estimated_monthly_kwh": round(monthly_kwh, 1),
        "estimated_annual_kwh": round(annual_kwh, 1),
        "is_metric": is_metric,
        "label_suffix": label_suffix,
    }


def estimate_all_roofs(
    polygons: List[Dict],
    config: SolarConfig = None,
) -> Tuple[List[Dict], Dict]:
    """
    Run solar estimation on all polygons.

    Args:
        polygons: list of dicts from vectorize.mask_to_polygons
                  (each must have 'area_value' and 'area_unit')
        config: SolarConfig

    Returns:
        (per_roof_results, aggregate_results)
    """
    if config is None:
        config = SolarConfig()

    per_roof = []
    for i, poly in enumerate(polygons):
        result = estimate_single_roof(
            area_m2=poly["area_value"],
            config=config,
            area_unit=poly["area_unit"],
        )
        result["polygon_id"] = i
        per_roof.append(result)

    # Aggregate
    total_roof = sum(r["roof_area"] for r in per_roof)
    total_usable = sum(r["usable_area"] for r in per_roof)
    total_kw = sum(r["estimated_system_kw"] for r in per_roof)
    total_monthly = sum(r["estimated_monthly_kwh"] for r in per_roof)
    total_annual = sum(r["estimated_annual_kwh"] for r in per_roof)

    # Determine if any roof is non-metric
    any_non_metric = any(not r["is_metric"] for r in per_roof)
    aggregate_unit = "m²" if not any_non_metric else per_roof[0]["roof_area_unit"]
    suffix = "" if not any_non_metric else " (demo estimate, pixel-based)"

    aggregate = {
        "num_roofs": len(per_roof),
        "total_roof_area": round(total_roof, 2),
        "total_roof_area_unit": aggregate_unit,
        "total_usable_area": round(total_usable, 2),
        "total_system_kw": round(total_kw, 2),
        "total_monthly_kwh": round(total_monthly, 1),
        "total_annual_kwh": round(total_annual, 1),
        "label_suffix": suffix,
    }

    return per_roof, aggregate


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(
    per_roof: List[Dict],
    aggregate: Dict,
    config: SolarConfig = None,
) -> str:
    """
    Format a human-readable report string with assumptions + disclaimer.
    """
    if config is None:
        config = SolarConfig()

    lines = []
    lines.append("=" * 60)
    lines.append("  ROOFTOP SOLAR PRE-ASSESSMENT REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Disclaimer first
    lines.append(config.pre_assessment_disclaimer)
    lines.append("")

    # Aggregate
    s = aggregate.get("label_suffix", "")
    lines.append(f"Total roofs detected: {aggregate['num_roofs']}")
    lines.append(f"Total roof area: {aggregate['total_roof_area']} {aggregate['total_roof_area_unit']}{s}")
    lines.append(f"Total usable area: {aggregate['total_usable_area']} {aggregate['total_roof_area_unit']}{s}")
    lines.append(f"Estimated system capacity: {aggregate['total_system_kw']} kW{s}")
    lines.append(f"Estimated monthly generation: {aggregate['total_monthly_kwh']} kWh{s}")
    lines.append(f"Estimated annual generation: {aggregate['total_annual_kwh']} kWh{s}")
    lines.append("")

    # Per-roof table
    lines.append("── Per-Roof Breakdown ──")
    for r in per_roof:
        lines.append(
            f"  Roof #{r['polygon_id']:>3}: "
            f"area={r['roof_area']:>8.1f} {r['roof_area_unit']}, "
            f"usable={r['usable_area']:>8.1f}, "
            f"kW={r['estimated_system_kw']:>6.2f}, "
            f"kWh/mo={r['estimated_monthly_kwh']:>8.1f}"
        )
    lines.append("")

    # Assumptions
    lines.append("── Assumptions Used ──")
    lines.append(f"  Label: {config.assumptions_label}")
    lines.append(f"  Roof usability factor: {config.roof_usability_factor}")
    lines.append(f"  Panel power density: {config.panel_power_density_kw_per_m2} kW/m²")
    lines.append(f"  Performance ratio: {config.performance_ratio}")
    lines.append(f"  Peak sun hours/day: {config.peak_sun_hours_per_day}")
    lines.append(f"  Monthly gen factor: {config.monthly_generation_kwh_per_kw} kWh/kW (delivered, already derated)")
    if config.annual_generation_kwh_per_kw is not None:
        lines.append(f"  Annual gen override: {config.annual_generation_kwh_per_kw} kWh/kW/yr")
    lines.append("")

    # Limitations
    lines.append("── Confidence / Limitations ──")
    for lim in config.limitations:
        lines.append(f"  • {lim}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
