"""
Market intelligence layer.

Loads city/neighborhood-level market data (adoption scores, maturity,
customer mix) from a local JSON/CSV config to support sales targeting
and government programme planning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MarketIntelligence:
    """City-level market data for solar targeting."""
    location_name: str
    adoption_density_score: Optional[float] = None      # 0–1
    adoption_momentum_score: Optional[float] = None     # 0–1
    installed_pv_density_per_km2: Optional[float] = None
    program_eligibility_tag: str = ""
    market_maturity_segment: str = "Unknown"  # Emerging / Growth / Mature
    customer_mix_proxy: str = "Unknown"       # Residential-heavy / C&I-heavy / Mixed
    notes: str = ""


def default_market_record(location_name: str = "Unknown") -> MarketIntelligence:
    """Return an empty placeholder market record."""
    return MarketIntelligence(location_name=location_name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path("data/config/market_intelligence_sample.json")


def load_market_intelligence(path: str | Path = None) -> Dict[str, MarketIntelligence]:
    """
    Load market intelligence records from a JSON file.

    Returns a dict keyed by location_name (case-insensitive lower).
    Returns empty dict if file is missing.
    """
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: Dict[str, MarketIntelligence] = {}
    for entry in data:
        rec = MarketIntelligence(
            location_name=entry.get("location_name", "Unknown"),
            adoption_density_score=entry.get("adoption_density_score"),
            adoption_momentum_score=entry.get("adoption_momentum_score"),
            installed_pv_density_per_km2=entry.get("installed_pv_density_per_km2"),
            program_eligibility_tag=entry.get("program_eligibility_tag", ""),
            market_maturity_segment=entry.get("market_maturity_segment", "Unknown"),
            customer_mix_proxy=entry.get("customer_mix_proxy", "Unknown"),
            notes=entry.get("notes", ""),
        )
        records[rec.location_name.lower()] = rec
    return records


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_market_summary(record: MarketIntelligence) -> str:
    """Format a human-readable market intelligence summary."""
    lines = [f"📊 Market Intelligence: {record.location_name}"]
    if record.adoption_density_score is not None:
        lines.append(f"  Adoption density: {record.adoption_density_score:.0%}")
    if record.adoption_momentum_score is not None:
        lines.append(f"  Adoption momentum: {record.adoption_momentum_score:.0%}")
    if record.installed_pv_density_per_km2 is not None:
        lines.append(f"  PV density: {record.installed_pv_density_per_km2:.1f} kW/km²")
    lines.append(f"  Market maturity: {record.market_maturity_segment}")
    lines.append(f"  Customer mix: {record.customer_mix_proxy}")
    if record.program_eligibility_tag:
        lines.append(f"  Programme: {record.program_eligibility_tag}")
    if record.notes:
        lines.append(f"  Notes: {record.notes}")
    return "\n".join(lines)


def market_to_dict(record: MarketIntelligence) -> dict:
    """Convert a MarketIntelligence to a plain dict for JSON serialization."""
    return {
        "location_name": record.location_name,
        "adoption_density_score": record.adoption_density_score,
        "adoption_momentum_score": record.adoption_momentum_score,
        "installed_pv_density_per_km2": record.installed_pv_density_per_km2,
        "program_eligibility_tag": record.program_eligibility_tag,
        "market_maturity_segment": record.market_maturity_segment,
        "customer_mix_proxy": record.customer_mix_proxy,
        "notes": record.notes,
    }
