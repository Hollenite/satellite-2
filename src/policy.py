"""
Policy & economics metadata layer.

Loads and serves local subsidy/tariff/compensation regime records
for each city so the app can display contextual economics information.
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
class PolicyRecord:
    """Compensation, subsidy, and tariff metadata for a location."""
    location_name: str
    state: str = ""
    country: str = ""
    compensation_regime: str = "Unknown"  # Net Metering / Net Billing / FIT / Self-consumption / Unknown
    subsidy_available: bool = False
    subsidy_notes: str = ""
    policy_source: str = ""
    policy_date: str = ""
    example_tariff_residential_per_kwh: Optional[float] = None
    example_tariff_unit: str = "INR"
    notes: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path("data/config/policy_metadata.json")


def load_policy_records(path: str | Path = None) -> Dict[str, PolicyRecord]:
    """
    Load policy records from a JSON file.

    Returns a dict keyed by location_name (case-insensitive lower).
    Returns empty dict if file is missing.
    """
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records: Dict[str, PolicyRecord] = {}
    for entry in data:
        rec = PolicyRecord(
            location_name=entry.get("location_name", "Unknown"),
            state=entry.get("state", ""),
            country=entry.get("country", ""),
            compensation_regime=entry.get("compensation_regime", "Unknown"),
            subsidy_available=entry.get("subsidy_available", False),
            subsidy_notes=entry.get("subsidy_notes", ""),
            policy_source=entry.get("policy_source", ""),
            policy_date=entry.get("policy_date", ""),
            example_tariff_residential_per_kwh=entry.get("example_tariff_residential_per_kwh"),
            example_tariff_unit=entry.get("example_tariff_unit", "INR"),
            notes=entry.get("notes", ""),
        )
        records[rec.location_name.lower()] = rec
    return records


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_policy_summary(record: PolicyRecord) -> str:
    """Format a human-readable policy summary string."""
    lines = [
        f"📋 Policy: {record.location_name} ({record.state}, {record.country})",
        f"  Compensation: {record.compensation_regime}",
        f"  Subsidy: {'Yes' if record.subsidy_available else 'No'}",
    ]
    if record.subsidy_notes:
        lines.append(f"  Subsidy details: {record.subsidy_notes}")
    if record.example_tariff_residential_per_kwh is not None:
        lines.append(
            f"  Example residential tariff: "
            f"{record.example_tariff_residential_per_kwh} {record.example_tariff_unit}/kWh"
        )
    if record.policy_source:
        lines.append(f"  Source: {record.policy_source} ({record.policy_date})")
    if record.notes:
        lines.append(f"  Notes: {record.notes}")
    return "\n".join(lines)


def policy_to_dict(record: PolicyRecord) -> dict:
    """Convert a PolicyRecord to a plain dict for JSON serialization."""
    return {
        "location_name": record.location_name,
        "state": record.state,
        "country": record.country,
        "compensation_regime": record.compensation_regime,
        "subsidy_available": record.subsidy_available,
        "subsidy_notes": record.subsidy_notes,
        "policy_source": record.policy_source,
        "policy_date": record.policy_date,
        "example_tariff_residential_per_kwh": record.example_tariff_residential_per_kwh,
        "example_tariff_unit": record.example_tariff_unit,
        "notes": record.notes,
    }
