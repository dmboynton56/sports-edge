"""Fetch MLB venue metadata used by weather and park feature joins."""

from __future__ import annotations

import argparse
import json
import os

import requests


MLB_VENUES_URL = "https://statsapi.mlb.com/api/v1/venues"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB venue metadata.")
    parser.add_argument(
        "--output",
        default="data-core/notebooks/cache/mlb_venue_meta.json",
        help="Destination JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = requests.get(
        MLB_VENUES_URL,
        params={"hydrate": "location,fieldInfo"},
        timeout=30,
    )
    response.raise_for_status()

    venue_meta = {}
    for venue in response.json().get("venues", []):
        venue_id = venue.get("id")
        if venue_id is None:
            continue
        location = venue.get("location") or {}
        field_info = venue.get("fieldInfo") or {}
        venue_meta[str(venue_id)] = {
            "name": venue.get("name"),
            "elevation": location.get("elevation"),
            "roofType": field_info.get("roofType"),
            "azimuthAngle": location.get("azimuthAngle"),
            "city": location.get("city"),
            "state": location.get("state"),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(venue_meta, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"Saved metadata for {len(venue_meta)} MLB venues to {args.output}")


if __name__ == "__main__":
    main()
