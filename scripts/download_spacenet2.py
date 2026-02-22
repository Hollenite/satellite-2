"""
Download a small subset of SpaceNet 2 Las Vegas building footprint data.

Requires AWS CLI configured with valid credentials.
The SpaceNet S3 bucket is requester-pays.

Usage:
    py scripts/download_spacenet2.py --num_tiles 30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# SpaceNet 2 Las Vegas paths on S3
BUCKET = "spacenet-dataset"
# RGB Pan-sharpened images (3-band GeoTIFF, 200x200m tiles)
IMAGE_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/PS-RGB"
# GeoJSON building footprint labels
LABEL_PREFIX = "spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings"


def list_s3_objects(prefix: str, max_keys: int = 200) -> list[str]:
    """List object keys under an S3 prefix using AWS CLI."""
    cmd = [
        sys.executable, "-m", "awscli", "s3api", "list-objects-v2",
        "--bucket", BUCKET,
        "--prefix", prefix,
        "--max-items", str(max_keys),
        "--request-payer", "requester",
        "--output", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    contents = data.get("Contents", [])
    return [obj["Key"] for obj in contents if obj["Key"].endswith((".tif", ".geojson"))]


def download_file(key: str, local_path: Path):
    """Download a single file from S3."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_uri = f"s3://{BUCKET}/{key}"
    cmd = [
        sys.executable, "-m", "awscli", "s3", "cp",
        s3_uri, str(local_path),
        "--request-payer", "requester",
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download SpaceNet 2 Las Vegas subset")
    parser.add_argument("--num_tiles", type=int, default=30,
                        help="Number of tiles to download")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    img_dir = output_dir / "images"
    label_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing image tiles from s3://{BUCKET}/{IMAGE_PREFIX} ...")
    image_keys = list_s3_objects(IMAGE_PREFIX, max_keys=args.num_tiles + 50)
    print(f"  Found {len(image_keys)} image keys")

    print(f"Listing label files from s3://{BUCKET}/{LABEL_PREFIX} ...")
    label_keys = list_s3_objects(LABEL_PREFIX, max_keys=args.num_tiles + 50)
    print(f"  Found {len(label_keys)} label keys")

    # Build a mapping from tile stem to (image_key, label_key)
    # SpaceNet 2 naming:
    #   Images: PS-RGB/SN2_buildings_train_AOI_2_Vegas_PS-RGB_img123.tif
    #   Labels: geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img123.geojson
    # We need to match by the trailing imgNNN part

    def extract_tile_id(key: str) -> str:
        """Extract tile ID like 'img123' from a key."""
        fname = key.rsplit("/", 1)[-1]
        # Find 'img' followed by digits
        for part in fname.replace(".", "_").split("_"):
            if part.startswith("img"):
                return part
        return fname

    image_by_id = {}
    for k in image_keys:
        tid = extract_tile_id(k)
        image_by_id[tid] = k

    label_by_id = {}
    for k in label_keys:
        tid = extract_tile_id(k)
        label_by_id[tid] = k

    # Find matched pairs
    matched_ids = sorted(set(image_by_id.keys()) & set(label_by_id.keys()))
    selected = matched_ids[:args.num_tiles]
    print(f"  Matched {len(matched_ids)} image-label pairs, downloading {len(selected)}")

    for i, tid in enumerate(selected, 1):
        img_key = image_by_id[tid]
        lbl_key = label_by_id[tid]

        img_fname = img_key.rsplit("/", 1)[-1]
        lbl_fname = lbl_key.rsplit("/", 1)[-1]

        # Rename label to match image stem for easy pairing
        img_stem = Path(img_fname).stem
        lbl_local = label_dir / f"{img_stem}.geojson"

        print(f"[{i}/{len(selected)}] Downloading {tid} ...")
        download_file(img_key, img_dir / img_fname)
        download_file(lbl_key, lbl_local)

    print(f"\n✅ Downloaded {len(selected)} tile pairs to {output_dir}")
    print(f"   Images: {img_dir}")
    print(f"   Labels: {label_dir}")


if __name__ == "__main__":
    main()
