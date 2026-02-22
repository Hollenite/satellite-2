"""
Download the WHU Building Dataset (aerial, raster labels, 0.3m resolution).

This is a free alternative to SpaceNet that requires NO AWS account.
Downloads from gpcv.whu.edu.cn — ~4.5 GB zip with:
  - 8,189 tiles (512×512) 
  - Binary raster masks (building / not-building)
  - Pre-split into train / val / test

We only extract a configurable subset of tiles to save time+space.

Usage:
    py scripts/download_whu.py --max_tiles 50
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen

# The 0.3m raster-label variant (smallest download w/ ready masks)
DATASET_URL = (
    "https://gpcv.whu.edu.cn/data/"
    "3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip"
)


def download_with_progress(url: str, dest: Path):
    """Download a file with a simple progress indicator."""
    print(f"⬇️  Downloading from:\n  {url}")
    print(f"  → {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {mb_done:.1f} / {mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)
        else:
            mb_done = downloaded / (1024 * 1024)
            print(f"\r  {mb_done:.1f} MB downloaded", end="", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(dest), reporthook=_progress)
    print()  # newline after progress


def extract_subset(zip_path: Path, output_dir: Path, max_tiles: int = 50):
    """
    Extract image + mask tile pairs from the WHU ZIP.

    WHU ZIP structure (typical):
      ├── image/  (or Images/ or img/)
      │   ├── 1.tif
      │   ├── 2.tif ...
      └── label/  (or Labels/ or gt/)
          ├── 1.tif
          ├── 2.tif ...

    We extract into:
      output_dir/images/*.tif
      output_dir/masks/*.tif
    """
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting up to {max_tiles} tile pairs from {zip_path.name} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()

        # Find image and label directories (case-insensitive)
        image_files = []
        label_files = []

        for name in all_names:
            lower = name.lower()
            # Skip directories and macOS metadata
            if name.endswith("/") or "__MACOSX" in name or ".DS_Store" in name:
                continue
            if not lower.endswith((".tif", ".tiff", ".png")):
                continue

            # Classify as image or label based on path
            if any(d in lower for d in ("/image/", "/images/", "/img/", "/train/image", "/train/images")):
                image_files.append(name)
            elif any(d in lower for d in ("/label/", "/labels/", "/gt/", "/mask/", "/masks/", "/train/label", "/train/labels")):
                label_files.append(name)

        if not image_files:
            # Try a more flexible approach: look at directory structure
            print("  ⚠️ Standard dirs not found. Listing ZIP contents for debug:")
            dirs = set()
            for n in all_names[:50]:
                parts = n.split("/")
                if len(parts) > 1:
                    dirs.add(parts[0] + "/" + (parts[1] if len(parts) > 2 else ""))
            for d in sorted(dirs):
                print(f"    {d}")
            print(f"  Total files in ZIP: {len(all_names)}")
            return 0

        # Sort for deterministic pairing
        image_files.sort()
        label_files.sort()

        # Build basename map for matching
        img_by_base = {}
        for f in image_files:
            base = Path(f).stem
            img_by_base[base] = f

        lbl_by_base = {}
        for f in label_files:
            base = Path(f).stem
            lbl_by_base[base] = f

        # Match pairs by basename
        matched = sorted(set(img_by_base.keys()) & set(lbl_by_base.keys()))
        selected = matched[:max_tiles]

        if not selected:
            print(f"  ⚠️ No matching pairs found!")
            print(f"  Image basenames sample: {list(img_by_base.keys())[:5]}")
            print(f"  Label basenames sample: {list(lbl_by_base.keys())[:5]}")
            return 0

        print(f"  Found {len(matched)} matched pairs, extracting {len(selected)}")

        for i, base in enumerate(selected, 1):
            img_src = img_by_base[base]
            lbl_src = lbl_by_base[base]
            ext_img = Path(img_src).suffix
            ext_lbl = Path(lbl_src).suffix

            # Extract image
            with zf.open(img_src) as src, open(img_dir / f"{base}{ext_img}", "wb") as dst:
                dst.write(src.read())

            # Extract mask
            with zf.open(lbl_src) as src, open(mask_dir / f"{base}{ext_lbl}", "wb") as dst:
                dst.write(src.read())

            if i % 10 == 0 or i == len(selected):
                print(f"  [{i}/{len(selected)}] extracted")

        return len(selected)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download WHU Building Dataset subset")
    parser.add_argument("--max_tiles", type=int, default=50,
                        help="Max tiles to extract (default: 50)")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Output directory for images/masks")
    parser.add_argument("--keep_zip", action="store_true",
                        help="Keep the downloaded ZIP file")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path = output_dir / "whu_building_dataset.zip"

    # Download
    if not zip_path.exists():
        download_with_progress(DATASET_URL, zip_path)
    else:
        print(f"✅ ZIP already exists: {zip_path}")

    # Extract subset
    n = extract_subset(zip_path, output_dir, max_tiles=args.max_tiles)
    print(f"\n✅ Extracted {n} tile pairs")
    print(f"   Images: {output_dir / 'images'}")
    print(f"   Masks:  {output_dir / 'masks'}")

    # Optionally clean up
    if not args.keep_zip and zip_path.exists():
        print(f"🗑️  Removing ZIP to save space ...")
        zip_path.unlink()

    print("\n🎉 Done! You can now train with:")
    print(f"   py -m src.train --data_dir {output_dir} --epochs 10")


if __name__ == "__main__":
    main()
