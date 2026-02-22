"""
Data loading, mask creation, synthetic tile generation, and PyTorch dataset.

Handles SpaceNet-style GeoTIFF + GeoJSON labels, multi-band imagery,
and provides a synthetic fallback for demos without external data.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape as shapely_shape

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> Tuple[np.ndarray, rasterio.Affine, Optional[object]]:
    """
    Load a GeoTIFF image.

    Returns:
        image  — np.ndarray with shape (bands, H, W)
        transform — rasterio Affine
        crs    — rasterio CRS or None
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with rasterio.open(path) as src:
        image = src.read()  # (bands, H, W)
        transform = src.transform
        crs = src.crs
    return image, transform, crs


# ---------------------------------------------------------------------------
# RGB display prep
# ---------------------------------------------------------------------------

def prepare_display_rgb(
    image: np.ndarray,
    rgb_bands: Tuple[int, int, int] = (0, 1, 2),
    clip_percentile: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Convert a raster array to a display-ready uint8 RGB image (H, W, 3).

    Handles:
    - (bands, H, W) → transposes to (H, W, bands)
    - (H, W, bands) → keeps as-is
    - >3 bands → selects *rgb_bands* indices
    - 1 band → pseudo-RGB (grayscale → 3-channel)
    - Percentile clipping for contrast stretch
    """
    img = image.copy().astype(np.float32)

    # Detect and normalise axis order
    if img.ndim == 2:
        # Single band (H, W) → (H, W, 1)
        img = img[:, :, np.newaxis]
    elif img.ndim == 3:
        # If first dim is small (≤ ~16) assume (bands, H, W)
        if img.shape[0] <= 16 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))  # → (H, W, bands)
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Band selection
    n_bands = img.shape[2]
    if n_bands == 1:
        img = np.repeat(img, 3, axis=2)
    elif n_bands >= 3:
        r, g, b = rgb_bands
        max_idx = max(r, g, b)
        if max_idx >= n_bands:
            warnings.warn(
                f"Requested RGB bands {rgb_bands} but image has {n_bands} bands. "
                f"Falling back to first 3 bands."
            )
            r, g, b = 0, 1, 2
        img = img[:, :, [r, g, b]]
    else:
        # 2-band edge case: pad with zeros
        pad = np.zeros((*img.shape[:2], 1), dtype=np.float32)
        img = np.concatenate([img, pad], axis=2)

    # Percentile stretch
    lo = np.percentile(img, clip_percentile[0])
    hi = np.percentile(img, clip_percentile[1])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)

    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_geojson_footprints(path: str | Path) -> list:
    """
    Load building footprints from a GeoJSON file.

    Returns a list of Shapely geometries.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    geometries = []
    features = data.get("features", [])
    for feat in features:
        geom = feat.get("geometry")
        if geom is not None:
            try:
                geometries.append(shapely_shape(geom))
            except Exception as e:
                warnings.warn(f"Skipping invalid geometry: {e}")
    return geometries


def rasterize_footprints(
    geometries: list,
    transform: rasterio.Affine,
    shape: Tuple[int, int],
) -> np.ndarray:
    """
    Burn vector footprints into a binary mask.

    Args:
        geometries: list of Shapely geometries
        transform: affine transform of the output raster
        shape: (height, width) of the output raster

    Returns:
        mask — np.ndarray uint8 (H, W), 1 = building, 0 = background
    """
    if not geometries:
        warnings.warn("No geometries to rasterize — returning empty mask.")
        return np.zeros(shape, dtype=np.uint8)

    mask = rio_rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def load_or_create_mask(
    image_path: str | Path,
    label_path: str | Path,
) -> Tuple[np.ndarray, rasterio.Affine, Optional[object]]:
    """
    Load image and create a binary mask from labels.

    - If *label_path* is a .tif/.tiff, loads it directly as a raster mask.
    - If *label_path* is a .geojson/.json, rasterizes footprints.

    Returns (mask, transform, crs).
    """
    image, transform, crs = load_image(image_path)
    h, w = image.shape[1], image.shape[2]

    label_path = Path(label_path)
    ext = label_path.suffix.lower()

    if ext in (".tif", ".tiff"):
        with rasterio.open(label_path) as src:
            mask = src.read(1)
        mask = (mask > 0).astype(np.uint8)
    elif ext in (".geojson", ".json"):
        geometries = load_geojson_footprints(label_path)
        mask = rasterize_footprints(geometries, transform, (h, w))
    else:
        raise ValueError(f"Unsupported label format: {ext}")

    return mask, transform, crs


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_tile(
    height: int = 256,
    width: int = 256,
    num_buildings: int = 8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine, None]:
    """
    Generate a synthetic satellite tile with fake buildings for demo.

    Returns:
        image     — (3, H, W) uint8 fake RGB
        mask      — (H, W) uint8 binary mask
        transform — affine from_bounds (pixel coordinates)
        crs       — None (synthetic data has no real CRS)
    """
    rng = np.random.RandomState(seed)

    # Background: noisy green/brown terrain
    image = np.zeros((3, height, width), dtype=np.uint8)
    image[0] = rng.randint(60, 100, (height, width), dtype=np.uint8)   # R
    image[1] = rng.randint(80, 130, (height, width), dtype=np.uint8)   # G
    image[2] = rng.randint(50, 80, (height, width), dtype=np.uint8)    # B

    mask = np.zeros((height, width), dtype=np.uint8)

    margin = 20
    for _ in range(num_buildings):
        bw = rng.randint(15, 50)
        bh = rng.randint(15, 50)
        bx = rng.randint(margin, width - bw - margin)
        by = rng.randint(margin, height - bh - margin)

        # Add a little rotation/skew via small random offsets
        noise_x = rng.randint(-3, 4)
        noise_y = rng.randint(-3, 4)
        x1, x2 = bx + noise_x, bx + bw
        y1, y2 = by + noise_y, by + bh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Building colour: light grey/brown roof
        roof_r = rng.randint(150, 200)
        roof_g = rng.randint(140, 180)
        roof_b = rng.randint(130, 170)
        image[0, y1:y2, x1:x2] = roof_r
        image[1, y1:y2, x1:x2] = roof_g
        image[2, y1:y2, x1:x2] = roof_b

        mask[y1:y2, x1:x2] = 1

    transform = from_bounds(0, 0, width, height, width, height)
    return image, mask, transform, None


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class SpaceNetDataset(Dataset):
        """
        Simple PyTorch dataset for binary building segmentation.

        Expects pairs of (image_path, mask_path) where mask is a
        single-band raster or can be derived from GeoJSON labels.
        """

        def __init__(
            self,
            image_paths: List[str | Path],
            mask_paths: List[str | Path],
            transform_fn=None,
            target_size: Tuple[int, int] = (256, 256),
        ):
            assert len(image_paths) == len(mask_paths), (
                "image_paths and mask_paths must have the same length"
            )
            self.image_paths = [Path(p) for p in image_paths]
            self.mask_paths = [Path(p) for p in mask_paths]
            self.transform_fn = transform_fn
            self.target_size = target_size

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int):
            # Load image
            img, tfm, crs = load_image(self.image_paths[idx])
            # Ensure 3-band
            if img.shape[0] > 3:
                img = img[:3]
            elif img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

            # Load mask
            mask_path = self.mask_paths[idx]
            ext = mask_path.suffix.lower()
            if ext in (".tif", ".tiff"):
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
            elif ext in (".geojson", ".json"):
                geoms = load_geojson_footprints(mask_path)
                mask = rasterize_footprints(geoms, tfm, (img.shape[1], img.shape[2]))
                mask = mask.astype(np.float32)
            else:
                raise ValueError(f"Unsupported mask format: {ext}")

            # Resize if needed
            from PIL import Image as PILImage
            if (img.shape[1], img.shape[2]) != self.target_size:
                # img: (C, H, W)
                pil_img = PILImage.fromarray(
                    np.transpose(img, (1, 2, 0)).astype(np.uint8)
                )
                pil_img = pil_img.resize(
                    (self.target_size[1], self.target_size[0]),
                    PILImage.BILINEAR,
                )
                img = np.transpose(np.array(pil_img), (2, 0, 1)).astype(np.float32)

                pil_mask = PILImage.fromarray(
                    (mask * 255).astype(np.uint8), mode="L"
                )
                pil_mask = pil_mask.resize(
                    (self.target_size[1], self.target_size[0]),
                    PILImage.NEAREST,
                )
                mask = (np.array(pil_mask) > 127).astype(np.float32)
            else:
                img = img.astype(np.float32)

            # Normalize to [0, 1]
            if img.max() > 1.0:
                img = img / 255.0

            img_tensor = torch.from_numpy(img)        # (3, H, W)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

            if self.transform_fn:
                img_tensor, mask_tensor = self.transform_fn(img_tensor, mask_tensor)

            return img_tensor, mask_tensor
else:
    # Stub so imports don't break
    class SpaceNetDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SpaceNetDataset.")
