"""
Inference script — run a trained U-Net on a single tile and produce a binary mask.

The predicted mask is saved as a GeoTIFF (preserving CRS/transform from the
input image) and can be fed directly into the vectorization pipeline.

CLI usage:
    python -m src.infer --image data/raw/images/tile.tif --checkpoint checkpoints/best.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

from src.data import load_image, prepare_display_rgb
from src.utils import ensure_output_dir


def load_model(
    checkpoint_path: str | Path,
    encoder: str = "resnet34",
    device: torch.device = None,
) -> torch.nn.Module:
    """Load a trained U-Net from a checkpoint file."""
    if not HAS_SMP:
        raise ImportError("segmentation_models_pytorch is required for inference.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device = None,
    threshold: float = 0.5,
    target_size: tuple = (256, 256),
) -> np.ndarray:
    """
    Run inference on a single image tile.

    Args:
        model: trained U-Net
        image: (bands, H, W) raster array
        device: torch device
        threshold: logit threshold for binary mask
        target_size: model input size (H, W)

    Returns:
        mask — (H, W) uint8 binary mask at ORIGINAL resolution
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_h, orig_w = image.shape[1], image.shape[2]

    # Prepare 3-band input
    img = image.copy().astype(np.float32)
    if img.shape[0] > 3:
        img = img[:3]
    elif img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)

    # Resize to target
    from PIL import Image as PILImage
    pil = PILImage.fromarray(np.transpose(img, (1, 2, 0)).astype(np.uint8))
    pil = pil.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
    img_resized = np.transpose(np.array(pil), (2, 0, 1)).astype(np.float32)

    # Normalize
    if img_resized.max() > 1.0:
        img_resized = img_resized / 255.0

    # Inference
    tensor = torch.from_numpy(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)

    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask_small = (probs > threshold).astype(np.uint8)

    # Resize back to original resolution
    pil_mask = PILImage.fromarray(mask_small * 255, mode="L")
    pil_mask = pil_mask.resize((orig_w, orig_h), PILImage.NEAREST)
    mask = (np.array(pil_mask) > 127).astype(np.uint8)

    return mask


def save_mask_geotiff(
    mask: np.ndarray,
    output_path: str | Path,
    transform=None,
    crs=None,
):
    """Save binary mask as a single-band GeoTIFF."""
    output_path = Path(output_path)
    ensure_output_dir(output_path.parent)

    h, w = mask.shape
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": w,
        "height": h,
        "count": 1,
        "nodata": 0,
    }
    if transform is not None:
        profile["transform"] = transform
    if crs is not None:
        profile["crs"] = crs

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"✅ Saved predicted mask → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run U-Net inference on a tile")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input GeoTIFF image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output mask path (default: outputs/pred_mask.tif)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--rgb_bands", type=int, nargs=3, default=[0, 1, 2])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load
    image, transform, crs = load_image(args.image)
    print(f"Image shape: {image.shape}, CRS: {crs}")

    model = load_model(args.checkpoint, encoder=args.encoder, device=device)

    # Predict
    mask = predict_mask(
        model, image,
        device=device,
        threshold=args.threshold,
        target_size=tuple(args.target_size),
    )
    print(f"Predicted mask shape: {mask.shape}, unique values: {np.unique(mask)}")

    # Save
    output_path = args.output or "outputs/pred_mask.tif"
    save_mask_geotiff(mask, output_path, transform=transform, crs=crs)


if __name__ == "__main__":
    main()
