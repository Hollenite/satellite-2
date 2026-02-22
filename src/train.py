"""
Minimal U-Net binary segmentation training pipeline.

Uses segmentation_models_pytorch with a ResNet-34 encoder.
Designed to train on a tiny SpaceNet subset and produce a plausible
building mask — not SOTA, just "works tonight".

CLI usage:
    python -m src.train --data_dir data/raw --epochs 10 --lr 1e-3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("⚠️  segmentation_models_pytorch not installed. Install with:")
    print("    pip install segmentation-models-pytorch")

from src.data import SpaceNetDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    """Create a U-Net model with the specified encoder."""
    if not HAS_SMP:
        raise ImportError("segmentation_models_pytorch is required for training.")

    weights = "imagenet" if pretrained else None
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=1,
        activation=None,  # raw logits
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate. Returns (avg_loss, avg_dice)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            # Dice metric
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            smooth = 1e-6
            inter = (preds * masks).sum()
            dice = (2 * inter + smooth) / (preds.sum() + masks.sum() + smooth)
            total_dice += dice.item()
            n += 1

    avg_loss = total_loss / max(n, 1)
    avg_dice = total_dice / max(n, 1)
    return avg_loss, avg_dice


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_pairs(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find image-mask pairs in a directory.

    Expects either:
        data_dir/images/*.tif  + data_dir/masks/*.tif
    or:
        data_dir/images/*.tif  + data_dir/labels/*.geojson

    Returns (image_paths, mask_paths).
    """
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    label_dir = data_dir / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Expected images directory at {img_dir}. "
            f"Organise your SpaceNet data as:\n"
            f"  {data_dir}/images/*.tif\n"
            f"  {data_dir}/masks/*.tif  OR  {data_dir}/labels/*.geojson"
        )

    image_paths = sorted(img_dir.glob("*.tif"))
    if not image_paths:
        image_paths = sorted(img_dir.glob("*.tiff"))
    if not image_paths:
        raise FileNotFoundError(f"No .tif files found in {img_dir}")

    # Prefer raster masks over vector labels
    if mask_dir.exists():
        mask_paths = sorted(mask_dir.glob("*.tif"))
        if not mask_paths:
            mask_paths = sorted(mask_dir.glob("*.tiff"))
    elif label_dir.exists():
        mask_paths = sorted(label_dir.glob("*.geojson"))
        if not mask_paths:
            mask_paths = sorted(label_dir.glob("*.json"))
    else:
        raise FileNotFoundError(
            f"Expected masks/ or labels/ directory in {data_dir}"
        )

    # Match by count (simple approach)
    if len(image_paths) != len(mask_paths):
        min_n = min(len(image_paths), len(mask_paths))
        print(f"⚠️  {len(image_paths)} images vs {len(mask_paths)} masks. Using first {min_n}.")
        image_paths = image_paths[:min_n]
        mask_paths = mask_paths[:min_n]

    print(f"Found {len(image_paths)} image-mask pairs in {data_dir}")
    return image_paths, mask_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for building segmentation")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory with images/ and masks/ or labels/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--target_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--rgb_bands", type=int, nargs=3, default=[0, 1, 2],
                        help="Band indices for RGB display (0-indexed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    image_paths, mask_paths = discover_pairs(data_dir)

    dataset = SpaceNetDataset(
        image_paths, mask_paths,
        target_size=tuple(args.target_size),
    )

    # Train/val split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = create_model(encoder=args.encoder).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = ckpt_dir / "best.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Saved best checkpoint (dice={best_dice:.4f})")

    # Save final checkpoint
    torch.save(model.state_dict(), ckpt_dir / "final.pth")
    print(f"\n✅ Training done. Best val dice: {best_dice:.4f}")
    print(f"   Checkpoints saved in {ckpt_dir}/")


if __name__ == "__main__":
    main()
