"""Quick end-to-end pipeline test with the trained model."""
from pathlib import Path
import numpy as np
import torch

from src.data import load_image
from src.infer import load_model, predict_mask, save_mask_geotiff
from src.vectorize import mask_to_polygons
from src.estimate import SolarConfig, estimate_all_roofs

# Pick a test tile (last one)
imgs = sorted(Path("data/raw/images").glob("*.tif"))
test_tile = imgs[-1]
print(f"Test tile: {test_tile.name}")

device = torch.device("cpu")

image, transform, crs = load_image(test_tile)
print(f"Image: {image.shape}, CRS: {crs}")

model = load_model("checkpoints/best.pth", device=device)
mask = predict_mask(model, image, device=device, threshold=0.5)
print(f"Predicted mask: shape={mask.shape}, building_pct={100*mask.mean():.1f}%")

# Save predicted mask
save_mask_geotiff(mask, "outputs/test_pred_mask.tif", transform=transform, crs=crs)

# Vectorize
polys = mask_to_polygons(mask, transform=transform, crs=crs, min_area=10)
print(f"Polygons found: {len(polys)}")
if polys:
    largest = polys[0]
    print(f"  Largest: {largest['area_value']:.1f} {largest['area_unit']}")

# Estimate
per_roof, agg = estimate_all_roofs(polys)
print(f"Total roofs: {agg['num_roofs']}")
print(f"Total area: {agg['total_roof_area']:.1f} {agg['total_roof_area_unit']}")
print(f"Estimated kW: {agg['total_system_kw']:.2f}")
print(f"Monthly kWh: {agg['total_monthly_kwh']:.0f}")
print(f"Annual kWh: {agg['total_annual_kwh']:.0f}")
print()
print("PIPELINE TEST PASSED")
