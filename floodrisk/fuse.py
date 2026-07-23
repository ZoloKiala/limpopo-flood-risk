"""FUSE: risk = susceptibility x rain factor, floored by observed SAR water."""
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from . import config

log = logging.getLogger(__name__)


def compute_rain_factor(forecast, thresholds):
    """How exceptional is tomorrow's rain vs the historical 95th percentile?"""
    value = 0.5 * (forecast["basin_mm"] + forecast["window_mm"])
    p95 = 0.5 * (thresholds["basin_p95_mm"] + thresholds["window_p95_mm"])
    return float(np.clip(value / max(p95, 1e-6), 0.0, config.RAIN_FACTOR_CAP))


def _observed_water_on_grid(observation, profile):
    """Reproject the SAR NOW water probability onto the risk grid.

    The now-water map lives in the Sentinel-1 native UTM grid over a small
    footprint; the risk grid is EPSG:4326 over the whole window. Returns
    (prob (H, W) 0-1, coverage (H, W) bool) on the risk grid, or (None, None)
    when there is no usable observation.
    """
    if not observation or not observation.get("geotiff"):
        return None, None
    path = Path(observation["geotiff"])
    if not path.exists():
        return None, None

    from rasterio.warp import Resampling, reproject

    h, w = profile["height"], profile["width"]
    prob = np.zeros((h, w), "float32")
    cov = np.zeros((h, w), "float32")
    with rasterio.open(path) as src:
        src_prob = src.read(1)                       # band 1 = P(open water)
        kw = dict(src_crs=src.crs, src_transform=src.transform,
                  dst_crs=profile["crs"], dst_transform=profile["transform"],
                  resampling=Resampling.average)
        reproject(source=src_prob, destination=prob, **kw)
        reproject(source=np.ones((src.height, src.width), "float32"),
                  destination=cov, **kw)            # where the footprint lands
    return prob, cov > 0.0


def fuse(rain_factor, valid_date, observation=None, static_dir=None,
         output_dir=None):
    static_dir = static_dir or config.STATIC_DIR
    output_dir = output_dir or config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(static_dir / "susceptibility.tif") as src:
        suscept = src.read(1)
        permanent = src.read(2).astype(bool)
        profile = src.profile

    risk = np.clip(suscept * rain_factor, 0.0, 1.0).astype("float32")

    # NOW layer as a floor on risk: SAR-confirmed open water (not the permanent
    # river) means it is *already* wet, so the index cannot read lower than the
    # observed water confidence there. Reported area therefore blends forecast
    # risk with observed inundation. Outside the S1 footprint, risk is untouched.
    obs_prob, coverage = _observed_water_on_grid(observation, profile)
    fused = obs_prob is not None
    if fused:
        obs_contrib = np.where(coverage & ~permanent, obs_prob, 0.0).astype("float32")
        risk = np.maximum(risk, obs_contrib)
    else:
        obs_contrib = np.zeros_like(risk)

    high = risk >= config.RISK_HIGH
    moderate = (risk >= config.RISK_MODERATE) & ~high

    out_path = output_dir / f"flood_risk_{valid_date:%Y%m%d}.tif"
    profile.update(count=4)
    # The inherited susceptibility profile carries a block size but not
    # TILED=YES; drop the stale block hints so GDAL picks a valid layout.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(risk, 1)
        dst.write(suscept, 2)
        dst.write(permanent.astype("float32"), 3)
        dst.write(obs_contrib, 4)
        for b, d in [(1, f"risk index valid {valid_date:%Y-%m-%d} "
                         "(forecast x susceptibility, floored by observed water)"),
                     (2, "static susceptibility"),
                     (3, "permanent water mask"),
                     (4, "observed open water now (SAR, reprojected to grid)")]:
            dst.set_band_description(b, d)

    km2 = (config.CELL / 1000.0) ** 2
    stats = {
        "rain_factor": rain_factor,
        "high_risk_fraction": float(high.mean()),
        "high_risk_km2": float(high.sum() * km2),
        "moderate_risk_fraction": float(moderate.mean()),
        "moderate_risk_km2": float(moderate.sum() * km2),
        "observation_fused": fused,
        "geotiff": str(out_path),
    }
    if fused:
        observed = (obs_prob >= config.WATER_PROB_THRESH) & coverage & ~permanent
        stats["observed_water_km2"] = float(observed.sum() * km2)
        stats["observed_coverage_km2"] = float(coverage.sum() * km2)
    log.info("risk product: %s", stats)
    return stats


def load_thresholds(static_dir=None):
    static_dir = static_dir or config.STATIC_DIR
    return json.loads((static_dir / "thresholds.json").read_text())
