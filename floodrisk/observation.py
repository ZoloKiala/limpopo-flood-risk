"""NOW layer: latest Sentinel-1 SAR scene -> open-water fraction (all-weather).

C-band radar sees through cloud, so unlike the old optical NDWI this layer keeps
working during the very cyclones that cause the floods. The latest Sentinel-1
GRD scene over the monitored reach is fetched from Microsoft Planetary Computer
(free, anonymous SAS signing), segmented by the Sen1Floods11-trained ViT, and
reported as a water fraction plus a georeferenced NOW water map.

Situational-awareness only; the pipeline degrades gracefully (returns None) when
no scene is reachable or the SAR model has not been trained yet.
"""
import datetime as dt
import logging

import numpy as np
import rasterio
import requests
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from . import config, sar

log = logging.getLogger(__name__)

_MODEL = None   # lazily loaded, cached across daily calls within a process


def _sign(href):
    """Anonymously sign a Planetary Computer blob URL (no key required)."""
    r = requests.get(config.MPC_SIGN_URL, params={"href": href}, timeout=60)
    r.raise_for_status()
    return r.json()["href"]


def _read_window(href, lon, lat, size):
    """Windowed read around (lon, lat) -> (data, window_transform, crs).

    VV and VH COGs of one GRD product share a grid, so independent calls return
    aligned windows and identical georeferencing.
    """
    with rasterio.open(href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        row = int(np.clip(row - size // 2, 0, src.height - size))
        col = int(np.clip(col - size // 2, 0, src.width - size))
        win = Window(col, row, size, size)
        data = src.read(1, window=win).astype("float32")
        return data, src.window_transform(win), src.crs


def _write_now_map(prob, water, transform, crs, date, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"now_water_{date}.tif"
    with rasterio.open(out, "w", driver="GTiff", height=prob.shape[0],
                       width=prob.shape[1], count=2, dtype="float32", crs=crs,
                       transform=transform, compress="deflate", predictor=3) as dst:
        dst.write(prob.astype("float32"), 1)
        dst.write(water.astype("float32"), 2)
        dst.set_band_description(1, "P(open water) 0-1 (Sentinel-1 SAR ViT)")
        dst.set_band_description(2, "open water mask")
    return out


def latest_water_extent(date=None, output_dir=None, region=None):
    """Open-water fraction in the monitored reach from the latest S1 scene."""
    global _MODEL
    reg = config.get_region(region)
    date = date or dt.datetime.now(dt.timezone.utc)
    output_dir = output_dir or config.OUTPUT_DIR
    lon, lat = reg["now_point"]

    if _MODEL is None:
        _MODEL = sar.load_model(config.STATIC_DIR / "sar_model.weights.h5")
    if _MODEL is None:
        return None   # SAR model not built - degrade gracefully

    start = date - dt.timedelta(days=config.NOW_LOOKBACK_DAYS)
    query = {
        "collections": [config.S1_COLLECTION],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{date:%Y-%m-%d}T23:59:59Z",
        "query": {"sar:instrument_mode": {"eq": "IW"}},
        "limit": 15,
    }
    try:
        items = requests.post(config.MPC_STAC_URL, json=query, timeout=60)\
            .json().get("features", [])
    except Exception as e:  # noqa: BLE001
        log.warning("Planetary Computer search failed: %s", e)
        return None

    items.sort(key=lambda it: it["properties"]["datetime"], reverse=True)
    for item in items:
        assets = item["assets"]
        if "vv" not in assets or "vh" not in assets:
            continue
        try:
            vv_dn, transform, crs = _read_window(
                _sign(assets["vv"]["href"]), lon, lat, config.NOW_WINDOW)
            vh_dn, _, _ = _read_window(
                _sign(assets["vh"]["href"]), lon, lat, config.NOW_WINDOW)

            vv_db = sar.amplitude_to_db(vv_dn)
            vh_db = sar.amplitude_to_db(vh_dn)
            valid = sar.valid_mask(vv_db, vh_db)
            if valid.mean() < 0.5:
                continue   # swath edge / mostly nodata

            prob = sar.predict_water(_MODEL, sar.standardise(vv_db, vh_db))
            water = (prob > config.WATER_PROB_THRESH) & valid
            scene_date = item["properties"]["datetime"][:10]
            geotiff = _write_now_map(prob, water, transform, crs,
                                     scene_date.replace("-", ""), output_dir)

            result = {
                "sensor": "Sentinel-1 GRD (C-band SAR)",
                "source": "Microsoft Planetary Computer",
                "scene": item["id"],
                "datetime": scene_date,
                "polarization": "VV+VH",
                "orbit": item["properties"].get("sat:orbit_state", "n/a"),
                "water_fraction": float(water.sum() / max(valid.sum(), 1)),
                "geotiff": str(geotiff),
            }
            log.info("observation: %s", result)
            return result
        except Exception as e:  # noqa: BLE001
            log.debug("scene %s unusable: %s", item.get("id"), e)
    log.info("no usable Sentinel-1 scene in the last %d days",
             config.NOW_LOOKBACK_DAYS)
    return None
