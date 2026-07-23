"""One-time static build: susceptibility map + rainfall thresholds + SAR model.

Products written to STATIC_DIR (cache these between runs):
  susceptibility.tif   band 1 = P(flood-prone) 0-1, band 2 = permanent water mask
  sus_model.weights.h5 trained ViT weights (for retraining/audit)
  thresholds.json      CHIRPS-derived rainfall percentiles + metadata
  sar_model.weights.h5 Sen1Floods11-trained SAR water-segmentation ViT
  sar_model.json       SAR training provenance (dataset, chip counts, metrics)
"""
import json
import logging

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from scipy import ndimage
import tensorflow as tf

from . import config, sar
from .models import PatchViT, masked_bce

log = logging.getLogger(__name__)


def _norm(x):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype("float32")


def build_susceptibility(static_dir):
    size, tile = config.SIZE, config.TILE

    log.info("downloading DEM tile ...")
    dem_path = tf.keras.utils.get_file("dem_S25_E033.tif", config.DEM_URL)
    with rasterio.open(dem_path) as src:
        win = Window(8, 8, size, size)
        elev = src.read(1, window=win).astype("float32")
        transform, crs = src.window_transform(win), src.crs
        bounds = rasterio.windows.bounds(win, src.transform)

    log.info("streaming GSW occurrence window ...")
    with rasterio.open(config.GSW_URL) as src:
        occ = src.read(1, window=from_bounds(*bounds, src.transform),
                       out_shape=(size, size),
                       resampling=rasterio.enums.Resampling.nearest)

    valid = occ != 255
    permanent = (occ >= 90) & valid
    label = np.full(occ.shape, -1.0, dtype="float32")
    sel = valid & ~permanent
    label[sel] = ((occ >= 2) & (occ < 90))[sel].astype("float32")
    log.info("labels: %.2f%% flood-affected, %.2f%% permanent water",
             100 * (label == 1).mean(), 100 * permanent.mean())

    sy, sx = np.gradient(elev, config.CELL)
    dist_px, idx = ndimage.distance_transform_edt(~permanent, return_indices=True)
    features = np.stack([
        _norm(elev),
        _norm(np.hypot(sx, sy)),
        _norm(ndimage.laplace(elev)),
        _norm(elev - ndimage.uniform_filter(elev, 15)),
        _norm(elev - elev[idx[0], idx[1]]),                    # HAND
        _norm(np.log1p(dist_px * config.CELL / 1000.0)),
    ], axis=-1)

    n = size // tile
    X = features.reshape(n, tile, n, tile, 6).swapaxes(1, 2).reshape(-1, tile, tile, 6)
    Y = label.reshape(n, tile, n, tile, 1).swapaxes(1, 2).reshape(-1, tile, tile, 1)
    col = np.tile(np.arange(n), n)
    tr, va = col <= 21, col >= 22                              # spatial split

    model = PatchViT(tile, tile, config.PATCH, 6)
    model(tf.zeros((1, tile, tile, 6)))
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
                  loss=masked_bce)
    log.info("training susceptibility ViT (%d epochs) ...", config.SUS_EPOCHS)
    model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
              batch_size=config.SUS_BATCH, epochs=config.SUS_EPOCHS, verbose=2)

    suscept = (tf.sigmoid(model.predict(X, batch_size=config.SUS_BATCH))
               .numpy()[..., 0]
               .reshape(n, n, tile, tile).swapaxes(1, 2).reshape(size, size))

    out = static_dir / "susceptibility.tif"
    with rasterio.open(out, "w", driver="GTiff", height=size, width=size,
                       count=2, dtype="float32", crs=crs, transform=transform,
                       compress="deflate", predictor=3) as dst:
        dst.write(suscept.astype("float32"), 1)
        dst.write(permanent.astype("float32"), 2)
        dst.set_band_description(1, "flood susceptibility 0-1")
        dst.set_band_description(2, "permanent water mask")
    model.save_weights(static_dir / "sus_model.weights.h5")
    log.info("wrote %s", out)


def build_thresholds(static_dir):
    """Historical rainfall percentiles from CHIRPS (basin + floodplain window)."""
    import pandas as pd
    import xarray as xr

    log.info("downloading CHIRPS record (~50 MB) ...")
    path = tf.keras.utils.get_file("chirps_limpopo_daily.nc", config.CHIRPS_URL)
    try:
        ds = xr.open_dataset(path)
    except Exception:
        ds = xr.open_dataset(path, decode_times=False)
    ds = ds.sortby("Y", ascending=False)
    rain = np.nan_to_num(ds["prcp"].values.astype("float32"), nan=0.0)

    lon_min, lon_max, lat_min, lat_max = config.WINDOW_BBOX
    c0 = int((lon_min - config.LON_MIN) / 0.25)
    c1 = int((lon_max - config.LON_MIN) / 0.25)
    r0 = int((config.LAT_MAX - lat_max) / 0.25)
    r1 = int((config.LAT_MAX - lat_min) / 0.25)

    basin = rain.mean(axis=(1, 2))
    window = rain[:, r0:r1, c0:c1].mean(axis=(1, 2))
    rainy = basin > 1.0

    thresholds = {
        "basin_p95_mm": float(np.percentile(basin[rainy], 95)),
        "basin_p99_mm": float(np.percentile(basin[rainy], 99)),
        "window_p95_mm": float(np.percentile(window[basin > 1.0], 95)),
        "n_days": int(len(basin)),
        "source": "CHIRPS v2.0 daily-improved 0.25deg",
        "note": "percentiles over rainy days (basin mean > 1 mm)",
    }
    (static_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    log.info("thresholds: %s", thresholds)


def _load_split(name):
    """Fetch a Sen1Floods11 split CSV -> list of (s1_filename, label_filename)."""
    import csv

    url = f"{config.SEN1FLOODS_BASE}/{config.SEN1FLOODS_SPLITS[name]}"
    path = tf.keras.utils.get_file(f"sen1floods_{name}.csv", url)
    with open(path, newline="") as fh:
        return [(row[0], row[1]) for row in csv.reader(fh) if len(row) >= 2]


def _load_chip(s1_name, label_name):
    """Download one Sen1Floods11 chip -> (features (H,W,2), label (H,W)) or None.

    S1Hand is VV/VH sigma0 in dB; LabelHand is {-1 nodata, 0 dry, 1 water}.
    Returns None (and logs) on any fetch/read failure so a few dead URLs don't
    sink the whole build - the caller counts what actually loaded.
    """
    try:
        s1_url = f"{config.SEN1FLOODS_BASE}/{config.SEN1FLOODS_S1DIR}/{s1_name}"
        lb_url = f"{config.SEN1FLOODS_BASE}/{config.SEN1FLOODS_LABELDIR}/{label_name}"
        s1_path = tf.keras.utils.get_file(s1_name, s1_url)
        lb_path = tf.keras.utils.get_file(label_name, lb_url)
        with rasterio.open(s1_path) as src:
            vv = src.read(1).astype("float32")   # band 1 = VV (dB)
            vh = src.read(2).astype("float32")   # band 2 = VH (dB)
        with rasterio.open(lb_path) as src:
            label = src.read(1).astype("float32")
    except Exception as e:  # noqa: BLE001 - skip unreadable chips
        log.debug("chip %s unusable: %s", s1_name, e)
        return None
    return sar.standardise(vv, vh), label


def _prepare_split(name):
    """All chips in a split, tiled to the ViT grid -> (X, Y) or (None, None)."""
    pairs = _load_split(name)
    log.info("Sen1Floods11 %s split: %d chips", name, len(pairs))
    xs, ys = [], []
    for i, (s1_name, label_name) in enumerate(pairs, 1):
        chip = _load_chip(s1_name, label_name)
        if chip is None:
            continue
        feats, label = chip
        if feats.shape[0] != config.SAR_CHIP or feats.shape[1] != config.SAR_CHIP:
            continue
        xs.append(sar.tile(feats, config.SAR_TILE))
        ys.append(sar.tile(label[..., None], config.SAR_TILE))
        if i % 50 == 0:
            log.info("  %s: fetched %d/%d", name, i, len(pairs))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def build_sar_model(static_dir):
    """Train the SAR water-segmentation ViT on Sen1Floods11 hand-labeled chips."""
    log.info("preparing Sen1Floods11 chips (downloads ~0.8 GB, cached) ...")
    Xtr, Ytr = _prepare_split("train")
    Xva, Yva = _prepare_split("valid")
    if Xtr is None:
        raise RuntimeError("no Sen1Floods11 training chips could be fetched")
    log.info("SAR training tiles: %d train, %d valid",
             len(Xtr), 0 if Xva is None else len(Xva))

    model = sar.build_model()
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
                  loss=masked_bce)
    log.info("training SAR ViT (%d epochs) ...", config.SAR_EPOCHS)
    val = (Xva, Yva) if Xva is not None else None
    hist = model.fit(Xtr, Ytr, validation_data=val,
                     batch_size=config.SAR_BATCH, epochs=config.SAR_EPOCHS,
                     verbose=2)

    model.save_weights(static_dir / "sar_model.weights.h5")
    provenance = {
        "dataset": "Sen1Floods11 v1.1 hand-labeled (Bonafilia et al. 2020)",
        "source": config.SEN1FLOODS_BASE,
        "n_train_tiles": int(len(Xtr)),
        "n_valid_tiles": 0 if Xva is None else int(len(Xva)),
        "channels": "VV, VH (per-scene standardised log backscatter)",
        "tile": config.SAR_TILE,
        "patch": config.SAR_PATCH,
        "epochs": config.SAR_EPOCHS,
        "final_train_loss": float(hist.history["loss"][-1]),
        "final_val_loss": (float(hist.history["val_loss"][-1])
                           if "val_loss" in hist.history else None),
        "note": "masked BCE ignores label -1 (nodata); water = label 1",
    }
    (static_dir / "sar_model.json").write_text(json.dumps(provenance, indent=2))
    log.info("SAR model provenance: %s", provenance)


def build_static(static_dir=None, only_missing=False):
    static_dir = static_dir or config.STATIC_DIR
    static_dir.mkdir(parents=True, exist_ok=True)
    have_sus = (static_dir / "susceptibility.tif").exists()
    have_thr = (static_dir / "thresholds.json").exists()
    have_sar = (static_dir / "sar_model.weights.h5").exists()
    if only_missing and have_sus and have_thr and have_sar:
        log.info("static products present - skipping build")
        return
    if not (only_missing and have_sus):
        build_susceptibility(static_dir)
    if not (only_missing and have_thr):
        build_thresholds(static_dir)
    if not (only_missing and have_sar):
        build_sar_model(static_dir)
