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
from rasterio.windows import from_bounds
from scipy import ndimage
import tensorflow as tf

from . import config, sar
from .models import PatchViT, masked_bce

log = logging.getLogger(__name__)


def _get_file(fname, url, tries=3):
    """tf.keras.utils.get_file with retry+backoff (IRI/CHIRPS + S3 can flake)."""
    import time

    last = None
    for attempt in range(tries):
        try:
            return tf.keras.utils.get_file(fname, url)
        except Exception as e:  # noqa: BLE001 - retry transient download failures
            last = e
            log.warning("download %s failed (attempt %d/%d): %s",
                        fname, attempt + 1, tries, e)
            if attempt < tries - 1:
                time.sleep(5 + 10 * attempt)   # 5s, 15s
    raise RuntimeError(f"could not download {url}: {last}")


def _norm(x):
    lo, hi = np.percentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1).astype("float32")


def _dem_tokens(bbox):
    """Copernicus GLO-30 tile tokens covering bbox (lon_min, lon_max, lat_min, lat_max).

    A tile token "S{|lat|}_00_E{lon}_00" has SW corner (lon, lat) and spans one
    degree N and E. We enumerate the integer SW corners inside the bbox.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    tokens = []
    for lat0 in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        ns = f"S{-lat0:02d}" if lat0 < 0 else f"N{lat0:02d}"
        for lon0 in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            ew = f"E{lon0:03d}" if lon0 >= 0 else f"W{-lon0:03d}"
            tokens.append(f"{ns}_00_{ew}_00")
    return tokens


def build_susceptibility(static_dir):
    tile = config.TILE
    from rasterio.merge import merge
    from rasterio.transform import array_bounds

    tokens = _dem_tokens(config.MOSAIC_BBOX)
    log.info("mosaicking %d DEM tiles: %s", len(tokens), ", ".join(tokens))
    srcs = []
    for tok in tokens:
        url = config.DEM_TILE_URL.format(tile=tok)
        srcs.append(rasterio.open(_get_file(f"dem_{tok}.tif", url)))
    mosaic, mtransform = merge(srcs)
    crs = srcs[0].crs
    for s in srcs:
        s.close()

    # Crop each dimension to a whole number of ViT tiles (top-left origin kept).
    full = mosaic[0].astype("float32")
    h = (full.shape[0] // tile) * tile
    w = (full.shape[1] // tile) * tile
    elev = np.ascontiguousarray(full[:h, :w])
    del mosaic, full
    transform = mtransform
    bounds = array_bounds(h, w, transform)          # (left, bottom, right, top)
    log.info("mosaic %d x %d px, extent %s", w, h, tuple(round(b, 3) for b in bounds))

    log.info("mosaicking WBM (ocean mask) ...")
    wsrcs = [rasterio.open(_get_file(f"wbm_{tok}.tif",
                                     config.WBM_TILE_URL.format(tile=tok)))
             for tok in tokens]
    wbm, _ = merge(wsrcs)
    for s in wsrcs:
        s.close()
    ocean = wbm[0, :h, :w] == 1                      # Copernicus WBM: 1 = ocean
    del wbm
    log.info("ocean pixels: %.2f%%", 100 * ocean.mean())

    log.info("streaming GSW occurrence over the mosaic ...")
    with rasterio.open(config.GSW_URL) as src:
        occ = src.read(1, window=from_bounds(*bounds, src.transform),
                       out_shape=(h, w),
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

    ny, nx = h // tile, w // tile
    X = features.reshape(ny, tile, nx, tile, 6).swapaxes(1, 2).reshape(-1, tile, tile, 6)
    Y = label.reshape(ny, tile, nx, tile, 1).swapaxes(1, 2).reshape(-1, tile, tile, 1)
    col = np.tile(np.arange(nx), ny)                           # tile column, row-major
    split = max(1, int(nx * 0.8))
    tr, va = col < split, col >= split                         # spatial (E strip) split

    model = PatchViT(tile, tile, config.PATCH, 6)
    model(tf.zeros((1, tile, tile, 6)))
    model.compile(optimizer=tf.keras.optimizers.AdamW(3e-4, weight_decay=1e-4),
                  loss=masked_bce)
    log.info("training susceptibility ViT (%d epochs, %d/%d train/val tiles) ...",
             config.SUS_EPOCHS, int(tr.sum()), int(va.sum()))
    model.fit(X[tr], Y[tr], validation_data=(X[va], Y[va]),
              batch_size=config.SUS_BATCH, epochs=config.SUS_EPOCHS, verbose=2)

    suscept = (tf.sigmoid(model.predict(X, batch_size=config.SUS_BATCH))
               .numpy()[..., 0]
               .reshape(ny, nx, tile, tile).swapaxes(1, 2).reshape(h, w))
    suscept[ocean] = 0.0                             # the sea is not flood-prone land

    out = static_dir / "susceptibility.tif"
    with rasterio.open(out, "w", driver="GTiff", height=h, width=w,
                       count=2, dtype="float32", crs=crs, transform=transform,
                       compress="deflate", predictor=3, tiled=True,
                       blockxsize=256, blockysize=256) as dst:
        dst.write(suscept.astype("float32"), 1)
        dst.write(permanent.astype("float32"), 2)
        dst.set_band_description(1, "flood susceptibility 0-1")
        dst.set_band_description(2, "permanent water mask")
    model.save_weights(static_dir / "sus_model.weights.h5")
    log.info("wrote %s (%d x %d)", out, w, h)


def build_thresholds(static_dir):
    """Historical rainfall percentiles from CHIRPS (basin + floodplain window)."""
    import pandas as pd
    import xarray as xr

    log.info("downloading CHIRPS record (~50 MB) ...")
    path = _get_file("chirps_limpopo_daily.nc", config.CHIRPS_URL)
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


def build_static(static_dir=None, only_missing=False, only=None):
    """Build the static products. ``only`` limits to one component
    (susceptibility|thresholds|sar), leaving the others as-is (must already
    exist, e.g. restored from cache)."""
    static_dir = static_dir or config.STATIC_DIR
    static_dir.mkdir(parents=True, exist_ok=True)
    steps = [
        ("susceptibility", "susceptibility.tif", build_susceptibility),
        ("thresholds", "thresholds.json", build_thresholds),
        ("sar", "sar_model.weights.h5", build_sar_model),
    ]
    for name, fname, fn in steps:
        if only and name != only:
            continue
        if only_missing and (static_dir / fname).exists():
            log.info("%s present - skipping", name)
            continue
        fn(static_dir)
