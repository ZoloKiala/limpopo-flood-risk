"""Sentinel-1 SAR helpers shared by training (static) and inference (daily).

The single source of truth for how backscatter becomes model input. Training
(Sen1Floods11 sigma0 dB) and inference (Planetary Computer GRD amplitude DN)
consume DIFFERENT physical quantities, so the preprocessing here deliberately
erases the absolute-calibration difference:

  amplitude/dB  --log-->  per-image z-score  -->  clip to +/-3, scale to [-1, 1]

Per-scene standardisation removes an additive calibration offset (subtracted by
the mean) and a multiplicative gain (divided out by the std). What survives is
the *shape* of the backscatter distribution - and open water is what it always
was in SAR: specular, so much darker than its surroundings. That relative
signature is what the ViT learns, which is why an uncalibrated GRD scene and a
calibrated training chip land in the same feature space. See README "Limits".
"""
import logging

import numpy as np

from . import config

log = logging.getLogger(__name__)


def amplitude_to_db(dn):
    """Planetary Computer GRD amplitude (uint16 DN, nodata 0) -> dB-like float.

    intensity ~ DN**2, so 10*log10(DN**2) = 20*log10(DN). The absolute scale is
    irrelevant here because standardise() z-scores it away; what matters is that
    it shares the (roughly log-normal) shape of the sigma0 dB training data.
    Zero DN is sensor nodata - flagged NaN so it is excluded downstream.
    """
    dn = dn.astype("float32")
    out = np.full(dn.shape, np.nan, dtype="float32")
    valid = dn > 0
    out[valid] = 20.0 * np.log10(dn[valid])
    return out


def standardise(vv_db, vh_db):
    """Two dB bands -> (H, W, 2) float32 model input, per-image standardised.

    NaNs (nodata) are ignored in the statistics and land at 0 (the post-z-score
    mean), so masked pixels read as "average", never as spuriously dark water.
    """
    feats = []
    for band in (vv_db, vh_db):
        b = band.astype("float32")
        m = np.nanmean(b)
        s = np.nanstd(b) + 1e-6
        z = np.clip((b - m) / s, -3.0, 3.0) / 3.0
        feats.append(np.nan_to_num(z, nan=0.0))
    return np.stack(feats, axis=-1).astype("float32")


def valid_mask(vv_db, vh_db):
    """True where both polarisations carry real data (used for water fraction)."""
    return np.isfinite(vv_db) & np.isfinite(vh_db)


def tile(arr, tile_px):
    """(H, W, C) -> (n*n, tile, tile, C), row-major, H and W multiples of tile."""
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1
    a = arr.reshape(h, w, c)
    ny, nx = h // tile_px, w // tile_px
    return (a.reshape(ny, tile_px, nx, tile_px, c)
             .swapaxes(1, 2)
             .reshape(ny * nx, tile_px, tile_px, c))


def untile(tiles, h, w):
    """Inverse of tile() for a single-channel prediction stack -> (H, W)."""
    tile_px = tiles.shape[1]
    ny, nx = h // tile_px, w // tile_px
    return (tiles.reshape(ny, nx, tile_px, tile_px)
                 .swapaxes(1, 2)
                 .reshape(h, w))


def build_model():
    """Fresh 2-channel water-segmentation ViT (built, ready for weights/train)."""
    import tensorflow as tf

    from .models import PatchViT
    model = PatchViT(config.SAR_TILE, config.SAR_TILE,
                     config.SAR_PATCH, config.SAR_CHANNELS)
    model(tf.zeros((1, config.SAR_TILE, config.SAR_TILE, config.SAR_CHANNELS)))
    return model


def load_model(weights_path):
    """Load the trained SAR ViT, or None if weights are absent."""
    if not weights_path.exists():
        log.warning("SAR model weights missing (%s) - run build-static",
                    weights_path)
        return None
    model = build_model()
    model.load_weights(weights_path)
    return model


def predict_water(model, features):
    """(H, W, 2) features -> P(water) map (H, W), tiling to the ViT grid."""
    import tensorflow as tf

    h, w = features.shape[:2]
    tiles = tile(features, config.SAR_TILE)
    logits = model.predict(tiles, batch_size=config.SAR_BATCH, verbose=0)[..., 0]
    prob = tf.sigmoid(logits).numpy()
    return untile(prob, h, w)
