"""WHEN layer: 24 h precipitation forecast.

Sourced from the Open-Meteo forecast API (free, no key, plain JSON - no GRIB
parsing). NOAA retired the NOMADS OPeNDAP GFS server (Service Change Notice
25-81), so GFS is now fetched *through* Open-Meteo: we pin its GFS global model
for provenance parity with the original design, and fall back to Open-Meteo's
default multi-model blend if that single model is unavailable.

Returns a dict with basin/window mean forecast (mm/day) and provenance.
"""
import datetime as dt
import logging

import numpy as np
import requests

from . import config

log = logging.getLogger(__name__)


def _bbox_indices(lats, lons, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    la = (lats >= lat_min) & (lats <= lat_max)
    lo = (lons >= lon_min) & (lons <= lon_max)
    return la, lo


def _openmeteo_precip(model, source):
    """Tomorrow's precipitation_sum on a 0.5deg grid via Open-Meteo.

    ``model`` pins a single NWP model (e.g. "gfs_global"); ``None`` uses
    Open-Meteo's default best-match blend. Raises on any transport/API error so
    get_forecast() can fall through to the next source.
    """
    lats = np.arange(config.LAT_MIN, config.LAT_MAX + 0.001, 0.5)
    lons = np.arange(config.LON_MIN, config.LON_MAX + 0.001, 0.5)
    points = [(la, lo) for la in lats for lo in lons]

    values = np.full(len(points), np.nan, dtype="float32")
    B = 50
    for start in range(0, len(points), B):
        chunk = points[start:start + B]
        params = {
            "latitude": ",".join(f"{la:.2f}" for la, _ in chunk),
            "longitude": ",".join(f"{lo:.2f}" for _, lo in chunk),
            "daily": "precipitation_sum",
            "forecast_days": 2,
            "timezone": "UTC",
        }
        if model:
            params["models"] = model
        r = requests.get(config.OPEN_METEO_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict):
            payload = [payload]
        for j, loc in enumerate(payload):
            days = loc["daily"]["precipitation_sum"]
            day1 = days[1] if len(days) > 1 else days[0]
            values[start + j] = np.nan if day1 is None else day1

    grid = values.reshape(len(lats), len(lons))
    la_m, lo_m = _bbox_indices(lats, lons, config.WINDOW_BBOX)
    result = {
        "source": source,
        "basin_mm": float(np.nanmean(grid)),
        "window_mm": float(np.nanmean(grid[np.ix_(la_m, lo_m)])),
        "valid_hours": "next calendar day (UTC)",
    }
    log.info("forecast: %s", result)
    return result


def get_forecast():
    """Best-available 24 h precipitation forecast with provenance.

    Primary: GFS global via Open-Meteo. Fallback: Open-Meteo default blend.
    """
    sources = (
        (config.OPEN_METEO_MODEL, f"GFS 0.25deg global via Open-Meteo "
                                  f"(models={config.OPEN_METEO_MODEL})"),
        (None, "Open-Meteo forecast API (default multi-model blend)"),
    )
    for model, source in sources:
        try:
            return _openmeteo_precip(model, source)
        except Exception as e:  # noqa: BLE001 - try the next source
            log.warning("forecast source '%s' failed: %s", source, e)
    raise RuntimeError("no forecast source reachable (Open-Meteo GFS + blend)")
