"""WHEN layer: 24 h precipitation forecast (date-aware).

Sourced from Open-Meteo (free, no key, plain JSON - no GRIB parsing). NOAA
retired the NOMADS OPeNDAP GFS server (Service Change Notice 25-81), so GFS is
fetched *through* Open-Meteo: its GFS global model is pinned for provenance
parity, with the default multi-model blend as fallback.

For a **future/near date** the live forecast API is used (day offset from today).
For a **past date** the historical-forecast API is used (Open-Meteo's archive of
past model runs) - this is what lets the dashboard reconstruct earlier days.

Returns a dict with basin/window mean forecast (mm/day) and provenance.
"""
import datetime as dt
import logging
import time

import numpy as np
import requests

from . import config

log = logging.getLogger(__name__)


def _bbox_indices(lats, lons, bbox):
    lon_min, lon_max, lat_min, lat_max = bbox
    la = (lats >= lat_min) & (lats <= lat_max)
    lo = (lons >= lon_min) & (lons <= lon_max)
    return la, lo


def _get_json(url, params, tries=3):
    """GET with retry+backoff. Open-Meteo throttles bursts with hung sockets;
    a couple of retries clears the transient read timeouts."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - retry transient failures
            last = e
            log.debug("open-meteo attempt %d/%d failed: %s", attempt + 1, tries, e)
            if attempt < tries - 1:
                time.sleep(3 + 4 * attempt)   # 3s, 7s
    raise last


def _openmeteo_precip(model, source, valid_date, today):
    """Precipitation_sum for ``valid_date`` on a 0.5deg grid via Open-Meteo.

    Live (valid_date >= today): forecast API, indexed by the day offset.
    Past (valid_date < today): historical-forecast API for that single day.
    ``model`` pins a single NWP model (e.g. "gfs_global"); ``None`` = best-match
    blend. Raises on transport/API error so get_forecast() can fall through.
    """
    historical = valid_date < today
    if historical:
        url = config.OPEN_METEO_ARCHIVE_URL
        day_params = {"start_date": valid_date.isoformat(),
                      "end_date": valid_date.isoformat()}
        idx = 0
    else:
        url = config.OPEN_METEO_URL
        idx = (valid_date - today).days              # 0 = today, 1 = tomorrow
        day_params = {"forecast_days": idx + 1}

    lats = np.arange(config.LAT_MIN, config.LAT_MAX + 0.001, 0.5)
    lons = np.arange(config.LON_MIN, config.LON_MAX + 0.001, 0.5)
    points = [(la, lo) for la in lats for lo in lons]

    values = np.full(len(points), np.nan, dtype="float32")
    B = 250   # all study-window points fit in one request (fewer calls = less throttling)
    for start in range(0, len(points), B):
        chunk = points[start:start + B]
        params = {
            "latitude": ",".join(f"{la:.2f}" for la, _ in chunk),
            "longitude": ",".join(f"{lo:.2f}" for _, lo in chunk),
            "daily": "precipitation_sum",
            "timezone": "UTC",
            **day_params,
        }
        if model:
            params["models"] = model
        payload = _get_json(url, params)
        if isinstance(payload, dict):
            payload = [payload]
        for j, loc in enumerate(payload):
            days = loc["daily"]["precipitation_sum"]
            val = days[idx] if len(days) > idx else days[-1]
            values[start + j] = np.nan if val is None else val

    grid = values.reshape(len(lats), len(lons))
    la_m, lo_m = _bbox_indices(lats, lons, config.WINDOW_BBOX)
    result = {
        "source": source,
        "basin_mm": float(np.nanmean(grid)),
        "window_mm": float(np.nanmean(grid[np.ix_(la_m, lo_m)])),
        "valid_hours": ("archived forecast" if historical
                        else "next calendar day (UTC)"),
    }
    log.info("forecast (%s): %s", valid_date, result)
    return result


def get_forecast(valid_date=None, today=None):
    """Best-available precipitation forecast for ``valid_date`` with provenance.

    Primary: GFS global via Open-Meteo. Fallback: Open-Meteo default blend.
    ``valid_date``/``today`` are ``datetime.date`` (default: tomorrow / today UTC).
    """
    today = today or dt.datetime.now(dt.timezone.utc).date()
    valid_date = valid_date or (today + dt.timedelta(days=1))
    historical = valid_date < today
    tag = "archived" if historical else "forecast"
    sources = (
        (config.OPEN_METEO_MODEL,
         f"GFS 0.25deg global via Open-Meteo ({tag}, models={config.OPEN_METEO_MODEL})"),
        (None, f"Open-Meteo API ({tag}, default multi-model blend)"),
    )
    for model, source in sources:
        try:
            return _openmeteo_precip(model, source, valid_date, today)
        except Exception as e:  # noqa: BLE001 - try the next source
            log.warning("forecast source '%s' failed: %s", source, e)
    raise RuntimeError(f"no forecast source reachable for {valid_date}")
