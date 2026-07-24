"""WHEN layer, part 2: GloFAS river discharge via Open-Meteo's Flood API.

River discharge integrates upstream rainfall and the channel's routing, so it
carries the flood wave that arrives *days after* the rain - the lead-time signal
local precipitation alone misses (e.g. the lower Limpopo crests days after
upstream storms). Free and keyless, same provider family as the precipitation
forecast; the historical-forecast/reanalysis covers past dates for backfill.
"""
import datetime as dt
import logging

from . import config
from .forecast import _get_json        # shared retry+backoff GET

log = logging.getLogger(__name__)


def get_discharge(valid_date=None, today=None):
    """River discharge (m3/s) at the gauge point for ``valid_date`` (or None)."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    valid_date = valid_date or (today + dt.timedelta(days=1))
    lon, lat = config.DISCHARGE_POINT

    params = {"latitude": lat, "longitude": lon,
              "daily": "river_discharge", "timezone": "UTC"}
    if valid_date < today:
        params["start_date"] = params["end_date"] = valid_date.isoformat()
        idx = 0
    else:
        idx = (valid_date - today).days           # 0 = today, 1 = tomorrow
        params["forecast_days"] = idx + 1

    try:
        vals = _get_json(config.FLOOD_API_URL, params)["daily"]["river_discharge"]
        q = vals[idx] if len(vals) > idx else vals[-1]
    except Exception as e:  # noqa: BLE001 - discharge is best-effort
        log.warning("discharge fetch failed for %s: %s", valid_date, e)
        return None
    if q is None:
        return None
    result = {"river_discharge_m3s": float(q),
              "source": "GloFAS via Open-Meteo Flood API",
              "point": [lon, lat]}
    log.info("discharge (%s): %s", valid_date, result)
    return result
