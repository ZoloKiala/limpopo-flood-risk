"""Render the daily product into a self-contained HTML dashboard.

One portable file (``outputs/dashboard.html``) with the risk map inlined as a
data URI, an alert banner, forecast + risk-area stat tiles, the SAR NOW status
and full provenance. No JavaScript, no external assets - it opens straight from
disk, ships as a CI artifact, and can be published to GitHub Pages as-is.

The map is a single composite raster: a neutral grey susceptibility base (so
the flood-prone terrain is visible even on a dry LOW day), a YlOrRd sequential
overlay where the risk index rises, the permanent river in deep blue, and
SAR-observed open water in bright cyan. A standalone ``risk_map_<date>.png`` is
written alongside for reuse.
"""
import datetime as dt
import json
import logging
import struct
import zlib
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)

MAP_PX = 720   # long-edge render size; the study window is ~square

# --- colour ramps -----------------------------------------------------------
# Susceptibility base: neutral light->slate (a hue-less context layer that does
# not compete with the warm risk overlay). Risk: ColorBrewer YlOrRd sequential,
# monotonic in lightness. Water layers validated for CVD separation (dataviz).
_GREY = [(0.0, (238, 240, 243)), (1.0, (94, 104, 120))]
_YLORRD = [(0.00, (255, 255, 178)), (0.25, (254, 204, 92)),
           (0.50, (253, 141, 60)), (0.75, (240, 59, 32)),
           (1.00, (189, 0, 38))]
_PERMANENT_WATER = (30, 58, 138)    # deep blue  #1e3a8a
_OBSERVED_WATER = (34, 211, 238)    # bright cyan #22d3ee
_RISK_FLOOR = 0.05                  # below this, show the susceptibility base

ALERT_COLOR = {"LOW": "#16a34a", "MODERATE": "#d97706", "HIGH": "#dc2626"}
ALERT_BLURB = {
    "HIGH": "Dangerous rainfall coincides with flood-prone terrain.",
    "MODERATE": "Elevated rainfall over susceptible terrain — watch conditions.",
    "LOW": "No significant rainfall forecast; flood risk is low across the window.",
}


def _ramp(v, stops):
    """Map v in [0, 1] (any shape) to an (..., 3) uint8 array via linear stops."""
    xs = np.array([s[0] for s in stops], dtype="float32")
    cols = np.array([s[1] for s in stops], dtype="float32")
    v = np.clip(v, 0.0, 1.0)
    out = np.stack([np.interp(v, xs, cols[:, c]) for c in range(3)], axis=-1)
    return out.round().astype("uint8")


def _png_data_uri(rgba):
    """Encode an (H, W, 4) uint8 array as a base64 PNG data URI (stdlib only)."""
    import base64

    h, w = rgba.shape[:2]
    rows = np.zeros((h, 1 + w * 4), dtype="uint8")     # filter byte 0 per scanline
    rows[:, 1:] = rgba.reshape(h, w * 4)

    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
           + chunk(b"IEND", b""))
    return png, "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _render_map(geotiff_path):
    """Composite the 4-band risk GeoTIFF -> (png_bytes, data_uri, bounds) or None."""
    path = Path(geotiff_path)
    if not path.exists():
        log.warning("risk GeoTIFF missing for dashboard: %s", path)
        return None

    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        w = MAP_PX
        h = max(1, round(MAP_PX * src.height / src.width))
        data = src.read(out_shape=(src.count, h, w),
                        resampling=Resampling.bilinear).astype("float32")
        bounds = src.bounds

    risk, suscept, permanent = data[0], data[1], data[2]
    observed = data[3] if data.shape[0] > 3 else np.zeros_like(risk)

    img = _ramp(suscept, _GREY)
    hot = risk >= _RISK_FLOOR
    img[hot] = _ramp(risk, _YLORRD)[hot]
    img[permanent >= 0.5] = _PERMANENT_WATER
    img[observed >= config.WATER_PROB_THRESH] = _OBSERVED_WATER

    rgba = np.dstack([img, np.full(img.shape[:2], 255, "uint8")])
    png_bytes, uri = _png_data_uri(rgba)
    return png_bytes, uri, bounds


# --- HTML -------------------------------------------------------------------
def _tile(value, label, sub=""):
    sub = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="tile"><div class="val">{value}</div>'
            f'<div class="lab">{label}</div>{sub}</div>')


def _legend():
    grad = ", ".join(f"rgb{c}" for _, c in _YLORRD)
    return f"""
    <div class="legend">
      <div class="lrow">
        <span class="lname">Risk index</span>
        <span class="bar" style="background:linear-gradient(90deg,{grad})"></span>
        <span class="lends">0 &nbsp; low → high &nbsp; 1</span>
      </div>
      <div class="lrow swatches">
        <span><i style="background:rgb{_PERMANENT_WATER}"></i>Permanent river</span>
        <span><i style="background:rgb{_OBSERVED_WATER}"></i>Observed water now (SAR)</span>
        <span><i style="background:linear-gradient(90deg,rgb(238,240,243),rgb(94,104,120))"></i>Susceptibility (base)</span>
      </div>
    </div>"""


def _observation_card(observation):
    if not observation:
        return ('<div class="card"><h3>SAR NOW layer</h3>'
                '<p class="muted">No usable Sentinel-1 scene in the lookback '
                'window — today is forecast-only. The all-weather radar '
                'observation floors risk when a scene is available.</p></div>')
    frac = observation.get("water_fraction", 0.0)
    return (f'<div class="card"><h3>SAR NOW layer</h3>'
            f'<p><b>{frac:.1%}</b> open water in the monitored reach</p>'
            f'<p class="muted">{observation.get("sensor", "Sentinel-1 SAR")} '
            f'· scene {observation.get("datetime", "")} '
            f'· {observation.get("polarization", "")} '
            f'{observation.get("orbit", "")}<br>{observation.get("source", "")}</p>'
            f'</div>')


def build_dashboard(payload, output_dir=None, valid_date=None):
    """Write ``dashboard.html`` (+ standalone map PNG) from a bulletin payload."""
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = payload["valid"]
    stamp = valid.replace("-", "")

    forecast, thresholds = payload["forecast"], payload["thresholds"]
    risk, observation = payload["risk"], payload.get("observation")
    level = payload["alert_level"]
    color = ALERT_COLOR.get(level, "#64748b")

    rendered = _render_map(risk.get("geotiff", ""))
    if rendered:
        png_bytes, map_uri, b = rendered
        (output_dir / f"risk_map_{stamp}.png").write_bytes(png_bytes)
        extent = (f"{b.left:g}–{b.right:g}°E, "
                  f"{-b.top:g}–{-b.bottom:g}°S · EPSG:4326 "
                  f"· ~{config.CELL:g} m")
        map_html = (f'<figure class="map"><img alt="Flood risk map" src="{map_uri}">'
                    f'{_legend()}<figcaption>{extent}</figcaption></figure>')
    else:
        map_html = ('<figure class="map"><div class="nomap">Risk GeoTIFF not '
                    'found — run <code>floodrisk daily</code> first.</div></figure>')

    tiles = "".join([
        _tile(f"{risk['rain_factor']:.2f}", "Rain factor", "forecast ÷ P95"),
        _tile(f"{forecast['basin_mm']:.1f} <span class=u>mm</span>",
              "Basin rainfall", f"95th pct {thresholds['basin_p95_mm']:.1f} mm"),
        _tile(f"{forecast['window_mm']:.1f} <span class=u>mm</span>",
              "Floodplain window", f"95th pct {thresholds['window_p95_mm']:.1f} mm"),
        _tile(f"{risk['high_risk_km2']:,.0f} <span class=u>km²</span>",
              "High-risk area", f"{risk['high_risk_fraction']:.1%} of window"),
        _tile(f"{risk['moderate_risk_km2']:,.0f} <span class=u>km²</span>",
              "Moderate-risk area", f"{risk['moderate_risk_fraction']:.1%} of window"),
    ])

    html = _TEMPLATE.format(
        valid=valid, issued=payload["issued"], level=level, color=color,
        blurb=ALERT_BLURB.get(level, ""), tiles=tiles, map_html=map_html,
        forecast_source=forecast["source"],
        thresholds_source=thresholds.get("source", "CHIRPS"),
        observation_card=_observation_card(observation),
    )
    out = output_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    log.info("wrote %s", out)
    return out


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Limpopo Flood Risk — {valid}</title>
<style>
  :root {{
    --bg:#f6f7f9; --surface:#ffffff; --ink:#1a2230; --muted:#61708a;
    --line:#e6e9ef; --accent:{color};
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0e1420; --surface:#161d2b; --ink:#e8edf5; --muted:#93a1b8;
             --line:#26303f; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:28px 20px 56px; }}
  header {{ display:flex; justify-content:space-between; align-items:baseline;
    flex-wrap:wrap; gap:8px; margin-bottom:18px; }}
  h1 {{ font-size:22px; margin:0; letter-spacing:-.01em; }}
  .place {{ color:var(--muted); font-size:13px; }}
  .dates {{ color:var(--muted); font-size:13px; text-align:right; }}
  .banner {{ display:flex; align-items:center; gap:16px; background:var(--surface);
    border:1px solid var(--line); border-left:6px solid var(--accent);
    border-radius:12px; padding:16px 20px; margin-bottom:20px; }}
  .badge {{ background:var(--accent); color:#fff; font-weight:700; font-size:15px;
    letter-spacing:.08em; padding:8px 16px; border-radius:8px; white-space:nowrap; }}
  .banner p {{ margin:0; color:var(--ink); }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin-bottom:22px; }}
  .tile {{ background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:16px; }}
  .tile .val {{ font-size:26px; font-weight:700; letter-spacing:-.02em; }}
  .tile .val .u {{ font-size:14px; font-weight:600; color:var(--muted); }}
  .tile .lab {{ font-size:13px; font-weight:600; margin-top:2px; }}
  .tile .sub {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  .map {{ margin:0 0 22px; background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:14px; }}
  .map img {{ width:100%; height:auto; border-radius:8px; display:block;
    image-rendering:auto; }}
  .map figcaption {{ color:var(--muted); font-size:12px; margin-top:10px;
    text-align:center; }}
  .nomap {{ padding:60px 20px; text-align:center; color:var(--muted); }}
  .legend {{ margin-top:12px; font-size:12px; }}
  .lrow {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    margin-top:8px; color:var(--muted); }}
  .lname {{ font-weight:600; color:var(--ink); }}
  .bar {{ flex:1; min-width:120px; height:12px; border-radius:6px;
    border:1px solid var(--line); }}
  .swatches {{ gap:18px; }}
  .swatches span {{ display:flex; align-items:center; gap:6px; }}
  .swatches i {{ width:14px; height:14px; border-radius:4px; display:inline-block;
    border:1px solid rgba(0,0,0,.15); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
    gap:12px; }}
  .card {{ background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:16px; }}
  .card h3 {{ margin:0 0 8px; font-size:13px; text-transform:uppercase;
    letter-spacing:.06em; color:var(--muted); }}
  .card p {{ margin:0 0 6px; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  footer {{ color:var(--muted); font-size:12px; margin-top:26px;
    border-top:1px solid var(--line); padding-top:14px; }}
  code {{ background:var(--line); padding:1px 5px; border-radius:4px; }}
</style></head>
<body><div class="wrap">
  <header>
    <div><h1>Limpopo Flood Risk</h1>
      <div class="place">Lower Limpopo floodplain · Chibuto reach</div></div>
    <div class="dates">valid <b>{valid}</b><br>issued {issued}</div>
  </header>

  <div class="banner">
    <span class="badge">{level}</span>
    <p>{blurb}</p>
  </div>

  <div class="tiles">{tiles}</div>

  {map_html}

  <div class="cards">
    <div class="card"><h3>Forecast</h3>
      <p>Next-day precipitation vs. climatology.</p>
      <p class="muted">Source: {forecast_source}<br>
      Thresholds: {thresholds_source}</p></div>
    {observation_card}
  </div>

  <footer>
    Risk index = susceptibility × rain factor, floored by observed SAR water.
    Uncalibrated against observed inundation — ranks pixels, not a probability.
    Generated by the <code>floodrisk</code> pipeline.
  </footer>
</div></body></html>
"""
