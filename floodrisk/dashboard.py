"""Render the daily product into an HTML dashboard.

Two rendering modes share one design:

* ``build_dashboard`` writes ``outputs/dashboard.html`` for GitHub Pages / local
  viewing — an **interactive Leaflet map** on an Esri World Imagery + labels
  basemap (a Google-Hybrid lookalike; Google's own tiles aren't licensed for
  this), with the risk and susceptibility fields as semi-transparent overlays,
  an opacity slider and layer toggles.
* ``render_fragment`` returns an embeddable ``<style>`` + markup string with a
  **self-contained static composite** map (inlined PNG, no external requests),
  for sandboxes that block tiles/CDNs (e.g. the Artifact preview).

Both carry the alert banner, forecast + risk-area stat tiles, the SAR NOW
status and provenance. A standalone ``risk_map_<date>.png`` (opaque composite)
is written alongside for reuse.
"""
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

# Esri basemap (free, attributed) - the Google-Hybrid substitute.
_ESRI_IMAGERY = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                 "World_Imagery/MapServer/tile/{z}/{y}/{x}")
_ESRI_LABELS = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}")
_LEAFLET_HEAD = (
    '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">\n'
    '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>')


def _ramp(v, stops):
    """Map v in [0, 1] (any shape) to an (..., 3) uint8 array via linear stops."""
    xs = np.array([s[0] for s in stops], dtype="float32")
    cols = np.array([s[1] for s in stops], dtype="float32")
    v = np.clip(v, 0.0, 1.0)
    out = np.stack([np.interp(v, xs, cols[:, c]) for c in range(3)], axis=-1)
    return out.round().astype("uint8")


def _png_data_uri(rgba):
    """Encode an (H, W, 4) uint8 array -> (png_bytes, base64 data URI); stdlib only."""
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


def _read_bands(geotiff_path):
    """Read the 4 risk bands decimated to MAP_PX -> (risk, suscept, perm, obs, bounds)."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(geotiff_path) as src:
        w = MAP_PX
        h = max(1, round(MAP_PX * src.height / src.width))
        data = src.read(out_shape=(src.count, h, w),
                        resampling=Resampling.bilinear).astype("float32")
        bounds = src.bounds
    obs = data[3] if data.shape[0] > 3 else np.zeros_like(data[0])
    return data[0], data[1], data[2], obs, bounds


def _render_map(geotiff_path):
    """Opaque composite (susceptibility base + overlays) -> (png, uri, bounds) | None."""
    path = Path(geotiff_path)
    if not path.exists():
        log.warning("risk GeoTIFF missing for dashboard: %s", path)
        return None
    risk, suscept, permanent, observed, bounds = _read_bands(path)

    img = _ramp(suscept, _GREY)
    img[risk >= _RISK_FLOOR] = _ramp(risk, _YLORRD)[risk >= _RISK_FLOOR]
    img[permanent >= 0.5] = _PERMANENT_WATER
    img[observed >= config.WATER_PROB_THRESH] = _OBSERVED_WATER

    rgba = np.dstack([img, np.full(img.shape[:2], 255, "uint8")])
    png_bytes, uri = _png_data_uri(rgba)
    return png_bytes, uri, bounds


def _render_overlays(geotiff_path):
    """Alpha-transparent overlays for the tiled basemap -> dict | None.

    Returns risk (risk + water, transparent elsewhere) and susceptibility
    (grey, alpha scaled by value) as data URIs, plus (w, s, e, n) bounds.
    """
    path = Path(geotiff_path)
    if not path.exists():
        log.warning("risk GeoTIFF missing for dashboard: %s", path)
        return None
    risk, suscept, permanent, observed, b = _read_bands(path)
    h, w = risk.shape

    rgb = _ramp(risk, _YLORRD).astype("float32")
    alpha = np.where(risk >= _RISK_FLOOR, 0.80, 0.0).astype("float32")
    pm = permanent >= 0.5
    rgb[pm], alpha[pm] = _PERMANENT_WATER, 0.85
    ob = observed >= config.WATER_PROB_THRESH
    rgb[ob], alpha[ob] = _OBSERVED_WATER, 0.90
    risk_rgba = np.dstack([rgb.round().astype("uint8"),
                           (alpha * 255).round().astype("uint8")])

    sus_rgb = _ramp(suscept, _GREY)
    sus_alpha = (np.clip(suscept, 0.0, 1.0) * 0.55 * 255).round().astype("uint8")
    sus_rgba = np.dstack([sus_rgb, sus_alpha])

    return {
        "risk": _png_data_uri(risk_rgba)[1],
        "suscept": _png_data_uri(sus_rgba)[1],
        "bounds": (b.left, b.bottom, b.right, b.top),
    }


# --- HTML pieces ------------------------------------------------------------
def _tile(value, label, sub=""):
    sub = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="tile"><div class="val">{value}</div>'
            f'<div class="lab">{label}</div>{sub}</div>')


def _legend():
    grad = ", ".join(f"rgb{c}" for _, c in _YLORRD)
    grey = "linear-gradient(90deg,rgb(238,240,243),rgb(94,104,120))"
    mod, high = config.RISK_MODERATE, config.RISK_HIGH
    return f"""
    <div class="legend">
      <div class="lblock">
        <div class="lhead">Risk index
          <span class="lmuted">forecast × susceptibility</span></div>
        <div class="rampwrap">
          <span class="rend">0</span>
          <span class="ramp" style="background:linear-gradient(90deg,{grad})">
            <span class="tick" style="left:{mod:.0%}"></span>
            <span class="tick" style="left:{high:.0%}"></span>
          </span>
          <span class="rend">1</span>
        </div>
        <div class="breaks">class breaks · {mod:.2f} moderate · {high:.2f} high</div>
      </div>
      <div class="lblock">
        <div class="lhead">Map layers</div>
        <div class="swatches">
          <span><i style="background:{grey}"></i>Susceptibility (terrain base)</span>
          <span><i style="background:rgb{_PERMANENT_WATER}"></i>Permanent river</span>
          <span><i style="background:rgb{_OBSERVED_WATER}"></i>Observed water now (SAR)</span>
        </div>
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


def _extent(left, bottom, right, top):
    return (f"{left:g}–{right:g}°E, {-top:g}–{-bottom:g}°S "
            f"· EPSG:4326 · ~{config.CELL:g} m")


_NOMAP = ('<figure class="map"><div class="nomap">Risk GeoTIFF not found — run '
          '<code>floodrisk daily</code> first.</div></figure>')


def _map_interactive(overlays):
    """Leaflet map figure (Esri hybrid basemap + risk/susceptibility overlays)."""
    w, s, e, n = overlays["bounds"]
    script = f"""<script>
(function() {{
  var bounds = [[{s:.6f}, {w:.6f}], [{n:.6f}, {e:.6f}]];
  var sat = L.tileLayer('{_ESRI_IMAGERY}', {{maxZoom: 18}});
  var lab = L.tileLayer('{_ESRI_LABELS}', {{maxZoom: 18}});
  var hybrid = L.layerGroup([sat, lab]);
  var map = L.map('fr-map', {{layers: [hybrid]}});
  map.fitBounds(bounds);
  map.attributionControl.addAttribution(
    'Imagery &copy; Esri, Maxar, Earthstar Geographics');
  var risk = L.imageOverlay("{overlays['risk']}", bounds, {{opacity: 0.8}}).addTo(map);
  var sus = L.imageOverlay("{overlays['suscept']}", bounds, {{opacity: 0.55}});
  L.control.layers({{'Satellite (hybrid)': hybrid}},
    {{'Risk &amp; water': risk, 'Susceptibility': sus}},
    {{collapsed: false}}).addTo(map);
  var Op = L.Control.extend({{options: {{position: 'topright'}},
    onAdd: function() {{
      var d = L.DomUtil.create('div', 'fr-op');
      d.innerHTML = '<label>Risk opacity</label>' +
        '<input type="range" min="0" max="100" value="80" aria-label="Risk opacity">';
      L.DomEvent.disableClickPropagation(d);
      d.querySelector('input').oninput = function(ev) {{
        risk.setOpacity(ev.target.value / 100);
      }};
      return d;
    }}}});
  map.addControl(new Op());
}})();
</script>"""
    return (f'<figure class="map"><div id="fr-map"></div>{script}'
            f'{_legend()}<figcaption>{_extent(w, s, e, n)}</figcaption></figure>')


def _map_static(geotiff, output_dir, write_png):
    """Self-contained composite <img> figure (no external requests)."""
    rendered = _render_map(geotiff)
    if not rendered:
        return _NOMAP
    png_bytes, uri, b = rendered
    if write_png:
        stamp = Path(geotiff).stem.replace("flood_risk_", "")
        (Path(output_dir) / f"risk_map_{stamp}.png").write_bytes(png_bytes)
    return (f'<figure class="map"><img alt="Flood risk map" src="{uri}">'
            f'{_legend()}<figcaption>{_extent(b.left, b.bottom, b.right, b.top)}'
            f'</figcaption></figure>')


def _build_subs(payload, output_dir, write_png, interactive):
    """Compute the template substitutions (renders + optionally saves the map)."""
    forecast, thresholds = payload["forecast"], payload["thresholds"]
    risk, observation = payload["risk"], payload.get("observation")
    level = payload["alert_level"]
    geotiff = risk.get("geotiff", "")

    if interactive:
        overlays = _render_overlays(geotiff)
        if write_png:
            _map_static(geotiff, output_dir, True)   # also save standalone PNG
        map_html = _map_interactive(overlays) if overlays else _NOMAP
    else:
        map_html = _map_static(geotiff, output_dir, write_png)

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

    return {
        "valid": payload["valid"], "issued": payload["issued"], "level": level,
        "color": ALERT_COLOR.get(level, "#64748b"),
        "blurb": ALERT_BLURB.get(level, ""), "tiles": tiles, "map_html": map_html,
        "forecast_source": forecast["source"],
        "thresholds_source": thresholds.get("source", "CHIRPS"),
        "observation_card": _observation_card(observation),
        "print_btn": ('<button class="printbtn" type="button" '
                      'onclick="window.print()">Print / PDF</button>'
                      if interactive else ""),
    }


def render_fragment(payload, output_dir=None, write_png=False):
    """Return the dashboard as an embeddable ``<style>`` + markup string.

    Static/self-contained map (no tiles or CDN) so it renders in sandboxes that
    block external requests, e.g. the Artifact preview.
    """
    subs = _build_subs(payload, output_dir or config.OUTPUT_DIR, write_png,
                       interactive=False)
    return _STYLE.format(color=subs["color"]) + "\n" + _BODY.format(**subs)


def build_dashboard(payload, output_dir=None, valid_date=None):
    """Write ``dashboard.html`` (interactive map) + standalone map PNG."""
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    subs = _build_subs(payload, output_dir, write_png=True, interactive=True)
    html = _DOC.format(title_date=subs["valid"], leaflet=_LEAFLET_HEAD,
                       style=_STYLE.format(color=subs["color"]),
                       body=_BODY.format(**subs))
    out = output_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    log.info("wrote %s", out)
    return out


# --- templates --------------------------------------------------------------
_STYLE = """<style>
  :root {{
    --bg:#f6f7f9; --surface:#ffffff; --ink:#1a2230; --muted:#61708a;
    --line:#e6e9ef; --accent:{color};
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0e1420; --surface:#161d2b; --ink:#e8edf5; --muted:#93a1b8;
             --line:#26303f; }}
  }}
  :root[data-theme="dark"] {{ --bg:#0e1420; --surface:#161d2b; --ink:#e8edf5;
    --muted:#93a1b8; --line:#26303f; }}
  :root[data-theme="light"] {{ --bg:#f6f7f9; --surface:#ffffff; --ink:#1a2230;
    --muted:#61708a; --line:#e6e9ef; }}
  .fr * {{ box-sizing:border-box; }}
  .fr {{ color:var(--ink); background:var(--bg); min-height:100vh;
    font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  .fr .wrap {{ max-width:1040px; margin:0 auto; padding:28px 20px 56px; }}
  .fr header {{ display:flex; justify-content:space-between; align-items:baseline;
    flex-wrap:wrap; gap:8px; margin-bottom:18px; }}
  .fr h1 {{ font-size:22px; margin:0; letter-spacing:-.01em; }}
  .fr .place {{ color:var(--muted); font-size:13px; }}
  .fr .dates {{ color:var(--muted); font-size:13px; text-align:right; }}
  .fr .hend {{ display:flex; align-items:center; gap:14px; }}
  .fr .printbtn {{ font:inherit; font-size:12px; cursor:pointer; padding:7px 13px;
    border:1px solid var(--line); border-radius:8px; background:var(--surface);
    color:var(--ink); white-space:nowrap; }}
  .fr .printbtn:hover {{ border-color:var(--accent); }}
  .fr .printbtn:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  .fr .banner {{ display:flex; align-items:center; gap:16px; background:var(--surface);
    border:1px solid var(--line); border-left:6px solid var(--accent);
    border-radius:12px; padding:16px 20px; margin-bottom:20px; }}
  .fr .badge {{ background:var(--accent); color:#fff; font-weight:700; font-size:15px;
    letter-spacing:.08em; padding:8px 16px; border-radius:8px; white-space:nowrap; }}
  .fr .banner p {{ margin:0; color:var(--ink); }}
  .fr .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin-bottom:22px; }}
  .fr .tile {{ background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:16px; }}
  .fr .tile .val {{ font-size:26px; font-weight:700; letter-spacing:-.02em; }}
  .fr .tile .val .u {{ font-size:14px; font-weight:600; color:var(--muted); }}
  .fr .tile .lab {{ font-size:13px; font-weight:600; margin-top:2px; }}
  .fr .tile .sub {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  .fr .map {{ margin:0 0 22px; background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:14px; }}
  .fr .map > img {{ width:100%; height:auto; border-radius:8px; display:block; }}
  .fr #fr-map {{ height:460px; width:100%; border-radius:8px; background:var(--line);
    z-index:0; }}
  .fr .leaflet-container, .fr .leaflet-container * {{ box-sizing:content-box; }}
  .fr .fr-op {{ background:var(--surface); color:var(--ink); padding:6px 9px;
    border-radius:8px; box-shadow:0 1px 5px rgba(0,0,0,.25); font-size:12px;
    display:flex; flex-direction:column; gap:3px; }}
  .fr .fr-op input {{ width:120px; }}
  .fr .map figcaption {{ color:var(--muted); font-size:12px; margin-top:10px;
    text-align:center; }}
  .fr .nomap {{ padding:60px 20px; text-align:center; color:var(--muted); }}
  .fr .legend {{ margin-top:14px; font-size:12px; display:flex;
    flex-direction:column; gap:12px; }}
  .fr .lhead {{ font-weight:600; color:var(--ink); margin-bottom:7px; }}
  .fr .lhead .lmuted {{ font-weight:400; color:var(--muted); margin-left:6px; }}
  .fr .rampwrap {{ display:flex; align-items:center; gap:8px; }}
  .fr .rend {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
  .fr .ramp {{ position:relative; flex:1; min-width:140px; height:14px;
    border-radius:5px; border:1px solid var(--line); }}
  .fr .ramp .tick {{ position:absolute; top:-3px; bottom:-3px; width:2px;
    background:#fff; box-shadow:0 0 0 .5px rgba(0,0,0,.55);
    transform:translateX(-1px); }}
  .fr .breaks {{ color:var(--muted); margin-top:6px;
    font-variant-numeric:tabular-nums; }}
  .fr .swatches {{ display:flex; gap:18px; flex-wrap:wrap; }}
  .fr .swatches span {{ display:flex; align-items:center; gap:6px;
    color:var(--muted); }}
  .fr .swatches i {{ width:18px; height:12px; border-radius:3px;
    display:inline-block; border:1px solid rgba(0,0,0,.15); }}
  .fr .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
    gap:12px; }}
  .fr .card {{ background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:16px; }}
  .fr .card h3 {{ margin:0 0 8px; font-size:13px; text-transform:uppercase;
    letter-spacing:.06em; color:var(--muted); }}
  .fr .card p {{ margin:0 0 6px; }}
  .fr .muted {{ color:var(--muted); font-size:13px; }}
  .fr footer {{ color:var(--muted); font-size:12px; margin-top:26px;
    border-top:1px solid var(--line); padding-top:14px; }}
  .fr code {{ background:var(--line); padding:1px 5px; border-radius:4px; }}
  @media print {{
    :root {{ --bg:#fff; --surface:#fff; --ink:#111; --muted:#444; --line:#ccc; }}
    .fr {{ background:#fff; min-height:0; }}
    .fr, .fr * {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
    .fr .wrap {{ max-width:none; padding:0; }}
    .fr .printbtn {{ display:none; }}
    .fr .leaflet-control-container {{ display:none !important; }}
    .fr #fr-map {{ height:430px; }}
    .fr .banner, .fr .tile, .fr .card, .fr .map {{ break-inside:avoid; }}
  }}
</style>"""

_BODY = """<div class="fr"><div class="wrap">
  <header>
    <div><h1>Limpopo Flood Risk</h1>
      <div class="place">Lower Limpopo floodplain · Chibuto reach</div></div>
    <div class="hend">
      <div class="dates">valid <b>{valid}</b><br>issued {issued}</div>
      {print_btn}
    </div>
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
</div></div>"""

_DOC = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Limpopo Flood Risk — {title_date}</title>
{leaflet}
<style>body {{ margin:0; background:var(--bg); }}</style>
{style}
</head><body>
{body}
</body></html>
"""
