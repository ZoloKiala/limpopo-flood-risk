# Limpopo Flood Risk — Operational Pipeline

Daily flood-risk products for the lower Limpopo floodplain, fusing a
ViT-derived **susceptibility map** (where terrain is flood-prone) with a
**GFS precipitation forecast** (when dangerous rain is coming) and a
**Sentinel-1 SAR observation layer** (what's wet now — a Sen1Floods11-trained
ViT that sees through the cyclone cloud which blinds optical sensors).

```
                 one-time (cached)                        daily (scheduled)
 ┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ Copernicus DEM ─┐                    │   │ GFS 0.25° via Open-Meteo (JSON)      │
 │                 ├─ terrain factors   │   │   └ fallback: Open-Meteo blend       │
 │ JRC GSW ────────┘   (incl. HAND)     │   │        │                             │
 │        │                             │   │        ▼                             │
 │        ▼                             │   │   rain factor  =  forecast / P95     │
 │  susceptibility ViT (trained once)   │   │        │                             │
 │        │                             │   │        ▼                             │
 │        ▼                             │   │  risk = susceptibility × factor      │
 │  susceptibility.tif  thresholds.json │   │        │            ┌ Sentinel-1 SAR │
 │  (CHIRPS 40-yr percentiles)          │   │        ▼            ▼  (best effort) │
 │  + SAR ViT (Sen1Floods11, once)      │   │  GeoTIFF + bulletin (txt/json)       │
 └──────────────────────────────────────┘   └──────────────────────────────────────┘
```

The SAR NOW layer is a ViT trained **once** (with the susceptibility model)
on the Sen1Floods11 hand-labeled dataset, then run each day on the newest
Sentinel-1 scene over the reach.

All data sources are free and unauthenticated: Copernicus DEM (AWS), JRC
Global Surface Water (GCS), CHIRPS (IRI Data Library),
Sen1Floods11 (GCS), Sentinel-1 GRD (Microsoft Planetary Computer, anonymous
SAS signing), Open-Meteo (GFS global + multi-model blend forecasts).

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m floodrisk selftest       # sanity check (~30 s)
python -m floodrisk build-static   # one-time: downloads ~1 GB, trains the
                                   # susceptibility ViT + SAR ViT
                                   # (~20 min GPU / ~2 h CPU)
python -m floodrisk daily          # today's product -> outputs/
python -m floodrisk dashboard      # rebuild dashboard.html from latest bulletin
```

Outputs per run:

| file | contents |
|---|---|
| `outputs/flood_risk_YYYYMMDD.tif` | band 1 risk index 0–1 (forecast × susceptibility, floored by observed SAR water), band 2 static susceptibility, band 3 permanent water, band 4 observed open water now (EPSG:4326, QGIS-ready) |
| `outputs/now_water_YYYYMMDD.tif` | SAR NOW layer: band 1 P(open water) 0–1, band 2 water mask (only when a Sentinel-1 scene was usable) |
| `outputs/dashboard.html` | self-contained visual dashboard: alert banner, forecast + risk-area stat tiles, composite risk map (inlined), SAR NOW status, provenance — no external assets, opens from disk / publishable to Pages |
| `outputs/risk_map_YYYYMMDD.png` | standalone composite map (susceptibility base + risk overlay + permanent/observed water) |
| `outputs/bulletin_YYYYMMDD.txt` | human-readable bulletin |
| `outputs/bulletin_YYYYMMDD.json` | machine-readable (feed to dashboards/APIs) |
| `outputs/ALERT_LEVEL` | `LOW` / `MODERATE` / `HIGH` (for CI alerting steps) |

`daily` writes the dashboard automatically; the standalone `dashboard` command
just re-renders it from the most recent bulletin JSON (no re-run needed).

## Deploy on GitHub Actions

1. Push this repo to GitHub.
2. Actions tab → **Rebuild static products** → *Run workflow* (one-time,
   ~90 min on the free CPU runner; caches `static/`).
3. Done. **Daily flood risk** runs every day at 04:30 UTC (06:30 CAT),
   posts the bulletin to the job summary, and uploads the GeoTIFF + bulletin
   as an artifact (30-day retention).

Notes:
- If the cache ever expires (7 days unused — daily runs keep it warm), the
  daily job rebuilds static products automatically (`build-static --if-missing`).
- To alert on HIGH risk, add a `WEBHOOK_URL` repository secret (Slack/Teams
  incoming webhook) and uncomment the final step in `daily-risk.yml`.
- Scheduled workflows pause after 60 days of repo inactivity on free plans —
  any commit re-enables them.

## Configuration

Everything tunable lives in `floodrisk/config.py`: susceptibility mosaic extent
(`MOSAIC_BBOX` — the DEM tiles are derived and mosaicked automatically), risk
class thresholds, SAR monitoring point, model size, alert thresholds. After
changing the mosaic or model,
re-run **Rebuild static products**.

## Design decisions

- **Susceptibility is static** — trained once on DEM terrain factors (elevation,
  slope, curvature, TPI, HAND, distance to water) against 40 years of JRC
  Landsat water history, with a **spatial** train/validation split. Retraining
  is only needed when the window or model changes (annual GSW refresh at most).
- **Forecast provenance is explicit** — every bulletin names its source. The GFS
  global model via Open-Meteo's JSON API is primary; Open-Meteo's default
  multi-model blend is the automatic fallback. (NOAA retired the NOMADS OPeNDAP
  GFS server — Service Change Notice 25-81 — so GFS is now fetched *through*
  Open-Meteo rather than direct.) No GRIB parsing, no keys.
- **Rain factor** = mean(basin, floodplain window) forecast ÷ CHIRPS 95th
  percentile of rainy days, capped at 1.5. Calibrated only by climatology —
  see *Limits*.
- **All-weather NOW layer, fused as a floor on risk** — the "what's wet now"
  observation is Sentinel-1 C-band SAR, segmented by a ViT trained on
  Sen1Floods11. Radar penetrates cloud, so the layer keeps reporting during the
  cyclones that cause the floods — exactly when optical NDWI goes dark. Scenes
  come from Microsoft Planetary Computer (free, anonymous SAS signing). The
  observed water (reprojected to the risk grid, permanent river excluded) is a
  *floor* on the risk index: where SAR confirms open water now, risk cannot read
  below that confidence — forecast and observation confirm each other.
- **Graceful degradation** — no reachable Sentinel-1 scene (or an untrained SAR
  model) just means the bulletin says so; the risk product still ships.
- **Everything auditable** — static products carry the training provenance
  (`sar_model.json` records the dataset, tile counts and final losses); daily
  products carry forecast source + thresholds + SAR scene id in the JSON
  bulletin.

## Limits (read before operational use)

- The risk index is **uncalibrated against damage or observed extent** — it
  ranks pixels well (that's what the AUC validation in the companion research
  notebooks shows) but "0.5" is a class boundary, not a probability of
  inundation. Fit against event records before decision use.
- **No hydrological routing.** The lower Limpopo crests *days* after upstream
  rain; this pipeline flags rainfall coincidence only. Couple the rain factor
  to GloFAS discharge (or gauge data) for lead-time-aware alerts.
- **No antecedent state** — soil moisture and Massingir dam levels modulate
  everything and are not yet inputs.
- Susceptibility labels are Landsat *history*: short floods between overpasses
  and water under cloud/vegetation are under-represented.
- **The SAR NOW layer floors the risk index only within its footprint** — one
  Sentinel-1 scene covers ~10 km around the monitoring point, so it can confirm
  inundation there but says nothing about the rest of the window; risk outside
  the footprint is forecast-only. It raises risk, never lowers it (a floor, not
  a correction), so a missed-water scene never suppresses a real forecast alert.
  Live Planetary Computer GRD is *uncalibrated amplitude*, while the
  Sen1Floods11 training chips are sigma0 in dB; the two are reconciled by
  **per-scene standardisation** (`sar.py`), not radiometric calibration. A
  terrain-corrected product (Sentinel-1 RTC) or SNAP calibration is the rigorous
  upgrade. SAR open-water detection is also confounded by wind-roughened water,
  saturated soil, and radar shadow/layover — treat it as a cross-check, not truth.
- **Susceptibility now mosaics a 2×2 DEM-tile window** (`MOSAIC_BBOX`, lon 32–34°E,
  lat 24–26°S — the lower Limpopo floodplain from Chókwè/Guijá to the Xai-Xai
  delta). The ViT trains once and is applied across the mosaic. Extending much
  further on the free CI runner is memory/time-bound; a bigger basin needs
  chunked/tiled inference or a paid runner.
- **Rain factor is a single scalar** applied across the whole mosaic — spatially
  uniform. Basin-wide rainfall varies, so couple to a gridded forecast (or GloFAS)
  before treating sub-regions differently.

## Roadmap

1. ~~**SAR NOW layer** — swap NDWI for the Sen1Floods11-trained ViT
   (all-weather), and fuse the observed water into the risk index.~~ **Done.**
   Next: a calibrated Sentinel-1 RTC input, and a wider/mosaicked S1 footprint
   so the observation floors risk across the whole window, not just the reach.
2. **GloFAS coupling** — discharge percentile as a second factor with routing lag.
3. ~~**Multi-window mosaic** — full lower-basin coverage.~~ **Done** for a 2×2
   tile window (lower Limpopo floodplain). Next: larger extent via chunked
   inference + spatially-varying rain factor.
4. **Bulletin-level verification** — replay 2000/2013/2017/2021/2023 events,
   count hits/misses/false alarms, tune thresholds.

## Provenance

Condensed from a research-notebook series (ViT classification → water
segmentation → SAR flood mapping → rainfall forecasting → susceptibility →
combined demo). The notebooks carry the full pedagogy, baselines and error
analysis behind each component.
