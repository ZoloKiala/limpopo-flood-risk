"""Central configuration for the Limpopo flood-risk pipeline."""
from pathlib import Path

# ---------------------------------------------------------------- paths
STATIC_DIR = Path("static")     # one-time products (cached between runs)
OUTPUT_DIR = Path("outputs")    # daily products

# ------------------------------------------------------- basin (WHEN grid)
LON_MIN, LON_MAX = 26.0, 35.0
LAT_MIN, LAT_MAX = -26.0, -20.0

# ----------------------------------------- susceptibility mosaic (WHERE)
# Lower Limpopo floodplain, covered by a grid of Copernicus GLO-30 1-deg DEM
# tiles mosaicked into one susceptibility raster. A tile is named by its SW
# corner degree: "S{|lat|}_00_E{lon}_00" spans [lat, lat+1) N, [lon, lon+1) E.
# build_susceptibility() derives the tile list from MOSAIC_BBOX.
DEM_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif"
)
MOSAIC_BBOX = (32.0, 34.0, -26.0, -24.0)   # lon_min, lon_max, lat_min, lat_max
GSW_URL = (
    "https://storage.googleapis.com/global-surface-water/"
    "downloads2021/occurrence/occurrence_30E_20Sv1_4_2021.tif"
)
WINDOW_BBOX = (33.0, 34.0, -25.0, -24.0)   # core reach, for the CHIRPS window pct
TILE = 128
PATCH = 16
CELL = 30.0          # ~meters per pixel

# ------------------------------------------------------ CHIRPS (thresholds)
CHIRPS_URL = (
    "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/"
    ".daily-improved/.global/.0p25/.prcp/"
    f"X/{LON_MIN:g}/{LON_MAX:g}/RANGEEDGES/"
    f"Y/{LAT_MIN:g}/{LAT_MAX:g}/RANGEEDGES/data.nc"
)

# ------------------------------------------------------ forecast (Open-Meteo)
# NOAA retired the NOMADS OPeNDAP GFS server (Service Change Notice 25-81), so
# GFS is now sourced through Open-Meteo's free JSON API (no key, no GRIB). We
# pin the GFS global model for provenance parity and fall back to Open-Meteo's
# default multi-model blend if that single model is unavailable.
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL = "gfs_global"

# --------------------------------------- Sentinel-1 SAR (NOW layer)
# All-weather "what's wet now" from C-band backscatter, segmented by a ViT
# trained on Sen1Floods11. Sentinel-1 sees through the cyclone cloud that
# blinds optical NDWI - i.e. it works precisely when a flood is happening.
#
# Live scenes come from Microsoft Planetary Computer (free, no key: assets are
# signed anonymously via the SAS endpoint). Earth Search's sentinel-1-grd is
# requester-pays S3, so it cannot satisfy the "free + unauthenticated" rule.
MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
MPC_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
S1_COLLECTION = "sentinel-1-grd"
NOW_POINT = (33.40, -24.70)   # Limpopo river, Chibuto reach
NOW_WINDOW = 1024             # px (8 x 128) at ~10 m -> ~10 km around the point
NOW_LOOKBACK_DAYS = 24        # S1 revisit is longer than S2; widen the search
WATER_PROB_THRESH = 0.5       # sigmoid P(water) above which a pixel counts wet

# --------------------------------------- Sen1Floods11 (SAR training data)
# 446 hand-labeled 512x512 chips, VV/VH sigma0 in dB, labels {-1 nodata,
# 0 dry, 1 water} - which map straight onto masked_bce. Free HTTP on GCS.
SEN1FLOODS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SEN1FLOODS_SPLITS = {
    "train": "splits/flood_handlabeled/flood_train_data.csv",
    "valid": "splits/flood_handlabeled/flood_valid_data.csv",
}
SEN1FLOODS_S1DIR = "data/flood_events/HandLabeled/S1Hand"
SEN1FLOODS_LABELDIR = "data/flood_events/HandLabeled/LabelHand"
SAR_CHIP = 512       # native Sen1Floods11 chip size (px)
SAR_TILE = 128       # ViT tile (4 x 4 tiles per chip); matches TILE/PATCH grid
SAR_PATCH = 16
SAR_CHANNELS = 2     # VV, VH
SAR_EPOCHS = 25
SAR_BATCH = 32

# ----------------------------------------------------------- risk fusion
RISK_HIGH = 0.5
RISK_MODERATE = 0.25
RAIN_FACTOR_CAP = 1.5

# --------------------------------------------------- susceptibility model
SUS_EPOCHS = 20
SUS_BATCH = 32
EMBED_DIM = 128
DEPTH = 6
NUM_HEADS = 8
MLP_DIM = 256
