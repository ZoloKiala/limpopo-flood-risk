"""Central configuration for the (multi-region) flood-risk pipeline."""
from pathlib import Path

# ---------------------------------------------------------------- paths
STATIC_DIR = Path("static")     # one-time products (cached between runs)
OUTPUT_DIR = Path("outputs")    # daily products

TILE = 128
PATCH = 16
CELL = 30.0          # ~meters per pixel

GSW_BASE = ("https://storage.googleapis.com/global-surface-water/"
            "downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif")


def chirps_url(bbox):
    """IRI CHIRPS daily subset URL for a (lon_min, lon_max, lat_min, lat_max) bbox."""
    lon_min, lon_max, lat_min, lat_max = bbox
    return ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/"
            ".daily-improved/.global/.0p25/.prcp/"
            f"X/{lon_min:g}/{lon_max:g}/RANGEEDGES/"
            f"Y/{lat_min:g}/{lat_max:g}/RANGEEDGES/data.nc")


# ------------------------------------------------------------- regions
# Each region is an independent study window: its own DEM mosaic (WHERE), its
# own forecast/CHIRPS bbox + river gauge (WHEN), and its own SAR reach point.
# Shared, region-independent settings (models, SAR training, MPC, risk classes)
# stay module-level below. The default region keeps its products at STATIC_DIR
# root (unchanged); others live under STATIC_DIR/<name>.
REGIONS = {
    "lower_limpopo": {
        "name": "lower_limpopo",
        "title": "Lower Limpopo floodplain",
        "place": "Chókwè · Chibuto · Xai-Xai (Mozambique)",
        "forecast_bbox": (26.0, 35.0, -26.0, -20.0),
        "mosaic_bbox": (32.0, 34.0, -26.0, -24.0),
        "window_bbox": (33.0, 34.0, -25.0, -24.0),
        "gsw_url": GSW_BASE.format(tile="30E_20S"),
        "discharge_point": (33.40, -24.70),
        "now_point": (33.40, -24.70),
    },
    "caprivi": {
        "name": "caprivi",
        "title": "Caprivi / Eastern Zambezi floodplain",
        "place": "Katima Mulilo · Zambezi–Chobe (Namibia)",
        "forecast_bbox": (22.0, 26.0, -19.0, -16.0),
        "mosaic_bbox": (24.0, 26.0, -19.0, -17.0),
        "window_bbox": (24.0, 26.0, -18.5, -17.5),
        "gsw_url": GSW_BASE.format(tile="20E_10S"),
        "discharge_point": (25.20, -17.80),   # main Zambezi, on the GloFAS network
        "now_point": (24.30, -17.55),          # Zambezi near Katima Mulilo
    },
}
DEFAULT_REGION = "lower_limpopo"


def get_region(name=None):
    return REGIONS[name or DEFAULT_REGION]


def region_static_dir(name=None):
    """Per-region static dir; the default region stays at STATIC_DIR root."""
    name = name or DEFAULT_REGION
    return STATIC_DIR if name == DEFAULT_REGION else STATIC_DIR / name


def region_output_dir(name=None):
    """Per-region daily-output dir; the default region stays at OUTPUT_DIR root."""
    name = name or DEFAULT_REGION
    return OUTPUT_DIR if name == DEFAULT_REGION else OUTPUT_DIR / name


# ----------------------------------------- susceptibility mosaic (WHERE)
# Copernicus GLO-30 1-deg DEM tiles are mosaicked per region. A tile is named
# by its SW corner degree: "S{|lat|}_00_E{lon}_00" spans [lat, lat+1) N,
# [lon, lon+1) E. build_susceptibility() derives the tile list from the region
# mosaic_bbox. The WBM aux layer (class 1 = ocean) masks the sea.
DEM_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif"
)
WBM_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{tile}_DEM/AUXFILES/Copernicus_DSM_COG_10_{tile}_WBM.tif"
)

# ------------------------------------------------------ forecast (Open-Meteo)
# NOAA retired the NOMADS OPeNDAP GFS server (Service Change Notice 25-81), so
# GFS is now sourced through Open-Meteo's free JSON API (no key, no GRIB). We
# pin the GFS global model for provenance parity and fall back to Open-Meteo's
# default multi-model blend if that single model is unavailable.
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_MODEL = "gfs_global"

# GloFAS river discharge via Open-Meteo's free Flood API (keyless). Discharge
# integrates upstream rain + routing, so it carries the days-later flood wave
# that local rain misses. The point must sit on the GloFAS river network.
FLOOD_API_URL = "https://flood-api.open-meteo.com/v1/flood"
DISCHARGE_POINT = (33.40, -24.70)   # Limpopo at the Chibuto reach

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
