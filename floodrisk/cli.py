"""Command-line interface.

  python -m floodrisk build-static [--if-missing]   one-time (train + thresholds)
  python -m floodrisk daily                          produce today's risk product
  python -m floodrisk selftest                       quick import/model sanity check
"""
import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

from . import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("floodrisk")


def cmd_build_static(args):
    from .static_build import build_static
    build_static(only_missing=args.if_missing, only=getattr(args, "only", None))


def cmd_daily(args):
    from . import bulletin as bl
    from .forecast import get_forecast
    from .fuse import compute_rain_factor, fuse, load_thresholds
    from .observation import latest_water_extent

    if getattr(args, "for_date", None):
        valid = dt.datetime.strptime(args.for_date, "%Y-%m-%d").replace(
            tzinfo=dt.timezone.utc)
        issue = valid - dt.timedelta(days=1)
    else:
        issue = dt.datetime.now(dt.timezone.utc)
        valid = issue + dt.timedelta(days=1)

    if not (config.STATIC_DIR / "susceptibility.tif").exists():
        log.error("static products missing - run `python -m floodrisk build-static`")
        sys.exit(2)

    thresholds = load_thresholds()
    forecast = get_forecast(valid_date=valid.date())
    rain_factor = compute_rain_factor(forecast, thresholds)
    observation = latest_water_extent(valid)          # SAR NOW layer (or None)
    stats = fuse(rain_factor, valid, observation=observation)

    text, payload = bl.build(issue, valid, forecast, thresholds, stats, observation)
    txt_path, json_path = bl.write(text, payload, config.OUTPUT_DIR, valid)
    print(text)
    log.info("wrote %s and %s", txt_path, json_path)

    from .dashboard import build_dashboard
    build_dashboard(payload, config.OUTPUT_DIR)

    # Non-zero-ish signal for CI: expose the alert level for downstream steps
    (config.OUTPUT_DIR / "ALERT_LEVEL").write_text(payload["alert_level"])


def cmd_dashboard(args):
    from .dashboard import build_dashboard

    bulletins = sorted(config.OUTPUT_DIR.glob("bulletin_*.json"))
    if not bulletins:
        log.error("no bulletin found in %s - run `python -m floodrisk daily`",
                  config.OUTPUT_DIR)
        sys.exit(2)
    payload = json.loads(bulletins[-1].read_text(encoding="utf-8"))
    out = build_dashboard(payload, config.OUTPUT_DIR)
    print(out)


def cmd_build_site(args):
    """Assemble a static site: one dated snapshot per bulletin + manifest + index.

    Renders a nav-enabled snapshot for every bulletin JSON in outputs/ (their
    GeoTIFFs must be present), preserving any snapshots already in the site dir
    (e.g. earlier days carried forward from a previous deploy).
    """
    from .dashboard import build_dashboard

    site = Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    for jf in sorted(config.OUTPUT_DIR.glob("bulletin_*.json")):
        payload = json.loads(jf.read_text(encoding="utf-8"))
        build_dashboard(payload, site, nav=True,
                        out_name=f"{payload['valid']}.html", write_png=False)

    dates = sorted(p.stem for p in site.glob("*.html") if p.stem != "index")
    if not dates:
        log.error("no snapshots in %s", site)
        sys.exit(2)
    latest = dates[-1]
    (site / "manifest.json").write_text(
        json.dumps({"dates": dates, "latest": latest}), encoding="utf-8")
    (site / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={latest}.html">'
        '<title>Limpopo Flood Risk</title>'
        f'<a href="{latest}.html">Latest flood-risk dashboard</a>',
        encoding="utf-8")
    log.info("site: %d dates, latest %s -> %s", len(dates), latest, site)
    print(site / "index.html")


def cmd_selftest(args):
    import numpy as np
    import tensorflow as tf
    from . import sar
    from .models import PatchViT, masked_bce

    m = PatchViT(128, 128, 16, 6, depth=1)
    out = m(tf.zeros((1, 128, 128, 6)))
    assert out.shape == (1, 128, 128, 1), out.shape
    loss = masked_bce(tf.constant(-np.ones((1, 128, 128, 1), "float32")), out)
    assert float(loss) == 0.0, "masked loss must ignore all -1 labels"

    # SAR NOW layer: 2-channel water-segmentation ViT + shared preprocessing
    sm = PatchViT(config.SAR_TILE, config.SAR_TILE, config.SAR_PATCH,
                  config.SAR_CHANNELS, depth=1)
    sout = sm(tf.zeros((1, config.SAR_TILE, config.SAR_TILE, config.SAR_CHANNELS)))
    assert sout.shape == (1, config.SAR_TILE, config.SAR_TILE, 1), sout.shape

    chip = np.random.default_rng(0).normal(-12, 3, (config.SAR_CHIP,
                                                    config.SAR_CHIP)).astype("float32")
    feats = sar.standardise(chip, chip + 2.0)
    assert feats.shape == (config.SAR_CHIP, config.SAR_CHIP, 2), feats.shape
    assert np.abs(feats).max() <= 1.0 + 1e-6, "standardised features must be in [-1, 1]"
    tiles = sar.tile(feats[..., :1], config.SAR_TILE)[..., 0]   # (n, tile, tile)
    roundtrip = sar.untile(tiles, config.SAR_CHIP, config.SAR_CHIP)
    assert np.allclose(roundtrip, feats[..., 0]), "tile/untile must round-trip"

    # nodata (amplitude 0) must not masquerade as water-dark backscatter
    db = sar.amplitude_to_db(np.array([[0, 100], [500, 0]], dtype="float32"))
    assert np.isnan(db[0, 0]) and np.isfinite(db[0, 1]), "DN 0 must map to NaN"

    print("selftest OK: susceptibility + SAR models and preprocessing behave "
          "as expected")


def main():
    parser = argparse.ArgumentParser(prog="floodrisk")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-static", help="train susceptibility + thresholds")
    p1.add_argument("--if-missing", action="store_true",
                    help="skip if static products already exist")
    p1.add_argument("--only", choices=["susceptibility", "thresholds", "sar"],
                    help="build only this component (others must already exist)")
    p1.set_defaults(func=cmd_build_static)

    p2 = sub.add_parser("daily", help="produce a risk GeoTIFF + bulletin")
    p2.add_argument("--for", dest="for_date", metavar="YYYY-MM-DD",
                    help="produce the product valid for this date "
                         "(past dates use the historical-forecast archive)")
    p2.set_defaults(func=cmd_daily)

    p3 = sub.add_parser("selftest", help="quick sanity check")
    p3.set_defaults(func=cmd_selftest)

    p4 = sub.add_parser("dashboard",
                        help="rebuild dashboard.html from the latest bulletin")
    p4.set_defaults(func=cmd_dashboard)

    p5 = sub.add_parser("build-site",
                        help="assemble dated snapshots + manifest into a site dir")
    p5.add_argument("--site", default="_site", help="output site directory")
    p5.set_defaults(func=cmd_build_site)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
