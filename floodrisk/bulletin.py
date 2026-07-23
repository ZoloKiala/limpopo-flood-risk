"""Compose and write the daily bulletin (text + JSON)."""
import datetime as dt
import json


def alert_level(rain_factor):
    if rain_factor >= 1.0:
        return "HIGH"
    if rain_factor >= 0.6:
        return "MODERATE"
    return "LOW"


def build(issue_date, valid_date, forecast, thresholds, stats, observation):
    level = alert_level(stats["rain_factor"])
    lines = [
        "=" * 62,
        f" LIMPOPO FLOOD RISK BULLETIN   issued {issue_date:%Y-%m-%d}, "
        f"valid {valid_date:%Y-%m-%d}",
        "=" * 62,
        f" Alert level             : {level}",
        f" Forecast source         : {forecast['source']}",
        f" Basin rainfall forecast : {forecast['basin_mm']:5.1f} mm/day "
        f"(95th pct = {thresholds['basin_p95_mm']:.1f})",
        f" Floodplain window       : {forecast['window_mm']:5.1f} mm/day "
        f"(95th pct = {thresholds['window_p95_mm']:.1f})",
        f" Rain factor             : {stats['rain_factor']:.2f}",
        f" Area at high risk       : {stats['high_risk_fraction']:.1%} "
        f"({stats['high_risk_km2']:,.0f} km2)",
        f" Area at moderate risk   : {stats['moderate_risk_fraction']:.1%} "
        f"({stats['moderate_risk_km2']:,.0f} km2)",
    ]
    if observation:
        lines.append(
            f" NOW layer (SAR)         : {observation['datetime']} "
            f"{observation['scene'][:24]}")
        lines.append(
            f"                           {observation.get('polarization', '')} "
            f"{observation.get('orbit', '')} - "
            f"{observation['water_fraction']:.1%} open water in monitored reach")
        if stats.get("observation_fused"):
            lines.append(
                f"                           fused into risk index: "
                f"{stats['observed_water_km2']:,.0f} km2 observed wet")
    else:
        lines.append(
            " NOW layer (SAR)         : none usable (no recent Sentinel-1 scene)")
    lines += [
        f" Product                 : {stats['geotiff']}",
        "=" * 62,
    ]
    text = "\n".join(lines)

    payload = {
        "issued": f"{issue_date:%Y-%m-%d}",
        "valid": f"{valid_date:%Y-%m-%d}",
        "alert_level": level,
        "forecast": forecast,
        "thresholds": thresholds,
        "risk": stats,
        "observation": observation,
    }
    return text, payload


def write(text, payload, output_dir, valid_date):
    txt_path = output_dir / f"bulletin_{valid_date:%Y%m%d}.txt"
    json_path = output_dir / f"bulletin_{valid_date:%Y%m%d}.json"
    txt_path.write_text(text + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return txt_path, json_path
