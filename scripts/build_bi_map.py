"""
Build visualization/bi_map.html as a deck.gl interactive map aligned
with visualization/abm_animation/index.html's design language.

Primary colour encoding: grid_status (Green: Sufficient / Amber: Moderate /
Red: Congested). DSO survives via marker shape and tooltip.

Layers (toggleable via checkboxes, rendered bottom -> top):
  - Grid congestion heatmap     -- red wash weighted by substation saturation.
  - Interurban corridors        -- full interurban network (AP-/A-/N-),
                                   per-segment TEN-T Core / Comprehensive /
                                   General styling.
  - Station -> Substation links -- one line per proposed station to its sub,
                                   colour-graded by connection distance.
  - Energy substations (~2,147) -- triangles by grid_status.
  - Existing fast chargers (~3,246) -- filled circles by nearest sub's grid_status.
  - Proposed stations (8)       -- pulsing circles by grid_status.

Hover tooltips expose contextual metadata (operator, power, connectors,
DSO, capacity, status, etc.). A legend, KPI dashboard and control panel
mirror the ABM animation chrome.

All data is inlined into the HTML as JSON, so the output is a single
self-contained file.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.neighbors import BallTree

try:
    import geopandas as gpd
    _HAVE_GPD = True
except ImportError:  # pragma: no cover
    _HAVE_GPD = False

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed"
OUTPUT = REPO / "output"
OUT = REPO / "visualization" / "bi_map.html"
CORRIDOR_SRC = REPO / "visualization" / "abm_animation" / "trajectories.json"
ROADS_PARQUET = DATA / "interurban_roads.parquet"

# Design tokens -- kept in sync with abm_animation/index.html.
DSO_HEX = {
    "i-DE":   "#4ade80",
    "Endesa": "#3b82f6",
    "Viesgo": "#ef4444",
}

# Grid status palette -- primary encoding across all point layers.
STATUS_COLORS = {
    "Sufficient": [ 74, 222, 128],
    "Moderate":   [251, 191,  36],
    "Congested":  [248, 113, 113],
}
STATUS_HEX = {
    "Sufficient": "#4ade80",
    "Moderate":   "#fbbf24",
    "Congested":  "#f87171",
}
UNKNOWN_STATUS_COLOR = [156, 163, 175]

# TEN-T corridor tier styling. Single blue hue across all tiers; width
# alone carries the AFIR hierarchy (wider = stricter spacing).
CORRIDOR_BLUE = [125, 211, 252, 180]
TIER_COLORS = {"Core": CORRIDOR_BLUE, "Comprehensive": CORRIDOR_BLUE, "General": CORRIDOR_BLUE}
TIER_WIDTHS = {"Core": 3500, "Comprehensive": 2200, "General": 1100}


# ---------------------------------------------------------------------------
# Icon atlas: triangle (substation). Tinted per-feature by IconLayer's
# getColor. Chargers and proposed stations use ScatterplotLayer directly.
# ---------------------------------------------------------------------------
def make_icon_atlas() -> str:
    """Return a data:image/png;base64,... URL for a 1-icon atlas (triangle)."""
    atlas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    draw.polygon([(64, 14), (12, 114), (116, 114)], fill=(255, 255, 255, 255))
    buf = io.BytesIO()
    atlas.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


ICON_MAPPING = {
    "triangle": {"x": 0, "y": 0, "width": 128, "height": 128, "mask": True},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_substations() -> pd.DataFrame:
    df = pd.read_csv(DATA / "grid_consolidated.csv")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df.loc[~df["distributor_network"].isin(DSO_HEX), "distributor_network"] = "i-DE"
    return df


def load_chargers() -> pd.DataFrame:
    df = pd.read_csv(DATA / "interurban_chargers_baseline.csv")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    keep = [
        "site_id", "name", "latitude", "longitude", "num_stations",
        "n_connectors", "max_power_kw", "connector_types", "province",
        "nearest_road",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    df = df[keep]
    df["max_power_kw"] = pd.to_numeric(df["max_power_kw"], errors="coerce").fillna(0)
    df = df[df["max_power_kw"] >= 50].copy()
    return df


def load_proposed() -> pd.DataFrame:
    """Enriched proposed-station frame. `n_chargers_proposed` is taken from
    the authoritative submission (output/File_2.csv) so any post-processing
    feedback-loop adjustments (e.g. STA_0003 AP-2 4 -> 6) are reflected."""
    df = pd.read_csv(DATA / "stations_with_grid_status.csv")
    f2 = OUTPUT / "File_2.csv"
    if f2.exists():
        sub = pd.read_csv(f2)[["location_id", "n_chargers_proposed"]]
        sub = sub.rename(columns={"n_chargers_proposed": "_n_chargers_submitted"})
        df = df.merge(sub, on="location_id", how="left")
        # Prefer the submitted count when present.
        mask = df["_n_chargers_submitted"].notna()
        df.loc[mask, "n_chargers_proposed"] = df.loc[mask, "_n_chargers_submitted"].astype(int)
        df = df.drop(columns=["_n_chargers_submitted"])
    return df


def load_friction_ids() -> set[str]:
    """Return the set of proposed-station location_ids present in File_3."""
    f3 = OUTPUT / "File_3.csv"
    if not f3.exists():
        return set()
    df = pd.read_csv(f3)
    col = "bottleneck_id" if "bottleneck_id" in df.columns else "location_id"
    return set(df[col].astype(str))


def load_dso_investment() -> dict:
    """Return {"total_mw": float, "by_dso": {dso: mw}} from the submission
    summary; empty dict if the file is missing."""
    f = OUTPUT / "dso_investment_summary.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    mw_col = "total_demand_mw" if "total_demand_mw" in df.columns else None
    by = {}
    if mw_col:
        for _, r in df.iterrows():
            by[str(r["distributor_network"])] = float(r.get(mw_col) or 0)
    total = float(sum(by.values()))
    return {"total_mw": round(total, 2), "by_dso": by}


def load_corridors_with_tiers() -> list[dict]:
    """Full interurban corridor geometry read from interurban_roads.parquet.
    Each segment keeps its own TEN-T tier (per-segment rather than per-road,
    so a road that changes classification mid-route renders faithfully).

    Geometry is simplified (~50 m tolerance) for payload size; at country
    zoom this is lossless to the eye and cuts vertex count ~10x.
    """
    if not (_HAVE_GPD and ROADS_PARQUET.exists()):
        return []

    rdf = gpd.read_parquet(ROADS_PARQUET)
    # interurban_roads.parquet is already AP-/A-/N- filtered, but guard anyway.
    if "road_prefix" in rdf.columns:
        rdf = rdf[rdf["road_prefix"].isin(["AP", "A", "N"])].copy()

    # Simplify: 0.0005 deg ~ 50 m at Spain's latitude. Visually lossless for
    # a national BI view, drops per-segment vertex count from ~200 to ~20.
    rdf["geometry"] = rdf.geometry.simplify(0.0005, preserve_topology=False)

    def _tier(val):
        if val == "Core":          return "Core"
        if val == "Comprehensive": return "Comprehensive"
        return "General"

    out = []
    for _, r in rdf.iterrows():
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        # Handle both LineString and the rare MultiLineString.
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = [(round(x, 5), round(y, 5)) for x, y in line.coords]
            if len(coords) < 2:
                continue
            out.append({
                "road": str(r.get("Carretera", "") or ""),
                "path": coords,
                "tier": _tier(r.get("TENT_red_basica")),
            })
    return out


def nearest_substation_status(chargers: pd.DataFrame, subs: pd.DataFrame) -> pd.DataFrame:
    """For each charger, tag the DSO, grid status and distance of its
    nearest substation (haversine, unbounded). Both fields land in the
    tooltip; grid_status is the colour driver."""
    sub_rad = np.radians(subs[["latitude", "longitude"]].values)
    chg_rad = np.radians(chargers[["latitude", "longitude"]].values)
    tree = BallTree(sub_rad, metric="haversine")
    _, idx = tree.query(chg_rad, k=1)
    status = subs["grid_status"].iloc[idx[:, 0]].values
    sub_name = subs["substation_name"].iloc[idx[:, 0]].values
    sub_dso = subs["distributor_network"].iloc[idx[:, 0]].values
    sub_lat = subs["latitude"].iloc[idx[:, 0]].values
    sub_lon = subs["longitude"].iloc[idx[:, 0]].values
    R_KM = 6371.0
    haversine_km = R_KM * np.arccos(
        np.clip(
            np.sin(chg_rad[:, 0]) * np.sin(np.radians(sub_lat))
            + np.cos(chg_rad[:, 0]) * np.cos(np.radians(sub_lat))
              * np.cos(chg_rad[:, 1] - np.radians(sub_lon)),
            -1.0, 1.0
        )
    )
    out = chargers.copy()
    out["grid_status"] = status
    out["nearest_substation"] = sub_name
    out["nearest_substation_dso"] = sub_dso
    out["nearest_substation_km"] = np.round(haversine_km, 1)
    return out


# ---------------------------------------------------------------------------
# Marker record builders (lean, with pre-computed colour arrays)
# ---------------------------------------------------------------------------
def substation_markers(subs: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in subs.iterrows():
        status = str(r.get("grid_status") or "Congested")
        out.append({
            "position": [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)],
            "dso": r["distributor_network"],
            "name": r["substation_name"],
            "capacity_mw": round(float(r.get("available_capacity_mw") or 0), 2),
            "voltage_kv": float(r.get("voltage_kv") or 0),
            "status": status,
            "color": STATUS_COLORS.get(status, UNKNOWN_STATUS_COLOR),
        })
    return out


def charger_markers(chargers: pd.DataFrame) -> list[dict]:
    """Existing fast-charger markers are coloured by their nearest
    substation's grid_status (proxy for local grid viability)."""
    out = []
    for _, r in chargers.iterrows():
        dso = r.get("nearest_substation_dso") or "i-DE"
        if dso not in DSO_HEX:
            dso = "i-DE"
        status = str(r.get("grid_status") or "Sufficient")
        out.append({
            "position": [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)],
            "name": str(r.get("name", "") or ""),
            "power_kw": float(r.get("max_power_kw") or 0),
            "connectors": int(r.get("n_connectors") or 1),
            "connector_types": str(r.get("connector_types", "") or ""),
            "road": str(r.get("nearest_road", "") or ""),
            "province": str(r.get("province", "") or ""),
            "nearest_sub": str(r.get("nearest_substation", "") or ""),
            "nearest_sub_km": float(r.get("nearest_substation_km") or 0),
            "dso": dso,
            "status": status,
            "color": STATUS_COLORS.get(status, UNKNOWN_STATUS_COLOR),
        })
    return out


def proposed_markers(proposed: pd.DataFrame, friction_ids: set[str]) -> list[dict]:
    out = []
    for _, r in proposed.iterrows():
        status = str(r.get("grid_status") or "Congested")
        loc_id = str(r["location_id"])
        out.append({
            "position": [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)],
            "id": loc_id,
            "road": str(r["route_segment"]),
            "chargers": int(r["n_chargers_proposed"]),
            "tent_tier": str(r.get("tent_tier", "") or ""),
            "is_tent": bool(r.get("is_tent", False)),
            "dso": str(r.get("distributor_network", "") or ""),
            "nearest_sub": str(r.get("nearest_substation_id", "") or ""),
            "connection_km": round(float(r.get("connection_distance_km") or 0), 1),
            "demand_kw": int(r.get("estimated_demand_kw") or 0),
            "status": status,
            "is_friction": loc_id in friction_ids,
            "color": STATUS_COLORS.get(status, UNKNOWN_STATUS_COLOR),
        })
    return out


def connection_markers(proposed: pd.DataFrame, subs: pd.DataFrame) -> list[dict]:
    """Build PathLayer records linking each proposed station to its matched
    substation. Colour-graded by connection distance: <=25 km green,
    25-50 km amber, >50 km red."""
    lookup = (
        subs.assign(key=subs["substation_name"].astype(str))
            .drop_duplicates("key")
            .set_index("key")[["latitude", "longitude", "available_capacity_mw",
                               "grid_status", "distributor_network"]]
            .to_dict("index")
    )
    out = []
    for _, r in proposed.iterrows():
        sub_id = str(r.get("nearest_substation_id", "") or "")
        sub = lookup.get(sub_id)
        if not sub:
            continue
        p = [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)]
        s = [round(float(sub["longitude"]), 5), round(float(sub["latitude"]), 5)]
        d = float(r.get("connection_distance_km") or 0)
        if d <= 25:
            color = [74, 222, 128, 210]
        elif d <= 50:
            color = [251, 191, 36, 210]
        else:
            color = [248, 113, 113, 230]
        out.append({
            "path": [p, s],
            "distance_km": round(d, 1),
            "proposed_id": str(r["location_id"]),
            "sub_id": sub_id,
            "sub_status": str(sub.get("grid_status") or ""),
            "sub_dso": str(sub.get("distributor_network") or ""),
            "sub_cap_mw": round(float(sub.get("available_capacity_mw") or 0), 2),
            "color": color,
        })
    return out


def heatmap_points(subs: pd.DataFrame) -> list[dict]:
    """Points for the congestion HeatmapLayer. Weight is saturation:
    1.0 at 0 MW, 0.0 at >=5 MW, linearly interpolated."""
    out = []
    for _, r in subs.iterrows():
        cap = float(r.get("available_capacity_mw") or 0)
        w = max(0.0, 1.0 - cap / 5.0)
        if w <= 0:
            continue
        out.append({
            "position": [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)],
            "weight": round(w, 3),
        })
    return out


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spain EV Network &mdash; BI Map</title>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" />
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', -apple-system, sans-serif; overflow: hidden; background: #0a0a0a; }
  #map { width: 100vw; height: 100vh; }

  .panel {
    position: absolute;
    background: rgba(10, 10, 10, 0.92);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    color: #fff;
    backdrop-filter: blur(12px);
    z-index: 100;
  }

  #controls { top: 20px; left: 20px; padding: 18px 20px; min-width: 280px; max-height: calc(100vh - 40px); overflow-y: auto; }
  #controls::-webkit-scrollbar { width: 6px; }
  #controls::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
  #controls h1 { font-size: 16px; font-weight: 600; margin-bottom: 4px; color: #f87171; }
  #controls .subtitle { font-size: 11px; color: rgba(255,255,255,0.5); margin-bottom: 16px; }

  .stat-group-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin: 12px 0 6px; }
  .stat-group-title:first-of-type { margin-top: 0; }
  .stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { font-size: 11px; color: rgba(255,255,255,0.65); }
  .stat-value { font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat-value.green { color: #4ade80; }
  .stat-value.amber { color: #fbbf24; }
  .stat-value.red   { color: #f87171; }
  .stat-value.blue  { color: #3b82f6; }
  .stat-value.dim   { color: rgba(255,255,255,0.85); }

  #layer-toggles { margin-top: 14px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.1); }
  .layer-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin-bottom: 8px; }
  .layer-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; }
  .layer-row input { cursor: pointer; accent-color: #f87171; }
  .layer-row span.icon {
    display: inline-block; width: 14px; height: 14px; flex-shrink: 0;
  }
  .layer-row .label { font-size: 12px; color: rgba(255,255,255,0.85); }
  .layer-row .count { font-size: 11px; color: rgba(255,255,255,0.45); margin-left: auto; font-variant-numeric: tabular-nums; }

  #legend { top: 20px; right: 20px; padding: 14px 18px; max-height: calc(100vh - 40px); overflow-y: auto; }
  #legend::-webkit-scrollbar { width: 6px; }
  #legend::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
  .legend-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin-bottom: 8px; margin-top: 12px; }
  .legend-title:first-child { margin-top: 0; }
  .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 11px; color: rgba(255,255,255,0.78); }
  .legend-item:last-child { margin-bottom: 0; }
  .legend-swatch { width: 14px; height: 14px; flex-shrink: 0; display: inline-block; }
  .legend-line { width: 22px; height: 3px; flex-shrink: 0; display: inline-block; border-radius: 2px; }

  #attribution {
    position: absolute; bottom: 6px; right: 10px; font-size: 10px;
    color: rgba(255,255,255,0.35); z-index: 50;
  }

  .pulse {
    border-radius: 50%;
    background: #f87171;
    box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.7);
    animation: pulse-ring 1.4s infinite;
  }
  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0   rgba(248, 113, 113, 0.7); }
    70%  { box-shadow: 0 0 0 8px rgba(248, 113, 113, 0); }
    100% { box-shadow: 0 0 0 0   rgba(248, 113, 113, 0); }
  }
</style>
</head>
<body>
  <div id="map"></div>

  <div id="controls" class="panel">
    <h1>EV Infrastructure Map</h1>
    <div class="subtitle">Spain Interurban &mdash; 2027 Proposal</div>

    <div class="stat-group-title">Network</div>
    <div class="stat-row">
      <span class="stat-label">Proposed stations</span>
      <span class="stat-value red" id="stat-proposed">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Proposed chargers</span>
      <span class="stat-value red" id="stat-chargers">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Proposed capacity</span>
      <span class="stat-value red" id="stat-proposed-mw">0 MW</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Remaining AFIR gaps</span>
      <span class="stat-value green" id="stat-gaps">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Existing fast chargers</span>
      <span class="stat-value dim" id="stat-existing">0</span>
    </div>

    <div class="stat-group-title">Grid status</div>
    <div class="stat-row">
      <span class="stat-label">Substations (physical)</span>
      <span class="stat-value dim" id="stat-subs">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Sufficient (&ge;5 MW)</span>
      <span class="stat-value green" id="stat-sufficient">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Moderate (1&ndash;5 MW)</span>
      <span class="stat-value amber" id="stat-moderate">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Congested (&lt;1 MW)</span>
      <span class="stat-value red" id="stat-congested">0</span>
    </div>

    <div class="stat-group-title">DSO investment required</div>
    <div class="stat-row">
      <span class="stat-label">i-DE (Iberdrola)</span>
      <span class="stat-value green" id="stat-ide-mw">0 MW</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Endesa</span>
      <span class="stat-value blue" id="stat-endesa-mw">0 MW</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Viesgo</span>
      <span class="stat-value red" id="stat-viesgo-mw">0 MW</span>
    </div>

    <div id="layer-toggles">
      <div class="layer-title">Layers</div>
      <label class="layer-row">
        <input type="checkbox" id="toggle-heatmap" checked>
        <span class="icon" id="swatch-heatmap"></span>
        <span class="label">Congestion heatmap</span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-corridors" checked>
        <span class="icon" id="swatch-corridors"></span>
        <span class="label">Interurban corridors</span>
        <span class="count" id="count-corridors"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-links" checked>
        <span class="icon" id="swatch-links"></span>
        <span class="label">Station &rarr; Substation links</span>
        <span class="count" id="count-links"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-subs" checked>
        <span class="icon" id="swatch-subs"></span>
        <span class="label">Substations by grid status</span>
        <span class="count" id="count-subs"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-chargers" checked>
        <span class="icon" id="swatch-chargers"></span>
        <span class="label">Existing chargers by grid status</span>
        <span class="count" id="count-chargers"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-proposed" checked>
        <span class="icon" id="swatch-proposed"></span>
        <span class="label">Proposed stations</span>
        <span class="count" id="count-proposed"></span>
      </label>
    </div>
  </div>

  <div id="legend" class="panel">
    <div class="legend-title">Grid status</div>
    <div class="legend-item">
      <span class="legend-swatch" style="background:#4ade80;"></span>
      <span>Sufficient (&ge;5 MW)</span>
    </div>
    <div class="legend-item">
      <span class="legend-swatch" style="background:#fbbf24;"></span>
      <span>Moderate (1&ndash;5 MW)</span>
    </div>
    <div class="legend-item">
      <span class="legend-swatch" style="background:#f87171;"></span>
      <span>Congested (&lt;1 MW)</span>
    </div>

    <div class="legend-title">Shapes</div>
    <div class="legend-item"><span class="legend-swatch" id="legend-shape-tri"></span><span>Substation</span></div>
    <div class="legend-item"><span class="legend-swatch" id="legend-shape-circle"></span><span>Existing fast charger</span></div>
    <div class="legend-item">
      <span class="legend-swatch"><span class="pulse" style="width:12px;height:12px;display:block;margin:1px;"></span></span>
      <span>Proposed station (pulsing)</span>
    </div>

    <div class="legend-title">Corridor tier (AFIR)</div>
    <div class="legend-item"><span class="legend-line" style="background:rgba(125,211,252,0.9);height:5px;"></span><span>TEN-T Core (60 km)</span></div>
    <div class="legend-item"><span class="legend-line" style="background:rgba(125,211,252,0.9);height:3px;"></span><span>TEN-T Comprehensive (100 km)</span></div>
    <div class="legend-item"><span class="legend-line" style="background:rgba(125,211,252,0.9);height:1px;"></span><span>General interurban (120 km)</span></div>

    <div class="legend-title">Station &larr; Substation link</div>
    <div class="legend-item"><span class="legend-line" style="background:repeating-linear-gradient(to right, #4ade80 0 4px, transparent 4px 7px);"></span><span>&le;25 km (economic)</span></div>
    <div class="legend-item"><span class="legend-line" style="background:repeating-linear-gradient(to right, #fbbf24 0 4px, transparent 4px 7px);"></span><span>25&ndash;50 km</span></div>
    <div class="legend-item"><span class="legend-line" style="background:repeating-linear-gradient(to right, #f87171 0 4px, transparent 4px 7px);"></span><span>&gt;50 km (remote site)</span></div>
  </div>

  <div id="attribution">Team Greenlabs &middot; Iberdrola Datathon 2026</div>

<script type="application/json" id="__bi_map_data__">__DATA_JSON__</script>

<script>
  const DATA = JSON.parse(document.getElementById('__bi_map_data__').textContent);
  const ICON_ATLAS = '__ICON_ATLAS__';
  const ICON_MAPPING = __ICON_MAPPING_JSON__;

  const DSO_HEX    = { 'i-DE': '#4ade80', 'Endesa': '#3b82f6', 'Viesgo': '#ef4444' };
  const STATUS_HEX = { Sufficient: '#4ade80', Moderate: '#fbbf24', Congested: '#f87171' };

  function setSwatch(id, color, shape) {
    const el = document.getElementById(id); if (!el) return;
    el.style.width = '14px'; el.style.height = '14px';
    el.style.background = color;
    if (shape === 'triangle') {
      el.style.clipPath = 'polygon(50% 0%, 0% 100%, 100% 100%)';
    } else if (shape === 'square') {
      el.style.clipPath = 'none';
      el.style.width = '12px'; el.style.height = '12px'; el.style.margin = '1px';
    } else if (shape === 'circle') {
      el.style.clipPath = 'none';
      el.style.borderRadius = '50%';
      el.style.width = '10px'; el.style.height = '10px'; el.style.margin = '2px';
    } else if (shape === 'line') {
      el.style.height = '3px'; el.style.margin = '5px 0';
    } else if (shape === 'dashed-line') {
      el.style.height = '3px'; el.style.margin = '5px 0';
      el.style.background = `repeating-linear-gradient(to right, ${color} 0 4px, transparent 4px 7px)`;
      return;
    } else if (shape === 'heatmap') {
      el.style.background = 'linear-gradient(90deg, rgba(248,113,113,0.15), rgba(248,113,113,0.9))';
      el.style.borderRadius = '3px';
    }
  }
  setSwatch('legend-shape-tri',    'rgba(255,255,255,0.7)', 'triangle');
  setSwatch('legend-shape-circle', 'rgba(255,255,255,0.7)', 'circle');

  setSwatch('swatch-heatmap', '', 'heatmap');
  setSwatch('swatch-corridors', 'rgba(125,211,252,0.9)', 'line');
  setSwatch('swatch-links', 'rgba(255,255,255,0.65)', 'dashed-line');
  setSwatch('swatch-subs', 'rgba(255,255,255,0.7)', 'triangle');
  setSwatch('swatch-chargers', 'rgba(255,255,255,0.7)', 'circle');
  setSwatch('swatch-proposed', 'rgba(255,255,255,0.7)', 'circle');

  // ----- Data references -----
  const corridors = DATA.corridors || [];
  const subs      = DATA.substations || [];
  const chgs      = DATA.chargers || [];
  const props     = DATA.proposed || [];
  const links     = DATA.connections || [];
  const heat      = DATA.heatmap || [];
  const kpi       = DATA.kpi || {};

  // ----- Stat panel population -----
  const nProp = props.length;
  const nChg  = props.reduce((s, p) => s + (p.chargers || 0), 0);
  const propMw = (nChg * 150 / 1000);
  document.getElementById('stat-proposed').textContent    = nProp.toLocaleString();
  document.getElementById('stat-chargers').textContent    = nChg.toLocaleString();
  document.getElementById('stat-proposed-mw').textContent = propMw.toFixed(1) + ' MW';
  document.getElementById('stat-gaps').textContent        = (kpi.remaining_afir_gaps ?? 0).toLocaleString();
  document.getElementById('stat-existing').textContent    = chgs.length.toLocaleString();
  document.getElementById('stat-subs').textContent        = subs.length.toLocaleString();
  document.getElementById('stat-sufficient').textContent  = subs.filter(s => s.status === 'Sufficient').length.toLocaleString();
  document.getElementById('stat-moderate').textContent    = subs.filter(s => s.status === 'Moderate').length.toLocaleString();
  document.getElementById('stat-congested').textContent   = subs.filter(s => s.status === 'Congested').length.toLocaleString();
  const dso = (kpi.dso_mw || {});
  document.getElementById('stat-ide-mw').textContent    = ((dso['i-DE']   || 0).toFixed(1)) + ' MW';
  document.getElementById('stat-endesa-mw').textContent = ((dso['Endesa'] || 0).toFixed(1)) + ' MW';
  document.getElementById('stat-viesgo-mw').textContent = ((dso['Viesgo'] || 0).toFixed(1)) + ' MW';

  document.getElementById('count-corridors').textContent = corridors.length.toLocaleString();
  document.getElementById('count-links').textContent     = links.length.toLocaleString();
  document.getElementById('count-subs').textContent      = subs.length.toLocaleString();
  document.getElementById('count-chargers').textContent  = chgs.length.toLocaleString();
  document.getElementById('count-proposed').textContent  = props.length.toLocaleString();

  // ----- Layer visibility state -----
  const visible = { heatmap: true, corridors: true, links: true, subs: true, chargers: true, proposed: true };
  Object.keys(visible).forEach(k => {
    const el = document.getElementById('toggle-' + k);
    if (!el) return;
    el.addEventListener('change', (e) => { visible[k] = e.target.checked; render(); });
  });

  // ----- Corridor tier styling -----
  const CORRIDOR_BLUE = [125, 211, 252, 180];
  const TIER_COLOR = { Core: CORRIDOR_BLUE, Comprehensive: CORRIDOR_BLUE, General: CORRIDOR_BLUE };
  const TIER_WIDTH = { Core: 3500, Comprehensive: 2200, General: 1100 };

  // ----- Tooltip builders -----
  function tooltipStyle() {
    return {
      background: 'rgba(10,10,10,0.94)',
      color: '#fff',
      border: '1px solid rgba(255,255,255,0.12)',
      borderRadius: '8px',
      padding: '10px 12px',
      fontFamily: 'Inter, -apple-system, sans-serif',
    };
  }
  function tooltipForSubstation(o) {
    return { html:
      `<div style="font-weight:600;color:${STATUS_HEX[o.status] || '#fff'};font-size:12px;">${o.name}</div>` +
      `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${o.dso} &middot; Substation</div>` +
      `<div style="font-size:11px;margin-top:6px;">` +
      `<span style="opacity:0.6;">Available capacity</span> ${o.capacity_mw} MW<br/>` +
      `<span style="opacity:0.6;">Voltage</span> ${o.voltage_kv} kV<br/>` +
      `<span style="opacity:0.6;">Grid status</span> <span style="color:${STATUS_HEX[o.status] || '#fff'};">${o.status}</span>` +
      `</div>`,
      style: tooltipStyle() };
  }
  function tooltipForCharger(o) {
    return { html:
      `<div style="font-weight:600;color:${STATUS_HEX[o.status] || '#fff'};font-size:12px;">${o.name || 'Fast charger'}</div>` +
      `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${o.road || ''} &middot; ${o.province || ''}</div>` +
      `<div style="font-size:11px;margin-top:6px;">` +
      `<span style="opacity:0.6;">Power</span> ${o.power_kw} kW<br/>` +
      `<span style="opacity:0.6;">Connectors</span> ${o.connectors}${o.connector_types ? ' &middot; ' + o.connector_types : ''}<br/>` +
      `<span style="opacity:0.6;">Provider</span> <span style="color:${DSO_HEX[o.dso] || '#fff'};">${o.dso}</span><br/>` +
      `<span style="opacity:0.6;">Nearest sub</span> ${o.nearest_sub} (${o.nearest_sub_km} km)<br/>` +
      `<span style="opacity:0.6;">Local grid</span> <span style="color:${STATUS_HEX[o.status] || '#fff'};">${o.status}</span>` +
      `</div>`,
      style: tooltipStyle() };
  }
  function tooltipForProposed(o) {
    const statusC = STATUS_HEX[o.status] || '#fff';
    return { html:
      `<div style="font-weight:600;color:${statusC};font-size:12px;">${o.id}${o.is_friction ? ' <span style="color:#fbbf24;">&#9888;</span>' : ''}</div>` +
      `<div style="font-size:11px;opacity:0.7;margin-top:2px;">Proposed station &middot; ${o.road}${o.is_tent ? ' &middot; TEN-T ' + o.tent_tier : ''}</div>` +
      `<div style="font-size:11px;margin-top:6px;">` +
      `<span style="opacity:0.6;">Chargers</span> ${o.chargers} &times; 150 kW = ${(o.chargers * 150 / 1000).toFixed(1)} MW<br/>` +
      `<span style="opacity:0.6;">Peak demand</span> ${o.demand_kw} kW<br/>` +
      `<span style="opacity:0.6;">DSO</span> <span style="color:${DSO_HEX[o.dso] || '#fff'};">${o.dso}</span><br/>` +
      `<span style="opacity:0.6;">Nearest sub</span> ${o.nearest_sub} (${o.connection_km} km)<br/>` +
      `<span style="opacity:0.6;">Grid status</span> <span style="color:${statusC};">${o.status}</span>` +
      (o.is_friction ? `<br/><span style="opacity:0.6;">Friction</span> <span style="color:#fbbf24;">in File_3 &middot; grid upgrade required</span>` : '') +
      `</div>`,
      style: tooltipStyle() };
  }
  function tooltipForCorridor(o) {
    const tierLabel = { Core: 'TEN-T Core', Comprehensive: 'TEN-T Comprehensive', General: 'General interurban' };
    const tierSpacing = { Core: '60 km', Comprehensive: '100 km', General: '120 km' };
    return { html:
      `<div style="font-weight:600;color:#7dd3fc;font-size:12px;">${o.road || 'Corridor'}</div>` +
      `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${tierLabel[o.tier] || o.tier}</div>` +
      `<div style="font-size:11px;margin-top:6px;">` +
      `<span style="opacity:0.6;">AFIR spacing</span> ${tierSpacing[o.tier] || '-'} max` +
      `</div>`,
      style: tooltipStyle() };
  }
  function tooltipForLink(o) {
    return { html:
      `<div style="font-weight:600;color:#fff;font-size:12px;">${o.proposed_id} &rarr; ${o.sub_id}</div>` +
      `<div style="font-size:11px;opacity:0.7;margin-top:2px;">Proposed &rarr; Substation</div>` +
      `<div style="font-size:11px;margin-top:6px;">` +
      `<span style="opacity:0.6;">Distance</span> ${o.distance_km} km<br/>` +
      `<span style="opacity:0.6;">Sub capacity</span> ${o.sub_cap_mw} MW<br/>` +
      `<span style="opacity:0.6;">Sub DSO</span> <span style="color:${DSO_HEX[o.sub_dso] || '#fff'};">${o.sub_dso}</span><br/>` +
      `<span style="opacity:0.6;">Sub grid</span> <span style="color:${STATUS_HEX[o.sub_status] || '#fff'};">${o.sub_status}</span>` +
      `</div>`,
      style: tooltipStyle() };
  }

  // ----- Deck.gl init -----
  const deckgl = new deck.DeckGL({
    container: 'map',
    mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initialViewState: { longitude: -3.7, latitude: 40.1, zoom: 5.6, pitch: 30, bearing: 0 },
    controller: true,
    getTooltip: ({object, layer}) => {
      if (!object || !layer) return null;
      const id = layer.id;
      if (id === 'substations')                                return tooltipForSubstation(object);
      if (id === 'chargers')                                   return tooltipForCharger(object);
      if (id === 'proposed-inner' || id === 'proposed-outer')  return tooltipForProposed(object);
      if (id === 'links')                                      return tooltipForLink(object);
      if (id === 'corridors')                                  return tooltipForCorridor(object);
      return null;
    }
  });

  const HEATMAP_COLOR_RANGE = [
    [  0,   0,   0,   0],
    [120,  60,  60,  40],
    [200,  90,  80,  85],
    [248, 113, 113, 140],
    [239,  68,  68, 180],
    [185,  28,  28, 210],
  ];

  function render() {
    const layers = [];

    // 1. Heatmap underlay -- weight = saturation (1 - cap/5 MW).
    if (visible.heatmap) {
      layers.push(new deck.HeatmapLayer({
        id: 'heatmap',
        data: heat,
        getPosition: d => d.position,
        getWeight:   d => d.weight,
        colorRange: HEATMAP_COLOR_RANGE,
        radiusPixels: 42,
        intensity: 0.9,
        threshold: 0.05,
        opacity: 0.55,
        aggregation: 'SUM',
      }));
    }

    // 2. Interurban corridors -- single blue hue, width alone carries tier.
    if (visible.corridors) {
      layers.push(new deck.PathLayer({
        id: 'corridors',
        data: corridors,
        getPath:  d => d.path,
        getColor: d => TIER_COLOR[d.tier] || TIER_COLOR.General,
        getWidth: d => TIER_WIDTH[d.tier] || TIER_WIDTH.General,
        widthMinPixels: 0.6,
        widthMaxPixels: 6,
        capRounded: true,
        jointRounded: true,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 255, 180],
        opacity: 1,
      }));
    }

    // 3. Station <- Substation connection lines -- a TripsLayer whose head
    //    travels substation -> station every 2.4 s, with a fading trail
    //    behind. The line itself is the animation (no separate dots / no
    //    static underlay). Legend stays dashed to hint at "marching flow".
    if (visible.links) {
      const FLOW_MS = 2400;
      const ct = (Date.now() % FLOW_MS) / FLOW_MS;
      layers.push(new deck.TripsLayer({
        id: 'links',
        data: links,
        getPath:       d => [d.path[1], d.path[0]],  // substation -> station
        getTimestamps: () => [0, 1],
        getColor:      d => d.color,
        getWidth:      1400,
        widthMinPixels: 1.3,
        widthMaxPixels: 3,
        trailLength:   0.55,
        currentTime:   ct,
        capRounded: true,
        jointRounded: true,
        pickable: true,
        opacity: 1,
        fadeTrail: true,
      }));
    }

    // 4. Substations (triangles, coloured by grid_status).
    if (visible.subs) {
      layers.push(new deck.IconLayer({
        id: 'substations',
        data: subs,
        iconAtlas: ICON_ATLAS,
        iconMapping: ICON_MAPPING,
        getIcon: () => 'triangle',
        getPosition: d => d.position,
        getColor: d => d.color,
        getSize: 8,
        sizeMinPixels: 3,
        sizeMaxPixels: 8,
        pickable: true,
        opacity: 0.9,
      }));
    }

    // 5. Existing fast chargers (circles, coloured by nearest sub's status).
    if (visible.chargers) {
      layers.push(new deck.ScatterplotLayer({
        id: 'chargers',
        data: chgs,
        getPosition: d => d.position,
        getFillColor: d => [d.color[0], d.color[1], d.color[2], 220],
        getLineColor: [10, 10, 10, 160],
        getRadius: 1800,
        radiusMinPixels: 3,
        radiusMaxPixels: 6,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 0.5,
        pickable: true,
        opacity: 0.9,
      }));
    }

    // 6. Proposed stations -- pulsing, coloured by grid_status.
    if (visible.proposed) {
      const phase = (Date.now() / 750) % (Math.PI * 2);
      const pulseScale = 1 + 0.45 * Math.sin(phase);
      const pulseAlpha = Math.round(80 + 100 * (0.5 + 0.5 * Math.sin(phase)));

      layers.push(new deck.ScatterplotLayer({
        id: 'proposed-outer',
        data: props,
        getPosition:  d => d.position,
        getFillColor: [0, 0, 0, 0],
        getLineColor: d => [d.color[0], d.color[1], d.color[2], pulseAlpha],
        getRadius: 9000 * pulseScale,
        radiusMinPixels: 10,
        radiusMaxPixels: 28,
        stroked: true,
        filled: false,
        lineWidthMinPixels: 2,
        pickable: true,
        opacity: 1,
      }));

      layers.push(new deck.ScatterplotLayer({
        id: 'proposed-inner',
        data: props,
        getPosition:  d => d.position,
        getFillColor: d => [d.color[0], d.color[1], d.color[2], 230],
        getLineColor: [255, 255, 255, 220],
        getRadius: 3500,
        radiusMinPixels: 6,
        radiusMaxPixels: 10,
        stroked: true,
        filled: true,
        lineWidthMinPixels: 1,
        pickable: true,
        opacity: 1,
      }));

    }

    deckgl.setProps({ layers });
  }

  // 60 Hz redraw so the pulsing ring stays alive even when idle.
  function tick() { render(); requestAnimationFrame(tick); }
  tick();
</script>
</body>
</html>
"""


def main() -> None:
    print("Loading datasets...")
    subs = load_substations()
    chargers = load_chargers()
    proposed = load_proposed()
    corridors = load_corridors_with_tiers()
    friction_ids = load_friction_ids()
    dso_inv = load_dso_investment()
    print(f"  corridors   : {len(corridors):,}")
    print(f"  substations : {len(subs):,}")
    print(f"  chargers    : {len(chargers):,}")
    print(f"  proposed    : {len(proposed):,}")
    print(f"  friction    : {len(friction_ids):,}")

    # Status mix sanity check -- should surface as three colours on the map.
    status_mix = subs["grid_status"].value_counts().to_dict()
    print(f"  sub status  : {status_mix}")
    tier_mix = {t: sum(1 for c in corridors if c['tier'] == t) for t in ('Core', 'Comprehensive', 'General')}
    print(f"  corridor tier mix: {tier_mix}")

    print("Joining chargers -> nearest substation for status + DSO...")
    chargers = nearest_substation_status(chargers, subs)

    print("Building marker records...")
    connections = connection_markers(proposed, subs)
    heat = heatmap_points(subs)
    print(f"  connections : {len(connections):,}")
    print(f"  heatmap pts : {len(heat):,}")

    kpi = {
        "remaining_afir_gaps": 0,
        "dso_mw": dso_inv.get("by_dso", {}),
        "total_mw": dso_inv.get("total_mw", 0),
    }

    data = {
        "corridors":   corridors,
        "substations": substation_markers(subs),
        "chargers":    charger_markers(chargers),
        "proposed":    proposed_markers(proposed, friction_ids),
        "connections": connections,
        "heatmap":     heat,
        "kpi":         kpi,
    }

    print("Encoding icon atlas...")
    atlas = make_icon_atlas()

    print("Rendering HTML...")
    data_json = json.dumps(data, separators=(",", ":"))
    data_json = data_json.replace("</script>", "<\\/script>")
    html = (
        HTML_TEMPLATE
        .replace("__DATA_JSON__", data_json)
        .replace("__ICON_ATLAS__", atlas)
        .replace("__ICON_MAPPING_JSON__", json.dumps(ICON_MAPPING))
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    size_kb = OUT.stat().st_size / 1024
    print(f"\nWrote {OUT}  ({size_kb:.0f} KB)")
    print(f"  corridors   : {len(data['corridors']):,}")
    print(f"  substations : {len(data['substations']):,}")
    print(f"  chargers    : {len(data['chargers']):,}")
    print(f"  proposed    : {len(data['proposed']):,}")
    print(f"  connections : {len(data['connections']):,}")
    print(f"  heatmap pts : {len(data['heatmap']):,}")


if __name__ == "__main__":
    main()
