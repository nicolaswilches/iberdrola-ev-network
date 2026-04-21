"""
Build visualization/bi_map.html as a deck.gl interactive map aligned
with visualization/abm_animation/index.html's design language.

Layers (toggleable via checkboxes):
  - Interurban corridors (~58 polylines) -- minimal translucent white,
    same geometry source as the ABM animation.
  - Energy substations (~2,147) -- triangles coloured by DSO.
  - Existing fast chargers (~3,246) -- squares coloured by the DSO of
    their nearest substation (a charger's power-supply provider).
  - Proposed stations (8) -- green blinking circles (same design as the
    ABM animation proposed markers).

Palette is provider-based for both substations and chargers so the two
layers never collide on colour. Corporate colour choices:
  - i-DE (Iberdrola) = green
  - Endesa           = blue
  - Viesgo           = red

Hover tooltips expose contextual metadata (operator, power, connectors,
DSO, capacity, status, etc.). A legend and control panel mirror the ABM
animation chrome.

All data is inlined into the HTML as JSON, so the output is a single
self-contained file (same portability as the Folium bi_map it replaces).
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

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed"
OUT = REPO / "visualization" / "bi_map.html"
CORRIDOR_SRC = REPO / "visualization" / "abm_animation" / "trajectories.json"

# Design tokens -- kept in sync with abm_animation/index.html
# Corporate palette by DSO:
#   i-DE (Iberdrola) green, Endesa blue, Viesgo red
DSO_COLORS = {
    "i-DE":   [74, 222, 128],   # Iberdrola green
    "Endesa": [59, 130, 246],   # Endesa blue
    "Viesgo": [239, 68, 68],    # Viesgo red
}
DSO_HEX = {
    "i-DE":   "#4ade80",
    "Endesa": "#3b82f6",
    "Viesgo": "#ef4444",
}
# Grid status colours are kept only for the tooltip text highlight, not
# for the base marker colour (which is DSO-based).
STATUS_HEX = {
    "Sufficient": "#4ade80",
    "Moderate":   "#fbbf24",
    "Congested":  "#f87171",
}
PROPOSED_GREEN = [74, 222, 128]


# ---------------------------------------------------------------------------
# Icon atlas: triangle (substation) + square (charger). Tinted per-feature
# by deck.gl IconLayer's getColor.
# ---------------------------------------------------------------------------
def make_icon_atlas() -> str:
    """Return a data:image/png;base64,... URL for a 2-icon atlas (triangle,
    square), each 128x128 px. IconLayer tints white pixels by getColor."""
    atlas = Image.new("RGBA", (256, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    # Triangle (up-pointing) in the left slot, inset slightly.
    draw.polygon([(64, 14), (12, 114), (116, 114)], fill=(255, 255, 255, 255))
    # Square in the right slot.
    draw.rectangle([(140, 20), (244, 108)], fill=(255, 255, 255, 255))
    buf = io.BytesIO()
    atlas.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


ICON_MAPPING = {
    "triangle": {"x": 0,   "y": 0, "width": 128, "height": 128, "mask": True},
    "square":   {"x": 128, "y": 0, "width": 128, "height": 128, "mask": True},
}


# ---------------------------------------------------------------------------
# Data loading & enrichment
# ---------------------------------------------------------------------------
def load_substations() -> pd.DataFrame:
    df = pd.read_csv(DATA / "grid_consolidated.csv")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    # Collapse rare unknown DSO labels to i-DE so the colour mapping covers
    # 100% of the points.
    df.loc[~df["distributor_network"].isin(DSO_COLORS), "distributor_network"] = "i-DE"
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
    df = pd.read_csv(DATA / "stations_with_grid_status.csv")
    return df


def load_corridors() -> list[dict]:
    """Pull the corridor polylines from the ABM animation bundle so both
    maps overlay the exact same road geometry."""
    if not CORRIDOR_SRC.exists():
        return []
    with open(CORRIDOR_SRC) as f:
        bundle = json.load(f)
    corridors = bundle.get("corridors", []) or []
    out = []
    for c in corridors:
        path = c.get("path") or []
        if len(path) < 2:
            continue
        out.append({"road": c.get("road", ""), "path": path})
    return out


def nearest_substation_status(
    chargers: pd.DataFrame, subs: pd.DataFrame
) -> pd.DataFrame:
    """For each charger, tag the DSO, grid status and distance of its
    nearest substation (haversine, unbounded). The DSO drives the marker
    colour; the status is surfaced in the tooltip."""
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
# Marker record -> JSON dicts (lean, with pre-computed colour arrays)
# ---------------------------------------------------------------------------
def substation_markers(subs: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in subs.iterrows():
        dso = r["distributor_network"]
        cap = float(r.get("available_capacity_mw") or 0)
        out.append({
            "position": [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)],
            "dso": dso,
            "name": r["substation_name"],
            "capacity_mw": round(cap, 2),
            "voltage_kv": float(r.get("voltage_kv") or 0),
            "status": r["grid_status"],
            "color": DSO_COLORS[dso],
        })
    return out


def charger_markers(chargers: pd.DataFrame) -> list[dict]:
    """Charger markers are coloured by their nearest substation's DSO
    (the provider supplying their power), matching the substation palette."""
    out = []
    for _, r in chargers.iterrows():
        dso = r.get("nearest_substation_dso") or "i-DE"
        if dso not in DSO_COLORS:
            dso = "i-DE"
        status = r.get("grid_status") or "Sufficient"
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
            "color": DSO_COLORS[dso],
        })
    return out


def proposed_markers(proposed: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in proposed.iterrows():
        out.append({
            "position": [round(float(r["longitude"]), 5), round(float(r["latitude"]), 5)],
            "id": str(r["location_id"]),
            "road": str(r["route_segment"]),
            "chargers": int(r["n_chargers_proposed"]),
            "tent_tier": str(r.get("tent_tier", "") or ""),
            "is_tent": bool(r.get("is_tent", False)),
            "dso": str(r.get("distributor_network", "") or ""),
            "nearest_sub": str(r.get("nearest_substation_id", "") or ""),
            "connection_km": round(float(r.get("connection_distance_km") or 0), 1),
            "demand_kw": int(r.get("estimated_demand_kw") or 0),
            "status": str(r.get("grid_status") or "Congested"),
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

  #controls { top: 20px; left: 20px; padding: 20px; min-width: 260px; }
  #controls h1 { font-size: 16px; font-weight: 600; margin-bottom: 4px; color: #4ade80; }
  #controls .subtitle { font-size: 11px; color: rgba(255,255,255,0.5); margin-bottom: 16px; }

  .stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06);
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-label { font-size: 11px; color: rgba(255,255,255,0.6); }
  .stat-value { font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .stat-value.green { color: #4ade80; }
  .stat-value.amber { color: #fbbf24; }
  .stat-value.red   { color: #ef4444; }
  .stat-value.blue  { color: #3b82f6; }

  #layer-toggles { margin-top: 14px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.1); }
  .layer-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin-bottom: 8px; }
  .layer-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; cursor: pointer; }
  .layer-row input { cursor: pointer; accent-color: #4ade80; }
  .layer-row span.icon {
    display: inline-block; width: 14px; height: 14px; flex-shrink: 0;
  }
  .layer-row .label { font-size: 12px; color: rgba(255,255,255,0.85); }
  .layer-row .count { font-size: 11px; color: rgba(255,255,255,0.45); margin-left: auto; font-variant-numeric: tabular-nums; }

  #legend { bottom: 20px; left: 20px; padding: 14px 18px; }
  .legend-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: rgba(255,255,255,0.4); margin-bottom: 8px; margin-top: 10px; }
  .legend-title:first-child { margin-top: 0; }
  .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 11px; color: rgba(255,255,255,0.75); }
  .legend-item:last-child { margin-bottom: 0; }
  .legend-swatch { width: 14px; height: 14px; flex-shrink: 0; display: inline-block; }

  #attribution {
    position: absolute; bottom: 6px; right: 10px; font-size: 10px;
    color: rgba(255,255,255,0.35); z-index: 50;
  }

  .pulse {
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7);
    animation: pulse-ring 1.4s infinite;
  }
  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0   rgba(74, 222, 128, 0.7); }
    70%  { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
    100% { box-shadow: 0 0 0 0   rgba(74, 222, 128, 0); }
  }
</style>
</head>
<body>
  <div id="map"></div>

  <div id="controls" class="panel">
    <h1>EV Infrastructure Map</h1>
    <div class="subtitle">Spain Interurban &mdash; 2027 Proposal</div>

    <div class="stat-row">
      <span class="stat-label">Proposed Stations</span>
      <span class="stat-value green" id="stat-proposed">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Proposed Chargers</span>
      <span class="stat-value green" id="stat-chargers">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Existing Fast Chargers</span>
      <span class="stat-value" id="stat-existing">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Substations (physical)</span>
      <span class="stat-value" id="stat-subs">0</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Congested substations</span>
      <span class="stat-value red" id="stat-congested">0</span>
    </div>

    <div id="layer-toggles">
      <div class="layer-title">Layers</div>
      <label class="layer-row">
        <input type="checkbox" id="toggle-corridors" checked>
        <span class="icon" id="swatch-corridors"></span>
        <span class="label">Interurban corridors</span>
        <span class="count" id="count-corridors"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-subs" checked>
        <span class="icon" id="swatch-subs"></span>
        <span class="label">Substations by DSO</span>
        <span class="count" id="count-subs"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-chargers" checked>
        <span class="icon" id="swatch-chargers"></span>
        <span class="label">Existing chargers by DSO</span>
        <span class="count" id="count-chargers"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-proposed" checked>
        <span class="icon"><span class="pulse" style="width:12px;height:12px;display:block;"></span></span>
        <span class="label">Proposed stations</span>
        <span class="count" id="count-proposed"></span>
      </label>
    </div>
  </div>

  <div id="legend" class="panel">
    <div class="legend-title">Provider (DSO)</div>
    <div class="legend-item">
      <span class="legend-swatch" id="legend-ide-tri"></span>
      <span class="legend-swatch" id="legend-ide-sq"></span>
      <span>i-DE (Iberdrola)</span>
    </div>
    <div class="legend-item">
      <span class="legend-swatch" id="legend-endesa-tri"></span>
      <span class="legend-swatch" id="legend-endesa-sq"></span>
      <span>Endesa</span>
    </div>
    <div class="legend-item">
      <span class="legend-swatch" id="legend-viesgo-tri"></span>
      <span class="legend-swatch" id="legend-viesgo-sq"></span>
      <span>Viesgo</span>
    </div>

    <div class="legend-title">Shapes</div>
    <div class="legend-item"><span class="legend-swatch" id="legend-shape-tri"></span><span>Substation</span></div>
    <div class="legend-item"><span class="legend-swatch" id="legend-shape-sq"></span><span>Existing charger</span></div>

    <div class="legend-title">Proposed network</div>
    <div class="legend-item">
      <span class="legend-swatch"><span class="pulse" style="width:12px;height:12px;display:block;margin:1px;"></span></span>
      <span>New station (blinking)</span>
    </div>
  </div>

  <div id="attribution">Team Greenlabs &middot; Iberdrola Datathon 2026</div>

<script type="application/json" id="__bi_map_data__">__DATA_JSON__</script>

<script>
  const DATA = JSON.parse(document.getElementById('__bi_map_data__').textContent);
  const ICON_ATLAS = '__ICON_ATLAS__';
  const ICON_MAPPING = __ICON_MAPPING_JSON__;

  // ----- Design tokens (must match build_bi_map.py) -----
  const DSO_HEX = { 'i-DE': '#4ade80', 'Endesa': '#3b82f6', 'Viesgo': '#ef4444' };
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
    } else if (shape === 'line') {
      el.style.height = '3px'; el.style.margin = '5px 0';
    }
  }
  setSwatch('legend-ide-tri', DSO_HEX['i-DE'], 'triangle');
  setSwatch('legend-ide-sq',  DSO_HEX['i-DE'], 'square');
  setSwatch('legend-endesa-tri', DSO_HEX['Endesa'], 'triangle');
  setSwatch('legend-endesa-sq',  DSO_HEX['Endesa'], 'square');
  setSwatch('legend-viesgo-tri', DSO_HEX['Viesgo'], 'triangle');
  setSwatch('legend-viesgo-sq',  DSO_HEX['Viesgo'], 'square');
  setSwatch('legend-shape-tri', 'rgba(255,255,255,0.6)', 'triangle');
  setSwatch('legend-shape-sq',  'rgba(255,255,255,0.6)', 'square');

  // Layer-toggle row swatches (neutral colours -- the map shows the palette)
  setSwatch('swatch-corridors', 'rgba(255,255,255,0.35)', 'line');
  setSwatch('swatch-subs', 'rgba(255,255,255,0.7)', 'triangle');
  setSwatch('swatch-chargers', 'rgba(255,255,255,0.7)', 'square');

  // ----- Stats -----
  const corridors = DATA.corridors || [];
  const subs = DATA.substations || [];
  const chgs = DATA.chargers || [];
  const props = DATA.proposed || [];
  document.getElementById('stat-proposed').textContent = props.length.toLocaleString();
  document.getElementById('stat-chargers').textContent = props.reduce((s, p) => s + (p.chargers || 0), 0).toLocaleString();
  document.getElementById('stat-existing').textContent = chgs.length.toLocaleString();
  document.getElementById('stat-subs').textContent = subs.length.toLocaleString();
  document.getElementById('stat-congested').textContent = subs.filter(s => s.status === 'Congested').length.toLocaleString();
  document.getElementById('count-corridors').textContent = corridors.length.toLocaleString();
  document.getElementById('count-subs').textContent = subs.length.toLocaleString();
  document.getElementById('count-chargers').textContent = chgs.length.toLocaleString();
  document.getElementById('count-proposed').textContent = props.length.toLocaleString();

  // ----- Layer visibility state -----
  const visible = { corridors: true, subs: true, chargers: true, proposed: true };
  ['corridors', 'subs', 'chargers', 'proposed'].forEach(k => {
    document.getElementById('toggle-' + k).addEventListener('change', (e) => {
      visible[k] = e.target.checked;
      render();
    });
  });

  // ----- Tooltip -----
  function tooltipForSubstation(o) {
    return {
      html:
        `<div style="font-weight:600;color:${DSO_HEX[o.dso] || '#fff'};font-size:12px;">${o.name}</div>` +
        `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${o.dso} &middot; Substation</div>` +
        `<div style="font-size:11px;margin-top:6px;">` +
        `<span style="opacity:0.6;">Available capacity</span> ${o.capacity_mw} MW<br/>` +
        `<span style="opacity:0.6;">Voltage</span> ${o.voltage_kv} kV<br/>` +
        `<span style="opacity:0.6;">Grid status</span> <span style="color:${STATUS_HEX[o.status] || '#fff'};">${o.status}</span>` +
        `</div>`,
      style: tooltipStyle()
    };
  }
  function tooltipForCharger(o) {
    return {
      html:
        `<div style="font-weight:600;color:${DSO_HEX[o.dso] || '#fff'};font-size:12px;">${o.name || 'Fast charger'}</div>` +
        `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${o.road || ''} &middot; ${o.province || ''}</div>` +
        `<div style="font-size:11px;margin-top:6px;">` +
        `<span style="opacity:0.6;">Power</span> ${o.power_kw} kW<br/>` +
        `<span style="opacity:0.6;">Connectors</span> ${o.connectors}${o.connector_types ? ' &middot; ' + o.connector_types : ''}<br/>` +
        `<span style="opacity:0.6;">Provider</span> <span style="color:${DSO_HEX[o.dso] || '#fff'};">${o.dso}</span><br/>` +
        `<span style="opacity:0.6;">Nearest sub</span> ${o.nearest_sub} (${o.nearest_sub_km} km)<br/>` +
        `<span style="opacity:0.6;">Local grid</span> <span style="color:${STATUS_HEX[o.status] || '#fff'};">${o.status}</span>` +
        `</div>`,
      style: tooltipStyle()
    };
  }
  function tooltipForProposed(o) {
    return {
      html:
        `<div style="font-weight:600;color:#4ade80;font-size:12px;">${o.id}</div>` +
        `<div style="font-size:11px;opacity:0.7;margin-top:2px;">Proposed station &middot; ${o.road}${o.is_tent ? ' &middot; TEN-T ' + o.tent_tier : ''}</div>` +
        `<div style="font-size:11px;margin-top:6px;">` +
        `<span style="opacity:0.6;">Chargers</span> ${o.chargers} &times; 150 kW<br/>` +
        `<span style="opacity:0.6;">Peak demand</span> ${o.demand_kw} kW<br/>` +
        `<span style="opacity:0.6;">DSO</span> <span style="color:${DSO_HEX[o.dso] || '#fff'};">${o.dso}</span><br/>` +
        `<span style="opacity:0.6;">Nearest sub</span> ${o.nearest_sub} (${o.connection_km} km)<br/>` +
        `<span style="opacity:0.6;">Grid status</span> <span style="color:${STATUS_HEX[o.status] || '#fff'};">${o.status}</span>` +
        `</div>`,
      style: tooltipStyle()
    };
  }
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

  // ----- Deck.gl init -----
  const deckgl = new deck.DeckGL({
    container: 'map',
    mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    initialViewState: { longitude: -3.7, latitude: 40.1, zoom: 5.6, pitch: 30, bearing: 0 },
    controller: true,
    getTooltip: ({object, layer}) => {
      if (!object || !layer) return null;
      const id = layer.id;
      if (id === 'substations') return tooltipForSubstation(object);
      if (id === 'chargers')    return tooltipForCharger(object);
      if (id === 'proposed-inner' || id === 'proposed-outer') return tooltipForProposed(object);
      return null;
    }
  });

  function render() {
    const layers = [];

    // 1. Interurban corridors -- minimal translucent white underlay, same
    //    geometry source as the ABM animation.
    if (visible.corridors) {
      layers.push(new deck.PathLayer({
        id: 'corridors',
        data: corridors,
        getPath: d => d.path,
        getColor: [255, 255, 255, 32],
        getWidth: 2200,
        widthMinPixels: 1,
        widthMaxPixels: 3,
        capRounded: true,
        jointRounded: true,
        opacity: 1,
      }));
    }

    // 2. Substations (triangles, coloured by DSO). Small sizes so they do
    //    not dominate the base map, matching the ABM charger dot scale.
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

    // 3. Existing chargers (squares, coloured by nearest-DSO -- same
    //    palette as substations so the provider read is instant).
    if (visible.chargers) {
      layers.push(new deck.IconLayer({
        id: 'chargers',
        data: chgs,
        iconAtlas: ICON_ATLAS,
        iconMapping: ICON_MAPPING,
        getIcon: () => 'square',
        getPosition: d => d.position,
        getColor: d => d.color,
        getSize: 7,
        sizeMinPixels: 3,
        sizeMaxPixels: 8,
        pickable: true,
        opacity: 0.85,
      }));
    }

    // 4. Proposed stations -- pulsing green circles (same design tokens
    //    as the ABM animation proposed markers).
    if (visible.proposed) {
      const phase = (Date.now() / 750) % (Math.PI * 2);
      const pulseScale = 1 + 0.45 * Math.sin(phase);
      const pulseAlpha = Math.round(80 + 100 * (0.5 + 0.5 * Math.sin(phase)));

      layers.push(new deck.ScatterplotLayer({
        id: 'proposed-outer',
        data: props,
        getPosition: d => d.position,
        getFillColor: [0, 0, 0, 0],
        getLineColor: [74, 222, 128, pulseAlpha],
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
        getPosition: d => d.position,
        getFillColor: [74, 222, 128, 230],
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

  // Drive a 60Hz redraw so the pulsing proposed ring stays alive even
  // when the user is not interacting with the map.
  function tick() {
    render();
    requestAnimationFrame(tick);
  }
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
    corridors = load_corridors()
    print(f"  corridors   : {len(corridors):,}")
    print(f"  substations : {len(subs):,}")
    print(f"  chargers    : {len(chargers):,}")
    print(f"  proposed    : {len(proposed):,}")

    print("Joining chargers -> nearest substation for DSO + status...")
    chargers = nearest_substation_status(chargers, subs)

    print("Building marker records...")
    data = {
        "corridors":   corridors,
        "substations": substation_markers(subs),
        "chargers":    charger_markers(chargers),
        "proposed":    proposed_markers(proposed),
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


if __name__ == "__main__":
    main()
