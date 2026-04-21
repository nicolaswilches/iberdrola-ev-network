"""
Build visualization/bi_map.html as a deck.gl interactive map aligned
with visualization/abm_animation/index.html's design language.

Layers (toggleable via checkboxes):
  - Energy substations (~2,147) — triangles coloured by DSO:
      Endesa = amber, i-DE = red, Viesgo = blue.
  - Existing chargers (~3,246) — squares coloured by nearest substation's
    grid_status: Sufficient = green, Moderate = amber, Congested = red.
  - Proposed stations (8) — green blinking circles (same design as the
    ABM animation proposed markers).

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

# Design tokens — kept in sync with abm_animation/index.html
DSO_COLORS = {
    "Endesa": [251, 191, 36],   # amber
    "i-DE":   [248, 113, 113],  # red
    "Viesgo": [96, 165, 250],   # blue
}
STATUS_COLORS = {
    "Sufficient": [74, 222, 128],  # green
    "Moderate":   [251, 191, 36],  # amber
    "Congested":  [248, 113, 113], # red
}
PROPOSED_GREEN = [74, 222, 128]


# ---------------------------------------------------------------------------
# Icon atlas: triangle (substation) + square (charger). Tinted per-feature
# by deck.gl IconLayer's getColor.
# ---------------------------------------------------------------------------
def make_icon_atlas() -> str:
    """Return a data:image/png;base64,... URL for a 2-icon atlas (triangle,
    square), each 128×128 px. IconLayer tints white pixels by getColor."""
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
    # Some rows have NaN coords — drop them.
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    # Collapse rare unknown DSO labels to i-DE so the colour mapping covers
    # 100% of the points (there are a handful in the wild).
    df.loc[~df["distributor_network"].isin(DSO_COLORS), "distributor_network"] = "i-DE"
    return df


def load_chargers() -> pd.DataFrame:
    df = pd.read_csv(DATA / "interurban_chargers_baseline.csv")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    # Only carry the columns we show in the tooltip; keeps payload lean.
    keep = [
        "site_id", "name", "latitude", "longitude", "num_stations",
        "n_connectors", "max_power_kw", "connector_types", "province",
        "nearest_road",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = ""
    df = df[keep]
    # Fast-charger filter mirrors NB04's ≥50 kW rule for "high-grade" dots.
    df["max_power_kw"] = pd.to_numeric(df["max_power_kw"], errors="coerce").fillna(0)
    df = df[df["max_power_kw"] >= 50].copy()
    return df


def load_proposed() -> pd.DataFrame:
    df = pd.read_csv(DATA / "stations_with_grid_status.csv")
    return df


def nearest_substation_status(
    chargers: pd.DataFrame, subs: pd.DataFrame
) -> pd.DataFrame:
    """For each charger, tag `grid_status` inherited from its nearest
    substation (haversine, unbounded) so the colour layer can reflect the
    local grid constraint."""
    # BallTree expects radians
    sub_rad = np.radians(subs[["latitude", "longitude"]].values)
    chg_rad = np.radians(chargers[["latitude", "longitude"]].values)
    tree = BallTree(sub_rad, metric="haversine")
    _, idx = tree.query(chg_rad, k=1)
    status = subs["grid_status"].iloc[idx[:, 0]].values
    sub_name = subs["substation_name"].iloc[idx[:, 0]].values
    sub_dso = subs["distributor_network"].iloc[idx[:, 0]].values
    sub_lat = subs["latitude"].iloc[idx[:, 0]].values
    sub_lon = subs["longitude"].iloc[idx[:, 0]].values
    # Distance in km
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
# Marker record → JSON dicts (lean, with pre-computed colour arrays)
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
    out = []
    for _, r in chargers.iterrows():
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
            "status": status,
            "color": STATUS_COLORS.get(status, STATUS_COLORS["Sufficient"]),
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
<title>Spain EV Network — BI Map</title>
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
  .stat-value.red   { color: #f87171; }
  .stat-value.blue  { color: #60a5fa; }

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

  /* Custom pulsing ring for proposed markers rendered as DOM overlay —
     actually done in deck.gl; this class is for the legend swatch. */
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
    <div class="subtitle">Spain Interurban — 2027 Proposal</div>

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
        <input type="checkbox" id="toggle-subs" checked>
        <span class="icon" id="swatch-subs"></span>
        <span class="label">Substations by DSO</span>
        <span class="count" id="count-subs"></span>
      </label>
      <label class="layer-row">
        <input type="checkbox" id="toggle-chargers" checked>
        <span class="icon" id="swatch-chargers"></span>
        <span class="label">Existing chargers by congestion</span>
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
    <div class="legend-title">Substation DSO</div>
    <div class="legend-item"><span class="legend-swatch" id="legend-endesa"></span><span>Endesa</span></div>
    <div class="legend-item"><span class="legend-swatch" id="legend-ide"></span><span>i-DE (Iberdrola)</span></div>
    <div class="legend-item"><span class="legend-swatch" id="legend-viesgo"></span><span>Viesgo</span></div>

    <div class="legend-title">Charger grid status</div>
    <div class="legend-item"><span class="legend-swatch" id="legend-suff"></span><span>Sufficient (≥5 MW)</span></div>
    <div class="legend-item"><span class="legend-swatch" id="legend-mod"></span><span>Moderate (1–5 MW)</span></div>
    <div class="legend-item"><span class="legend-swatch" id="legend-cong"></span><span>Congested (&lt;1 MW)</span></div>

    <div class="legend-title">Proposed network</div>
    <div class="legend-item">
      <span class="legend-swatch"><span class="pulse" style="width:12px;height:12px;display:block;margin:1px;"></span></span>
      <span>New station (blinking)</span>
    </div>
  </div>

  <div id="attribution">Team Greenlabs · Iberdrola Datathon 2026</div>

<script type="application/json" id="__bi_map_data__">__DATA_JSON__</script>

<script>
  const DATA = JSON.parse(document.getElementById('__bi_map_data__').textContent);
  const ICON_ATLAS = '__ICON_ATLAS__';
  const ICON_MAPPING = __ICON_MAPPING_JSON__;

  // ----- Paint legend/stat swatches from the design tokens embedded in JS -----
  const DSO_HEX = { Endesa: '#fbbf24', 'i-DE': '#f87171', Viesgo: '#60a5fa' };
  const STATUS_HEX = { Sufficient: '#4ade80', Moderate: '#fbbf24', Congested: '#f87171' };
  function paintSwatch(id, color, shape) {
    const el = document.getElementById(id); if (!el) return;
    if (shape === 'triangle') {
      el.style.width = '14px'; el.style.height = '14px';
      el.style.background = color;
      el.style.clipPath = 'polygon(50% 0%, 0% 100%, 100% 100%)';
    } else if (shape === 'square') {
      el.style.width = '12px'; el.style.height = '12px';
      el.style.background = color; el.style.margin = '1px';
    }
  }
  paintSwatch('legend-endesa', DSO_HEX.Endesa, 'triangle');
  paintSwatch('legend-ide', DSO_HEX['i-DE'], 'triangle');
  paintSwatch('legend-viesgo', DSO_HEX.Viesgo, 'triangle');
  paintSwatch('legend-suff', STATUS_HEX.Sufficient, 'square');
  paintSwatch('legend-mod', STATUS_HEX.Moderate, 'square');
  paintSwatch('legend-cong', STATUS_HEX.Congested, 'square');
  // swatches in the layer toggles (use DSO-neutral style)
  const sSubs = document.getElementById('swatch-subs');
  sSubs.style.width='14px'; sSubs.style.height='14px';
  sSubs.style.background='#fbbf24'; sSubs.style.clipPath='polygon(50% 0%, 0% 100%, 100% 100%)';
  const sChg = document.getElementById('swatch-chargers');
  sChg.style.width='12px'; sChg.style.height='12px'; sChg.style.background='#f87171'; sChg.style.margin='1px';

  // ----- Stats -----
  const subs = DATA.substations || [];
  const chgs = DATA.chargers || [];
  const props = DATA.proposed || [];
  document.getElementById('stat-proposed').textContent = props.length.toLocaleString();
  document.getElementById('stat-chargers').textContent = props.reduce((s, p) => s + (p.chargers || 0), 0).toLocaleString();
  document.getElementById('stat-existing').textContent = chgs.length.toLocaleString();
  document.getElementById('stat-subs').textContent = subs.length.toLocaleString();
  document.getElementById('stat-congested').textContent = subs.filter(s => s.status === 'Congested').length.toLocaleString();
  document.getElementById('count-subs').textContent = subs.length.toLocaleString();
  document.getElementById('count-chargers').textContent = chgs.length.toLocaleString();
  document.getElementById('count-proposed').textContent = props.length.toLocaleString();

  // ----- Layer visibility state -----
  const visible = { subs: true, chargers: true, proposed: true };
  ['subs', 'chargers', 'proposed'].forEach(k => {
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
        `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${o.dso} · Substation</div>` +
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
        `<div style="font-weight:600;color:${STATUS_HEX[o.status] || '#fff'};font-size:12px;">${o.name || 'Fast charger'}</div>` +
        `<div style="font-size:11px;opacity:0.7;margin-top:2px;">${o.road || '—'} · ${o.province || ''}</div>` +
        `<div style="font-size:11px;margin-top:6px;">` +
        `<span style="opacity:0.6;">Power</span> ${o.power_kw} kW<br/>` +
        `<span style="opacity:0.6;">Connectors</span> ${o.connectors}${o.connector_types ? ' · ' + o.connector_types : ''}<br/>` +
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
        `<div style="font-size:11px;opacity:0.7;margin-top:2px;">Proposed station · ${o.road}${o.is_tent ? ' · TEN-T ' + o.tent_tier : ''}</div>` +
        `<div style="font-size:11px;margin-top:6px;">` +
        `<span style="opacity:0.6;">Chargers</span> ${o.chargers} × 150 kW<br/>` +
        `<span style="opacity:0.6;">Peak demand</span> ${o.demand_kw} kW<br/>` +
        `<span style="opacity:0.6;">DSO</span> ${o.dso}<br/>` +
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

    if (visible.subs) {
      layers.push(new deck.IconLayer({
        id: 'substations',
        data: subs,
        iconAtlas: ICON_ATLAS,
        iconMapping: ICON_MAPPING,
        getIcon: () => 'triangle',
        getPosition: d => d.position,
        getColor: d => d.color,
        getSize: 18,
        sizeMinPixels: 8,
        sizeMaxPixels: 22,
        pickable: true,
        opacity: 0.9,
      }));
    }

    if (visible.chargers) {
      layers.push(new deck.IconLayer({
        id: 'chargers',
        data: chgs,
        iconAtlas: ICON_ATLAS,
        iconMapping: ICON_MAPPING,
        getIcon: () => 'square',
        getPosition: d => d.position,
        getColor: d => d.color,
        getSize: 14,
        sizeMinPixels: 6,
        sizeMaxPixels: 16,
        pickable: true,
        opacity: 0.85,
      }));
    }

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
    print("Loading datasets…")
    subs = load_substations()
    chargers = load_chargers()
    proposed = load_proposed()
    print(f"  substations : {len(subs):,}")
    print(f"  chargers    : {len(chargers):,}")
    print(f"  proposed    : {len(proposed):,}")

    print("Joining chargers → nearest substation for grid status…")
    chargers = nearest_substation_status(chargers, subs)

    print("Building marker records…")
    data = {
        "substations": substation_markers(subs),
        "chargers":    charger_markers(chargers),
        "proposed":    proposed_markers(proposed),
    }

    print("Encoding icon atlas…")
    atlas = make_icon_atlas()

    print("Rendering HTML…")
    data_json = json.dumps(data, separators=(",", ":"))
    # Escape </script> to keep the enclosing JSON script tag intact.
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
    print(f"  substations : {len(data['substations']):,}")
    print(f"  chargers    : {len(data['chargers']):,}")
    print(f"  proposed    : {len(data['proposed']):,}")


if __name__ == "__main__":
    main()
