# Datathon Data Gap Audit

Audit date: 2026-03-14

This note maps the datathon brief to the datasets currently present in `data/raw/`, highlights what was added during this audit, and lists the exact official source URLs used.

## Filled During This Audit

### 1. Ministry road network
- Local file: `data/raw/ministry_roads/hermes_roads.geojson`
- Status: usable
- Features: 1,602 road geometries
- Official source:
  - Viewer: `https://mapas.fomento.gob.es/VisorHermes/`
  - Service layer: `https://mapas.fomento.gob.es/arcgis2/rest/services/Hermes/0_CARRETERAS/MapServer/19`

### 2. DGT IMD traffic counts
- Local file: `data/raw/additional/dgt_imd_traffic/hermes_traffic_count_stations.geojson`
- Status: usable
- Features: 14,066 traffic count stations
- Official source:
  - Viewer: `https://mapas.fomento.gob.es/VisorHermes/`
  - Service layer: `https://mapas.fomento.gob.es/arcgis2/rest/services/Hermes/2_INSTALACIONES_Y_SERVICIOS/MapServer/1010`

### 3. TEN-T corridors
- Local file: `data/raw/additional/tent_corridors/hermes_tent_roads_2024.geojson`
- Status: usable
- Features: 77 corridor segments
- Official source:
  - Viewer: `https://mapas.fomento.gob.es/VisorHermes/`
  - Service layer: `https://mapas.fomento.gob.es/arcgis/rest/services/Hermes/NEW_RED_TRANSEUROPEA_DE_TRANSPORTE__TEN_T_2024/MapServer/5`

### 4. Candidate rest areas / service areas
- Local file: `data/raw/additional/osm_rest_areas/hermes_service_areas.geojson`
- Status: usable
- Features: 113 service areas
- Official source:
  - Viewer: `https://mapas.fomento.gob.es/VisorHermes/`
  - Service layer: `https://mapas.fomento.gob.es/arcgis2/rest/services/Hermes/2_INSTALACIONES_Y_SERVICIOS/MapServer/1356`
- Note: this folder was originally intended for OpenStreetMap extractions, but the official Hermes service-area layer is more reliable and more appropriate for the "trustworthy and usable" requirement.

### 5. INE municipal population
- Local files:
  - `data/raw/additional/ine_population/ine_pobmun.zip`
  - `data/raw/additional/ine_population/ine_pobmun_2025.xlsx`
  - `data/raw/additional/ine_population/ine_population_municipal_2025.csv`
- Status: usable
- Rows in CSV: 8,132 municipalities
- Official source:
  - INE municipal detail page: `https://www.ine.es/dynt3/inebase/es/index.htm?padre=525`
  - Direct national ZIP: `https://www.ine.es/pob_xls/pobmun.zip`
- Notes:
  - The CSV was derived directly from `pobmun25.xlsx`.
  - Columns: `cpro`, `provincia`, `cmun`, `nombre`, `pob_2025`, `hombres`, `mujeres`, `municipality_code`

### 6. INE tourism seasonality
- Local files:
  - `data/raw/additional/ine_population/ine_hotel_occupancy_provinces_2024_raw.csv`
  - `data/raw/additional/ine_population/ine_hotel_occupancy_provinces_2024_monthly.csv`
  - `data/raw/additional/ine_population/ine_hotel_travelers_overnights_provinces_raw.csv`
  - `data/raw/additional/ine_population/ine_hotel_travelers_overnights_provinces_2025_monthly.csv`
- Status: usable
- Official sources:
  - Occupancy table page: `https://www.ine.es/jaxi/Tabla.htm?tpx=75843`
  - Occupancy direct CSV: `https://www.ine.es/jaxi/files/tpx/es/csv_bd/75843.csv`
  - Travelers / overnights table page: `https://www.ine.es/jaxiT3/Tabla.htm?t=2074&L=0`
  - Travelers / overnights direct CSV: `https://www.ine.es/jaxiT3/files/t/es/csv_bd/2074.csv`
- Notes:
  - The occupancy file is the cleanest source for `seasonal_multiplier`.
  - The travelers / overnights file adds a second tourism-pressure signal at province-month level.

### 7. REE transmission context
- Local files:
  - `data/raw/additional/ree_grid/ree_generation_capacity_nodes_2026_03_02_raw.csv`
  - `data/raw/additional/ree_grid/ree_generation_capacity_nodes_2026_03_02_key_fields.csv`
  - `data/raw/additional/ree_grid/ree_demand_requests_2026_02_raw.xlsx`
  - `data/raw/additional/ree_grid/ree_demand_requests_2026_02.csv`
- Status: usable with limitations
- Official sources:
  - Generator access page: `https://www.ree.es/es/clientes/generador/acceso-conexion/conoce-la-capacidad-de-acceso`
  - Generator direct CSV: `https://www.ree.es/sites/default/files/12_CLIENTES/Documentos/2026_03_02_GRT_generacion.csv`
  - Consumer access page: `https://www.ree.es/es/clientes/consumidor/acceso-conexion/conoce-el-estado-de-las-solicitudes`
  - Consumer direct XLSX: `https://www.ree.es/sites/default/files/12_CLIENTES/Documentos/publicacion-solicitudes-acceso-demanda-febrero-2026.xlsx`
- Notes:
  - These files are official and useful for transmission-context analysis.
  - They are still not a geocoded transport-grid asset layer, so they should not replace DSO data for nearest-substation spatial joins.

## Already Present Before This Audit

### Mandatory datasets already available in the repo
- NAP charging points:
  - Local file: `data/raw/nap_charging_points/nap_ev_charging_points.xml`
  - Official source: `https://nap.dgt.es/dataset/puntos-de-recarga-electrica-para-vehiculos`
- datos.gob EV forecast output:
  - Local folder: `data/raw/datos_gob_ev_forecast/`
  - Source page cited in the brief: `https://datos.gob.es/es/conocimiento/ruta-la-electrificacion-descifrando-el-crecimiento-del-vehiculo-electrico-en-espana`
- Grid capacity from distributors:
  - Local folders: `data/raw/grid_capacity/ide_iberdrola/`, `data/raw/grid_capacity/endesa/`, `data/raw/grid_capacity/viesgo/`
  - Official source pages from the datathon brief:
    - i-DE: `https://www.i-de.es/conexion-red-electrica/suministro-electrico/mapa-capacidad-consumo`
    - e-distribucion: `https://www.edistribucion.com/es/red-electrica/nodos-capacidad-red/capacidad-generacion.html`
    - Viesgo: `https://www.viesgodistribucion.com/mapa-interactivo-de-la-red`
- DGT registrations:
  - Local folder: `data/raw/dgt_registrations/`
  - Official source page from the brief: `https://www.dgt.es/menusecundario/dgt-en-cifras/matraba-listados/matriculaciones-automoviles-mensual.html`

## Still Missing Or Still Limited

### 1. REE spatial transmission geometry
- Target folder: `data/raw/additional/ree_grid/`
- Status: still missing as a geocoded layer
- Why it is still limited:
  - The REE publications now in the repo are official and usable, but they remain tabular and not spatially geocoded.
  - They add context and evidence, not a plug-in replacement for geospatial substation layers.

### 2. AFIR regulation reference
- Target folder: `data/raw/additional/afir_requirements/`
- Status: optional reference, not required for modeling
- Official source:
  - `https://eur-lex.europa.eu/eli/reg/2023/1804/oj/eng`
- Recommendation:
  - Use this as a policy reference for the 60 km TEN-T spacing rule, but do not prioritize downloading it over modeling datasets.

## Practical Takeaway

If you want the minimum trustworthy dataset set to keep moving in notebooks, no key notebook input folder is empty anymore.

Everything else needed by the current notebook structure is now either:
- already in the repo, or
- added during this audit in a directly usable format.
