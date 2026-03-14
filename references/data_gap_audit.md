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

## Still Missing Or Not Yet Reliable Enough To Ingest

### 1. REE transmission-level grid context
- Target folder: `data/raw/additional/ree_grid/`
- Status: still missing as a clean spatial dataset
- Why it is still missing:
  - The official REE public files we found are trustworthy, but the ones exposed publicly here are tabular node-capacity files, not a clean geocoded transmission network layer ready for spatial joins.
  - The REE API endpoint tested during the audit was unstable from this environment.
- Official source pages:
  - Consumer access status: `https://www.ree.es/es/clientes/consumidor/acceso-conexion/conoce-el-estado-de-las-solicitudes`
  - Generator access capacity: `https://www.ree.es/es/clientes/generador/acceso-conexion/conoce-la-capacidad-de-acceso`
- Direct official files found:
  - `https://www.ree.es/sites/default/files/12_CLIENTES/Documentos/2026_02_28_GRT_demanda.xlsx`
  - `https://www.ree.es/sites/default/files/12_CLIENTES/Documentos/2026_03_02_GRT_generacion.csv`
- Recommendation:
  - Only ingest REE once you have a spatially joinable layer or a clearly documented method to match REE nodes to coordinates/substations. Otherwise it will add churn without helping the optimization notebooks.

### 2. AFIR regulation reference
- Target folder: `data/raw/additional/afir_requirements/`
- Status: optional reference, not required for modeling
- Official source:
  - `https://eur-lex.europa.eu/eli/reg/2023/1804/oj/eng`
- Recommendation:
  - Use this as a policy reference for the 60 km TEN-T spacing rule, but do not prioritize downloading it over modeling datasets.

## Practical Takeaway

If you want the minimum trustworthy dataset set to keep moving in notebooks, the critical missing data need is now mostly reduced to `ree_grid/`.

Everything else needed by the current notebook structure is now either:
- already in the repo, or
- added during this audit in a directly usable format.
