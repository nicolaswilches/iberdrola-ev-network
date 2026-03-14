# Data Sources

## Mandatory Sources

1. **Road Routes** — Ministry of Transport and Sustainable Mobility
   - URL: https://www.mitma.gob.es/ (exact dataset link TBD)
   - Content: Interurban road network geometry (autopistas, autovías, carreteras nacionales)
   - Used in: Notebooks 03, 06, 07

2. **Electric Vehicle Charging Points** — National Access Point for Traffic and Mobility
   - URL: https://nap.mitma.es/
   - Content: Existing EV charging station locations across Spain
   - Used in: Notebooks 04, 07

3. **Route to Electrification (datos.gob.es)** — GitHub Fork (MANDATORY)
   - URL: https://datos.gob.es/ (link to specific repo TBD)
   - Content: EV growth projection model → 2027 fleet estimate
   - Used in: Notebooks 02, 06, 09

4. **i-DE Consumption Capacity Map** — Iberdrola Group
   - URL: https://www.i-de.es/conexion-red/mapa-capacidad-consumo
   - Content: Substation-level available capacity (MW) + coordinates
   - Used in: Notebooks 05, 08

5. **e-distribución Historical Access Capacity** — Endesa
   - URL: https://www.edistribucion.com/ (capacity documents section)
   - Content: Node-level access capacity data (CSV/XLSX)
   - Used in: Notebooks 05, 08

6. **Viesgo Interactive Grid Map** — Viesgo Distribución
   - URL: https://www.viesgodistribucion.com/
   - Content: Substation-level capacity for northern Spain
   - Used in: Notebooks 05, 08

7. **DGT Vehicle Registrations** — Dirección General de Tráfico
   - URL: https://www.dgt.es/microdatos/
   - Content: Monthly registration data by province (Jun-Nov 2025)
   - Used in: Notebooks 01, 06

## Additional Sources (Innovation)

8. **DGT IMD Traffic Counts** — Intensidad Media Diaria
   - Content: Average daily traffic volume per road segment
   - Used in: Notebooks 03, 06

9. **INE Population & Tourism Data** — Instituto Nacional de Estadística
   - URL: https://www.ine.es/
   - Content: Municipal population, tourism flow by province
   - Used in: Notebook 06

10. **REE Grid Data** — Red Eléctrica de España
    - URL: https://www.ree.es/ / https://www.esios.ree.es/
    - Content: Transmission-level demand/capacity by region
    - Used in: Notebook 05

11. **OpenStreetMap Rest Areas** — via Overpass API
    - Content: Service stations and rest areas on Spanish motorways
    - Used in: Notebook 07

12. **EU TEN-T Core Network Corridors** — European Commission
    - Content: Trans-European Transport Network maps for Spain
    - Used in: Notebooks 03, 07 (AFIR compliance check)

13. **AFIR Regulation Requirements** — EU Alternative Fuels Infrastructure Regulation
    - Content: Mandatory charging pool spacing (every 60 km on TEN-T core by end 2025)
    - Used in: Notebooks 07, Report

14. **IDAE Studies** — Instituto para la Diversificación y Ahorro de la Energía
    - URL: https://www.idae.es/
    - Content: EV charging patterns and infrastructure planning studies
    - Used in: Report, assumptions

15. **ANFAC / ACEA EV Statistics** — Manufacturer associations
    - Content: Average EV range by model, fleet composition data
    - Used in: Assumptions (autonomy range justification)
