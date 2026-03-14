# Assumptions Register

All assumptions used in this model are documented below with justification.

## A1 — Average EV Autonomy Range
- **Value used:** 300 km (WLTP)
- **Effective range (80% usable):** 240 km
- **Justification:** Based on weighted average of top-selling EVs in Spain (2024-2025 fleet mix). Sources: ANFAC registration data, ACEA EV report.
- **Impact on model:** Determines maximum station spacing (~120-150 km between chargers to ensure no driver falls below 20% battery).

## A2 — Maximum Station Spacing
- **Value used:** 150 km
- **Justification:** Derived from A1. Conservative buffer: 240 km effective range ÷ ~1.6 safety factor = 150 km. Also aligned with AFIR requirement of charging pools every 60 km on TEN-T core network (our spacing is more conservative for non-TEN-T roads).
- **Impact on model:** Core parameter for the optimization algorithm's coverage constraint.

## A3 — Standard Power per Charger
- **Value used:** 150 kW (FIXED — mandated by datathon rules)
- **Impact on model:** estimated_demand_kw = n_chargers_proposed × 150 kW

## A4 — Grid Status Classification Thresholds
- **Sufficient:** Available capacity ≥ 5 MW at nearest substation
- **Moderate:** Available capacity between 1 MW and 5 MW
- **Congested:** Available capacity < 1 MW
- **Justification:** A typical interurban charging station with 4 fast chargers requires 0.6 MW (4 × 150 kW). "Sufficient" means the substation can comfortably host the station plus future expansion. "Moderate" means it can host a station but with limited headroom. "Congested" means grid reinforcement is needed before deployment.
- **Spatial matching method:** Each proposed station is matched to the nearest substation within a [X] km radius using haversine distance. If no substation exists within radius, classified as Congested by default.
- **Sources:** [TBD — cite industry standards, CNMC reports, or IDAE planning guides]

## A5 — Number of Chargers per Station
- **Method:** Based on projected daily EV traffic at that road segment × average charging probability × average charging duration.
- **Formula:** n_chargers = ceil((daily_ev_traffic × charging_probability × avg_charge_hours) / operating_hours_per_day)
- **Default range:** 2-8 chargers per station depending on corridor traffic volume.
- **Assumptions within:**
  - Charging probability per passing EV: ~5-10% on any given trip
  - Average charging session: 20-30 minutes for DC fast charging
  - Operating hours: 18 hours/day effective utilization window
- **Sources:** [TBD — IDAE, ChargePoint utilization studies]

## A6 — EV Fleet Projection 2027
- **Value used:** [OUTPUT FROM DATOS.GOB.ES FORK]
- **Source:** Mandatory GitHub fork of datos.gob.es Exercise 3
- **Used for:** total_ev_projected_2027 in File 1, and as input to demand model

## A7 — Seasonal Traffic Adjustment
- **Method:** Apply seasonal multiplier to IMD traffic data for summer months (Jul-Aug) on Mediterranean and coastal corridors.
- **Multiplier:** 1.5x - 2.5x depending on corridor (AP-7, A-7 highest)
- **Justification:** DGT historical traffic data shows peak summer volumes on coastal routes.
- **Impact on model:** Station sizing on tourist corridors accounts for peak demand, not just annual average.

## A8 — Interurban Road Filter
- **Rule:** Only autopistas (AP-), autovías (A-), and carreteras nacionales (N-) are included.
- **Exclusion:** Urban road sections excluded regardless of municipality size, per datathon rules.
- **Method:** Filter using Ministry of Transport road classification field.
