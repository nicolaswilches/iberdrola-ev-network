SELECT
    station_id,
    node_id,
    name,
    num_connectors,
    peak_queue,
    total_wait_min,
    total_charge_min,
    total_wait_min / (total_charge_min / 21.7) AS wait_min_per_charge_session
FROM
    './src/new-abm/outputs/debug/20kagents/baseline_station_summary.csv'
WHERE
    total_wait_min > 100
    AND wait_min_per_charge_session > 5
    AND peak_queue > 5
ORDER BY
    wait_min_per_charge_session DESC;

SELECT
    SUM(num_connectors) / COUNT(*) AS avg_connectors
FROM
    './src/new-abm/outputs/debug/20kagents/baseline_station_summary.csv';

SELECT
    *
FROM
    './src/new-abm/outputs/debug/20kagents/baseline_station_summary.csv'
WHERE
    name LIKE '%Proposed%';

SELECT
    *
FROM
    './src/new-abm/outputs/debug/20kagents/baseline_station_summary.csv'
WHERE
    total_sessions = 0;