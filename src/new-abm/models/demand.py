"""Travel demand models.

TripRequest is the unit of demand: one BEV making one interurban trip.

ODMatrix holds zone-to-zone demand volumes and can generate TripRequest
lists with sampled departure times.  Real demand data (MiTMA study routes,
DGT vehicle flows) can be loaded here instead of using synthetic generation.

Calibration hook:
    observed_link_counts is the interface through which IMD / AADT data
    enters the model.  A future calibrator would adjust OD flows until
    simulated link counts match observed ones (standard transport planning
    procedure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TripRequest:
    """
    A single interurban BEV trip to be simulated.

    Created by ODMatrix.generate_trips() or loaded from a CSV.
    Becomes the seed for a VehicleAgent in the simulation engine.
    """

    trip_id: str
    origin: str          # node_id of origin city / zone
    destination: str     # node_id of destination city / zone
    departure_time_min: float  # minutes from midnight (simulation epoch)

    # Optional metadata for calibration / reporting
    od_pair_id: str = ""
    num_passengers: int = 1
    trip_purpose: str = "leisure"  # leisure | business | commute

    def __repr__(self) -> str:
        return (
            f"TripRequest({self.trip_id!r}, "
            f"{self.origin}→{self.destination}, "
            f"dep={self.departure_time_min:.0f} min)"
        )


@dataclass
class ODPair:
    """Volume and attributes for a single origin-destination pair."""

    origin: str
    destination: str
    daily_bev_trips: float     # expected BEV trips per day
    purpose: str = "leisure"

    @property
    def pair_id(self) -> str:
        return f"{self.origin}_{self.destination}"


class ODMatrix:
    """
    Origin-destination demand matrix.

    Stores daily BEV trip volumes between zone pairs and generates
    time-stamped TripRequest objects for the simulation.

    Calibration interface
    ---------------------
    Observed link counts (IMD / AADT style data) are passed in via
    ``set_observed_counts``.  A calibrator (see calibration/interfaces.py)
    can call ``adjust_od_flows`` to shift volumes until simulated flows
    match observations.  In the MVP the synthetic OD is used directly.
    """

    def __init__(self) -> None:
        self.pairs: List[ODPair] = []
        # Calibration targets: link_id → observed daily BEV count
        self.observed_link_counts: Dict[str, float] = {}

    def add_pair(self, pair: ODPair) -> None:
        self.pairs.append(pair)

    def set_observed_counts(self, counts: Dict[str, float]) -> None:
        """
        Store observed IMD / link-count targets for calibration.

        Args:
            counts: mapping from road segment / link id to daily observed
                    BEV count.  In practice these come from DGT loop detectors
                    or MiTMA permanent stations.
        """
        self.observed_link_counts = counts
        logger.info("Loaded %d observed link counts", len(counts))

    def total_daily_trips(self) -> float:
        return sum(p.daily_bev_trips for p in self.pairs)

    def generate_trips(
        self,
        rng: np.random.Generator,
        num_trips: Optional[int] = None,
        peak_config: Optional[Dict] = None,
    ) -> List[TripRequest]:
        """
        Sample trip requests from the OD matrix with realistic departure times.

        Args:
            rng:         NumPy random generator (for reproducibility).
            num_trips:   Override total trips (useful for scaling demo runs).
            peak_config: Dict with morning/evening peak parameters (from config).

        Returns:
            Sorted list of TripRequest objects.
        """
        if not self.pairs:
            logger.warning("ODMatrix has no pairs; returning empty trip list")
            return []

        total = num_trips or int(self.total_daily_trips())
        if total <= 0:
            return []

        # Distribute trips proportionally across OD pairs
        weights = np.array([p.daily_bev_trips for p in self.pairs], dtype=float)
        weights /= weights.sum()
        pair_counts = rng.multinomial(total, weights)

        trips: List[TripRequest] = []
        trip_counter = 0

        for pair, count in zip(self.pairs, pair_counts):
            for _ in range(count):
                dep = _sample_departure_time(rng, peak_config)
                trips.append(
                    TripRequest(
                        trip_id=f"TRIP_{trip_counter:05d}",
                        origin=pair.origin,
                        destination=pair.destination,
                        departure_time_min=dep,
                        od_pair_id=pair.pair_id,
                        trip_purpose=pair.purpose,
                    )
                )
                trip_counter += 1

        trips.sort(key=lambda t: t.departure_time_min)
        logger.info(
            "Generated %d trips from %d OD pairs", len(trips), len(self.pairs)
        )
        return trips

    def __repr__(self) -> str:
        return (
            f"ODMatrix({len(self.pairs)} pairs, "
            f"{self.total_daily_trips():.0f} daily BEV trips)"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_departure_time(
    rng: np.random.Generator,
    peak_config: Optional[Dict],
) -> float:
    """
    Sample a departure time (minutes from midnight) using a bimodal
    morning / evening peak distribution.

    Default parameters replicate typical Spanish interurban BEV traffic:
    - Morning peak: 7:00 (±60 min)
    - Evening peak: 17:00 (±90 min)
    - Off-peak: uniform 5:00 – 22:00
    """
    cfg = peak_config or {}
    morning_center = cfg.get("peak_morning_center_min", 420)    # 7:00
    morning_std    = cfg.get("peak_morning_std_min", 60)
    evening_center = cfg.get("peak_evening_center_min", 1020)   # 17:00
    evening_std    = cfg.get("peak_evening_std_min", 90)
    morning_share  = cfg.get("peak_morning_share", 0.35)
    evening_share  = cfg.get("peak_evening_share", 0.35)

    roll = rng.random()
    if roll < morning_share:
        t = rng.normal(morning_center, morning_std)
    elif roll < morning_share + evening_share:
        t = rng.normal(evening_center, evening_std)
    else:
        t = rng.uniform(300, 1320)   # 5:00 – 22:00 off-peak

    return float(np.clip(t, 0, 1439))
