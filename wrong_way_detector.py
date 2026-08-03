"""Wrong-way driving detection from tracked vehicle positions.

This module is pure geometry and bookkeeping. It consumes structured track
data -- a track id and a pixel position, once per vehicle per frame -- and
never touches raw video, the detector, or the tracker (CLAUDE.md
principle 5). That is what makes it testable against synthetic
trajectories with no GPU and no footage.

The detector is self-calibrating (principle 1): it has no map and no GPS.
It learns the normal direction of travel for each region of the image by
watching the traffic that actually passes through it, and only starts
judging vehicles once that learned baseline is reliable.
"""

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Any, Deque, Dict, Optional, Tuple

ALERT_TYPE = "wrong_way"

Position = Tuple[float, float]
Zone = Tuple[int, int]


@dataclass
class DetectorConfig:
    """Tunable thresholds for :class:`WrongWayDetector`.

    Per CLAUDE.md principle 3 a false positive is worse than a false
    negative, so when tuning any of these, move toward the quieter side.
    These values are validated against synthetic trajectories only -- they
    are starting points for tuning on real footage, not final values.
    """

    zone_size: int = 120                    # grid cell size in pixels, for the per-zone baseline
    heading_window: int = 8                 # frames to look back when computing the heading vector
    baseline_alpha: float = 0.05            # EMA update rate of the learned baseline
    baseline_min_samples: int = 30          # votes a zone needs before it may be trusted
    opposite_cos_threshold: float = -0.3    # below this, a heading counts as opposed
    violation_frames_required: int = 10     # consecutive opposed frames before alerting
    min_speed_px: float = 1.5               # below this speed the heading is noise; ignore it

    # How much a zone's votes must agree before it is trusted, 0..1. The
    # EMA of unit vectors keeps length ~1 while traffic is consistent and
    # collapses toward 0 when it is not, so this one number rejects
    # junctions, roundabouts and parking areas -- places with no single
    # correct direction, and the likeliest source of false accusations.
    baseline_min_coherence: float = 0.6
    # Cap on stored positions per track, so long-lived tracks do not grow
    # without limit.
    max_history: int = 60


@dataclass
class _ZoneStats:
    """The running baseline for one zone of the image grid."""

    dir_x: float = 0.0
    dir_y: float = 0.0
    samples: int = 0


class DirectionBaseline:
    """Learns the dominant direction of travel per image zone.

    The image is divided into a fixed grid of square zones. Every moving
    vehicle contributes its unit heading vector as one vote for the zone
    it currently occupies, and each zone keeps an exponential moving
    average of the votes it has seen.

    A zone is trusted only once it has seen enough vehicles *and* those
    vehicles agree with one another. Averaging unit vectors gives both
    numbers: the count, and the average's length as a measure of
    agreement.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._zones: Dict[Zone, _ZoneStats] = {}

    def zone_of(self, position: Position) -> Zone:
        """Return the grid cell containing a pixel position."""
        size = self.config.zone_size
        return (int(position[0] // size), int(position[1] // size))

    def state(self, position: Position) -> Tuple[bool, Optional[Position]]:
        """Read a zone's current baseline without modifying it.

        Returns ``(trusted, direction)``, where direction is a unit vector,
        and is None whenever the zone is not trusted.

        Callers must read this *before* feeding in the heading of the
        vehicle they are about to judge, so that a vehicle can never
        contribute to the baseline it is measured against. See
        md/ARCHITECTURE.md, diagram 3.
        """
        stats = self._zones.get(self.zone_of(position))
        if stats is None or stats.samples < self.config.baseline_min_samples:
            return False, None

        length = hypot(stats.dir_x, stats.dir_y)
        if length < self.config.baseline_min_coherence:
            return False, None

        return True, (stats.dir_x / length, stats.dir_y / length)

    def update(self, position: Position, heading: Position) -> None:
        """Record one unit heading vector as a vote for its zone."""
        zone = self.zone_of(position)
        stats = self._zones.get(zone)
        if stats is None:
            self._zones[zone] = _ZoneStats(heading[0], heading[1], 1)
            return

        alpha = self.config.baseline_alpha
        stats.dir_x = (1.0 - alpha) * stats.dir_x + alpha * heading[0]
        stats.dir_y = (1.0 - alpha) * stats.dir_y + alpha * heading[1]
        stats.samples += 1


@dataclass
class TrackState:
    """Per-vehicle position history and violation bookkeeping."""

    positions: Deque[Position]
    # Consecutive frames this vehicle has been opposed to its zone baseline.
    violation_streak: int = 0
    # Sum of the cosine similarities over the current streak, so an alert
    # can report how opposed the vehicle was on average rather than on the
    # single frame that happened to trip the counter.
    cos_sum: float = 0.0
    # One alert per track, ever.
    already_reported: bool = False


class WrongWayDetector:
    """Flags vehicles travelling against the learned flow of their zone."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config if config is not None else DetectorConfig()
        self.baseline = DirectionBaseline(self.config)
        self.tracks: Dict[int, TrackState] = {}

    def update(
        self, track_id: int, position: Position
    ) -> Optional[Dict[str, Any]]:
        """Feed one vehicle's position for one frame.

        Call once per tracked vehicle per frame. Returns an alert payload
        the first time this vehicle is confirmed to be driving the wrong
        way, and None on every other call -- which is the overwhelming
        majority of calls.
        """
        track = self.tracks.get(track_id)
        if track is None:
            track = TrackState(positions=deque(maxlen=self.config.max_history))
            self.tracks[track_id] = track
        track.positions.append(position)

        heading = self._heading(track)
        if heading is None:
            return None
        unit_x, unit_y, speed = heading
        if speed < self.config.min_speed_px:
            return None

        # Read the baseline first, then vote. A wrong-way vehicle must not
        # be allowed to drag the baseline toward itself in the same frame
        # it is judged in -- in an empty zone it would otherwise define the
        # baseline single-handedly and prove itself innocent.
        trusted, baseline_dir = self.baseline.state(position)
        # A vehicle already mid-violation stops voting entirely. Without
        # this, a sustained wrong-way run rewrites the very baseline it is
        # being measured against and talks its way out of the alert well
        # before the streak completes.
        if track.violation_streak == 0:
            self.baseline.update(position, (unit_x, unit_y))
        if not trusted or baseline_dir is None:
            return None

        cosine = unit_x * baseline_dir[0] + unit_y * baseline_dir[1]
        if cosine >= self.config.opposite_cos_threshold:
            track.violation_streak = 0
            track.cos_sum = 0.0
            return None

        track.violation_streak += 1
        track.cos_sum += cosine
        if track.violation_streak < self.config.violation_frames_required:
            return None
        if track.already_reported:
            return None

        track.already_reported = True
        return {
            "type": ALERT_TYPE,
            "track_id": track_id,
            "confidence": self._confidence(track.cos_sum / track.violation_streak),
            "position": (float(position[0]), float(position[1])),
        }

    def _heading(self, track: TrackState) -> Optional[Tuple[float, float, float]]:
        """Return ``(unit_x, unit_y, speed_px_per_frame)`` for a track.

        The heading spans ``heading_window`` frames rather than the last
        two: bounding-box centres jitter frame to frame, and a longer
        baseline turns that jitter into a small angular error instead of a
        random direction.

        Returns None when the track has too little history, or has not
        moved at all across the window.
        """
        window = self.config.heading_window
        if len(track.positions) <= window:
            return None

        start_x, start_y = track.positions[-1 - window]
        end_x, end_y = track.positions[-1]
        delta_x, delta_y = end_x - start_x, end_y - start_y
        distance = hypot(delta_x, delta_y)
        if distance == 0.0:
            return None

        return delta_x / distance, delta_y / distance, distance / window

    def _confidence(self, mean_cosine: float) -> float:
        """Score the evidence in 0.0-1.0 from how opposed the vehicle was.

        A vehicle that only just crossed the threshold scores near 0.5; a
        clean head-on reversal approaches 1.0. Per CLAUDE.md principle 4
        this is evidence, not a verdict -- the consumer decides what a 0.55
        is worth versus a 0.95.
        """
        threshold = self.config.opposite_cos_threshold
        span = threshold + 1.0
        directness = (threshold - mean_cosine) / span if span > 0.0 else 1.0
        directness = max(0.0, min(1.0, directness))
        return round(0.5 + 0.5 * directness, 3)
