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

    # Minimum speed for a heading to be believed, as a fraction of the
    # vehicle's own apparent width per frame.
    #
    # An absolute pixel threshold cannot express this. The vehicle you are
    # following sits at a near-constant distance, so it barely moves in the
    # image -- a few pixels per frame, which is smaller than the jitter of
    # its own bounding box. Measured in pixels that looked like motion;
    # measured against its width it is plainly noise. Meanwhile a distant
    # vehicle, only twenty pixels wide, can genuinely be travelling fast
    # while moving those same few pixels.
    #
    # Provisional: 2-3% per frame was jitter on both clips examined, so 5%
    # clears it with margin. Confirm against the evaluation set.
    min_speed_fraction: float = 0.05
    # Fallback for callers that cannot supply a vehicle size. Absolute, and
    # therefore resolution-dependent -- prefer passing `scale`.
    min_speed_px: float = 1.5

    # How much a zone's votes must agree before it is trusted, 0..1. The
    # EMA of unit vectors keeps length ~1 while traffic is consistent and
    # collapses toward 0 when it is not, so this one number rejects
    # junctions, roundabouts and parking areas -- places with no single
    # correct direction, and the likeliest source of false accusations.
    baseline_min_coherence: float = 0.6
    # Cap on stored positions per track, so long-lived tracks do not grow
    # without limit.
    max_history: int = 60

    # --- Peer agreement ---
    # Disagreeing with the learned baseline is not enough to accuse a
    # driver, because the baseline can simply be out of date: on a dashcam
    # the apparent flow reverses whenever the ego vehicle turns, and then
    # every vehicle present contradicts a direction learned a minute ago.
    # A violation additionally requires disagreeing with the traffic
    # travelling alongside the vehicle *at that moment*.
    #
    # How many frames back a peer's heading still counts as "now".
    # This has to be measured in frames, not in samples: a quiet zone
    # collects samples slowly, so a fixed-size sample window there reaches
    # tens of frames into the past and reports history as though it were
    # the present. That is precisely the staleness this rule exists to
    # defeat.
    peer_max_age: int = 12
    # Hard cap on stored headings per zone, so a busy zone cannot grow
    # without limit. Age is what decides relevance; this only bounds memory.
    peer_window: int = 64
    # Distinct other vehicles needed before their agreement counts as
    # evidence. One is enough: per principle 3 a single contemporary moving
    # the same way is reason to stay quiet, and staying quiet is the safe
    # failure.
    peer_min_tracks: int = 1
    # How much those peers must agree with each other to count at all.
    peer_min_coherence: float = 0.6
    # Cosine above which the vehicle counts as agreeing with its peers.
    peer_agree_cos: float = 0.0


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
        # Two views of the same question, at different time scales.
        # `_zones` is what traffic has done here over the long run;
        # `_recent` is what it is doing right now. A baseline can go stale;
        # contemporaries cannot.
        self._recent: Dict[Zone, Deque[Tuple[int, int, float, float]]] = {}

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

    def record(
        self, position: Position, heading: Position, track_id: int, frame: int
    ) -> None:
        """Note what one vehicle is doing right now, for peer comparison.

        Unlike :meth:`update`, this records *every* moving vehicle,
        including one already mid-violation. Its job is to measure what
        traffic is actually doing, suspects included -- excluding them is
        what let a stale baseline stand unchallenged.
        """
        zone = self.zone_of(position)
        window = self._recent.get(zone)
        if window is None:
            window = deque(maxlen=self.config.peer_window)
            self._recent[zone] = window
        window.append((frame, track_id, heading[0], heading[1]))

    def peer_consensus(
        self, position: Position, exclude_track_id: int, frame: int
    ) -> Optional[Position]:
        """Direction this vehicle's contemporaries are travelling, if they agree.

        Searches the vehicle's zone and the eight around it. Zones are small
        relative to the image, so traffic moving alongside a vehicle
        routinely sits in a neighbouring cell -- restricting this to one
        zone would miss most genuine peers.

        The judged vehicle is excluded: it may not vouch for itself.

        Returns None when too few other vehicles have been seen recently, or
        when the ones that have disagree among themselves. Both mean there
        is no consensus to appeal to, and the caller falls back to the
        learned baseline alone.
        """
        zone_x, zone_y = self.zone_of(position)
        oldest = frame - self.config.peer_max_age

        # One vote per vehicle, and only its latest heading. Averaging every
        # sample instead would let a single long-lived track outvote the
        # rest, and would smear its own history across the result: a vehicle
        # turning through 180 degrees over ten frames averages out to a
        # direction nobody ever travelled, and the suspect is then measured
        # against a fiction.
        latest: Dict[int, Tuple[int, float, float]] = {}
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                window = self._recent.get((zone_x + offset_x, zone_y + offset_y))
                if window is None:
                    continue
                for seen_at, track_id, unit_x, unit_y in window:
                    if track_id == exclude_track_id or seen_at < oldest:
                        continue
                    known = latest.get(track_id)
                    if known is None or seen_at > known[0]:
                        latest[track_id] = (seen_at, unit_x, unit_y)

        if len(latest) < self.config.peer_min_tracks:
            return None
        sum_x = sum(entry[1] for entry in latest.values())
        sum_y = sum(entry[2] for entry in latest.values())
        length = hypot(sum_x, sum_y)
        if length / len(latest) < self.config.peer_min_coherence:
            return None
        return (sum_x / length, sum_y / length)

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
    # Zone this track was first seen in. Compared against the zone at alert
    # time, it shows whether the vehicle was judged by the same baseline it
    # helped train, or drifted into a zone taught by different traffic.
    first_zone: Optional[Zone] = None
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
        self._call_counter = 0

    def update(
        self,
        track_id: int,
        position: Position,
        frame: Optional[int] = None,
        scale: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Feed one vehicle's position for one frame.

        Call once per tracked vehicle per frame. Returns an alert payload
        the first time this vehicle is confirmed to be driving the wrong
        way, and None on every other call -- which is the overwhelming
        majority of calls.

        `frame` is the video frame number. Pass it: peer agreement is
        measured in frames, and without a real one the detector falls back
        to counting calls, which drifts as soon as more than one vehicle is
        visible. The parameter is optional only so that a caller feeding a
        single vehicle can ignore it.

        `scale` is the vehicle's apparent size in pixels -- its bounding box
        width. Pass it: it is what separates a vehicle that is genuinely
        crawling from one that merely looks slow because you are following
        it. Without it the detector falls back to an absolute pixel
        threshold, which is resolution-dependent and cannot tell those two
        apart.
        """
        if frame is None:
            self._call_counter += 1
            frame = self._call_counter
        track = self.tracks.get(track_id)
        if track is None:
            track = TrackState(positions=deque(maxlen=self.config.max_history))
            track.first_zone = self.baseline.zone_of(position)
            self.tracks[track_id] = track
        track.positions.append(position)

        heading = self._heading(track)
        if heading is None:
            return None
        unit_x, unit_y, speed = heading
        if scale is not None and scale > 0.0:
            required_speed = self.config.min_speed_fraction * scale
        else:
            required_speed = self.config.min_speed_px
        if speed < required_speed:
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
        # The peer record takes everyone, suspects included -- see `record`.
        self.baseline.record(position, (unit_x, unit_y), track_id, frame)
        if not trusted or baseline_dir is None:
            return None

        cosine = unit_x * baseline_dir[0] + unit_y * baseline_dir[1]
        if cosine >= self.config.opposite_cos_threshold:
            track.violation_streak = 0
            track.cos_sum = 0.0
            return None

        # Contradicting the baseline is necessary but not sufficient. If the
        # traffic alongside this vehicle right now is doing the same thing,
        # the flow has changed and the baseline is what is out of date --
        # accusing the driver would be accusing them of the scene moving.
        peer_dir = self.baseline.peer_consensus(position, track_id, frame)
        peer_cosine: Optional[float] = None
        if peer_dir is not None:
            peer_cosine = unit_x * peer_dir[0] + unit_y * peer_dir[1]
            if peer_cosine > self.config.peer_agree_cos:
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
        mean_cosine = track.cos_sum / track.violation_streak
        return {
            "type": ALERT_TYPE,
            "track_id": track_id,
            "confidence": self._confidence(mean_cosine),
            "position": (float(position[0]), float(position[1])),
            # Everything needed to reconstruct the decision afterwards. An
            # alert nobody can audit is a verdict, which principle 4
            # forbids; it is also impossible to debug, which is how a
            # false positive survives to accuse the next driver.
            "evidence": {
                "heading": (round(unit_x, 4), round(unit_y, 4)),
                "baseline": (round(baseline_dir[0], 4), round(baseline_dir[1], 4)),
                "zone": self.baseline.zone_of(position),
                "zone_first_seen": self.tracks[track_id].first_zone,
                "mean_cosine": round(mean_cosine, 4),
                "streak": track.violation_streak,
                "speed_px": round(speed, 2),
                # The bar this vehicle's speed had to clear, and what set
                # it. A `scale` of None means the alert rests on an absolute
                # pixel threshold, which is weaker evidence.
                "scale_px": None if scale is None else round(scale, 1),
                "required_speed_px": round(required_speed, 2),
                # What the surrounding traffic was doing, and how far this
                # vehicle departed from it. `None` means there were no
                # contemporaries to compare against, so the alert rests on
                # the learned baseline alone -- weaker evidence, and worth
                # seeing in the payload.
                "peers": (
                    None
                    if peer_dir is None
                    else (round(peer_dir[0], 4), round(peer_dir[1], 4))
                ),
                "peer_cosine": None if peer_cosine is None else round(peer_cosine, 4),
            },
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
