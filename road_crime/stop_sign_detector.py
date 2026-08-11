"""Stop-sign violation detection from tracked vehicle positions.

The module consumes geometry only: a vehicle track id, its road-contact
position, and stop-sign bounding boxes produced by the shared perception
pipeline.  It never runs the vision model itself.

``StopZone`` deliberately uses a simple v1 heuristic.  The expected stopping
area is a configurable rectangle below a sign, scaled from the sign's apparent
size and clamped to configurable minimum dimensions.  This is an image-space
assumption, not lane geometry; real-footage tuning and sign-to-lane association
are still required before production use.

The decision is conservative: a vehicle is judged exactly once, when it exits
a zone.  It can only be reported after enough consecutive frames in the zone
and at least one finite speed sample.  Any complete stop observed later in the
visit therefore clears an earlier period of faster motion.
"""

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

ALERT_TYPE = "stop_sign_violation"

Position = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class StopSignConfig:
    """All tunable geometry, speed, and sign-tracking thresholds."""

    zone_below_scale: float = 1.5
    zone_width_scale: float = 2.0
    zone_min_width_px: float = 60.0
    zone_min_height_px: float = 60.0
    max_stop_speed_px: float = 1.0
    heading_window: int = 8
    violation_frames_required: int = 8
    max_history: int = 60

    # Kept so configs written against the first prototype still construct.
    # Stop-sign logic intentionally records speeds below this value, including
    # zero; filtering them would turn a real stop into a false violation.
    min_speed_px: float = 0.5

    tracker_iou_threshold: float = 0.25
    tracker_max_center_distance_px: float = 40.0
    tracker_max_missed_frames: int = 2

    # Restricts judgement to a vertical corridor of image space, intended to
    # cover the ego vehicle's own carriageway. Both bounds are required; with
    # either left unset the filter is off and every vehicle inside a zone is
    # judged, which is the behaviour every earlier config had.
    #
    # Why it exists: a zone wide enough to reach our own carriageway also
    # sweeps in the crossing road. Vehicles there answer to a sign whose back
    # faces this camera, and COCO's "stop sign" class is trained on the red
    # face, not the blank rear -- so their obligation is unreadable, and
    # accusing them would be a guess (principles 3 and 4).
    #
    # In pixels rather than a frame fraction, so the module stays pure
    # geometry with no notion of a frame; `pipeline.run` converts.
    ego_corridor_center_x: Optional[float] = None
    ego_corridor_half_width_px: Optional[float] = None

    def __post_init__(self) -> None:
        positive_floats = {
            "zone_below_scale": self.zone_below_scale,
            "zone_width_scale": self.zone_width_scale,
            "zone_min_width_px": self.zone_min_width_px,
            "zone_min_height_px": self.zone_min_height_px,
        }
        for name, value in positive_floats.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError("{0} must be finite and greater than zero".format(name))

        if not isfinite(self.max_stop_speed_px) or self.max_stop_speed_px < 0.0:
            raise ValueError("max_stop_speed_px must be finite and non-negative")
        if not isfinite(self.min_speed_px) or self.min_speed_px < 0.0:
            raise ValueError("min_speed_px must be finite and non-negative")
        if self.heading_window < 1:
            raise ValueError("heading_window must be at least one frame")
        if self.violation_frames_required < 1:
            raise ValueError("violation_frames_required must be at least one frame")
        if self.max_history <= self.heading_window:
            raise ValueError("max_history must be greater than heading_window")
        if not 0.0 <= self.tracker_iou_threshold <= 1.0:
            raise ValueError("tracker_iou_threshold must be between zero and one")
        if (
            not isfinite(self.tracker_max_center_distance_px)
            or self.tracker_max_center_distance_px < 0.0
        ):
            raise ValueError(
                "tracker_max_center_distance_px must be finite and non-negative"
            )
        if self.tracker_max_missed_frames < 0:
            raise ValueError("tracker_max_missed_frames must be non-negative")

        if self.ego_corridor_center_x is not None and not isfinite(
            self.ego_corridor_center_x
        ):
            raise ValueError("ego_corridor_center_x must be finite")
        if self.ego_corridor_half_width_px is not None and (
            not isfinite(self.ego_corridor_half_width_px)
            or self.ego_corridor_half_width_px <= 0.0
        ):
            raise ValueError(
                "ego_corridor_half_width_px must be finite and greater than zero"
            )


# The notebook and early integration code imported this name.  It is an alias,
# rather than a subclass, so both names describe exactly the same config type.
DetectorConfig = StopSignConfig


def _bbox(value: Sequence[float]) -> BBox:
    """Return a validated, plain-float bounding box."""

    if len(value) != 4:
        raise ValueError("a bounding box must contain exactly four values")
    result = tuple(float(component) for component in value)
    if not all(isfinite(component) for component in result):
        raise ValueError("bounding-box values must be finite")
    x1, y1, x2, y2 = result
    if x2 <= x1 or y2 <= y1:
        raise ValueError("a bounding box must have positive width and height")
    return x1, y1, x2, y2


def _position(value: Sequence[float]) -> Position:
    if len(value) != 2:
        raise ValueError("a position must contain exactly two values")
    result = float(value[0]), float(value[1])
    if not all(isfinite(component) for component in result):
        raise ValueError("position values must be finite")
    return result


@dataclass(frozen=True)
class StopZone:
    """An image-space rectangle in which a vehicle is expected to stop."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_sign_bbox(
        cls, sign_bbox: BBox, config: StopSignConfig
    ) -> "StopZone":
        sx1, sy1, sx2, sy2 = _bbox(sign_bbox)
        sign_width = sx2 - sx1
        sign_height = sy2 - sy1
        center_x = (sx1 + sx2) / 2.0

        zone_width = max(
            sign_width * config.zone_width_scale, config.zone_min_width_px
        )
        zone_height = max(
            sign_height * config.zone_below_scale, config.zone_min_height_px
        )
        return cls(
            center_x - zone_width / 2.0,
            sy2,
            center_x + zone_width / 2.0,
            sy2 + zone_height,
        )

    def contains(self, position: Position) -> bool:
        px, py = _position(position)
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def as_bbox(self) -> BBox:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class _TrackedSign:
    bbox: BBox
    missed_frames: int = 0


class StopSignTracker:
    """Assign stable local ids to stop-sign detections.

    RF-DETR detections do not carry persistent ids.  Matching by overlap (or,
    for a larger one-frame jitter, by centre distance) makes ids independent
    of detector output order.  A last-known box is retained for a short,
    configurable dropout so downstream zone state does not disappear after a
    single missed detection.
    """

    def __init__(self, config: Optional[StopSignConfig] = None) -> None:
        self.config = config if config is not None else StopSignConfig()
        self._tracks: Dict[int, _TrackedSign] = {}
        self._next_id = 1

    def update(self, detections: Sequence[BBox]) -> Dict[int, BBox]:
        boxes = [_bbox(detection) for detection in detections]

        # Build all plausible old/new pairs, then choose the strongest
        # non-conflicting pairs.  The final tie breakers keep tests and logs
        # deterministic without depending on input order.
        candidates = []
        for sign_id, tracked in self._tracks.items():
            for detection_index, detection in enumerate(boxes):
                overlap = self._iou(tracked.bbox, detection)
                distance = self._center_distance(tracked.bbox, detection)
                if (
                    overlap >= self.config.tracker_iou_threshold
                    or distance <= self.config.tracker_max_center_distance_px
                ):
                    candidates.append(
                        (-overlap, distance, sign_id, detection_index)
                    )

        matched_ids = set()
        matched_detections = set()
        for _negative_overlap, _distance, sign_id, detection_index in sorted(
            candidates
        ):
            if sign_id in matched_ids or detection_index in matched_detections:
                continue
            self._tracks[sign_id].bbox = boxes[detection_index]
            self._tracks[sign_id].missed_frames = 0
            matched_ids.add(sign_id)
            matched_detections.add(detection_index)

        for sign_id in list(self._tracks):
            if sign_id in matched_ids:
                continue
            tracked = self._tracks[sign_id]
            tracked.missed_frames += 1
            if tracked.missed_frames > self.config.tracker_max_missed_frames:
                del self._tracks[sign_id]

        for detection_index, detection in enumerate(boxes):
            if detection_index in matched_detections:
                continue
            sign_id = self._next_id
            self._next_id += 1
            self._tracks[sign_id] = _TrackedSign(detection)

        return {
            sign_id: self._tracks[sign_id].bbox for sign_id in sorted(self._tracks)
        }

    @staticmethod
    def _center_distance(first: BBox, second: BBox) -> float:
        first_x = (first[0] + first[2]) / 2.0
        first_y = (first[1] + first[3]) / 2.0
        second_x = (second[0] + second[2]) / 2.0
        second_y = (second[1] + second[3]) / 2.0
        return hypot(first_x - second_x, first_y - second_y)

    @staticmethod
    def _iou(first: BBox, second: BBox) -> float:
        intersection_width = max(
            0.0, min(first[2], second[2]) - max(first[0], second[0])
        )
        intersection_height = max(
            0.0, min(first[3], second[3]) - max(first[1], second[1])
        )
        intersection = intersection_width * intersection_height
        if intersection == 0.0:
            return 0.0
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / (first_area + second_area - intersection)


@dataclass
class _ZoneEvalState:
    """One non-repeatable visit by one vehicle to one sign's zone."""

    min_speed_px: Optional[float] = None
    frames_in_zone: int = 0
    evaluated: bool = False
    last_zone_bbox: Optional[BBox] = None
    inside_positions: Deque[Position] = field(default_factory=deque)
    has_full_window_sample: bool = False


@dataclass
class TrackState:
    positions: Deque[Position]
    evaluations: Dict[int, _ZoneEvalState] = field(default_factory=dict)


class StopSignDetector:
    """Report vehicles that fully traverse a stop zone without stopping."""

    def __init__(self, config: Optional[StopSignConfig] = None) -> None:
        self.config = config if config is not None else StopSignConfig()
        self.tracks: Dict[int, TrackState] = {}

    def update(
        self,
        track_id: int,
        position: Position,
        stop_signs: Dict[int, BBox],
    ) -> Optional[Dict[str, Any]]:
        """Process one vehicle observation and return at most one alert.

        ``stop_signs`` should normally be the output of
        :class:`StopSignTracker`, so its keys remain stable across frames.
        """

        vehicle_id = int(track_id)
        current_position = _position(position)
        track = self.tracks.get(vehicle_id)
        if track is None:
            track = TrackState(positions=deque(maxlen=self.config.max_history))
            self.tracks[vehicle_id] = track
        track.positions.append(current_position)

        # Treated as part of being "inside": a vehicle that leaves the
        # corridor while still within the zone has left the traffic this
        # sign's face governs, so its visit closes there rather than
        # continuing to accumulate speed samples from the crossing road.
        in_corridor = self._in_ego_corridor(current_position)

        alert_to_return: Optional[Dict[str, Any]] = None
        for raw_sign_id, raw_sign_bbox in stop_signs.items():
            sign_id = int(raw_sign_id)
            zone = StopZone.from_sign_bbox(raw_sign_bbox, self.config)
            inside = zone.contains(current_position) and in_corridor
            state = track.evaluations.get(sign_id)

            # Merely seeing a sign must not allocate per-zone evaluation state.
            # A state begins only when this vehicle actually enters that zone.
            if state is None:
                if not inside:
                    continue
                state = _ZoneEvalState()
                track.evaluations[sign_id] = state

            if state.evaluated:
                continue

            if inside:
                state.frames_in_zone += 1
                state.last_zone_bbox = zone.as_bbox()
                state.inside_positions.append(current_position)
                while len(state.inside_positions) > self.config.heading_window + 1:
                    state.inside_positions.popleft()
                speed, is_full_window = self._zone_speed(state)
                if speed is not None and isfinite(speed):
                    if state.min_speed_px is None:
                        state.min_speed_px = speed
                    else:
                        state.min_speed_px = min(state.min_speed_px, speed)
                    if is_full_window:
                        state.has_full_window_sample = True
                continue

            # State existence means the vehicle was previously inside.  The
            # first exit locks the pair regardless of whether it accumulated
            # enough evidence to report, preventing later double-counting.
            state.evaluated = True
            if state.frames_in_zone < self.config.violation_frames_required:
                continue
            if not state.has_full_window_sample:
                continue
            if state.min_speed_px is None or not isfinite(state.min_speed_px):
                continue
            if state.min_speed_px <= self.config.max_stop_speed_px:
                continue
            if state.last_zone_bbox is None:  # Defensive; entry sets this.
                continue

            if alert_to_return is None:
                alert_to_return = self._alert(
                    vehicle_id, sign_id, current_position, state
                )

        return alert_to_return

    def _in_ego_corridor(self, position: Position) -> bool:
        """Whether a position sits in the carriageway we are entitled to judge.

        Unset bounds mean no corridor, so every position qualifies.
        """

        center = self.config.ego_corridor_center_x
        half_width = self.config.ego_corridor_half_width_px
        if center is None or half_width is None:
            return True
        return abs(position[0] - center) <= half_width

    def _zone_speed(
        self, state: _ZoneEvalState
    ) -> Tuple[Optional[float], bool]:
        """Return speed measured exclusively from positions inside one zone.

        Short-window samples let a later stationary observation lower the
        visit's minimum speed immediately.  A violation still requires at
        least one complete ``heading_window`` sample; this keeps a very short
        visit silent instead of turning sparse evidence into an accusation.
        """

        if len(state.inside_positions) < 2:
            return None, False
        intervals = min(
            self.config.heading_window, len(state.inside_positions) - 1
        )
        start_x, start_y = state.inside_positions[-1 - intervals]
        end_x, end_y = state.inside_positions[-1]
        speed = hypot(end_x - start_x, end_y - start_y) / intervals
        return speed, intervals == self.config.heading_window

    def _speed(self, track: TrackState) -> Optional[float]:
        """Return smoothed pixels-per-frame, or ``None`` without history."""

        window = self.config.heading_window
        if len(track.positions) <= window:
            return None
        start_x, start_y = track.positions[-1 - window]
        end_x, end_y = track.positions[-1]
        return hypot(end_x - start_x, end_y - start_y) / window

    def _alert(
        self,
        track_id: int,
        sign_id: int,
        position: Position,
        state: _ZoneEvalState,
    ) -> Dict[str, Any]:
        # The caller proves these are present and finite before reaching here.
        min_speed = float(state.min_speed_px)  # type: ignore[arg-type]
        zone_bbox = state.last_zone_bbox
        assert zone_bbox is not None
        evidence = {
            "stop_sign_id": sign_id,
            "min_speed_px": round(min_speed, 3),
            "max_allowed_stop_speed": float(self.config.max_stop_speed_px),
            "frames_in_zone": state.frames_in_zone,
            "stop_zone_bbox": tuple(round(value, 3) for value in zone_bbox),
        }
        return {
            "type": ALERT_TYPE,
            "track_id": track_id,
            "confidence": self._confidence(min_speed),
            "position": (float(position[0]), float(position[1])),
            "evidence": evidence,
        }

    def _confidence(self, min_speed_px: float) -> float:
        threshold = self.config.max_stop_speed_px
        if min_speed_px <= threshold:
            return 0.0
        scale = max(threshold, 1.0)
        excess_ratio = (min_speed_px - threshold) / (4.0 * scale)
        return round(min(1.0, 0.5 + 0.5 * excess_ratio), 3)
