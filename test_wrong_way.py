"""Synthetic-trajectory tests for the wrong-way module.

No camera, no GPU, no RF-DETR (CLAUDE.md principle 6). Every test builds
fake tracks frame by frame and asserts on the returned alert dict, so the
whole file runs in milliseconds on any machine.

Runs under pytest, or standalone with ``python test_wrong_way.py``.

Geometry note: the default ``zone_size`` is 120, so every trajectory below
is kept inside the single zone (0, 0) -- x and y in [0, 120) -- to keep
the baseline learning in one place and the reasoning easy to follow.
"""

from collections import deque
from typing import Dict, List, Sequence, Tuple

from wrong_way_detector import (
    DetectorConfig,
    DirectionBaseline,
    TrackState,
    WrongWayDetector,
)

Alert = Dict[str, object]
Vehicle = Tuple[int, Tuple[float, float], Tuple[float, float]]

# A lane's worth of straight-line travel that stays inside zone (0, 0):
# 38 frames at 3 px/frame runs x from 5 to 116.
LANE_FRAMES = 38
FORWARD: Vehicle = (0, (5.0, 60.0), (3.0, 0.0))
BACKWARD: Vehicle = (0, (116.0, 60.0), (-3.0, 0.0))


def _drive(
    detector: WrongWayDetector,
    track_id: int,
    start: Tuple[float, float],
    step: Tuple[float, float],
    frames: int,
) -> List[Alert]:
    """Feed one straight-line trajectory and collect every alert it produces."""
    alerts: List[Alert] = []
    x, y = start
    for _ in range(frames):
        alert = detector.update(track_id, (x, y))
        if alert is not None:
            alerts.append(alert)
        x += step[0]
        y += step[1]
    return alerts


def _drive_together(
    detector: WrongWayDetector, vehicles: Sequence[Vehicle], frames: int
) -> List[Alert]:
    """Advance several vehicles in lockstep, one frame at a time."""
    alerts: List[Alert] = []
    live = [(track_id, list(start), step) for track_id, start, step in vehicles]
    for _ in range(frames):
        for track_id, position, step in live:
            alert = detector.update(track_id, (position[0], position[1]))
            if alert is not None:
                alerts.append(alert)
            position[0] += step[0]
            position[1] += step[1]
    return alerts


def _trained_detector(vehicles: int = 2) -> WrongWayDetector:
    """A detector whose zone (0, 0) baseline is trusted and points +x.

    Each vehicle contributes ``LANE_FRAMES - heading_window`` = 30 votes,
    so two of them clear ``baseline_min_samples`` with margin.
    """
    detector = WrongWayDetector()
    for index in range(vehicles):
        _drive(detector, 100 + index, FORWARD[1], FORWARD[2], LANE_FRAMES)
    trusted, direction = detector.baseline.state((60.0, 60.0))
    assert trusted, "fixture failed to establish a trusted baseline"
    assert direction is not None and direction[0] > 0.99
    return detector


# --- Required cases from CLAUDE.md -----------------------------------------


def test_normal_traffic_is_never_flagged() -> None:
    """Vehicles going with the flow produce nothing, however many there are."""
    detector = WrongWayDetector()
    alerts = []
    for index in range(6):
        alerts += _drive(detector, index, FORWARD[1], FORWARD[2], LANE_FRAMES)
    assert alerts == []


def test_wrong_way_vehicle_is_flagged_once_baseline_is_trusted() -> None:
    """The core positive case, plus the shape of the alert payload."""
    detector = _trained_detector()
    alerts = _drive(detector, 7, BACKWARD[1], BACKWARD[2], LANE_FRAMES)

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["type"] == "wrong_way"
    assert alert["track_id"] == 7
    assert isinstance(alert["confidence"], float)
    assert 0.0 <= alert["confidence"] <= 1.0
    # A clean head-on reversal is the strongest evidence the scale allows.
    assert alert["confidence"] > 0.9
    position = alert["position"]
    assert isinstance(position, tuple) and len(position) == 2


def test_no_false_alarm_during_warm_up() -> None:
    """A wrong-way vehicle seen before any baseline exists is not accused.

    It is the only evidence available, so it defines the zone's normal
    direction rather than violating it. Staying silent here is the whole
    point of principle 3.
    """
    detector = WrongWayDetector()
    alerts = _drive(detector, 1, BACKWARD[1], BACKWARD[2], LANE_FRAMES)
    assert alerts == []


def test_no_duplicate_alerts_for_one_track() -> None:
    """A track is reported once, no matter how long it keeps offending."""
    detector = _trained_detector()
    alerts = _drive(detector, 7, BACKWARD[1], BACKWARD[2], LANE_FRAMES)
    # Keep the same track offending for another full lane.
    alerts += _drive(detector, 7, BACKWARD[1], BACKWARD[2], LANE_FRAMES)
    assert len(alerts) == 1


# --- Supporting cases -------------------------------------------------------


def test_streak_must_reach_the_configured_length() -> None:
    """One frame short of the requirement stays silent; the next frame fires.

    Principle 2 in its most literal form -- this is the boundary the whole
    state machine exists to enforce.
    """
    config = DetectorConfig()
    # First heading lands on frame heading_window + 1, and the streak needs
    # violation_frames_required frames after that.
    firing_frame = config.heading_window + config.violation_frames_required

    quiet = _drive(
        _trained_detector(), 7, BACKWARD[1], BACKWARD[2], firing_frame - 1
    )
    assert quiet == []

    firing = _drive(_trained_detector(), 7, BACKWARD[1], BACKWARD[2], firing_frame)
    assert len(firing) == 1


def test_slow_and_stationary_vehicles_are_ignored() -> None:
    """Below min_speed_px a heading is noise, so it must not be judged."""
    detector = _trained_detector()
    assert _drive(detector, 7, (116.0, 60.0), (-0.5, 0.0), LANE_FRAMES) == []
    assert _drive(detector, 8, (60.0, 60.0), (0.0, 0.0), LANE_FRAMES) == []


def test_zone_with_two_way_traffic_is_never_trusted() -> None:
    """Bidirectional traffic in one zone has no single correct direction.

    Without the coherence gate this is where false accusations would come
    from: whichever direction happened to arrive first would become the
    law, and everyone else would be a violator.
    """
    detector = WrongWayDetector()
    alerts = _drive_together(
        detector,
        [(1, FORWARD[1], FORWARD[2]), (2, BACKWARD[1], BACKWARD[2])],
        LANE_FRAMES,
    )
    assert alerts == []
    trusted, _ = detector.baseline.state((60.0, 60.0))
    assert not trusted


def test_offender_stops_voting_into_the_baseline() -> None:
    """A vehicle mid-violation must not rewrite the baseline it is judged by.

    Guards the ordering encoded in md/ARCHITECTURE.md diagram 3: the
    baseline stays pointing +x throughout a sustained wrong-way run.
    """
    detector = _trained_detector()
    _drive(detector, 7, BACKWARD[1], BACKWARD[2], LANE_FRAMES)
    trusted, direction = detector.baseline.state((60.0, 60.0))
    assert trusted
    assert direction is not None and direction[0] > 0.8


def test_heading_needs_history_before_it_reports_anything() -> None:
    """_heading is the shared primitive the stop-sign module will reuse."""
    detector = WrongWayDetector()
    track = TrackState(positions=deque(maxlen=60))
    assert detector._heading(track) is None
    for index in range(detector.config.heading_window + 1):
        track.positions.append((float(index * 3), 0.0))
    heading = detector._heading(track)
    assert heading is not None
    unit_x, unit_y, speed = heading
    assert round(unit_x, 6) == 1.0
    assert round(unit_y, 6) == 0.0
    assert round(speed, 6) == 3.0


def test_empty_zone_is_not_trusted() -> None:
    """A zone nobody has driven through yet reports no direction at all."""
    baseline = DirectionBaseline(DetectorConfig())
    trusted, direction = baseline.state((60.0, 60.0))
    assert not trusted
    assert direction is None


if __name__ == "__main__":
    import traceback

    cases = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    failures = 0
    for name, function in cases:
        try:
            function()
        except Exception:  # noqa: BLE001 - standalone runner, report and continue
            failures += 1
            print("FAIL " + name)
            traceback.print_exc()
        else:
            print("ok   " + name)
    print("\n{0} passed, {1} failed".format(len(cases) - failures, failures))
    raise SystemExit(1 if failures else 0)
