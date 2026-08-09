"""Synthetic-trajectory tests for the wrong-way module.

No camera, no GPU, no RF-DETR (CLAUDE.md principle 6). Every test builds
fake tracks frame by frame and asserts on the returned alert dict, so the
whole file runs in milliseconds on any machine.

Run from the repository root, either way:

    python -m tests.test_wrong_way
    pytest tests/test_wrong_way.py

Geometry note: the default ``zone_size`` is 120, so every trajectory below
is kept inside the single zone (0, 0) -- x and y in [0, 120) -- to keep
the baseline learning in one place and the reasoning easy to follow.
"""

import os
import sys
from collections import deque
from typing import Dict, List, Sequence, Tuple

# Running this file directly puts tests/ on the path rather than the
# repository root, so the package would not be importable. Three lines
# here keep `python tests/test_wrong_way.py` working, which matters: the
# suite is meant to run on any laptop with nothing installed.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

FIXTURES = os.path.join(REPO_ROOT, "fixtures")


def _fixture(name: str) -> str:
    """Locate a recorded run, and refuse to pretend it is optional.

    These used to skip silently when the file was missing. That is the
    failure mode this project has been bitten by twice: a check that
    degrades quietly reports success while testing nothing. The fixtures
    are committed, so absence means the layout moved under us and the
    suite should say so.
    """
    path = os.path.join(FIXTURES, name)
    assert os.path.isfile(path), (
        "Missing fixture {0}.\nLooked in {1}. Fixtures are committed to the "
        "repository; if this fails the path resolution is wrong, not the "
        "data.".format(name, FIXTURES)
    )
    return path

from road_crime.replay import load  # noqa: E402
from road_crime.wrong_way_detector import (  # noqa: E402
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
        detector._test_frame = getattr(detector, "_test_frame", 0) + 1
        alert = detector.update(track_id, (x, y), frame=detector._test_frame)
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
        detector._test_frame = getattr(detector, "_test_frame", 0) + 1
        for track_id, position, step in live:
            alert = detector.update(
                track_id, (position[0], position[1]), frame=detector._test_frame
            )
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


def test_alert_carries_auditable_evidence() -> None:
    """Every alert must be reconstructable after the fact (principle 4).

    Without these fields an alert is an unexplained accusation, and a
    false positive is impossible to diagnose from the payload alone.
    """
    detector = _trained_detector()
    alert = _drive(detector, 7, BACKWARD[1], BACKWARD[2], LANE_FRAMES)[0]
    evidence = alert["evidence"]

    assert set(evidence) == {
        "heading",
        "baseline",
        "zone",
        "zone_first_seen",
        "mean_cosine",
        "streak",
        "speed_px",
        "peers",
        "peer_cosine",
        "scale_px",
        "required_speed_px",
    }
    # Travelling -x against a +x baseline: a clean reversal.
    assert evidence["heading"][0] < -0.99
    assert evidence["baseline"][0] > 0.99
    assert evidence["mean_cosine"] < -0.99
    assert evidence["streak"] == DetectorConfig().violation_frames_required
    assert evidence["speed_px"] > 0.0
    # This trajectory never leaves zone (0, 0), so the vehicle was judged
    # by the baseline of the zone it started in.
    assert evidence["zone"] == evidence["zone_first_seen"]


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


def test_vehicle_being_followed_is_not_judged() -> None:
    """The car ahead in your own lane must never be accused.

    Reproduces the false positive found on real footage. Following a
    vehicle at a near-constant distance leaves it almost stationary in the
    image -- a couple of pixels per frame, less than the jitter of its own
    bounding box. Read as pixels that looks like motion; read against its
    width it is plainly noise.

    This is the most common object in any dashcam clip, so getting it
    wrong means firing constantly in production.
    """
    detector = _trained_detector()
    # 90 px wide, drifting 2 px/frame against the +x baseline: 2.2% of its
    # own width, well inside detector jitter.
    alerts: List[Alert] = []
    x = 116.0
    for frame in range(LANE_FRAMES):
        alert = detector.update(9, (x, 60.0), frame=1000 + frame, scale=90.0)
        if alert is not None:
            alerts.append(alert)
        x -= 2.0
    assert alerts == []


def test_distant_vehicle_moving_the_same_pixels_is_judged() -> None:
    """The other half: the same pixel speed on a small vehicle is real motion.

    Guards against 'fixing' the previous case by raising an absolute
    threshold, which would go blind to genuinely fast traffic far down the
    road. 2 px/frame is 10% of a 20 px wide vehicle's width -- motion, not
    jitter.
    """
    detector = _trained_detector()
    alerts: List[Alert] = []
    x = 116.0
    for frame in range(LANE_FRAMES):
        alert = detector.update(9, (x, 60.0), frame=1000 + frame, scale=20.0)
        if alert is not None:
            alerts.append(alert)
        x -= 2.0
    assert len(alerts) == 1
    assert alerts[0]["evidence"]["scale_px"] == 20.0


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


def test_traffic_that_reverses_together_is_not_accused() -> None:
    """When the whole flow reverses, nobody is driving the wrong way.

    Reproduces the false positives found on real dashcam footage: the
    scene's apparent flow reversed (ego motion), every vehicle present was
    moving the new way, and the module accused them of contradicting a
    baseline learned minutes earlier.

    The failure was self-reinforcing. Each vehicle that disagreed entered a
    violation streak, and a vehicle mid-violation stops voting -- so the
    baseline could never learn the new reality, and the streaks ran to
    completion. The rule that makes a lone offender detectable is exactly
    what makes a reversed flow indefensible.

    The distinction is not vehicle-versus-history but vehicle-versus-peers:
    a driver is only a violator if the traffic around them at that moment
    disagrees too.
    """
    detector = _trained_detector()
    alerts = _drive_together(
        detector,
        [
            (20, (116.0, 55.0), (-3.0, 0.0)),
            (21, (116.0, 60.0), (-3.0, 0.0)),
            (22, (116.0, 65.0), (-3.0, 0.0)),
        ],
        LANE_FRAMES,
    )
    assert alerts == [], "accused {0} vehicle(s) that all moved together".format(
        len(alerts)
    )


def test_lone_offender_is_still_caught_among_normal_traffic() -> None:
    """The other half of the peer rule: disagreeing with peers is a violation.

    Guards against 'fixing' the false positive by simply going quiet. One
    vehicle drives against three that are travelling normally, all in the
    same place at the same time, and it must still be reported.
    """
    detector = _trained_detector()
    alerts = _drive_together(
        detector,
        [
            (30, (5.0, 55.0), (3.0, 0.0)),
            (31, (5.0, 60.0), (3.0, 0.0)),
            (32, (5.0, 65.0), (3.0, 0.0)),
            (33, (116.0, 70.0), (-3.0, 0.0)),
        ],
        LANE_FRAMES,
    )
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 33


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


def test_real_footage_flow_reversal_regression() -> None:
    """Replays the dashcam clip that produced the original false positives.

    Perception output only -- no GPU, no model, no video. `fixtures/` holds
    what the modules actually consume, so this case stays reproducible for
    as long as the repo exists.

    Tracks 7 and 11 were accused while every vehicle around them moved the
    same way; the peer rule now clears both. Track 2 still fires and is
    *not* asserted away here: its heading was 1.79 px/frame, barely over
    `min_speed_px`, which is a threshold too permissive for a 2040 px wide
    frame rather than a flaw in the decision logic. Tuning it belongs with
    the evaluation set, where it can be measured across many clips instead
    of fitted to this one.
    """
    fixture = _fixture("flow_reversal_false_positive.jsonl")

    detector = WrongWayDetector()
    flagged = set()
    for frame_index, tracks in load(fixture):
        for track_id, x, y, width in tracks:
            alert = detector.update(
                track_id, (x, y), frame=frame_index, scale=width
            )
            if alert is not None:
                flagged.add(track_id)

    assert 7 not in flagged, "track 7 moved with the flow and must not be accused"
    assert 11 not in flagged, "track 11 moved with the flow and must not be accused"


def test_real_footage_divided_highway_stays_silent() -> None:
    """A full clip of ordinary motorway driving must produce nothing.

    825 frames, 50 tracks: the vehicle we are following, traffic ahead in
    our own lane, and an oncoming carriageway across the barrier. Every
    one of them is driving legally, so silence is the only correct output.

    This is the case that decides whether the module is usable at all.
    Most motorways are divided, and a detector that flags lawful oncoming
    traffic is worthless on all of them.
    """
    fixture = _fixture("divided_highway_clean.jsonl")

    detector = WrongWayDetector()
    alerts = []
    for frame_index, tracks in load(fixture):
        for track_id, x, y, width in tracks:
            alert = detector.update(
                track_id, (x, y), frame=frame_index, scale=width
            )
            if alert is not None:
                alerts.append(alert)

    assert alerts == [], "accused {0} lawful driver(s)".format(len(alerts))


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
