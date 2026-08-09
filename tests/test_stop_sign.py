"""Synthetic, GPU-free tests for :mod:`road_crime.stop_sign_detector`.

Run from the repository root, either way:

    python -m tests.test_stop_sign
    pytest tests/test_stop_sign.py
"""

import json
import math
import os
import sys
from typing import Dict, Iterable, List, Tuple

# Running this file directly puts tests/ on the path rather than the
# repository root, so the package would not be importable. Three lines
# here keep `python tests/test_stop_sign.py` working, which matters: the
# suite is meant to run on any laptop with nothing installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from road_crime.stop_sign_detector import (  # noqa: E402
    DetectorConfig,
    StopSignConfig,
    StopSignDetector,
    StopSignTracker,
    StopZone,
)

Alert = Dict[str, object]
Position = Tuple[float, float]

STOP_SIGN_ID = 42
STOP_SIGN_BBOX = (10.0, 10.0, 50.0, 50.0)
STOP_SIGNS = {STOP_SIGN_ID: STOP_SIGN_BBOX}


def _feed(
    detector: StopSignDetector,
    track_id: int,
    positions: Iterable[Position],
    signs: Dict[int, Tuple[float, float, float, float]] = STOP_SIGNS,
) -> List[Alert]:
    alerts: List[Alert] = []
    for position in positions:
        alert = detector.update(track_id, position, signs)
        if alert is not None:
            alerts.append(alert)
    return alerts


def _line(start_x: float, step_x: float, count: int, y: float = 70.0):
    return [(start_x + step_x * index, y) for index in range(count)]


def _fast_config(**overrides) -> StopSignConfig:
    values = {
        "heading_window": 3,
        "violation_frames_required": 5,
        "max_stop_speed_px": 1.0,
    }
    values.update(overrides)
    return StopSignConfig(**values)


# --- The original six prototype scenarios ---------------------------------


def test_vehicle_that_stops_inside_zone_is_not_flagged() -> None:
    detector = StopSignDetector(_fast_config())
    positions = [(20.0, 70.0), (24.0, 70.0), (28.0, 70.0), (32.0, 70.0)]
    positions += [(32.0, 70.0)] * 6
    positions += [(80.0, 70.0)]
    assert _feed(detector, 1, positions) == []


def test_vehicle_maintaining_speed_is_flagged_on_exit() -> None:
    detector = StopSignDetector(_fast_config())
    alerts = _feed(detector, 2, _line(30.0, 4.0, 12))
    assert len(alerts) == 1
    assert alerts[0]["type"] == "stop_sign_violation"
    assert alerts[0]["track_id"] == 2
    assert alerts[0]["confidence"] > 0.5


def test_vehicle_outside_zone_is_never_evaluated() -> None:
    detector = StopSignDetector(_fast_config())
    assert _feed(detector, 3, _line(30.0, 4.0, 15, y=200.0)) == []
    assert detector.tracks[3].evaluations == {}


def test_rolling_stop_above_threshold_is_flagged() -> None:
    detector = StopSignDetector(_fast_config())
    alerts = _feed(detector, 4, _line(30.0, 2.0, 22))
    assert len(alerts) == 1
    assert alerts[0]["evidence"]["min_speed_px"] > 1.0


def test_reentering_an_evaluated_zone_does_not_duplicate_alert() -> None:
    detector = StopSignDetector(_fast_config())
    assert len(_feed(detector, 5, _line(30.0, 4.0, 12))) == 1
    assert _feed(detector, 5, _line(30.0, 4.0, 12)) == []


def test_alert_has_complete_auditable_evidence() -> None:
    detector = StopSignDetector(_fast_config())
    alert = _feed(detector, 6, _line(30.0, 4.0, 12))[0]
    evidence = alert["evidence"]
    assert set(evidence) == {
        "stop_sign_id",
        "min_speed_px",
        "max_allowed_stop_speed",
        "frames_in_zone",
        "stop_zone_bbox",
    }
    assert evidence["stop_sign_id"] == STOP_SIGN_ID
    assert evidence["frames_in_zone"] >= 5


# --- False-positive and state-machine regressions --------------------------


def test_default_config_stationary_vehicle_never_alerts() -> None:
    detector = StopSignDetector()
    positions = [(30.0, 70.0)] * 20 + [(80.0, 70.0)]
    assert _feed(detector, 7, positions) == []


def test_default_required_length_stop_never_uses_pre_zone_speed() -> None:
    """A full default-length stop cannot inherit motion from before entry."""

    detector = StopSignDetector()
    outside = [(-90.0 + 5.0 * index, 70.0) for index in range(9)]
    stopped_inside = [(30.0, 70.0)] * detector.config.violation_frames_required
    alerts = _feed(detector, 70, outside + stopped_inside + [(80.0, 70.0)])

    assert alerts == []
    state = detector.tracks[70].evaluations[STOP_SIGN_ID]
    assert state.min_speed_px == 0.0
    assert state.evaluated


def test_late_stop_clears_earlier_fast_motion() -> None:
    detector = StopSignDetector(_fast_config())
    positions = _line(20.0, 4.0, 6)
    positions += [positions[-1]] * 5
    positions += [(80.0, 70.0)]
    assert _feed(detector, 8, positions) == []


def test_alert_is_emitted_only_after_vehicle_exits_zone() -> None:
    detector = StopSignDetector(_fast_config())
    inside_positions = _line(30.0, 4.0, 11)  # x=30..70, inclusive zone edge
    assert _feed(detector, 9, inside_positions) == []
    alert = detector.update(9, (74.0, 70.0), STOP_SIGNS)
    assert alert is not None


def test_insufficient_history_never_becomes_infinite_speed_evidence() -> None:
    detector = StopSignDetector(
        _fast_config(heading_window=8, violation_frames_required=2)
    )
    alerts = _feed(detector, 10, [(30.0, 70.0), (34.0, 70.0), (80.0, 70.0)])
    assert alerts == []


def test_alert_and_evidence_are_finite_strict_json() -> None:
    detector = StopSignDetector(_fast_config())
    alert = _feed(detector, 11, _line(30.0, 4.0, 12))[0]
    json.dumps(alert, allow_nan=False)

    def assert_finite(value) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for nested in value.values():
                assert_finite(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                assert_finite(nested)

    assert_finite(alert)


def test_short_visit_is_locked_without_alert_or_later_rejudgment() -> None:
    detector = StopSignDetector(_fast_config(violation_frames_required=5))
    assert _feed(detector, 12, [(30.0, 70.0), (34.0, 70.0), (80.0, 70.0)]) == []
    assert _feed(detector, 12, _line(30.0, 4.0, 12)) == []


# --- Stop-sign id stability -------------------------------------------------


def test_stop_sign_tracker_handles_jitter_reordering_and_dropout() -> None:
    config = _fast_config(
        tracker_iou_threshold=0.2,
        tracker_max_center_distance_px=20.0,
        tracker_max_missed_frames=2,
    )
    tracker = StopSignTracker(config)
    left = (10.0, 10.0, 30.0, 30.0)
    right = (100.0, 10.0, 120.0, 30.0)
    first = tracker.update([left, right])
    assert len(first) == 2
    left_id = min(first, key=lambda sign_id: first[sign_id][0])
    right_id = max(first, key=lambda sign_id: first[sign_id][0])

    # Detector order reverses and each box jitters, but ids do not swap.
    second = tracker.update(
        [(102.0, 9.0, 122.0, 29.0), (8.0, 11.0, 28.0, 31.0)]
    )
    assert second[left_id][0] < 50.0
    assert second[right_id][0] > 50.0

    # Both last-known detections survive the configured short dropout.
    assert set(tracker.update([])) == {left_id, right_id}
    assert set(tracker.update([])) == {left_id, right_id}

    # Reappearance still binds to the original id.
    recovered = tracker.update([(9.0, 10.0, 29.0, 30.0)])
    assert left_id in recovered
    assert recovered[left_id][0] < 50.0


def test_stop_zone_heuristic_is_below_sign_and_configurable() -> None:
    config = _fast_config(
        zone_width_scale=3.0,
        zone_below_scale=2.0,
        zone_min_width_px=1.0,
        zone_min_height_px=1.0,
    )
    zone = StopZone.from_sign_bbox(STOP_SIGN_BBOX, config)
    assert zone.as_bbox() == (-30.0, 50.0, 90.0, 130.0)


def test_detector_config_is_a_compatibility_alias() -> None:
    assert DetectorConfig is StopSignConfig


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
        except Exception:  # noqa: BLE001 - standalone runner reports all cases
            failures += 1
            print("FAIL " + name)
            traceback.print_exc()
        else:
            print("ok   " + name)
    print("\n{0} passed, {1} failed".format(len(cases) - failures, failures))
    raise SystemExit(1 if failures else 0)
