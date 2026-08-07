"""Score the wrong-way module over a corpus of track dumps.

Until now "does it work" meant a person watching a video. That is neither
measurable nor repeatable, and it makes threshold tuning guesswork:
`CLAUDE.md` principle 3 demands minimising false positives, and you cannot
minimise what you do not measure.

Two measurements, neither of which needs a labelled dataset:

**False positives, from unlabelled footage.** Ordinary driving contains no
wrong-way driving to any useful approximation, so every alert raised on it
is one that should not have been. Point this at any pile of ordinary
dashcam dumps and the count is the false-positive rate.

**Detection rate, from injected violations.** A real trajectory replayed
backwards is a vehicle travelling against its own lane, at a realistic
speed and in a real scene. Injecting those measures sensitivity without
waiting years for a genuine wrong-way event to be filmed.

    python evaluate.py fixtures/
    python evaluate.py fixtures/ --zone-size 240 --baseline-min-samples 15
    python evaluate.py fixtures/ --json          # for a tuner to parse

The single score is deliberately asymmetric:

    loss = false_positive_weight * false_positives + missed_injections

That weight *is* principle 3, written as a number. Making it explicit
turns "bias toward the safer side" from a slogan into something you can
argue about and change on purpose.
"""

import argparse
import glob
import json
import os
from math import hypot
from typing import Any, Dict, List, Optional, Sequence, Tuple

from replay import Track, load
from wrong_way_detector import DetectorConfig, WrongWayDetector

HEADING_WINDOW_FOR_RANKING = 8
GHOST_ID = 999_000


def _alerts_for(frames: Sequence[Tuple[int, List[Track]]],
                config: DetectorConfig) -> List[Dict[str, Any]]:
    """Replay a clip untouched and collect whatever it reports."""
    detector = WrongWayDetector(config)
    alerts = []
    for frame_index, tracks in frames:
        for track_id, x, y, width in tracks:
            alert = detector.update(track_id, (x, y), frame=frame_index, scale=width)
            if alert is not None:
                alert["frame"] = frame_index
                alerts.append(alert)
    return alerts


def _median_speed_ratio(points: Sequence[Tuple[int, float, float, Optional[float]]]) -> float:
    """Typical speed as a fraction of the vehicle's own width."""
    values = []
    window = HEADING_WINDOW_FOR_RANKING
    for index in range(window, len(points)):
        _, x0, y0, _ = points[index - window]
        _, x1, y1, width = points[index]
        if width:
            values.append((hypot(x1 - x0, y1 - y0) / window) / width)
    if not values:
        return 0.0
    values.sort()
    return values[len(values) // 2]


def _injection_donors(frames: Sequence[Tuple[int, List[Track]]],
                      count: int) -> List[List[Tuple[int, float, float, Optional[float]]]]:
    """Pick trajectories worth reversing: long enough, and actually moving.

    Choosing the longest track is a trap -- on dashcam footage that is
    usually the vehicle being followed, which barely moves. Reversed it
    still barely moves, so it tests nothing while looking like a miss.
    """
    per_track: Dict[int, List[Tuple[int, float, float, Optional[float]]]] = {}
    for frame_index, tracks in frames:
        for track_id, x, y, width in tracks:
            per_track.setdefault(track_id, []).append((frame_index, x, y, width))

    candidates = [
        (points, _median_speed_ratio(points))
        for points in per_track.values()
        if len(points) >= 20
    ]
    candidates.sort(key=lambda item: -item[1])
    return [points for points, _ in candidates[:count]]


def _injection_caught(frames: Sequence[Tuple[int, List[Track]]],
                      donor: Sequence[Tuple[int, float, float, Optional[float]]],
                      config: DetectorConfig) -> bool:
    """Replay the clip with one extra vehicle retracing donor in reverse."""
    detector = WrongWayDetector(config)
    reversed_path = list(donor)[::-1]
    start = donor[0][0]
    for frame_index, tracks in frames:
        for track_id, x, y, width in tracks:
            detector.update(track_id, (x, y), frame=frame_index, scale=width)
        offset = frame_index - start
        if 0 <= offset < len(reversed_path):
            _, gx, gy, gwidth = reversed_path[offset]
            alert = detector.update(GHOST_ID, (gx, gy), frame=frame_index, scale=gwidth)
            if alert is not None and alert["track_id"] == GHOST_ID:
                return True
    return False


def evaluate(
    directory: str,
    config: Optional[DetectorConfig] = None,
    injections_per_clip: int = 3,
    false_positive_weight: float = 10.0,
) -> Dict[str, Any]:
    """Score every dump in `directory`. Returns a summary dict."""
    config = config or DetectorConfig()
    paths = sorted(glob.glob(os.path.join(directory, "*.jsonl")))
    if not paths:
        raise SystemExit("No dumps in {0}".format(directory))

    rows = []
    total_false = 0
    total_caught = 0
    total_injected = 0

    for path in paths:
        frames = list(load(path))
        alerts = _alerts_for(frames, config)
        donors = _injection_donors(frames, injections_per_clip)
        caught = sum(_injection_caught(frames, donor, config) for donor in donors)

        tracks = {track_id for _, ts in frames for (track_id, _, _, _) in ts}
        rows.append({
            "clip": os.path.basename(path),
            "frames": len(frames),
            "tracks": len(tracks),
            "false_positives": len(alerts),
            "caught": caught,
            "injected": len(donors),
        })
        total_false += len(alerts)
        total_caught += caught
        total_injected += len(donors)

    missed = total_injected - total_caught
    return {
        "clips": rows,
        "false_positives": total_false,
        "caught": total_caught,
        "injected": total_injected,
        "detection_rate": (total_caught / total_injected) if total_injected else None,
        "loss": false_positive_weight * total_false + missed,
        "false_positive_weight": false_positive_weight,
    }


def _print(summary: Dict[str, Any]) -> None:
    print("{0:<40} {1:>7} {2:>7} {3:>7} {4:>9}".format(
        "clip", "frames", "tracks", "FALSE", "caught"))
    print("-" * 74)
    for row in summary["clips"]:
        print("{0:<40} {1:>7} {2:>7} {3:>7} {4:>9}".format(
            row["clip"][:40], row["frames"], row["tracks"],
            row["false_positives"] or "", "{0}/{1}".format(row["caught"], row["injected"])))
    print("-" * 74)
    rate = summary["detection_rate"]
    print("{0:<40} {1:>23} {2:>9}".format(
        "TOTAL",
        "{0} false positive(s)".format(summary["false_positives"]),
        "{0}/{1}".format(summary["caught"], summary["injected"])))
    if rate is not None:
        print("\n  detection rate      : {0:.0%}".format(rate))
    print("  loss ({0:g}xFP + miss): {1:g}".format(
        summary["false_positive_weight"], summary["loss"]))
    print("\n  Lower loss is better. False positives are weighted heavily on")
    print("  purpose: accusing an innocent driver is the failure this system")
    print("  is built to avoid (CLAUDE.md principle 3).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default="fixtures")
    parser.add_argument("--injections", type=int, default=3)
    parser.add_argument("--fp-weight", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="machine-readable output")

    defaults = DetectorConfig()
    parser.add_argument("--zone-size", type=int, default=defaults.zone_size)
    parser.add_argument("--heading-window", type=int, default=defaults.heading_window)
    parser.add_argument("--baseline-alpha", type=float, default=defaults.baseline_alpha)
    parser.add_argument("--baseline-min-samples", type=int,
                        default=defaults.baseline_min_samples)
    parser.add_argument("--baseline-min-coherence", type=float,
                        default=defaults.baseline_min_coherence)
    parser.add_argument("--opposite-cos-threshold", type=float,
                        default=defaults.opposite_cos_threshold)
    parser.add_argument("--violation-frames-required", type=int,
                        default=defaults.violation_frames_required)
    parser.add_argument("--min-speed-fraction", type=float,
                        default=defaults.min_speed_fraction)
    parser.add_argument("--peer-max-age", type=int, default=defaults.peer_max_age)
    args = parser.parse_args()

    config = DetectorConfig(
        zone_size=args.zone_size,
        heading_window=args.heading_window,
        baseline_alpha=args.baseline_alpha,
        baseline_min_samples=args.baseline_min_samples,
        baseline_min_coherence=args.baseline_min_coherence,
        opposite_cos_threshold=args.opposite_cos_threshold,
        violation_frames_required=args.violation_frames_required,
        min_speed_fraction=args.min_speed_fraction,
        peer_max_age=args.peer_max_age,
    )
    summary = evaluate(args.directory, config, args.injections, args.fp_weight)
    if args.json:
        print(json.dumps(summary))
    else:
        _print(summary)


if __name__ == "__main__":
    main()
