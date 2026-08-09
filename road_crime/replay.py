"""Re-run violation logic over cached perception output.

`python -m road_crime.pipeline --dump-tracks` records what the modules actually consume:
per frame, each track's id and road-contact point. That is the whole
input surface a violation module has (CLAUDE.md principle 5), so
everything downstream of perception can be reproduced from it exactly --
with no GPU, no model, no video, and no third-party packages.

Why this exists: iterating on the logic through the full pipeline costs
minutes per attempt (edit, push, pull on the runtime, reload the model,
decode video). Replaying the same run costs well under a second, and runs
on a laptop. When the thing being debugged is the logic rather than the
perception -- which is nearly always -- use this.

    python -m road_crime.replay tracks.jsonl
    python -m road_crime.replay tracks.jsonl --zone-size 240
    python -m road_crime.replay tracks.jsonl --track 16 --verbose

`--verbose` with `--track` prints that vehicle's per-frame decision trail,
which is the fastest way to see why one specific alert fired.
"""

import argparse
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from road_crime.wrong_way_detector import DetectorConfig, WrongWayDetector

Track = Tuple[int, float, float, Optional[float]]
Frame = Tuple[int, List[Track]]


def load(path: str) -> Iterator[Frame]:
    """Yield ``(frame_index, [(track_id, x, y, width), ...])`` in recorded order.

    Feed order within a frame is preserved deliberately: the baseline is
    updated as tracks are processed, so a different order would produce a
    different result and the replay would stop being faithful.

    Dumps written before vehicle width was recorded have three fields per
    track instead of four; those yield a width of None, and the detector
    falls back to its absolute speed threshold.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tracks: List[Track] = [
                (
                    int(entry[0]),
                    float(entry[1]),
                    float(entry[2]),
                    float(entry[3]) if len(entry) > 3 else None,
                )
                for entry in record["tracks"]
            ]
            yield int(record["frame"]), tracks


def replay(
    path: str,
    config: Optional[DetectorConfig] = None,
    watch: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Run the wrong-way module over cached tracks. Returns every alert."""
    detector = WrongWayDetector(config)
    alerts: List[Dict[str, Any]] = []
    frames = 0
    track_ids = set()

    for frame_index, tracks in load(path):
        frames += 1
        for track_id, x, y, width in tracks:
            track_ids.add(track_id)
            alert = detector.update(
                track_id, (x, y), frame=frame_index, scale=width
            )

            if verbose and watch is not None and track_id == watch:
                _trace(detector, frame_index, track_id, (x, y))

            if alert is not None:
                alert["frame"] = frame_index
                alerts.append(alert)

    print(
        "Replayed {0} frames, {1} distinct track(s), {2} alert(s).".format(
            frames, len(track_ids), len(alerts)
        )
    )
    for alert in alerts:
        print(json.dumps(alert, indent=2, default=list))
    return alerts


def _trace(
    detector: WrongWayDetector,
    frame_index: int,
    track_id: int,
    position: Tuple[float, float],
) -> None:
    """Print one frame of a single track's decision trail.

    Reads the detector's state rather than duplicating its arithmetic, so
    the trail cannot drift away from what `update` actually did.
    """
    track = detector.tracks[track_id]
    heading = detector._heading(track)
    zone = detector.baseline.zone_of(position)
    trusted, baseline_dir = detector.baseline.state(position)

    if heading is None:
        print(
            "f{0:<5} #{1}  ({2:7.1f},{3:7.1f})  zone={4}  warming up".format(
                frame_index, track_id, position[0], position[1], zone
            )
        )
        return

    unit_x, unit_y, speed = heading
    cosine = (
        unit_x * baseline_dir[0] + unit_y * baseline_dir[1]
        if baseline_dir is not None
        else float("nan")
    )
    print(
        "f{0:<5} #{1}  ({2:7.1f},{3:7.1f})  zone={4}  head=({5:+.2f},{6:+.2f})"
        "  spd={7:5.1f}  trusted={8!s:<5}  cos={9:+.3f}  streak={10}".format(
            frame_index,
            track_id,
            position[0],
            position[1],
            zone,
            unit_x,
            unit_y,
            speed,
            trusted,
            cosine,
            track.violation_streak,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tracks", help="JSONL file from pipeline.py --dump-tracks")
    parser.add_argument("--track", type=int, help="track id to trace")
    parser.add_argument(
        "--verbose", action="store_true", help="print the traced track's decision trail"
    )

    # Threshold overrides, so a sweep is a shell loop rather than an edit.
    defaults = DetectorConfig()
    parser.add_argument("--zone-size", type=int, default=defaults.zone_size)
    parser.add_argument("--heading-window", type=int, default=defaults.heading_window)
    parser.add_argument("--baseline-alpha", type=float, default=defaults.baseline_alpha)
    parser.add_argument(
        "--baseline-min-samples", type=int, default=defaults.baseline_min_samples
    )
    parser.add_argument(
        "--baseline-min-coherence", type=float, default=defaults.baseline_min_coherence
    )
    parser.add_argument(
        "--opposite-cos-threshold", type=float, default=defaults.opposite_cos_threshold
    )
    parser.add_argument(
        "--violation-frames-required",
        type=int,
        default=defaults.violation_frames_required,
    )
    parser.add_argument("--min-speed-px", type=float, default=defaults.min_speed_px)
    parser.add_argument(
        "--min-speed-fraction", type=float, default=defaults.min_speed_fraction
    )
    args = parser.parse_args()

    config = DetectorConfig(
        zone_size=args.zone_size,
        heading_window=args.heading_window,
        baseline_alpha=args.baseline_alpha,
        baseline_min_samples=args.baseline_min_samples,
        baseline_min_coherence=args.baseline_min_coherence,
        opposite_cos_threshold=args.opposite_cos_threshold,
        violation_frames_required=args.violation_frames_required,
        min_speed_px=args.min_speed_px,
        min_speed_fraction=args.min_speed_fraction,
    )
    replay(args.tracks, config=config, watch=args.track, verbose=args.verbose)


if __name__ == "__main__":
    main()
