"""Explain what the wrong-way module decided, and where it stopped deciding.

`replay.py` answers "did it alert". This answers the question that follows
every time the answer is no: **why not**. Zero alerts has six different
causes and only one of them is the decision logic -- the vehicle may never
have had enough history, never cleared the speed gate, never stood in a
trusted zone, or may have been judged and found compliant. Those are not
the same result, and a bare `0 alert(s)` cannot tell them apart.

Why this lives in the package rather than in a notebook cell: it has to
mirror `WrongWayDetector.update()`'s sequence of gates exactly, and a copy
of that sequence pasted into a notebook drifts away from the original
without anything failing. So every number below is read from the detector
itself -- `_heading`, `required_speed`, `baseline.state`,
`baseline.peer_consensus` -- in the same order `update` consults them, and
always *before* `update` is allowed to mutate anything. Same discipline as
`replay._trace`, same reason.

    python -m road_crime.diagnose tracks.jsonl
    python -m road_crime.diagnose tracks.jsonl --min-speed-fraction 0.01

No GPU, no model, no video, no third-party imports.
"""

import argparse
from math import hypot
from typing import Any, Dict, List, Optional, Tuple

from road_crime.replay import load
from road_crime.wrong_way_detector import DetectorConfig, WrongWayDetector

# In the order `update` applies them. The last entry is the only one that
# represents the decision logic actually running; everything above it is a
# frame the module never got to judge.
WARMING_UP = "warming up"
BELOW_SPEED_GATE = "below speed gate"
ZONE_NOT_TRUSTED = "zone not trusted"
AGREES = "judged: agrees"
CLEARED_BY_PEERS = "judged: cleared by peers"
OPPOSED = "judged: OPPOSED"

STAGES = (
    WARMING_UP,
    BELOW_SPEED_GATE,
    ZONE_NOT_TRUSTED,
    AGREES,
    CLEARED_BY_PEERS,
    OPPOSED,
)


def _stage_for(
    detector: WrongWayDetector,
    track_id: int,
    position: Tuple[float, float],
    scale: Optional[float],
    frame: int,
) -> str:
    """Which gate this frame will die at, asked before `update` mutates state.

    Deliberately reads the detector rather than recomputing: the only
    things expressed here are the comparisons themselves, and the values
    being compared all come out of the detector's own methods.
    """
    track = detector.tracks.get(track_id)
    if track is None or len(track.positions) <= detector.config.heading_window:
        return WARMING_UP

    heading = detector._heading(track)
    if heading is None:
        return WARMING_UP

    unit_x, unit_y, speed = heading
    if speed < detector.required_speed(scale):
        return BELOW_SPEED_GATE

    trusted, baseline_dir = detector.baseline.state(position)
    if not trusted or baseline_dir is None:
        return ZONE_NOT_TRUSTED

    cosine = unit_x * baseline_dir[0] + unit_y * baseline_dir[1]
    if cosine >= detector.config.opposite_cos_threshold:
        return AGREES

    peer_dir = detector.baseline.peer_consensus(position, track_id, frame)
    if peer_dir is not None:
        peer_cosine = unit_x * peer_dir[0] + unit_y * peer_dir[1]
        if peer_cosine > detector.config.peer_agree_cos:
            return CLEARED_BY_PEERS

    return OPPOSED


def diagnose(
    path: str, config: Optional[DetectorConfig] = None
) -> Dict[str, Any]:
    """Replay a dump and record, per track, where every frame ended up.

    Returns the counts, the zone baseline that was learned, and any alerts,
    so a caller can assert on them instead of reading printed text.
    """
    detector = WrongWayDetector(config)
    per_track: Dict[int, Dict[str, int]] = {}
    widths: Dict[int, List[float]] = {}
    alerts: List[Dict[str, Any]] = []
    frames = 0

    for frame_index, tracks in load(path):
        frames += 1
        for track_id, x, y, width in tracks:
            stage = _stage_for(detector, track_id, (x, y), width, frame_index)
            counts = per_track.setdefault(track_id, {name: 0 for name in STAGES})
            counts[stage] += 1
            if width:
                widths.setdefault(track_id, []).append(width)

            alert = detector.update(track_id, (x, y), frame=frame_index, scale=width)
            if alert is not None:
                alert["frame"] = frame_index
                alerts.append(alert)

    zones = []
    for zone, stats in detector.baseline._zones.items():
        coherence = hypot(stats.dir_x, stats.dir_y)
        zones.append({
            "zone": zone,
            "samples": stats.samples,
            "coherence": coherence,
            "trusted": (
                stats.samples >= detector.config.baseline_min_samples
                and coherence >= detector.config.baseline_min_coherence
            ),
        })
    zones.sort(key=lambda row: -row["samples"])

    return {
        "frames": frames,
        "tracks": per_track,
        "widths": widths,
        "zones": zones,
        "alerts": alerts,
        "config": detector.config,
    }


def report(path: str, config: Optional[DetectorConfig] = None) -> Dict[str, Any]:
    """Print the diagnosis. Returns the same dict `diagnose` produces."""
    summary = diagnose(path, config)
    config = summary["config"]

    print("{0}: {1} frames, {2} track(s), {3} alert(s)\n".format(
        path, summary["frames"], len(summary["tracks"]), len(summary["alerts"])))

    print("Where each frame's decision ended up")
    print("-" * 78)
    print("{0:<7}{1:>9}{2:>9}{3:>11}{4:>9}{5:>9}{6:>10}".format(
        "track", "warming", "too slow", "untrusted", "agrees", "peers", "OPPOSED"))
    judged_total = 0
    for track_id, counts in sorted(
        summary["tracks"].items(), key=lambda item: -sum(item[1].values())
    ):
        print("#{0:<6}{1:>9}{2:>9}{3:>11}{4:>9}{5:>9}{6:>10}".format(
            track_id, *(counts[name] for name in STAGES)))
        judged_total += counts[AGREES] + counts[CLEARED_BY_PEERS] + counts[OPPOSED]
    print("\n  warming   too little history for a heading      "
          "agrees    judged, moving with the flow")
    print("  too slow  below required_speed(width)            "
          "peers     judged, contemporaries agreed")
    print("  untrusted no learned baseline in that zone       "
          "OPPOSED   judged, counted toward a streak")

    if judged_total == 0:
        print(
            "\n  NOT ONE FRAME REACHED THE DECISION LOGIC.\n"
            "  Zero alerts here says nothing about whether the wrong-way rule\n"
            "  works -- it was never consulted. Read the columns to the left\n"
            "  for the gate that stopped it."
        )

    print("\nBaseline learned per zone (zone_size={0})".format(config.zone_size))
    print("-" * 78)
    if not summary["zones"]:
        print("  no zone ever received a vote")
    else:
        print("{0:<12} {1:>9} {2:>12} {3:>10}".format(
            "zone", "samples", "coherence", "trusted"))
        for row in summary["zones"]:
            print("{0:<12} {1:>9} {2:>12.2f} {3:>10}".format(
                str(row["zone"]), row["samples"], row["coherence"],
                "yes" if row["trusted"] else ""))
        print("\n  needs samples >= {0} and coherence >= {1}".format(
            config.baseline_min_samples, config.baseline_min_coherence))

    # Apparent size over a track's life. An approaching vehicle expands; one
    # holding station ahead does not. This column is not used by any decision
    # today -- it is here because it is the measurement that separates the two
    # cases on real footage, and the dumps already carry it.
    print("\nApparent size over each track's life")
    print("-" * 78)
    print("{0:<7} {1:>10} {2:>10} {3:>10}".format("track", "first", "last", "growth"))
    for track_id, values in sorted(summary["widths"].items()):
        if not values:
            continue
        print("#{0:<6} {1:>10.0f} {2:>10.0f} {3:>10}".format(
            track_id, values[0], values[-1],
            "x{0:.2f}".format(values[-1] / values[0]) if values[0] else "-"))

    for alert in summary["alerts"]:
        print("\nALERT: track #{0}, confidence {1}".format(
            alert["track_id"], alert["confidence"]))
        for key, value in alert["evidence"].items():
            print("    {0:<20} {1}".format(key, value))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tracks", help="JSONL file from pipeline.py --dump-tracks")

    defaults = DetectorConfig()
    parser.add_argument("--zone-size", type=int, default=defaults.zone_size)
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
        "--min-speed-fraction", type=float, default=defaults.min_speed_fraction
    )
    parser.add_argument("--min-speed-px", type=float, default=defaults.min_speed_px)
    args = parser.parse_args()

    report(args.tracks, DetectorConfig(
        zone_size=args.zone_size,
        baseline_min_samples=args.baseline_min_samples,
        baseline_min_coherence=args.baseline_min_coherence,
        opposite_cos_threshold=args.opposite_cos_threshold,
        min_speed_fraction=args.min_speed_fraction,
        min_speed_px=args.min_speed_px,
    ))


if __name__ == "__main__":
    main()
