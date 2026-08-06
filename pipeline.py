"""Wires the perception stack to the violation modules.

This is the only file in the system that touches video, a model, or a GPU.
Everything downstream of it -- `wrong_way_detector.py` today, the stop-sign
and red-light modules later -- consumes structured track data and stays
testable with no camera and no footage (CLAUDE.md principle 6).

Perception runs exactly once per frame and its output fans out to every
violation module (principle 5). No module may call the detector itself.

Usage:
    python pipeline.py --video dashcam.mp4 --output annotated.mp4
    python pipeline.py --video dashcam.mp4 --limit-frames 300
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from wrong_way_detector import DetectorConfig, WrongWayDetector

# COCO ids for car, motorcycle, bus, truck.
#
# DO NOT TRUST THIS SET until you have seen the report that
# `_report_class_ids` prints on the first run against a real video. Class
# numbering is a property of the specific model build, not a law of
# nature, and silently mismatched ids would make the whole pipeline look
# like it is working while it tracks the wrong objects.
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# "base" and "large" still load, but rfdetr deprecated them in v1.7.0 and
# will drop them in v2.0.0, so the default below is a current one.
MODEL_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}
DEFAULT_VARIANT = "medium"

BOX_COLOR = (0, 200, 0)
ALERT_COLOR = (0, 0, 255)
ANCHOR_COLOR = (255, 200, 0)


def send_alert(alert: Dict[str, Any]) -> None:
    """Deliver one violation alert.

    TODO: replace this print with a real HTTP POST once an API destination
    exists. The payload shape is already the contract every module emits
    (CLAUDE.md, "Repo conventions"), so only the transport changes here.

    Note what is *not* in the payload: any claim that the driver is
    guilty. This is evidence -- a type, a confidence, and where it
    happened -- and the consumer decides what it is worth (principle 4).
    """
    print("[ALERT] " + json.dumps(alert))


def _bottom_center(xyxy: Sequence[float]) -> Tuple[float, float]:
    """Reduce a bounding box to the point where the vehicle meets the road.

    Deliberately not the box centre: the centre floats above the road by
    half the vehicle's apparent height, so a tall truck and the car beside
    it in the same lane would land in different zones of the learned
    baseline grid. The bottom edge is on the road surface for both.

    The solid-line module needs this exact point too, for the same reason.
    """
    x1, _, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return ((x1 + x2) / 2.0, y2)


def _report_class_ids(counts: Counter, frames: int) -> None:
    """Print observed class ids against the model's own names.

    This exists to settle the open question in CLAUDE.md: whether
    VEHICLE_CLASS_IDS actually matches this model build. Read the table,
    then either confirm the constant or correct it.
    """
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES
    except ImportError:  # pragma: no cover - depends on rfdetr version
        COCO_CLASSES = {}

    print("\n--- class ids seen in the first {0} frame(s) ---".format(frames))
    if not counts:
        print("  no detections at all -- lower --conf, or check the video")
    for class_id, count in sorted(counts.items(), key=lambda item: -item[1]):
        name = COCO_CLASSES.get(class_id, "<unknown>") if COCO_CLASSES else "<unknown>"
        marker = "TRACKED" if class_id in VEHICLE_CLASS_IDS else "       "
        print("  [{0}] id={1:<4} {2:<18} {3} detections".format(marker, class_id, name, count))
    print(
        "  VEHICLE_CLASS_IDS is currently {0} -- confirm the names above are\n"
        "  the vehicles you meant before trusting any alert.\n".format(
            sorted(VEHICLE_CLASS_IDS)
        )
    )


def _load_model(variant: str) -> Any:
    """Instantiate an RF-DETR variant by short name."""
    import rfdetr

    class_name = MODEL_VARIANTS[variant]
    if not hasattr(rfdetr, class_name):
        available = ", ".join(
            sorted(name for name in MODEL_VARIANTS.values() if hasattr(rfdetr, name))
        )
        raise RuntimeError(
            "This rfdetr build has no {0}. Available: {1}".format(class_name, available)
        )
    return getattr(rfdetr, class_name)()


def _draw(
    frame: "np.ndarray",
    xyxy: Sequence[float],
    track_id: int,
    anchor: Tuple[float, float],
    alerted: bool,
) -> None:
    """Annotate one tracked vehicle, including its road-contact anchor."""
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    color = ALERT_COLOR if alerted else BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = "WRONG WAY #{0}".format(track_id) if alerted else "#{0}".format(track_id)
    cv2.putText(
        frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )
    cv2.circle(frame, (int(anchor[0]), int(anchor[1])), 3, ANCHOR_COLOR, -1)


def run(
    video: str,
    output: Optional[str] = None,
    variant: str = DEFAULT_VARIANT,
    conf: float = 0.5,
    limit_frames: Optional[int] = None,
    probe_frames: int = 30,
    config: Optional[DetectorConfig] = None,
    dump_tracks: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run perception + the wrong-way module over a video file.

    Returns every alert produced, so a caller can assert on them. Alerts
    are also handed to `send_alert` as they happen.

    `dump_tracks` writes the perception output -- one JSON line per frame,
    holding each track's id and road-contact point -- to a file. That file
    is everything a violation module ever sees (principle 5), so `replay.py`
    can re-run the logic from it with no GPU, no model and no video, in
    under a second. Use it whenever the thing being debugged is the logic
    rather than the perception, which is most of the time.
    """
    from trackers import ByteTrackTracker

    # Check the path before loading a model, so a typo costs a second
    # rather than a weight download.
    if not os.path.isfile(video):
        raise FileNotFoundError(
            "No such video: {0}\n"
            "If you meant the bundled sample, re-run the sample cell so that "
            "VIDEO points at it again.".format(video)
        )

    model = _load_model(variant)
    tracker = ByteTrackTracker()
    detector = WrongWayDetector(config)

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise RuntimeError(
            "OpenCV could not open {0}. The file exists but the codec may be "
            "unsupported -- try re-encoding to H.264 mp4.".format(video)
        )

    writer: Optional[Any] = None
    if output is not None:
        writer = cv2.VideoWriter(
            output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            capture.get(cv2.CAP_PROP_FPS) or 30.0,
            (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

    vehicle_ids = np.array(sorted(VEHICLE_CLASS_IDS))
    class_counts: Counter = Counter()
    reported = False
    alerted_ids: Set[int] = set()
    alerts: List[Dict[str, Any]] = []
    frame_index = 0
    dump = open(dump_tracks, "w", encoding="utf-8") if dump_tracks else None

    try:
        while True:
            if limit_frames is not None and frame_index >= limit_frames:
                break
            success, frame = capture.read()
            if not success:
                break
            frame_index += 1

            detections = model.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=conf
            )

            # Count every class, not just the vehicles, so the report can
            # show what we are throwing away as well as what we keep.
            if not reported and detections.class_id is not None:
                class_counts.update(int(value) for value in detections.class_id)
                if frame_index >= probe_frames:
                    _report_class_ids(class_counts, frame_index)
                    reported = True

            if detections.class_id is not None and len(detections) > 0:
                detections = detections[np.isin(detections.class_id, vehicle_ids)]
            tracked = tracker.update(detections)

            frame_tracks: List[List[float]] = []
            if tracked.tracker_id is not None:
                for xyxy, raw_id in zip(tracked.xyxy, tracked.tracker_id):
                    if raw_id is None or int(raw_id) < 0:
                        continue
                    track_id = int(raw_id)
                    anchor = _bottom_center(xyxy)
                    # Recorded before the detector sees it, and in feed
                    # order, so a replay reproduces this run exactly.
                    frame_tracks.append([track_id, anchor[0], anchor[1]])
                    alert = detector.update(track_id, anchor)
                    if alert is not None:
                        alert["frame"] = frame_index
                        alerts.append(alert)
                        alerted_ids.add(track_id)
                        send_alert(alert)
                    if writer is not None:
                        _draw(frame, xyxy, track_id, anchor, track_id in alerted_ids)

            if dump is not None:
                dump.write(
                    json.dumps({"frame": frame_index, "tracks": frame_tracks}) + "\n"
                )

            if writer is not None:
                writer.write(frame)
            if frame_index % 100 == 0:
                print("  frame {0}, {1} alert(s) so far".format(frame_index, len(alerts)))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if dump is not None:
            dump.close()

    if not reported:
        _report_class_ids(class_counts, frame_index)
    print("Done: {0} frames, {1} alert(s).".format(frame_index, len(alerts)))
    if output is not None:
        print("Annotated video written to {0}".format(output))
    if dump_tracks is not None:
        print(
            "Track data written to {0} -- replay it locally with:\n"
            "    python replay.py {0}".format(dump_tracks)
        )
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="path to a dashcam video file")
    parser.add_argument("--output", help="write an annotated video here (optional)")
    parser.add_argument(
        "--model",
        default=DEFAULT_VARIANT,
        choices=sorted(MODEL_VARIANTS),
        help="RF-DETR variant",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="detection threshold")
    parser.add_argument(
        "--limit-frames", type=int, help="stop after N frames, for a quick check"
    )
    parser.add_argument(
        "--probe-frames",
        type=int,
        default=30,
        help="frames to sample before printing the class-id report",
    )
    parser.add_argument(
        "--dump-tracks",
        help="write per-frame track data here, for replay.py to re-run offline",
    )
    args = parser.parse_args()

    run(
        video=args.video,
        output=args.output,
        variant=args.model,
        conf=args.conf,
        limit_frames=args.limit_frames,
        probe_frames=args.probe_frames,
        dump_tracks=args.dump_tracks,
    )


if __name__ == "__main__":
    main()
