"""Wires the perception stack to the violation modules.

This is the only file in the system that touches video, a model, or a GPU.
Everything downstream of it -- the wrong-way and stop-sign detectors today,
and the red-light module later -- consumes structured track data and stays
testable with no camera and no footage (CLAUDE.md principle 6).

Perception runs exactly once per frame and its output fans out to every
violation module (principle 5). No module may call the detector itself.

Usage:
    python pipeline.py --video dashcam.mp4 --output annotated.mp4
    python pipeline.py --video dashcam.mp4 --limit-frames 300
    python pipeline.py --video dashcam.mp4 --modules all
"""

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from wrong_way_detector import DetectorConfig, WrongWayDetector
from stop_sign_detector import (
    StopSignConfig,
    StopSignDetector,
    StopSignTracker,
    StopZone,
)

# COCO has two numbering schemes and they disagree about every vehicle.
# The 80-class contiguous scheme is what most YOLO code uses; the 91-class
# scheme is the original COCO annotation ids, with gaps where categories
# were dropped.
COCO_80 = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
}
COCO_91 = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
}

# Car, motorcycle, bus, truck under the 91-class scheme.
#
# Established from footage, not assumed. Two clips reported id=3 for the
# cars, id=10 for tunnel signals and motorway gantries, and id=6 for a
# coach. Under the 80-class scheme those same ids mean motorcycle, fire
# hydrant and train -- there were no fire hydrants or trains in either
# clip, and there was plainly a coach.
#
# The previous value, {2, 3, 5, 7}, meant bicycle/car/airplane/train here:
# cars were tracked by coincidence while every bus, truck and motorcycle
# was silently discarded.
#
# If you change model or model version, verify again. The report that
# `_report_class_ids` prints exists for exactly this and will not stay
# quiet when it cannot resolve a name.
VEHICLE_CLASS_IDS = {3, 4, 6, 8}

# Stop sign under RF-DETR's 91-class COCO scheme. Keep this independent of
# VEHICLE_CLASS_IDS: the same prediction is split after inference and fans out
# to the vehicle tracker and the stop-sign tracker.
STOP_SIGN_CLASS_IDS = {13}

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
STOP_ZONE_COLOR = (0, 165, 255)

AVAILABLE_MODULES = {"wrong_way", "stop_sign"}


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


def _model_class_names(model: Any) -> Optional[Dict[int, str]]:
    """Ask the model itself what its class ids mean, or admit it cannot say.

    Returns None rather than an empty mapping. The distinction matters: an
    earlier version swallowed the failed import and printed a tidy table of
    `<unknown>`, which looks like verification and contains none. That
    single silent fallback is why the wrong class ids survived two runs.
    """
    for module_path, attribute in (
        ("rfdetr.assets.coco_classes", "COCO_CLASSES"),
        ("rfdetr.assets.coco_classes", "COCO_CLASS_NAMES"),
        ("rfdetr.util.coco_classes", "COCO_CLASSES"),
        ("rfdetr.util.coco_classes", "COCO_CLASSES_91"),
    ):
        try:
            module = __import__(module_path, fromlist=[attribute])
            names = getattr(module, attribute)
            if names:
                return {int(key): str(value) for key, value in dict(names).items()}
        except Exception:  # noqa: BLE001 - any failure here means "cannot say"
            pass

    for attribute in ("class_names", "classes", "names"):
        names = getattr(model, attribute, None)
        if isinstance(names, dict) and names:
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)) and names:
            return {index: str(value) for index, value in enumerate(names)}
    return None


def _report_class_ids(counts: Counter, frames: int, model: Any) -> None:
    """Print observed class ids beside what they would mean under each scheme.

    Settles the open question in CLAUDE.md: whether VEHICLE_CLASS_IDS
    matches this model build. Where the model can name its own classes,
    that is authoritative. Where it cannot, both COCO interpretations are
    shown side by side and the ambiguity is stated rather than hidden --
    you resolve it by looking at the footage and asking which column
    describes what is actually on the road.
    """
    names = _model_class_names(model)

    print("\n--- class ids seen in the first {0} frame(s) ---".format(frames))
    if not counts:
        print("  NO DETECTIONS AT ALL -- lower --conf, or check the video.\n")
        return

    if names is not None:
        print("  names come from the model itself\n")
        print("  {0:<9} {1:<6} {2:<22} {3}".format("", "id", "name", "detections"))
        for class_id, count in sorted(counts.items(), key=lambda item: -item[1]):
            marker = "[TRACKED]" if class_id in VEHICLE_CLASS_IDS else "[       ]"
            print("  {0} {1:<6} {2:<22} {3}".format(
                marker, class_id, names.get(class_id, "?? not in model table"), count))
    else:
        print("  THIS MODEL BUILD DOES NOT EXPOSE ITS CLASS NAMES.")
        print("  Both COCO schemes are shown. Decide which column matches what")
        print("  is actually visible in the footage -- they disagree about every")
        print("  vehicle, so guessing silently corrupts every alert downstream.\n")
        print("  {0:<9} {1:<6} {2:<22} {3:<22} {4}".format(
            "", "id", "if 91-class", "if 80-class", "detections"))
        for class_id, count in sorted(counts.items(), key=lambda item: -item[1]):
            marker = "[TRACKED]" if class_id in VEHICLE_CLASS_IDS else "[       ]"
            print("  {0} {1:<6} {2:<22} {3:<22} {4}".format(
                marker,
                class_id,
                COCO_91.get(class_id, "?"),
                COCO_80.get(class_id, "?"),
                count,
            ))

    print(
        "\n  VEHICLE_CLASS_IDS is {0} (car, motorcycle, bus, truck under the\n"
        "  91-class scheme). If the names beside those ids are not those\n"
        "  vehicles, fix the constant before trusting a single alert.\n".format(
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
    wrong_way_alerted: bool,
    stop_sign_alerted: bool = False,
) -> None:
    """Annotate one tracked vehicle, including its road-contact anchor."""
    x1, y1, x2, y2 = (int(value) for value in xyxy)
    alerted = wrong_way_alerted or stop_sign_alerted
    color = ALERT_COLOR if alerted else BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    violations: List[str] = []
    if wrong_way_alerted:
        violations.append("WRONG WAY")
    if stop_sign_alerted:
        violations.append("STOP SIGN")
    label = (
        " + ".join(violations) + " #{0}".format(track_id)
        if violations
        else "#{0}".format(track_id)
    )
    cv2.putText(
        frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )
    cv2.circle(frame, (int(anchor[0]), int(anchor[1])), 3, ANCHOR_COLOR, -1)


def _draw_stop_zone(
    frame: "np.ndarray", sign_id: int, zone: StopZone
) -> None:
    """Draw the image-space region in which vehicles must stop."""
    x1, y1, x2, y2 = (
        int(zone.x1),
        int(zone.y1),
        int(zone.x2),
        int(zone.y2),
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), STOP_ZONE_COLOR, 2)
    cv2.putText(
        frame,
        "STOP ZONE #{0}".format(sign_id),
        (x1, max(14, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        STOP_ZONE_COLOR,
        2,
    )


def _selected_modules(modules: Sequence[str]) -> Set[str]:
    """Validate module names and expand ``all`` to the live modules."""
    requested = {modules} if isinstance(modules, str) else set(modules)
    allowed = AVAILABLE_MODULES | {"all"}
    invalid = requested - allowed
    if invalid:
        raise ValueError(
            "Unknown module(s): {0}. Choose from: {1}".format(
                ", ".join(sorted(invalid)), ", ".join(sorted(allowed))
            )
        )
    if not requested:
        raise ValueError("At least one module must be selected")
    if "all" in requested:
        return set(AVAILABLE_MODULES)
    return requested


def run(
    video: str,
    output: Optional[str] = None,
    variant: str = DEFAULT_VARIANT,
    conf: float = 0.5,
    limit_frames: Optional[int] = None,
    probe_frames: int = 30,
    config: Optional[DetectorConfig] = None,
    dump_tracks: Optional[str] = None,
    stop_sign_config: Optional[StopSignConfig] = None,
    modules: Sequence[str] = ("wrong_way",),
) -> List[Dict[str, Any]]:
    """Run shared perception and the selected violation modules over a video.

    Returns every alert produced, so a caller can assert on them. Alerts
    are also handed to `send_alert` as they happen. The wrong-way module is
    the backwards-compatible default; pass ``modules=("all",)`` to run both
    live modules from the same perception output.

    `dump_tracks` writes the perception output -- one JSON line per frame,
    holding each track's id, road-contact point and apparent width -- to a
    file. That file is everything a violation module ever sees (principle
    5), so `replay.py` can re-run the logic from it with no GPU, no model
    and no video, in under a second. Use it whenever the thing being
    debugged is the logic rather than the perception, which is most of the
    time.
    """
    from trackers import ByteTrackTracker

    selected_modules = _selected_modules(modules)

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
    wrong_way_detector = (
        WrongWayDetector(config) if "wrong_way" in selected_modules else None
    )
    stop_sign_detector = (
        StopSignDetector(stop_sign_config) if "stop_sign" in selected_modules else None
    )
    stop_sign_tracker = (
        StopSignTracker(stop_sign_config) if "stop_sign" in selected_modules else None
    )

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
    stop_sign_ids = np.array(sorted(STOP_SIGN_CLASS_IDS))
    class_counts: Counter = Counter()
    reported = False
    wrong_way_alerted_ids: Set[int] = set()
    stop_sign_alerted_ids: Set[int] = set()
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
                    _report_class_ids(class_counts, frame_index, model)
                    reported = True

            stop_sign_bboxes: List[Tuple[float, float, float, float]] = []
            vehicle_detections = detections
            if detections.class_id is not None and len(detections) > 0:
                vehicle_detections = detections[
                    np.isin(detections.class_id, vehicle_ids)
                ]
                if stop_sign_tracker is not None:
                    sign_detections = detections[
                        np.isin(detections.class_id, stop_sign_ids)
                    ]
                    stop_sign_bboxes = [
                        tuple(float(value) for value in xyxy)
                        for xyxy in sign_detections.xyxy
                    ]

            # This update is deliberately unconditional while the module is
            # enabled. Empty frames let the sign tracker age its retained
            # identities instead of freezing their state during dropouts.
            stable_signs = (
                stop_sign_tracker.update(stop_sign_bboxes)
                if stop_sign_tracker is not None
                else {}
            )
            tracked = tracker.update(vehicle_detections)

            if writer is not None and stop_sign_detector is not None:
                for sign_id, sign_bbox in stable_signs.items():
                    zone = StopZone.from_sign_bbox(
                        sign_bbox, stop_sign_detector.config
                    )
                    _draw_stop_zone(frame, sign_id, zone)

            frame_tracks: List[List[float]] = []
            if tracked.tracker_id is not None:
                for xyxy, raw_id in zip(tracked.xyxy, tracked.tracker_id):
                    if raw_id is None or int(raw_id) < 0:
                        continue
                    track_id = int(raw_id)
                    anchor = _bottom_center(xyxy)
                    # Apparent vehicle size. The detector needs it to tell a
                    # vehicle that is genuinely crawling from one that only
                    # looks slow because the ego car is following it.
                    width = float(xyxy[2]) - float(xyxy[0])
                    # Recorded before the detector sees it, and in feed
                    # order, so a replay reproduces this run exactly.
                    frame_tracks.append([track_id, anchor[0], anchor[1], width])
                    if wrong_way_detector is not None:
                        wrong_way_alert = wrong_way_detector.update(
                            track_id, anchor, frame=frame_index, scale=width
                        )
                        if wrong_way_alert is not None:
                            wrong_way_alert["frame"] = frame_index
                            alerts.append(wrong_way_alert)
                            wrong_way_alerted_ids.add(track_id)
                            send_alert(wrong_way_alert)
                    if stop_sign_detector is not None:
                        stop_sign_alert = stop_sign_detector.update(
                            track_id, anchor, stable_signs
                        )
                        if stop_sign_alert is not None:
                            stop_sign_alert["frame"] = frame_index
                            alerts.append(stop_sign_alert)
                            stop_sign_alerted_ids.add(track_id)
                            send_alert(stop_sign_alert)
                    if writer is not None:
                        _draw(
                            frame,
                            xyxy,
                            track_id,
                            anchor,
                            track_id in wrong_way_alerted_ids,
                            track_id in stop_sign_alerted_ids,
                        )

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
        _report_class_ids(class_counts, frame_index, model)
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
    parser.add_argument(
        "--modules",
        nargs="+",
        default=("wrong_way",),
        choices=sorted(AVAILABLE_MODULES | {"all"}),
        help="violation modules to run (default: wrong_way; use all for both)",
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
        modules=args.modules,
    )


if __name__ == "__main__":
    main()
