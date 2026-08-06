# Road crime detector

Real-time detection of traffic-law violations from a forward-facing car
dashcam, with alerts sent via API. This system **detects and documents —
it does not judge.** Every alert is evidence (video clip, timestamp,
confidence score), never a binary "guilty" verdict.

## Scope assumption (read first)

The camera is a forward-facing dashcam on the ego vehicle. The target of
detection is **other vehicles visible in the ego camera's field of view**
(a "witness dashcam" model) — not the ego vehicle's own compliance. All
modules below assume this unless a task says otherwise.

## Non-negotiable design principles

Do not deviate from these without checking with the user first:

1. **No GPS / external map dependency for v1.** Everything is either
   vision-only or self-calibrating from observed traffic.
2. **State machine, not single-frame classification.** No module may fire
   an alert from one frame. Every module requires a sustained condition
   over multiple consecutive frames (a "violation streak") before firing.
3. **Error asymmetry.** A false positive (accusing an innocent driver) is
   worse than a false negative (missing a real violation). When tuning any
   threshold, bias toward the safer (less aggressive) side.
4. **Evidence, not verdict.** Every alert payload includes a confidence
   score and supporting evidence. Never output a bare "guilty" boolean.
5. **Shared perception, independent modules.** RF-DETR + ByteTrack run
   once per frame. Each violation module is a separate, swappable,
   independently testable unit that consumes structured track data — never
   raw video, never the model directly.
6. **Pure logic before real video.** Every new module's core logic must
   ship with a synthetic-trajectory test (no camera, no GPU, no RF-DETR)
   that proves correctness before it is wired into `pipeline.py`.
7. **Language:** all code, comments, docstrings, and this file: English.

See `md/ARCHITECTURE.md` for Mermaid diagrams of the data flow, module
build order, and a worked internal-logic diagram of the wrong-way module.

See `md/WORKPLAN.md` for who is building what, in what order, and the
acceptance criteria for each step. Two developers work in parallel:
Track A finishes the wrong-way module, Track B builds the stop-sign
module.

## Architecture layers

```
Perception (RF-DETR detection + ByteTrack tracking)  <- runs every frame
        |
        v
Violation modules (wrong-way, stop-sign, solid-line, red-light)
        |  each independent, each ~one alert type
        v
Orchestration layer (NOT BUILT YET — only needed once 2+ modules run
together and events need merging/prioritizing; may use an LLM here
specifically, since alerts are rare — see "Orchestration" section below)
        |
        v
API alert (payload: evidence + confidence, never a verdict)
```

Detector choice: **RF-DETR** (not YOLO26/YOLOv12) — Apache 2.0 license,
no AGPL obligations if this ever ships as a product. Tracker: `trackers`
library (Roboflow), ByteTrack algorithm, Apache 2.0, detector-agnostic.

## Current status

| Component | Status |
|---|---|
| `wrong_way_detector.py` | **Done, tested.** `DetectorConfig`, `DirectionBaseline`, `TrackState`, `WrongWayDetector`. |
| `test_wrong_way.py` | **Passing.** Synthetic trajectories, no GPU/video needed. |
| `pipeline.py` | Written, wires RF-DETR + ByteTrack to `WrongWayDetector`. **Not yet run on real video** — needs GPU (local RTX 5070 or Colab). |
| Stop-sign module | Not started. |
| Solid/double line module | Not started. |
| Red-light module | Not started. |
| Orchestration layer | Not started — out of scope until 2+ modules are live. |

## Repo conventions

- Python type hints everywhere. Use `Optional[X]`, not `X | None`, for
  broader Python version compatibility.
- Every violation module gets its own file: `<violation_name>_detector.py`.
  No shared base class yet — duplication across modules is acceptable at
  this stage; do not prematurely abstract.
- Every violation module gets a matching `test_<violation_name>.py` with
  synthetic-trajectory tests, following the pattern in `test_wrong_way.py`:
  build fake tracks frame-by-frame, assert on the returned alert dict.
- Alert payload shape (keep consistent across all modules):
  ```python
  {
      "type": str,           # e.g. "wrong_way", "stop_sign_violation"
      "track_id": int,
      "confidence": float,   # 0.0-1.0
      "position": tuple,     # (x, y) pixel coords at time of violation
  }
  ```

---

## Task list: wrong-way driving module

**Status: core logic done. Remaining work below.**

- [x] `DetectorConfig` dataclass with tunable thresholds
- [x] `DirectionBaseline` — self-calibrating per-zone direction learner
- [x] `TrackState` — per-vehicle position history + violation streak
- [x] `WrongWayDetector.update()` — main entry point, one call per vehicle
      per frame
- [x] Synthetic tests: normal traffic never flagged, wrong-way vehicle
      flagged after baseline is trusted, no false alarms during warm-up,
      no duplicate alerts for an already-reported track
- [ ] Run `pipeline.py` against a real dashcam video file (not synthetic
      data) and manually verify RF-DETR class IDs actually match
      `VEHICLE_CLASS_IDS = {2, 3, 5, 7}` for this specific model/version —
      do not assume, print `detections.class_id` and cross-check against
      `model.class_names` (or equivalent) on first run
- [ ] Tune `DetectorConfig` thresholds against real footage (the synthetic
      test values are reasonable starting points, not final)
- [ ] Replace the `print()` placeholder in `send_alert()` with a real HTTP
      call once an API destination exists

---

## Task list: stop-sign violation module

**Status: not started.** Create `stop_sign_detector.py` +
`test_stop_sign.py`.

- [ ] Confirm the STOP sign class ID in RF-DETR's output (COCO includes
      "stop sign" as a standard class — verify the exact ID the same way
      as the wrong-way module's `VEHICLE_CLASS_IDS`, do not hardcode
      without checking)
- [ ] Define a "stop zone": a small region in image space directly in
      front of a detected stop sign, where a vehicle is expected to reach
      near-zero speed. Start with a simple heuristic (e.g. a fixed-size
      box below the sign's bounding box) — this does not need to be
      perfect for v1, just documented as a config-level assumption
- [ ] Reuse the heading/speed calculation pattern from
      `WrongWayDetector._heading()` to compute a tracked vehicle's speed
      while its position falls inside a stop zone
- [ ] Track, per `(track_id, stop_sign_id)` pair, the minimum speed
      observed while inside the zone
- [ ] Violation condition: vehicle passes fully through the zone without
      the minimum recorded speed dropping below a near-zero threshold for
      at least N consecutive frames (mirror `violation_frames_required`
      from wrong-way module for consistency)
- [ ] Guard against double-counting: once a `(track_id, stop_sign_id)`
      pair has been evaluated (vehicle has exited the zone), do not
      re-evaluate it again even if the same vehicle re-enters the frame
- [ ] Synthetic tests:
  - [ ] Vehicle that decelerates to ~0 inside the zone -> no alert
  - [ ] Vehicle that maintains speed through the zone -> alert fires
  - [ ] Vehicle that never enters any stop zone -> detector never even
        evaluates it (no crash, no false alert)
  - [ ] Vehicle that slows down but not close enough to zero -> alert
        fires (partial stops still count as violations)

---

## Task list: solid/double line crossing module

**Status: not started.** This module has a real open dependency — flag
it early rather than discovering it mid-implementation.

- [ ] **Open dependency:** RF-DETR does object detection, not lane-marking
      segmentation. This module needs a *different* model that can
      distinguish solid/double lines from dashed lines specifically (most
      standard lane-detection datasets like CULane/TuSimple label lane
      *position* but not always solid-vs-dashed *type*). Research and pick
      one of:
  - [ ] A pretrained lane segmentation model with marking-type labels
        (search current options — do not assume one exists without
        checking, this space moves fast)
  - [ ] A small amount of manually labeled data + fine-tuning, if no
        suitable pretrained model is found
- [ ] Once a segmentation source exists: extract solid/double line
      segments as polylines (pixel coordinate sequences) per frame
- [ ] Track each vehicle's bottom-center point (not full bbox — the point
      where the vehicle contacts the road) across frames
- [ ] Detect a crossing event: the bottom-center point transitions from
      one side of a solid/double polyline to the other between consecutive
      frames
- [ ] Filter: only flag crossings of lines classified as solid/double —
      dashed-line crossings (legal lane changes) must never be flagged
- [ ] Debounce: require the crossed-side state to persist for a few
      frames after the crossing, to reject bounding-box jitter at the
      line boundary
- [ ] Synthetic tests:
  - [ ] Vehicle trajectory that crosses a synthetic "solid line" polyline
        -> alert fires
  - [ ] Vehicle trajectory that crosses a synthetic "dashed line" polyline
        -> no alert
  - [ ] Vehicle trajectory that stays on one side throughout -> no alert

---

## Task list: red-light running module

**Status: not started. Hardest module — build last.**

- [ ] Detect traffic-light objects (RF-DETR/COCO includes "traffic light"
      as a class — verify exact ID, same process as the other modules)
- [ ] Add a small secondary classifier (or simple color-heuristic on the
      cropped bounding box — e.g. dominant color in the top/middle/bottom
      third of the light) to determine state: red / yellow / green
- [ ] Define stop-line position in image space (start with a manually
      configured coordinate for v1 — same simplifying-assumption approach
      as the stop-sign zone; do not try to auto-detect stop lines yet)
- [ ] **Open problem, do not skip past it:** light-to-lane association —
      when multiple traffic lights are visible in one frame, determine
      which light governs which lane/direction. For v1, scope this down:
      assume a single relevant light and a single lane of interest, and
      document this as a known limitation rather than solving the general
      case
- [ ] Build a per-vehicle state machine: track whether the relevant light
      was red at the moment the vehicle's position crossed the stop-line
      position
- [ ] Violation condition: vehicle crosses the stop line while the
      associated light is red, with the red state confirmed for at least
      a few consecutive frames before the crossing (to reject flicker/
      misclassification of a single frame)
- [ ] Synthetic tests:
  - [ ] Light sequence green->red, vehicle crosses stop-line after red
        confirmed -> alert fires
  - [ ] Vehicle crosses stop-line while light is still green -> no alert
  - [ ] Vehicle crosses just as light turns red (single-frame flicker,
        not sustained) -> no alert (avoids punishing borderline timing)

---

## Orchestration layer (do not build until 2+ modules above are done)

Only relevant once wrong-way + at least one more module run in parallel
and can produce overlapping/simultaneous events on the same vehicle.

- [ ] Define the structured JSON event shape passed *into* orchestration
      (list of `{type, confidence}` events per `track_id` per time window)
- [ ] Decide merge vs. separate-alert logic for correlated events (e.g. a
      wrong-way maneuver that also triggers a solid-line crossing in the
      same window — likely one root cause, one alert)
- [ ] This is the one place an LLM call is appropriate in this system —
      event volume is low (a handful per hour of driving, not per frame),
      so latency/cost of an LLM call is acceptable here specifically
- [ ] Do not start this until at least two violation modules above are
      implemented and tested — there is nothing to orchestrate with one
