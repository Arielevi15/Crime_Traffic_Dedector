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
| `wrong_way_detector.py` | Implemented. `DetectorConfig`, `DirectionBaseline`, `TrackState`, `WrongWayDetector`. **Produced a false positive on real video — see "What the first real run taught us".** |
| `test_wrong_way.py` | **17 tests passing.** Synthetic trajectories and regression fixtures, no GPU/video needed. Runs under pytest or standalone. |
| `stop_sign_detector.py` | Implemented. Stop-zone heuristic, per-vehicle/sign state, evidence payload and duplicate-evaluation guard are in place. Real stop-sign footage validation and tuning are still pending. |
| `test_stop_sign.py` | **16 tests passing.** Canonical pure-Python regression suite for the stop-sign module; no GPU, video or RF-DETR required. |
| `pipeline.py` | **Wrong-way path run successfully on GPU (Colab T4).** Stop-sign detection is wired into the shared inference/tracking path, but has not yet passed real stop-sign video acceptance. |
| `run_on_colab.ipynb` | Working. Fetches code from GitHub, video from the `supervision` sample set or Drive. |
| `run_stop_sign_on_colab.ipynb` | Added as a separate stop-sign Colab runner. Its existence does not imply that a full real-footage Colab run has completed. |
| Stop-sign module | Implementation and synthetic regression tests added; pipeline integration is wired. Real stop-sign footage validation, threshold tuning and measured acceptance remain pending. |
| Solid/double line module | Not started. Open dependency unresolved. |
| Red-light module | Not started. |
| Orchestration layer | Not started — out of scope until 2+ modules are live. |

### Runtime environment

GPU work runs on **Google Colab**, driven from VS Code via the official
`google.colab` extension. The Colab runtime is a remote machine that
cannot see the local disk, so `run_on_colab.ipynb` pulls the code from
GitHub. **After every push, re-run the git cell** or the runtime keeps
executing the previous version. Synthetic tests run locally with no GPU.

## What the first real run taught us

The chain works. Both defects it exposed are in our own logic, not in
RF-DETR or ByteTrack.

**1. Class ids do not match the assumption.** The run reported `id=3`
(120 detections) and `id=8` (1 detection), with names unresolved because
the `COCO_CLASSES` import failed and was swallowed. The pattern suggests
RF-DETR uses the original 91-class COCO numbering (3=car, 8=truck), not
the contiguous 80-class one that `VEHICLE_CLASS_IDS = {2, 3, 5, 7}`
assumes. If so, **trucks and buses are being missed entirely**, and cars
are tracked only by coincidence. Not yet confirmed — confirm before
changing the constant.

**2. A legal lane change was flagged**, confidence 0.903, implying a mean
cosine near -0.86 (~150° deviation). A lane change is 10-20°. That gap
means this is **not a tuning problem**. Two hypotheses, unresolved:

- **Zone hopping** — the vehicle crossed into a zone whose baseline was
  learned from other traffic and was judged against a stranger's rule.
- **Dashcam perspective** — under ego motion, a same-direction vehicle
  that is slower than the ego car closes in (bottom-centre moves down the
  image) while a faster one recedes (moves up). Both legal, opposite in
  image space. If this is the cause, image-space heading is not a valid
  proxy for real-world direction and the module's premise needs revisiting.

Alerts now carry an `evidence` block specifically to tell these apart.

**The meta-lesson:** all synthetic tests passed while both defects were
present. The tests prove the code does what we designed; they do not
prove the design matches reality. Every surprise reality delivers becomes
a new synthetic test — write the failing test *before* the fix.

## Where the wrong-way module actually stands

Both defects above are fixed and both fixes are pinned by tests. Class
ids are corrected to the 91-class scheme, which roughly six-folded the
number of tracked vehicles on the same clip once buses and trucks stopped
being discarded.

Measured on a full 825-frame motorway clip — 50 tracks, a vehicle being
followed, traffic ahead in lane, and an oncoming carriageway across the
barrier:

| | |
|---|---|
| False positives | **0** |
| Injected wrong-way drivers caught | **2 of 6** |
| Zones that ever became trusted | **6 of 21** |

The silence is real, not the detector having stopped working: replaying
the same clip with a synthetic wrong-way vehicle spliced in still
produces alerts, at confidence 0.99-1.00.

**But sensitivity is low, and the reason is measurable.** Only six zones
ever reached `baseline_min_samples`, so most of the image is never judged
at all. Passing vehicles cross a 120 px zone too quickly to leave 30
samples in it. Four of the six injected violations went unreported for
this reason rather than through any decision the module made.

That is the correct direction to err — principle 3 says an unjudged
vehicle beats an accused innocent one — but "we catch about a third of
blatant violations" is a limitation to fix, not a result to accept. The
zone geometry is the lever: `zone_size` and `baseline_min_samples` are
both absolute numbers that behave differently at every resolution, and
neither has been tuned against anything. That work needs the evaluation
set, which is why it is Track C's blocker and not a quick edit.

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
      "evidence": dict,      # module-specific; see below
  }
  ```
  `evidence` holds whatever a human needs to reconstruct the decision
  afterwards — for the wrong-way module: heading, baseline compared
  against, zone, zone first seen, mean cosine, streak, speed. The fields
  differ per module; the requirement does not. An alert nobody can audit
  is a verdict (principle 4 forbids it) and is also undebuggable, which
  is how a false positive survives to accuse the next driver.

---

## Task list: wrong-way driving module

**Status: runs end to end on real video; correctness not yet established.**
Track A in `md/WORKPLAN.md` owns the remaining work.

- [x] `DetectorConfig` dataclass with tunable thresholds
- [x] `DirectionBaseline` — self-calibrating per-zone direction learner
- [x] `TrackState` — per-vehicle position history + violation streak
- [x] `WrongWayDetector.update()` — main entry point, one call per vehicle
      per frame
- [x] Synthetic tests: normal traffic never flagged, wrong-way vehicle
      flagged after baseline is trusted, no false alarms during warm-up,
      no duplicate alerts for an already-reported track
- [x] Every alert carries an auditable `evidence` block
- [x] Run `pipeline.py` against real video on a GPU — done, Colab T4.
      `_report_class_ids` prints observed ids over the first N frames
- [ ] **Finish the class-id verification.** The report ran but printed
      `<unknown>` for every name, so the cross-check never happened. Fix
      the name lookup, then correct `VEHICLE_CLASS_IDS` — the evidence
      points at the 91-class numbering, but confirm before changing it
- [ ] **Resolve the false positive.** Diagnose with the `evidence` block,
      reproduce it as a *failing* synthetic test, then fix
- [ ] Tune `DetectorConfig` thresholds against real footage. Blocked on
      the evaluation set (Track C) — tuning without measurement is
      guessing, and principle 3 demands minimising false positives
- [ ] Replace the `print()` placeholder in `send_alert()` with a real HTTP
      call once an API destination exists

---

## Task list: stop-sign violation module

**Status: implementation and pure-Python regression tests added; wired
into the pipeline, but not yet validated or tuned on real stop-sign
footage.** The canonical test filename is `test_stop_sign.py`.

- [x] Verify the installed RF-DETR class mapping: it uses sparse COCO ids,
      with `13 = stop sign` and `11 = fire hydrant`. This corrects the
      contiguous-map assumption that would label id 11 as a stop sign.
- [ ] Confirm id 13 is actually emitted on real footage containing a stop
      sign before treating the mapping check as end-to-end validation
- [x] Define a "stop zone": a small region in image space directly in
      front of a detected stop sign, where a vehicle is expected to reach
      near-zero speed. Start with a simple heuristic (e.g. a fixed-size
      box below the sign's bounding box) — this does not need to be
      perfect for v1, just documented as a config-level assumption
- [x] Reuse the heading/speed calculation pattern from
      `WrongWayDetector._heading()` to compute a tracked vehicle's speed
      while its position falls inside a stop zone
- [x] Track, per `(track_id, stop_sign_id)` pair, the minimum speed
      observed while inside the zone
- [x] Violation condition: vehicle passes fully through the zone without
      the minimum recorded speed dropping below a near-zero threshold for
      at least N consecutive frames (mirror `violation_frames_required`
      from wrong-way module for consistency)
- [x] Guard against double-counting: once a `(track_id, stop_sign_id)`
      pair has been evaluated (vehicle has exited the zone), do not
      re-evaluate it again even if the same vehicle re-enters the frame
- [x] Pure-Python synthetic regression tests in `test_stop_sign.py`:
  - [x] Vehicle that decelerates to ~0 inside the zone -> no alert
  - [x] Vehicle that maintains speed through the zone -> alert fires
  - [x] Vehicle that never enters any stop zone -> detector never even
        evaluates it (no crash, no false alert)
  - [x] Vehicle that slows down but not close enough to zero -> alert
        fires (partial stops still count as violations)
- [x] Wire the detector into the shared `pipeline.py` perception pass and
      add the separate `run_stop_sign_on_colab.ipynb` runner
- [ ] Run the stop-sign path end to end on real footage, inspect the
      annotated result and verify standard alerts with populated evidence
- [ ] Tune thresholds and measure false positives/false negatives on a
      labelled stop-sign evaluation set

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
