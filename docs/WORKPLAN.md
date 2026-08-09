# Work plan — two developers, parallel tracks

Companion to `../CLAUDE.md` (the design contract) and `ARCHITECTURE.md`
(the diagrams). Read `../CLAUDE.md` first; nothing here overrides it.

**Owners**

| Track | Owner | Deliverable |
|---|---|---|
| A | Ariel | Wrong-way module, finished and measured |
| B | Partner | Stop-sign module, from scratch |
| C | Both | Evaluation set + open questions |

The two tracks touch disjoint files by design (`CLAUDE.md` principle 5:
one file per module, no shared base class). Merge conflicts should be
close to zero. If you find yourself editing the other track's module,
stop and talk first — it means an interface is wrong.

---

## 0. Rules both tracks follow

These are not style preferences. Breaking them costs real time later.

**0.1 The alert payload is frozen.** Every module emits exactly this:

```python
{
    "type": str,           # "wrong_way", "stop_sign_violation", ...
    "track_id": int,
    "confidence": float,   # 0.0-1.0
    "position": tuple,     # (x, y) at time of violation
    "evidence": dict,      # module-specific, see 0.2
}
```

Changing this shape requires both owners to agree, because the
orchestration layer will consume all of them together.

**0.2 `evidence` is mandatory, and it is module-specific.** It holds
whatever a human needs to reconstruct the decision afterwards. The
wrong-way module reports heading, baseline, zone, mean cosine, streak and
speed. The stop-sign module will report different fields. What matters is
that an alert can always be audited. An alert nobody can explain is a
verdict, which principle 4 forbids, and it is also undebuggable — which
is how a false positive survives to accuse the next driver.

**0.3 Synthetic tests before real video, always.** Principle 6. A module
without a passing `test_<name>.py` does not get wired into
`road_crime/pipeline.py`. Not "should not" — does not.

**0.4 When reality contradicts the tests, write the test first.** If a
real run produces a wrong result, reproduce it as a synthetic test that
*fails*, and only then fix the code. Fixing first leaves you with no
evidence the fix works and no protection against regression.

**0.5 Never assume an ID.** Class ids, sign ids, light ids: print what
the model actually emits and cross-check against the model's own names.
This has already cost us once — see A1.

**0.6 Branches.** `track-a/<topic>`, `track-b/<topic>`. `main` stays
green: no merge unless every test passes. Both test files must pass, not
just your own.

---

## The tooling, and why you should not iterate through Colab

This section did not exist when the plan was written. It is the single
biggest time saver in the repository, and it was built because Track A
spent hours per bug going the slow way.

**The problem.** Debugging logic through the notebook costs minutes per
attempt: edit, push, pull on the runtime, reload the model, decode video.
Most of that pays for perception, which is almost never what is broken.

**The insight.** A violation module only ever sees structured track data
(principle 5). That is a few hundred kilobytes per clip. So record it
once, and everything downstream replays from it with no GPU, no model, no
video and no third-party imports, in well under a second, on a laptop.

```
python -m road_crime.pipeline --video clip.mp4 --dump-tracks clip.jsonl   # once, on Colab
python -m road_crime.replay clip.jsonl                                    # forever, locally
python -m road_crime.replay clip.jsonl --track 7 --verbose                # why did THAT fire
python -m road_crime.evaluate fixtures/                                   # score the corpus
```

**`fixtures/`** holds those recordings, committed, so a bug stays
reproducible for as long as the repository exists and reaches the other
developer through `git pull`. Two of them are named regressions: a
false-positive case and a clean divided-highway run.

**`road_crime/corpus.py`** fetches clips from `url:`, `hf:`, `kaggle:` or
a path and turns them into dumps, so nobody downloads video by hand.

**`road_crime/evaluate.py`** scores a whole corpus without any labelling.
Ordinary driving contains no violations to any useful approximation, so
every alert raised on it is a false positive; sensitivity comes from
replaying real trajectories backwards. The two combine into one
deliberately asymmetric number, which is principle 3 written as
arithmetic:

```
loss = 10 * false_positives + missed_injections
```

**0.7 Tune against the corpus, never against one clip.** Fitting
thresholds to a single video is how you get numbers that look perfect and
break on the next road. If a change does not move the loss over
`fixtures/`, you do not know that it helped.

### Known gap, and Track B's cheapest big win

**The dump records vehicles only** — id, road-contact point, width. It
does not record stop signs, so none of the above works for the stop-sign
module yet. Extending `dump_tracks` to carry sign boxes, and teaching
`evaluate.py` to score stop-sign alerts, would hand Track B the same
second-long iteration loop that Track A has.

Do that before tuning anything. It is a small change that pays for itself
within the first afternoon of threshold work.

**0.7 `CLAUDE.md` is the contract.** Changes to principles, payload
shape or module boundaries need both owners. Changes to your own module's
task list do not.

---

## Track A — Ariel: finish the wrong-way module

The module works end to end but produced a false positive on its first
real video. Everything below is about turning "it runs" into "we know
what it does."

### A1. Close the class-id question

**Problem.** The first run reported:

```
[TRACKED] id=3    <unknown>    120 detections
[       ] id=8    <unknown>      1 detections
```

Two failures. Names did not resolve (`COCO_CLASSES` import failed and was
swallowed by an `except`), and `VEHICLE_CLASS_IDS = {2, 3, 5, 7}` looks
wrong — the pattern suggests RF-DETR uses the original 91-class COCO
numbering (3=car, 8=truck), not the contiguous 80-class one. If that
holds, **trucks and buses are being missed entirely**, and cars are being
tracked by coincidence.

**Steps**

1. Run the diagnostic cell that imports `COCO_CLASSES` and prints ids 2-8
   with their names. If the import fails, find where the names actually
   live in this `rfdetr` build.
2. Fix `_report_class_ids` in `road_crime/pipeline.py` so it never silently degrades
   to `<unknown>`. If names cannot be resolved, it should say so loudly,
   not print a table that looks fine.
3. Correct `VEHICLE_CLASS_IDS` to the verified ids.
4. Re-run and confirm trucks and buses now appear as `[TRACKED]`.

**Done when.** The class-id report prints real names, the constant
matches them, and a video containing a truck shows that truck tracked.
Record the finding in `CLAUDE.md` so the open task can be ticked.

### A2. Diagnose the false positive

**Problem.** A vehicle changing lanes legally was flagged with confidence
0.903, implying a mean cosine near -0.86 — roughly a 150° deviation. A
lane change is 10-20°. That gap means this is **not a threshold-tuning
problem**; something structural is wrong.

Two hypotheses:

- **H1 — zone hopping.** `zone_size = 120px`. The vehicle crossed into a
  zone whose baseline was learned from different traffic, and was judged
  against a stranger's rule. It never changed direction; it changed judge.
- **H2 — dashcam perspective.** A vehicle ahead travelling the same
  direction but *slower* than the ego car closes in, so its bottom-centre
  point moves **down** the image. One travelling *faster* moves **up**.
  Both are legal and identical in the real world, opposite in image space.

H2 is the more serious of the two: it would mean image-space heading is
not a valid proxy for real-world direction under ego motion, and the
module's core premise needs revisiting.

**Steps**

1. Re-run with the `evidence` block (already pushed) on the same clip.
2. Compare `zone` against `zone_first_seen`.
   - Different → H1 is in play.
   - Same → H1 is ruled out for this case; H2 becomes the leading theory.
3. Extract frames around the alert and confirm visually what the vehicle
   and the ego camera were doing.
4. Write the conclusion down — one paragraph, with the numbers that
   support it. This is the input to A3, and guessing here poisons
   everything after it.

**Done when.** A written root cause naming H1, H2, or something else,
supported by evidence values from an actual run.

### A3. Reproduce it synthetically

Per rule 0.4. Before touching the fix.

- H1 → a test where a legal vehicle crosses a zone boundary into a zone
  trained by traffic moving another way, and is wrongly flagged.
- H2 → a test modelling relative motion under a moving camera: several
  vehicles all travelling the same real direction, whose image-space
  headings differ by sign because of differing relative speed.

**Done when.** A new test in `tests/test_wrong_way.py` fails, and fails for the
diagnosed reason rather than by accident.

### A4. Fix it

Only now. Possible directions, to be chosen from A2's conclusion — do not
pre-commit to one:

- Larger zones, or zones keyed to something more stable than a pixel grid
- Requiring a track to have contributed to a zone's baseline before that
  zone may judge it
- Ego-motion compensation: estimate global image flow and subtract it, so
  headings become relative to the scene rather than to the camera
- Restricting judgement to image regions where perspective distortion is
  least severe

**Done when.** The A3 test passes, every other test still passes, and the
same clip no longer flags that vehicle.

### A5. Tune against real footage

Blocked on Track C's evaluation set. Tuning without measurement is
guessing.

- Sweep `opposite_cos_threshold`, `violation_frames_required`,
  `baseline_min_samples`, `baseline_min_coherence`, `zone_size`
- Record false positives and false negatives per setting
- Per principle 3, prefer the quieter setting whenever two are close

Note that `opposite_cos_threshold = -0.3` (~107°) is currently on the
permissive side; a sharply turning vehicle can plausibly reach it. It was
left as-is deliberately, pending real measurement.

**Done when.** Chosen values are recorded in `CLAUDE.md` with the
measured FP/FN counts that justify them.

### A6. Real alert delivery

`send_alert()` is still a `print`. Blocked on an API destination
existing.

- Replace with an HTTP POST
- Decide and document: retry policy, what happens when the endpoint is
  down, whether alerts are queued or dropped
- Never block the frame loop on a network call

---

## Track B — Partner: stop-sign module

Fully independent of Track A. New files only: `road_crime/stop_sign_detector.py` and
`tests/test_stop_sign.py`.

### B0. Onboarding

Before writing anything:

1. Read `../CLAUDE.md` end to end, then `ARCHITECTURE.md`, then the
   tooling section above — it is what stops you repeating Track A's
   slowest week.
2. Clone and run the existing tests locally — no GPU needed:
   `python -m tests.test_wrong_way` → expect `17 passed, 0 failed`.
3. Replay a recorded run, also with no GPU:
   `python -m road_crime.replay fixtures/divided_highway_clean.jsonl`
   → 825 frames, 50 tracks, 0 alerts. Then
   `python -m road_crime.evaluate fixtures/` for the corpus score. Seeing
   how fast this is, is the point.
4. Run `notebooks/run_on_colab.ipynb` once on a GPU runtime, all the way
   to an annotated video. You need to have seen the pipeline work before
   you extend it — but note that this is the *only* step that needs a GPU.
5. Read `road_crime/wrong_way_detector.py`. It is the reference shape every module
   follows: config dataclass → per-frame `update()` → state machine →
   alert dict. Diagram 3 in `ARCHITECTURE.md` walks its internal logic.
   The `evidence` block and the peer-agreement rule both exist because of
   specific false positives on real footage; `CLAUDE.md` records which.

**Done when.** Tests pass locally, a fixture replays locally, and you have
produced one annotated video yourself.

### B1. Confirm the STOP sign class id

Same discipline as A1, and read A1 first — it exists because this step
was skipped once already.

- Print every class id RF-DETR emits on footage containing a stop sign
- Cross-check against the model's own class names
- Hardcode nothing until the name is confirmed

**Done when.** The id is documented alongside the name the model gives it.

### B2. Define the stop zone

A region in image space in front of a detected sign where a vehicle is
expected to reach near-zero speed.

Start simple — a fixed-size box below the sign's bounding box, scaled by
the sign's apparent size. This does **not** need to be geometrically
correct for v1. It needs to be **documented as an explicit assumption**
in the module docstring, with its parameters exposed in the config so
they can be tuned rather than rediscovered.

**Done when.** The heuristic and its limitations are written down before
any code depends on them.

### B3. Core logic and synthetic tests

Write the tests alongside the logic, not after.

**State to keep**, per `(track_id, sign_id)` pair:

- minimum speed observed while inside the zone
- whether the pair has already been evaluated (a vehicle that leaves the
  zone is never re-judged, even if it re-enters the frame)

**Violation condition.** The vehicle passes fully through the zone
without its minimum speed dropping below a near-zero threshold for at
least N consecutive frames. Mirror `violation_frames_required` from the
wrong-way module for consistency.

Reuse the heading/speed pattern from `WrongWayDetector._heading()` — copy
it. Duplication across modules is explicitly accepted at this stage
(`CLAUDE.md`, repo conventions); do not build a shared base class.

**Required test cases** (from `CLAUDE.md`):

| Scenario | Expected |
|---|---|
| Decelerates to ~0 inside the zone | no alert |
| Maintains speed through the zone | alert |
| Never enters any zone | never evaluated, no crash |
| Slows but not close to zero | alert — partial stops count |

Add at least one more: the same vehicle re-entering a zone it already
cleared must not produce a second alert.

**Done when.** `tests/test_stop_sign.py` passes with no GPU, no video and no
RF-DETR.

### B4. Wire into the pipeline

Only after B3 is green.

- Stop signs are detected in the same `model.predict` call as vehicles —
  do not add a second inference pass. Perception runs once per frame
  (principle 5) and its output fans out.
- Feed the module structured data only, exactly as `WrongWayDetector` is
  fed.
- Extend the annotated output so stop zones are drawn — you cannot tune a
  zone you cannot see.

**Done when.** A real video produces stop-sign alerts in the standard
payload, with a populated `evidence` block.

### B5. Tune and measure

Same as A5, on your own evaluation clips. Same rule: quieter wins ties.

---

## Track C — Shared

### C1. Evaluation set — the highest-value missing piece

Right now "does it work" is decided by a human watching a video. That is
neither measurable nor repeatable, and it makes A5 and B5 impossible to
do honestly.

**Build:** 10-20 short clips (20-60s), each labelled by hand:

```
clip_04.mp4 | wrong_way   | violation at 00:12, vehicle in left lane
clip_05.mp4 | wrong_way   | no violation
clip_06.mp4 | stop_sign   | violation at 00:31, silver car
```

Cover the hard cases deliberately, not just the easy ones: lane changes,
turns at junctions, the ego vehicle turning, heavy traffic, poor light.
The lane-change clip that produced our false positive belongs in here as
a permanent regression case.

**Split:** each owner collects and labels clips for their own module.
Agree the label format first so the two sets stay comparable.

**Then:** every threshold change is scored — how many real violations
caught, how many innocent drivers accused. Principle 3 demands minimising
false positives, and you cannot minimise what you do not measure.

### C2. Open question — privacy

Real footage contains number plates and faces. While this is a learning
project that is a background concern; with commercial intent it becomes a
legal one.

Not to be solved now. To be **written down as an open question** before a
library of real clips accumulates on someone's laptop: where evidence is
stored, who can access it, how long it is retained, and whether faces and
plates are blurred outside the flagged vehicle.

### C3. Deferred — the other modules

- **Solid/double line.** Has a genuine open dependency: RF-DETR does not
  segment lane markings, and most lane datasets label position but not
  solid-vs-dashed *type*. Worth a research spike early — if no suitable
  pretrained model exists, this module needs labelled data and
  fine-tuning, which is a significant scope change. Finding that out now
  is worth more than finding it out in a month.
- **Red light.** Hardest. Needs a colour-state classifier plus
  light-to-lane association. Build last.
- **Orchestration.** Blocked until two modules run together. Nothing to
  orchestrate with one.

---

## Sequencing

```
Track A:  A1 → A2 → A3 → A4 → A5 → A6
                              ↑
Track C:  C1 (evaluation set) ┘
                              ↓
Track B:  B0 → B1 → B2 → B3 → B4 → B5
                                    ↓
                          Orchestration layer
```

- A and B run fully in parallel. Neither blocks the other.
- C1 blocks the tuning steps of both. Start it early, not when you reach
  A5 and discover you cannot proceed.
- Orchestration is blocked on both A4 and B4.
- Share the A1 finding with Track B immediately — B1 is the same trap.

## Definition of done, per module

A module is finished when all of the following hold:

1. Its own file, its own test file, no shared base class
2. Synthetic tests pass with no GPU, camera or model
3. Wired into `road_crime/pipeline.py`, producing alerts on real video
4. Alerts carry a populated, auditable `evidence` block
5. Thresholds measured against the evaluation set, not guessed
6. Assumptions and known limitations written in the module docstring
7. `CLAUDE.md` status updated to match reality
