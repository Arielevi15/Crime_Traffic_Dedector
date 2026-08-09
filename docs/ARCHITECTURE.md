# Architecture diagrams

These diagrams use Mermaid syntax — plain text that Claude Code can parse
directly as structure (nodes and edges), and that also renders visually in
GitHub, VS Code preview, and most markdown viewers. This file is a
companion to `../CLAUDE.md`, not a replacement — read that first for the
full task lists and design principles.

## 1. Data flow

How a single frame moves through the system, from camera to API alert.

```mermaid
flowchart TD
    A[Dashcam video frame] --> B[RF-DETR: vehicle + sign + light detection]
    B --> C[ByteTrack: assigns a stable track_id per vehicle]
    C --> D[Wrong-way module]
    C --> E[Stop-sign module]
    C --> F[Solid-line module]
    C --> G[Red-light module]
    D --> H[Orchestration layer]
    E --> H
    F --> H
    G --> H
    H --> I[API alert: evidence + confidence]
```

Key point encoded in this diagram: detection and tracking (B, C) run once
per frame and feed all four modules in parallel — never duplicate the
RF-DETR/ByteTrack call inside a module.

## 2. Module build order and current status

```mermaid
flowchart LR
    A[Wrong-way<br/>TRACK A<br/>runs end to end<br/>false positive open] --> C[Solid-line<br/>NOT STARTED<br/>open dependency]
    B[Stop-sign<br/>TRACK B<br/>NOT STARTED] --> C
    C --> D[Red-light<br/>NOT STARTED]
    A --> E[Orchestration<br/>BLOCKED<br/>needs A and B done]
    B --> E

    classDef partial fill:#fff3cd,stroke:#8a6d1a,color:#1b1b1b
    classDef todo fill:#eeeeee,stroke:#757575,color:#1b1b1b
    class A,B partial
    class C,D,E todo
```

Wrong-way and stop-sign are now built **in parallel by two developers** —
see `WORKPLAN.md` for ownership and acceptance criteria. They touch
disjoint files, which is what principle 5 buys us.

This is a **suggested build order, not a hard dependency** — each module's
code is independent (see `../CLAUDE.md` principle 5). The order reflects
implementation difficulty (wrong-way is self-calibrating and vision-only;
red-light needs a secondary color classifier and light-to-lane
association) and the fact that the orchestration layer has nothing to do
until at least two modules exist. If Claude Code is asked to work on a
later module before an earlier one is finished, that's fine — the ordering
is a recommendation, not a blocker.

## 3. Wrong-way module internals

The one module that's actually built — useful as a concrete reference
before implementing the others, since they follow the same shape
(config -> per-frame update -> state machine -> alert).

Two ordering details in this diagram are load-bearing, and both exist to
stop a vehicle from rewriting the rule it is judged by:

- **F before G** — the baseline is read before this vehicle votes, so in
  an empty zone a wrong-way driver cannot define the baseline
  single-handedly and clear itself.
- **The V branch** — a track already mid-violation stops voting
  altogether. Without it, a sustained wrong-way run drags the baseline
  round to face it within a few frames, well before
  `violation_frames_required` is reached, and the alert never fires.

```mermaid
flowchart TD
    A[WrongWayDetector.update called once per vehicle per frame] --> B[Append position to TrackState.positions]
    B --> C{Enough history for heading_window?}
    C -->|No| Z[Return None]
    C -->|Yes| D[Compute heading vector]
    D --> E{Speed above min_speed_px?}
    E -->|No| Z
    E -->|Yes| F[Capture trusted + baseline_dir from CURRENT zone state, before this vehicle's own vote]
    F --> V{Is this track already mid-violation?}
    V -->|Yes, streak > 0| H2[Skip the vote entirely]
    V -->|No, streak == 0| G[DirectionBaseline.update: this vehicle votes, for future frames only]
    G --> H2
    H2 --> H3{Was zone trusted, per F?}
    H3 -->|No| Z
    H3 -->|Yes| H[Compare heading to captured baseline_dir via cosine similarity]
    H --> I{Below opposite_cos_threshold?}
    I -->|No| J[Reset violation_streak to 0] --> Z
    I -->|Yes| K[Increment violation_streak]
    K --> L{streak >= violation_frames_required AND not already_reported?}
    L -->|No| Z
    L -->|Yes| M[Mark already_reported, return alert dict]
```
