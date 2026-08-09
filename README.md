# Road crime detector

Detects traffic-law violations from a forward-facing car dashcam. The
system **documents; it does not judge** — every alert carries a confidence
score and an evidence block a human can audit, never a bare verdict.

`CLAUDE.md` is the design contract and the place to start.
`docs/WORKPLAN.md` says who is building what.

## Layout

```
road_crime/    code: violation modules, perception, tooling
tests/         synthetic-trajectory suites; no GPU, no footage
fixtures/      recorded perception output, replayable forever
notebooks/     Colab runners
docs/          architecture and work plan
```

## Running it

Everything runs from the repository root. There is no install step: the
detector, the replay tool and the test suites are standard library only,
by design.

```bash
python -m tests.test_wrong_way                    # 17 tests, milliseconds
python -m road_crime.evaluate fixtures/           # score the whole corpus
python -m road_crime.replay fixtures/divided_highway_clean.jsonl
python -m road_crime.replay fixtures/divided_highway_clean.jsonl --track 7 --verbose
```

Only the perception stage needs a GPU, and only once per clip. Open
`notebooks/run_on_colab.ipynb` in Colab, which clones this repository and
writes a track dump — every track's id, road-contact point and apparent
width, per frame. That dump is the entire input surface a violation module
has, so everything downstream replays from it locally in under a second.

```bash
python -m pyflakes road_crime/*.py tests/*.py     # before every push
```

## How it decides

No GPS and no map. The image is divided into zones, and each zone learns
its normal direction of travel from the traffic that passes through it. A
vehicle is reported only when it disagrees with **both** that learned
baseline **and** the traffic travelling alongside it at that moment, for
ten consecutive frames.

The peer condition is not decoration. On a dashcam the apparent flow
reverses whenever the ego vehicle turns, and without it every vehicle
present gets accused of contradicting a direction learned a minute
earlier. Guilt has to be established against the present, not the
archive.

## Measuring it

Two numbers, neither needing a labelled dataset:

- **False positives.** Ordinary driving contains no wrong-way driving to
  any useful approximation, so every alert raised on ordinary footage is
  one that should not have been raised.
- **Sensitivity.** A real trajectory replayed backwards is a vehicle
  travelling against its own lane, at a realistic speed in a real scene.

They combine into one deliberately asymmetric score:

```
loss = 10 * false_positives + missed_injections
```

That weight is the project's error-asymmetry principle written as a
number: accusing an innocent driver is worse than missing a violation.
Stated explicitly, it becomes something to argue about and change on
purpose rather than something hidden inside a threshold.
