"""Road crime detector.

Layout, and why it is this way:

    road_crime/   the code -- one file per violation module, plus the
                  perception and tooling that serve them
    tests/        synthetic-trajectory suites, no GPU or footage
    fixtures/     recorded perception output, replayable forever
    notebooks/    Colab runners
    docs/         architecture and work plan
    CLAUDE.md     the design contract, at the root by convention

A plain package rather than role folders (`detectors/`, `tools/`) because
role folders would need a `sys.path` fix at the top of every file. This
works as-is from the repository root, which is where everything already
runs -- locally, and in Colab after the notebook cds into the clone.

Deliberately no `pip install` step: `wrong_way_detector`, `replay` and the
test suites are standard library only, and must stay runnable on any
laptop with nothing set up (CLAUDE.md principle 6).
"""
