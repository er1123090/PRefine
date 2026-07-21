# Integration tests

Tests in this directory will exercise complete prepare, run, and evaluate
flows with deterministic no-network fixtures.

`test_run_suite_artifacts.py` injects a fake condition runner to lock the run
tree, atomic status transitions, verified manifest snapshot, no-overwrite and
resume behavior, failure state, and side-effect-free dry runs without API use.
