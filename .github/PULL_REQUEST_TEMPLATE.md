<!--
  Per-task PR template — DEVPLAN.md §8.
  One PR per task; <400 LOC diff.
-->

## Task ID
<!-- e.g. T0.1, T1.4, T3.6 -->

## Goal (copied from plan)
<!-- Paste the relevant task description from docs/DEVPLAN.md verbatim. -->

## Acceptance tests run
<!--
  Paste the exact pytest invocation and its summary line, e.g.

      pytest tests/test_masking.py tests/test_loss.py -q  ->  PASS (12/12)

  If the task has additional gates (lint/typecheck, scalar metric thresholds,
  Playwright e2e) list them too, with their pass/fail line.
-->

## New / changed files
<!--
  - path/to/file  (new|modified, ~LOC)
-->

## Metrics / artifacts produced
<!--
  - none, OR
  - artifacts/<name>.png  (1-line interpretation)
  - runs/<id>/metrics.json (key numbers)
  - W&B run URL
-->

## Open questions for human
<!--
  - none, OR
  - explicit questions; if there are any, this PR likely should not merge yet.
-->

## Next task
<!-- e.g. T0.2 -->
