# Non-cheating protocol

1. Candidate design uses only the PReFine paper, copied method code, public
   configuration, structural ID-overlap counts, and already-sealed aggregate
   dev6 outcomes. It does not use dev6 rows, query answers, prediction text,
   or per-slice scores.
2. The validation set is the deterministic `dev_3` subset whose `example_id`
   is absent from `1229_dev_6`; its expected cardinality is 95. The selector
   reads IDs only.
3. The evaluator-owned preparer is the sole code that accesses target-bearing
   source fields. It emits only sanitized history/public tasks to the runner
   and writes target calls only to `evaluator_vault/gold.jsonl`.
4. GPU memory construction and both action arms run without a gold import,
   gold path, or evaluator API. Each arm gets exactly one action call per
   public case under the same seed/model/schema/budget.
5. The runner seals prediction hashes before evaluation. The evaluator checks
   that receipt, opens gold once, emits aggregate metrics only, and compares a
   registered implementation with an independent counting implementation.
6. Any completed evaluation, pass or fail, ends tuning for this candidate.
