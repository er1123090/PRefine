# ours-memory3

`ours-memory3` is a standalone extension of the two-stage `ours_memory2`
method.  It preserves the PReFine-style latent-memory baseline and appends a
small **query-conditioned, evidence-calibrated preference overlay** only when
public session history supplies a safe, repeated preference fact.

The core package has no runtime import, path lookup, or resource dependency on
`ours_memory`, `ours_memory2`, or any prior experiment directory.
`eval/prepare_1229.py` is a one-time, explicit evaluator migration utility:
it accepts a caller-supplied sealed source root and creates an isolated
ours_memory3 evaluator root that no longer needs that source.

## Hypothesis

PReFine's free-form latent memory can correctly express broad preferences but
can be hard for an action model to ground in a concrete current-tool slot.
Repeated API choices provide a complementary, high-precision signal.  The
candidate therefore keeps the exact baseline memory and appends a fact only
when all of these conditions hold:

1. The value is observed in at least two independent sessions.
2. It is the unique, high-consensus value for a declared preference slot.
3. The current public query selects that schema domain.
4. The user has not explicitly supplied that slot or a known slot value now.
5. The fact fits the compact overlay cap.

If no fact survives, `ours_memory3` returns the baseline memory byte-for-byte.
This is intended to increase useful slot grounding while avoiding negative
transfer, explicit-query override, and prompt bloat.  The action prompt,
model, decoding, and one-action-call budget are shared by baseline and
candidate arms.

## Public-only input

The overlay accepts a JSONL stream of public cases.  Each row has:

```json
{
  "case_id": "opaque-case-id",
  "baseline_memory": "PReFine latent memory plus historical API calls",
  "history": {"sessions": [{"api_calls": ["Restaurant(price=\"cheap\")"]}]},
  "query": "Find a restaurant",
  "mode": "singleturn",
  "schema_domain": "Restaurant",
  "preference_slots": {"Restaurant": ["price"]},
  "schema": []
}
```

Rows containing answer, gold, label, reference, or prediction keys are
rejected before processing.  The package never needs those fields.

```bash
ours-memory3 overlay build --input public_cases.jsonl --output candidate_memory.jsonl
```

The output contains only the candidate memory, selected public domain, and
safe fact metadata.  It never serializes raw dialogue, raw API calls,
per-session provenance, source literals, answers, or model predictions.

## Evaluation

`eval/run_paired.py` is a sealed paired-evaluation runner.  It constructs
baseline and candidate prompts from the same public cases, makes exactly one
action call per arm/case, verifies the pre-gold pairing receipt, and opens the
evaluator-owned gold vault only afterwards.  It reports aggregate metrics only.

The target 1229_dev6 final command is deliberately one-shot:

```bash
cd /data/minseo/experiments4/ours_memory3
PYTHONDONTWRITEBYTECODE=1 python3 -B eval/run_paired.py \
  --root /data/minseo/experiments4/ours_memory3/eval_1229_dev6_v1 --final
```

The final evaluator passes only if both single-turn and multi-turn exact match
improve by at least 0.5 percentage points over the paired PReFine baseline.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p 'test_*.py' -v
```
