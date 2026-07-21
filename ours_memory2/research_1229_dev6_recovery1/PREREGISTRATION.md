# ECPR on 1229_dev6 — preregistration

Locked on 2026-07-16 before this research package read or computed any target
metric, per-example target error, expected API call, or gold preference value.

## Confirmatory claim

The sole confirmatory candidate is `ecpr_v1` (Evidence-Calibrated Preference
Routing). It is compared pairwise with the source-faithful standalone
PREFINE/`ours_memory2` baseline on one frozen single-turn and multi-turn case
universe. Both arms use the exact same case keys, target request, action prompt
template, tool schema, cached Qwen snapshot, seed, decoding settings, maximum
generated tokens, retry policy, and one action-model call per case. Only the
serialized memory block differs.

The primary endpoint is task-balanced pooled micro-F1:

`BMF1 = 0.5 * pooled_micro_F1(singleturn) + 0.5 * pooled_micro_F1(multiturn)`.

## Frozen inputs

- history/evaluator source: `/data/minseo/experiments4/data/1229_dev_6.json`
  (`911f37c195c52722ecf85a23bfa55be22c4c843112f49c633267f9d09b5d64ae`)
- single query: `/data/minseo/experiments5/config/query_singleturn.json`
  (`e8a25828cf319e63c573fbcb8a485e7ece1ed3a7d49120d41ac5d716adc897c7`)
- single schema: `/data/minseo/experiments5/config/schema_easy.json`
  (`ef388951dcff703e3f3a41be0a20315e54125c43e885dca6aeb85e6fbd77dfb9`)
- multi query: legacy 13-domain
  `/data/minseo/experiments5/config/query_multiturn-domain.json`
  (`745f578f71b63c38fd052904a0a9e5a7601df6e9e52c1335ab251fc8d720c627`)
- multi schema: `/data/minseo/experiments5/config/schema_all.json`
  (`c3639e58e42c8cbe2f202ea9798dd158da305da8229614e3db09f829b945628f`)
- preference slots: `/data/minseo/experiments5/config/pref_list.json`
  (`f5994dab209909ed31d15cbabe6a88a84ef5d95f85c156795549c96fd5126091`)
- preference groups: `/data/minseo/experiments5/config/pref_group.json`
  (`aac0cc9210a59b15e1422d0946bab0e37c9c38b538f29a2d7c24b84ccd8348fb`)
- action model: `Qwen/Qwen3-8B`, cached snapshot
  `b968826d9c46dd6066d109eabc6255188de91218`

The multi-turn condition above is not the distinct 25-template
`query_multiturn.json` condition.

## Information firewall

The memory constructor receives only a separately written, validated history
JSONL. Its allowlist is `example_id`, chronological
`sessions[].dialogue[].role`, `sessions[].dialogue[].message`, and
`sessions[].api_call[]`. Nonsemantic metadata is empty by default. It never
receives top-level `api_calls`, `api_calls_pref`, query/template material,
expected/reference/gold/ground-truth fields, `pref_group`, `value_group`,
`difficulty`, or evaluator/verifier output.

An evaluator-only builder may read the frozen sources to create two sealed
artifacts: a gold-free inference task stream and a separate gold vault. The
inference process accepts only the task stream, schema bundle, memories, and
run contract. It has no gold-vault argument. Predictions contain no query,
prompt, reference, or gold. Gold is joined only after prediction files close,
in a separate evaluator process.

Case IDs are opaque SHA-256-derived keys. Examples and memories join only by
normalized `example_id`; query templates join by domain/slot. `query_id` is
never joined to `example_id`.

## Candidate fixed before evaluation

ECPR combines the baseline's verified latent abstraction with deterministic
typed evidence from historical preference-compatible API slots. It retains
multiple independent value hypotheses per `(domain, slot)`. Every hypothesis
contains support count, counterevidence count, last-seen session, confidence,
and digest-only provenance. Raw historical API text is forbidden in the
candidate action memory.

Routing is fail-closed:

- keep only the current target domain and slots present in its current schema;
- keep only preference-compatible slots;
- current explicit/non-preference values override memory;
- apply memory only to missing preference slots;
- minimum support: 2;
- minimum confidence: 0.67;
- abstain when competing-value conflict ratio exceeds 0.34;
- recency breaks ties but cannot create support;
- at most 24 stored hypotheses, 6 routed hypotheses;
- routed overlay cap: 384 deterministic lexical tokens;
- complete memory block cap for both arms: 1,536 lexical tokens.

Registered diagnostic ablations are `no_latent`, `no_typed`,
`no_counterevidence`, and `no_routing`. They are exploratory and cannot replace
`ecpr_v1` as the confirmatory candidate after target results are observed.

## Equal inference budget

- temperature: 0
- seed: `2026071600`, deterministically mixed with the opaque case key
- maximum action output: 1,024 tokens
- action calls: exactly 1 per case per arm
- retries: 0
- provider errors: recorded once and scored as empty predictions
- model snapshot, prompt, schema, timeout, and call order are recorded in the
  run manifest and must match across arms

## Fail-closed evaluation

The paired key universe is frozen before action inference. Duplicate keys fail.
Every expected key must occur exactly once in each arm. Missing, error,
malformed, and unparsable outputs are scored as empty predictions; they are
never removed from the denominator. Recorded coverage must be 100%.

Guardrails, in absolute F1-rate units:

- each single-turn and multi-turn delta is at least `-0.005`;
- preference-slot pooled micro-F1 drop is at most `0.005`;
- non-preference pooled micro-F1 drop is at most `0.005`;
- parse-failure-rate increase is at most `0.005`;
- paired coverage is exactly 1.0 and action budgets are equal.

Statistics cluster by `example_id`: paired cluster bootstrap, 10,000 draws,
seed `2026071601`; and one-sided cluster sign-flip randomization, 100,000 draws,
seed `2026071602`. PASS requires all guardrails, `delta BMF1 >= 0.010`, bootstrap
95% CI lower bound strictly greater than zero, and randomization `p < 0.05`.

Development stops after 20 registered candidates or 5 consecutive gains below
`0.0025` BMF1, whichever occurs first. This lock initially registers one. Final
test is run exactly once after all code/config/model/budget hashes are frozen.

If the cached model cannot run safely, no target metric, CI, p-value, PASS, or
superiority claim will be fabricated. The deliverable is then limited to
implementation, unit/static/smoke/leakage evidence and an explicit GPU blocker.
