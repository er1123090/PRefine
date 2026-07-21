# mix600-v1 dataset contract

`scripts/prepare.py` is the only canonical materialization entrypoint. It uses
the pure builder under `src/exp7/datasets`; active preparation does not import or
execute historical experiment runners.

## Inputs and outputs

The 600-row mix600 source remains external by default at
`/data/minseo/experiments4/data/mix600.json`. The source and all small query,
preference, schema, and config inputs are admitted only when their byte size and
SHA-256 match the values under `configs/`.

The current output boundary is transitional:
`artifacts/prepared/mix600-v1`, not the audited top-level `data/` tree. It
contains:

| Turn | Easy | Medium | Hard | Total |
| --- | ---: | ---: | ---: | ---: |
| singleturn | 554 | 293 | 472 | 1,319 |
| multiturn | 554 | 293 | 472 | 1,319 |
| all | 1,108 | 586 | 944 | 2,638 |

Each condition is a JSONL file under `<turn>/<difficulty>.jsonl`. The directory
also contains `manifest.json` and `validation_report.json`.

## Query and ground-truth semantics

- Easy projects preference slots from each source `api_calls` entry through
  `pref_list.json`.
- Medium applies supported `pref_group.json` rules to matching preference
  evidence domains and slots.
- Hard transfers those rules to eligible domains not used by the evidence.
- Single-turn uses the domain query string. Multi-turn selects the template with
  the largest target-slot overlap and merges its base API arguments with the
  preference arguments.

The supported medium/hard groups are `low_cost` (93 annotations), `high_cost`
(4), and `solo_usage` (39). The source also contains `eco` (130) and
`prefers_star` (468); because these groups have no configured rules, they are
reported as ignored and generate no medium/hard instances. This leaves 123
source examples supported for medium/hard and 477 unsupported-only examples.

## Identity

`instance_id` is canonical and globally unique:

```text
mix600-v1:<turn>:<difficulty>:<source-example-id>:<semantic-sha256>:<occurrence>
```

It is condition-scoped and content-derived. `legacy_example_id_sub` preserves
the old `<example_id>_<sub-index>` shape only for compatibility; positional
legacy IDs are not semantic identity and must not be used to join new results.

## Validation boundary

Generation parity retains, rather than hides, 463 GT/schema warnings:

| Scenario | Warning instances |
| --- | ---: |
| singleturn easy / medium / hard | 39 / 12 / 27 |
| multiturn easy / medium / hard | 63 / 129 / 193 |

The missing single-turn slot is `number_of_seats`; the missing multi-turn slots
are `city`, `departure_date`, `destination`, and `pickup_time`. The 25
multi-turn templates have zero base/preference slot conflicts. These are
evaluation-boundary warnings, not builder errors, so GT is not silently removed
or rewritten.

For the frozen historical evidence and regeneration tests, see
[the legacy query/GT contract](mix600-legacy-contract.md) and
[the legacy validation audit](mix600-legacy-validation.md).
