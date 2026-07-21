# mix600 Legacy Query/GT Contract

This document freezes the experiments4 VanillaLLM instance-generation behavior used as the regression baseline for the experiments7 cleanup. The machine-readable contract is `tests/fixtures/dataset/mix600_legacy_contract.json`; the test regenerates every instance by executing the selected function definitions from the hashed legacy builders.

## Frozen totals

| Turn | Easy | Medium | Hard | Total |
| --- | ---: | ---: | ---: | ---: |
| singleturn | 554 | 293 | 472 | 1,319 |
| multiturn | 554 | 293 | 472 | 1,319 |
| all | 1,108 | 586 | 944 | 2,638 |

Easy instances come from 335 of the 600 source examples. Medium and hard instances each come from the same 123 source examples. The fixture records the SHA-256 and byte size of mix600, both query files, both preference files, both schemas, and both legacy builders.

## GT construction

- Easy reads each source `api_calls` entry, retains only slots declared for that domain in `pref_list.json`, and emits their Cartesian-product API calls. Single-turn uses the domain string from `query_singleturn.json`. Multi-turn selects the query template with the largest target-slot overlap and merges its base API arguments into every GT call.
- Medium reads `api_calls_pref`. For each supported `value_group`, it follows every evidence `(domain, slot)` and collects every matching rule value from `pref_group.json`. It emits one query/GT group per evidence domain. Multi-turn adds the selected template's base arguments.
- Hard reads the same preference evidence but transfers the value-group rules to every rule domain absent from the evidence domains. It emits all rule slots and values for each such domain. Multi-turn again adds template base arguments.

Only `low_cost`, `high_cost`, and `solo_usage` exist in the frozen `pref_group.json`. mix600 also contains 468 `prefers_star` and 130 `eco` preference annotations; those annotations generate no medium/hard instances under this baseline.

## Identity and canonicalization

The legacy runner writes `example_id_sub = "{example_id}_{sub_idx}"`. All IDs are unique inside each scenario, and the fixture locks the exact ID multiset. It does not treat the ID-to-record mapping as stable: the medium builder turns values into sets, and the hard builder stores candidate domains in a set, so Python hash seed can change their iteration order and therefore change `sub_idx` assignment.

The semantic regression digest normalizes this implementation accident. It sorts each GT alternative list, serializes `(example_id, query, ground_truth)` with sorted JSON keys, sorts the complete record list while retaining duplicates, and hashes it with a final newline. A replacement builder must match this semantic multiset and should introduce a content-derived stable ID rather than inherit positional `sub_idx` identity.

This contract proves query/GT generation parity only. Schema/tool compatibility and evaluator behavior are separate validation gates and must not be inferred from these passing tests.

## Verification

Run:

```bash
python -m unittest tests.dataset.test_mix600_legacy_contract -v
```
