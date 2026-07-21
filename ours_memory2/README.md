# ours-memory2

`ours-memory2` is a self-contained, standard-library reproduction of the
two-stage preference-memory method. It has no runtime dependency on another
project tree and does not resolve resources relative to the current directory.

## Inputs

Step 1 accepts strict UTF-8 JSONL. Each nonblank line is one object with a
nonempty scalar `example_id` and nonempty ordered `sessions`; every session has
nonempty ordered `dialogue` (`role`/`message`) and a required string
`api_call` list, which may be empty. The complete file is validated before any
provider is constructed or output is created.

Step 2 accepts a version-1 JSON manifest whose relative resource paths are
resolved against the manifest directory:

```json
{"version":1,"examples":"examples.json","memories":"memories.jsonl",
 "single_query_map":"single.json","multiturn_templates":"multi.json",
 "preference_slots":"slots.json","preference_groups":"groups.json",
 "tool_schema":"tools.json"}
```

All seven resources, shapes, normalized identities, joins, referenced domains,
groups, slots, templates, and the nonzero derived-case set are validated before
provider construction. Raw example API calls are parsed at their exact logical
path (`resources.examples[E].api_calls[A]`), so malformed envelopes or
arguments fail before provider construction or output creation.

## CLI

```text
ours-memory2 step1 build --input EXAMPLES.jsonl --output-root DIR \
  --endpoint URL --model MODEL [--key-env ENV_NAME] [--timeout SECONDS]

ours-memory2 step2 single --manifest MANIFEST.json --difficulty all \
  --context memory_api --output-root DIR --endpoint URL --model MODEL

ours-memory2 step2 multi --manifest MANIFEST.json --input-shape auto \
  --difficulty all --context memory_api --output-root DIR \
  --endpoint URL --model MODEL
```

`python -m ours_memory2` exposes the identical parser. `--timeout` must be
positive and finite. The optional `--key-env` names an environment variable;
secret values are never command arguments or diagnostics. The bundled HTTP
adapter accepts explicit loopback HTTP endpoints, performs no implicit retry,
and is constructed only after command inputs pass validation.

## Difficulty and contexts

`easy` derives cases from ordered `api_calls`; `medium` and `hard` derive from
ordered `api_calls_pref` evidence and preference groups. `all` emits easy,
medium, then hard. Null hard-rule values are skipped (including the all-null
zero-case failure); medium retains the source's separate value behavior. The
four context values are:

- `memory_only`: preference;
- `memory_api`: preference plus accumulated API history;
- `memory_diag`: preference plus prior dialogue;
- `api_only`: accumulated API history without preference or dialogue.

Every context also includes the tool schema and current utterance.

Step 2 bundles the source action-inference methodology: schema-only slot
filtering, evidence/relevance reasoning, cross-domain preference reasoning,
anti-hallucination constraints, the exact output shape, and final-API-only
instruction. Each inference call sends exactly one `user` message containing
the full prompt and omits both `temperature` and `response_format`. Step 1 is
intentionally distinct: generation/refinement and verification retain their
system/user roles, source temperatures, and JSON response intent. A consumed
Step 1 slot refines only when both a parsed prior draft and nonempty verifier
feedback exist; otherwise it retries the initial generation role.

## Outputs and exit codes

All writes are beneath the validated explicit output root. Step 1 appends one
`memories.jsonl` row per completed example plus correlated `drafts.jsonl` and
`verifiers.jsonl` attempt rows. Step 2 writes correlated `results.jsonl` and
`diagnostics.jsonl` only after every selected provider call succeeds. Each
Step 2 diagnostic includes redacted request provenance (model, purpose,
ordered roles/messages, prompt, temperature presence/value, and JSON intent)
alongside request ID, usage, and raw-response diagnostics. Redaction is
recursive and token-aware for nested/compound credential keys such as
`aws_access_key_id`, while benign keys such as `monkey` are retained.

- `0`: success or help;
- `1`: provider/transport or output-processing failure;
- `2`: argument, input, manifest, join, context, difficulty, or path failure.

Provider failures do not write the current Step 1 example or any current Step 2
run rows. Help is side-effect free, and commands never change the process CWD.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v

OURS_MEMORY_REFERENCE_ROOT=/path/to/read-only/ours_memory \
REQUIRE_REFERENCE_DIFFERENTIAL=1 PYTHONHASHSEED=0 \
PYTHONDONTWRITEBYTECODE=1 \
python3 -m unittest -v tests.test_differential_reference
```

The reference root is accepted only by subprocess development probes under
`tests/`; it is never read, imported, or packaged by runtime code. Declared
adapter departures remain limited to `intentional_normalizations.json`.
