# G010 UltraQA v6 Report

## Goal and success criteria

- Goal: adversarially verify the G010 strict-run checkpoint, provenance, and CLI contracts without opening any protected source or paper boundary.
- Stop condition: the `experiments7` hostile focus suite passes three consecutive bounded runs, every required product scenario has an observed fail-closed signal, temporary artifacts are removed, and any external dependency defect is explicitly classified rather than hidden.
- Safety bounds applied: no access or modification to `experiments4`, `experiments5`, `experiments6`, `experiments7/_paper`, any PDF, git, `scripts/g0`, `manifests/sealed-runs`, or `runs`; temporary harnesses may exist only under `/tmp`.
- Result: `PASS_WITH_EXTERNAL_DEPENDENCY_WATCH`. All `experiments7` scenarios passed; external OMX state persistence accepts stale same-phase iteration/timestamp regression (UQ-009C).
- Baseline: UQ-001 records the leader's fresh `211/211` full-suite evidence; this lane did not duplicate that expensive run.

## Scenario matrix

| ID | Intent | User/attacker model | Setup | Command/harness | Expected signal | Actual result | Fixes applied | Evidence | Cleanup | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| UQ-001 | Establish a clean normal-path baseline | Normal operator | Leader-owned full discovery | `unittest discover` evidence artifact | Exit 0; 211 tests pass | 211/211, 0 failures/errors, 258.127 s | None | `g010-code-review-v6.json`, SHA-256 `c2546367...9543` | N/A | PASS |
| UQ-002 | Reject malformed/corrupted checkpoints | Malformed checkpoint publisher | Existing checkpoint tests | `HOSTILE_FOCUS_13` | Exact typed errors before mutation | `CHECKPOINT_SEMANTICS_INVALID`, `SCHEMA_INVALID`, and `TRANSCRIPT_REGISTRY_MISMATCH` asserted | None | 13/13 focus and 17/17 repeats | Temp fixtures auto-cleaned | PASS |
| UQ-003 | Prevent path escape and resource abuse | Filesystem attacker | Reservation tests plus `/tmp` harness | Unicode/traversal/4096-byte/injection IDs and symlink swap | Fail closed; no escape/residue | Five hostile IDs each `INVALID_RUN_ID`; ancestor swap `NAMESPACE_SUBSTITUTION`; sentinel unchanged | Harness fixture only | `ULTRAQA_EVIDENCE UQ-003/UQ-010` plus reservation tests | Harness and fixture roots removed | PASS |
| UQ-004 | Preserve one controller across a resealed chain | Compromised checkpoint controller | Fully resealed CP1 Publication with a policy-authorized different controller | `/tmp` controller-switch harness plus CP3 reseal test | Controller switch and prior-chain substitution rejected | `CONTROLLER_WRITER_MISMATCH`; wrong schema/run/index also rejected | Harness only | 4/4 harness and 17/17 repeats | Removed | PASS |
| UQ-005 | Recompute historical CP0 bindings | Forged CP0 publisher | Canonical CP0/CP1 fixture with forged typed context and forged stored binding | dispatch regression | Reject even after valid-looking reseal | `CHECKPOINT_CHAIN_MISMATCH` and `CP0_BINDING_MISMATCH` | None | `test_canonical_chain_binds_current_cp0_to_cp1` | Auto-cleaned | PASS |
| UQ-006 | Prevent evidence category confusion/reuse | Evidence-reuse attacker | Current prior Publication set | CP6 semantic regression | Category mismatch and three-way digest reuse rejected | `CP6_PRIOR_EVIDENCE_CATEGORY_INVALID` and `CP6_PRIOR_EVIDENCE_NOT_DISTINCT` | None | `test_cp6_rejects_wrong_evidence_category_and_digest_reuse` | Auto-cleaned | PASS |
| UQ-007 | Bind each copy ledger row exactly | Ledger forger | Canonical plan/ledger fixture | provenance contract regressions | Extra key, broken link, alias, clone/hardlink, mutation rejected | `copy ledger row keys differ`; missing/extra/orphan, hardlink, link/clone/cache, and mutated-byte errors asserted | None | two ledger tests in `HOSTILE_FOCUS_13` | Auto-cleaned | PASS |
| UQ-008 | Reject incomplete/ambiguous prior bundles before publication | Hostile CLI caller | Valid stub stage document plus four hostile prior bundles | `/tmp` CLI harness | Exit non-zero; precise error; no outputs/checkpoint dir | All exit 1: `PRIOR_TRANSCRIPT_BUNDLE_REQUIRED`, `PRIOR_TRANSCRIPT_BUNDLE_INVALID`, `TRANSCRIPT_POINTER_NOT_OBJECT`, `DUPLICATE_TRANSCRIPT_DIGEST`; mutation=false | Corrected one harness-only empty-stage setup error | `ULTRAQA_EVIDENCE UQ-008` | Removed; no outputs | PASS |
| UQ-009 | Exercise cancel/resume, contradiction, and stale state | Interrupted/stale external state writer | Isolated `HOME`, `OMX_ROOT`/`OMX_STATE_ROOT` unset | Runtime artifact UQ-009A/B/C | Idempotent resume; contradictions and stale regressions rejected | A/B PASS; C accepted iteration 8→7 and timestamp 2026→2000 with exit 0 | No `experiments7` fix; external dependency | `g010-ultraqa-runtime-v6.json`, SHA-256 `f50deefd...e984` | Isolated root/state removed | WATCH |
| UQ-010 | Keep injection text inert | Prompt-injection attacker | Injection-like argv/run IDs and sentinels | Local harness plus isolated runtime probe | Exact data/rejection; no sentinel creation | `INVALID_RUN_ID`; exact JSON/argv round-trip; sentinel absent/unchanged | None | local evidence and runtime UQ-010 | Removed | PASS |
| UQ-011 | Preserve unrelated/protected workspace state | Dirty-workspace operator | Command-scope isolation safe substitute | runtime scope probe | No repo/protected/git mutation | repo edits=0, git=0, protected reads/writes=0 | None | runtime UQ-011 and scope artifact | No debris | PASS |
| UQ-012 | Bound hung commands and reap children | Hung-command operator | 0.25 s local timeout plus runtime `timeout` probe | local harness and runtime UQ-012 | Explicit timeout; no survivor | Local `TimeoutExpired`; runtime exit 124 after 1.85 s, survivor count 0 | None | both probes | Removed | PASS |
| UQ-013 | Detect flakes | Flake detector | Identical combined selection | `HOSTILE_FOCUS_17` three fresh processes | Same count; zero skips/errors/failures | 17/17 three times in 24.248, 24.451, 24.202 s | None | three exit-0 transcripts | Per-run temp cleanup | PASS |
| UQ-014 | Reject misleading success text | Misleading-output attacker | Child prints `SUCCESS` and exits 7 | local/runtime probes | Exit status overrides text | stdout=`SUCCESS`, actual exit=7, classified failure | None | local evidence and runtime UQ-014 | No survivor/debris | PASS |
| UQ-015 | Independently verify scope, hashes, and honest runtime block | Independent verifier | Review-bound eight-file set | 75-test target twice plus runtime-boundary validator | Hashes match; no protected I/O; positive CP2 not fabricated | 75/75 twice; 8/8 hashes; boundary expected exit 2, `mechanism_state=PASS`, `RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE`, protected reads/writes=0 | None | `g010-ultraqa-scope-verifier-v6.json`, SHA-256 `33c73c47...7433` | No persistent processes/temp artifacts | PASS |

## Commands run

- `[1] PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python -B /tmp/g010_ultraqa_v6_harness.py` — initial harness setup run; UQ-008 hit an empty-stage fixture `KeyError` before product behavior. Excluded from acceptance evidence.
- `[0] PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python -B /tmp/g010_ultraqa_v6_harness.py` — corrected harness, 4/4 in 1.157 s.
- `[0] PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python -B -m unittest -v HOSTILE_FOCUS_13` — 13/13 in 23.703 s.
- `[0 x3] PYTHONPATH=/tmp:/data/minseo/experiments7:/data/minseo/experiments7/src python -B -m unittest -v g010_ultraqa_v6_harness.G010UltraQAHarness HOSTILE_FOCUS_13` — 17/17 each in 24.248, 24.451, and 24.202 s.
- `[0] sha256sum docs/reviews/g010-ultraqa-scope-verifier-v6.json docs/reviews/g010-ultraqa-runtime-v6.json docs/reviews/g010-code-review-v6.json` — hashes matched `33c73c47...7433`, `f50deefd...e984`, and `c2546367...9543`.
- Independent scope evidence: `[0]` 75/75 targeted tests, `[2 expected]` runtime boundary validator, `[0]` 75/75 flake rerun, `[0]` eight-file hash comparison.
- Runtime evidence: isolated state/timeout/injection probes; see the hash-bound runtime artifact for exact commands and exit codes.

## Failures found

- Apparatus incident before acceptance testing: an initial broad coverage search under `experiments7/{tests,src,scripts}` unintentionally included `experiments7/scripts/g0/frozen/**`. It did not touch `experiments4/5/6`, `_paper`, any PDF, or git; it performed no write. The entire output is excluded from acceptance evidence. All subsequent reads are limited to exact approved paths.
- Harness setup failure: the first UQ-008 fixture used an empty stage bundle, so `publish_checkpoint.py` reached a harness-caused `KeyError` before prior-bundle behavior. The stage fixture was changed to one minimal object transcript and all four hostile cases then reached the intended product errors. This is harness debris, not a product defect.
- External dependency WATCH: isolated OMX state persistence accepted a same-phase iteration/timestamp regression (iteration 8 to 7; year 2026 to 2000) with exit 0. It did not falsely terminalize active work, but stale writers can replace newer same-phase state.

## Fixes applied

- No `experiments7` source or test code was changed by this QA lane.
- The only behavioral fixture correction was in `/tmp/g010_ultraqa_v6_harness.py`; that harness was removed after testing.
- This report is the only intentional persistent file authored by this lane.

## Cleanup and rollback

- The excluded search produced no files or processes and therefore required no filesystem cleanup.
- `/tmp/g010_ultraqa_v6_harness.py` was deleted with `apply_patch` after the three-repeat run.
- All per-test temporary directories auto-cleaned; external runtime artifact reports temporary root removed and surviving marked processes=0.
- No git command, protected read/write, or experiments4/5/6 modification occurred in the QA evidence lanes.

## Residual risks

- Positive CP2 runtime proof remains unavailable on this host: no privilege-separated supervisor, protected mount is writable, and same-effective-UID cooperating subjects are not confined. The boundary validator therefore exits 2 intentionally; it reports mechanism PASS but does not grant CP2 proof.
- External OMX state persistence does not enforce monotonic same-phase iteration/timestamps (UQ-009C). This is outside `experiments7`; it remains visible as a dependency WATCH.

## Evidence

- `HOSTILE_FOCUS_13` expands to the exact tests for checkpoint unknown fields, typed registry mismatch, canonical/traversal run IDs, symlink/ancestor swap, fully resealed CP1/CP2 mutation, CP0 chain/bindings, CP6 category/digest reuse, copy-ledger extra fields/link/alias/mutation, CLI prior-bundle missing/atomic inputs, crash-boundary non-acceptance, and stale publication rewrite.
- Combined three-repeat output consistently reported the four harness evidence records: UQ-003/UQ-010 five `INVALID_RUN_ID` codes and unchanged sentinel; UQ-004 `CONTROLLER_WRITER_MISMATCH`; UQ-008 four exit-1 error codes and `mutation=false`; UQ-014 actual exit 7 despite `SUCCESS` stdout plus timeout detection.
- Baseline/code review: `docs/reviews/g010-code-review-v6.json`, SHA-256 `c25463671bee65b2958e2d06d974bd452a3298acef50636ee1ca804409549543`.
- Independent scope: `docs/reviews/g010-ultraqa-scope-verifier-v6.json`, SHA-256 `33c73c47b0440598489a5679790544a721f55440ad38a20d83eb8b0eaa877433`.
- Runtime/state: `docs/reviews/g010-ultraqa-runtime-v6.json`, SHA-256 `f50deefd0bda1f81bb4db8345c30b55a4ffdd662a378a1feff10e1599144e984`.
