# ADR 0001: Versioned strict evidence runs

Status: Proposed, revision 2 awaiting independent critic approval

## Context

The existing canonical evidence paths were published under a relaxed contract:

- `paper_outputs/admission/precopy` reports `terminal_state=PASS` while
  `unresolved_count=238`.
- `paper_outputs/raw/verified` and `manifests/copies.jsonl` were populated
  before the required zero-unresolved CP3 gate.
- the CP0 seal and G3/copy artifacts do not share one sealed-run identity.
- 34 Table 10 admissions inherit an incomplete one-parent raw union.
- one copied raw artifact supports only declared-drift results and therefore is
  not exact-only evidence.

The last two findings are reproducible from the immutable audit
`paper_outputs/admission/audits/legacy-v1-defects/defects.jsonl` (SHA-256
`eafde9d16051be316cbdb8240d01e417f5177aeec0c03a4930c93106c38117cf`)
and its `summary.json` (SHA-256
`c105dfd897d0be638e2c1406812e4e61d8885e2a101d1f39e12b6b7a85dd2ec4`).
The 238-result classification is preserved in
`paper_outputs/admission/audits/unresolved-v1/results.jsonl` (SHA-256
`d408eef89d648ce2f03377b17f94f707aaeb68ad56174343c9a2c74414d58d09`)
and `summary.json` (SHA-256
`55c30778cd2f0e25d19ab16d97c87fe02d79ba5d7ec301a518f0525e563c3733`).

Those artifacts are immutable historical evidence. The no-replace contract
forbids editing, replacing, renaming, deleting, or reusing their precopy seal
as a strict final result.

## Scope

This ADR changes only the canonical publication location from occupied legacy
paths to a fresh, versioned strict-run root. Every other v4 MUST and test
remains binding, including zero unresolved results, exact typed provenance,
immutable publication, separated single writers, and the CP0 -> CP6 order.
A reviewed plan-and-test amendment that binds the new run-root contract is a
prerequisite to starting any strict run; this ADR alone authorizes no protected
read, copy, publication, or checkpoint transition.

## Decision

1. Preserve every existing canonical admission, provenance, copy, and raw
   artifact byte-for-byte. Treat them as `legacy_relaxed_v1`, never as strict
   release evidence.

2. Every strict attempt reserves exactly one fresh path:

   `paper_outputs/strict-runs/<sealed_run_id>/`

   `sealed_run_id` is canonical ASCII matching
   `^exp7-strict-v[0-9]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$`; normalization or
   case folding is forbidden. Reject an absolute path, slash, `.` or `..`
   component, NUL, symlink component, or noncanonical spelling. Open the
   trusted `strict-runs` directory using no-follow directory semantics, create
   the single child atomically with exclusive `mkdir` relative to that
   directory descriptor, and verify by descriptor plus canonical real path
   that it remains beneath `strict-runs`. An already existing child blocks the
   attempt regardless of its contents; it is never adopted, completed, or
   replaced.

3. A strict run owns all evidence beneath that path:

   - `manifests/{source-pre.jsonl,paper-pre.json,source-post.jsonl,paper-post.json}`
   - `checkpoints/cp0..cp6`
   - `inventory`
   - `provenance`
   - `admission/precopy`
   - `raw/verified`
   - `copies.jsonl`
   - `admission/final`
   - `verification`

   Every artifact, including checkpoint seals, graph, ledger, copied raw,
   precopy, final admission, and verification evidence, is append-once and
   atomically published with no replacement. Precopy and final admission are
   distinct immutable artifacts. Preserve the v4 single-writer split across
   the entire run root: G0 owns only pre/post manifests and envelope evidence;
   Registry owns only the registry subtree; Inventory owns only the inventory
   subtree; Adapter/Snapshot owns only adapters, snapshots, and golden evidence;
   Provenance owns only graph/recomputation evidence; Admission owns only
   precopy/final validator output; Copy owns only copied raw plus the copy
   ledger; Verification owns only verification evidence; and the Checkpoint
   controller owns only checkpoint seals. Each lane publishes only its own
   subtree. The Checkpoint controller references hashes already published by
   other lanes and may neither generate nor modify their results.

4. The same `sealed_run_id` and canonical run root bind every phase. Each
   checkpoint seals the immediately preceding checkpoint hash and only the
   artifacts newly accepted at that stage:

   - CP0: source-pre, paper-pre, G0 envelope, frozen tool bundle, and CP0
     configuration/validator hashes;
   - CP1: registry and sealed inventory, plus the CP0 seal hash;
   - CP2: adapters, snapshots, and golden evidence, plus the CP1 seal hash;
   - CP3: provenance graph, recomputation evidence, and precopy admission,
     plus the CP2 seal hash;
   - CP4: copied raw artifacts and copy ledger, plus the CP3 seal hash;
   - CP5: source-post, paper-post, and final admission, plus the CP4 seal hash;
   - CP6: verification reports and final seal, plus the CP5 seal hash.

   This forms one CP0 -> CP6 hash chain. Mixed-run, stale, implicit-latest, or
   unsealed inputs fail closed.

5. CP3 binds the result count to the SHA-256 digest of the sealed paper
   inventory; `1,908` is accepted only when both the sealed count and digest
   match the CP0/CP1 chain. The Admission validator is the sole state producer
   and emits only the three v4 states: `UNRESOLVED`, `ADMITTED_FOR_COPY`, and
   `VERIFIED` (with `VERIFIED` valid only in final admission). CP3 passes only
   when every sealed inventory result is `ADMITTED_FOR_COPY`, unresolved is
   zero, and every result has exact recomputation plus typed
   producer/config/script/log/raw provenance closure. A declared drift,
   derived-only result, ambiguous candidate, missing source-baseline path,
   unrelated contributor, incomplete producer graph, or incomplete contributor
   set is `UNRESOLVED` and blocks the entire run.

6. Copy runs only after the immutable CP3 precopy seal passes. Its sources are
   exclusively protected experiments4/5/6 paths present in and byte-identical
   to the run's G0 `source-pre` seal. Existing experiments7 legacy copies may
   not be used as a copy source, hardlink, reflink, cache, or provenance
   substitute. Copy admits exactly the contributor set named by the sealed
   graph, creates one physical strict-run copy per raw node, and records
   `copied_artifact` nodes and `copied_to` edges that close every admitted
   result-to-source-to-destination path.

7. Final admission passes only when every sealed inventory result is
   validator-derived `VERIFIED`, `verified_count == total_result_count > 0`,
   and `unresolved_count == 0`. CP5 requires `source-post` and `paper-post` to
   equal the complete CP0 pre-baseline record sets exactly in path, type, bytes,
   symlink target, and all declared metadata; a summary or baseline hash alone
   cannot replace this record-by-record comparison. CP6 requires verifier,
   code-reviewer, and adversarial QA to independently PASS against the same
   immutable CP5 seal. Those three reviewers execute with experiments7 mounted
   read-only and complete/hash their reports in external staging. Only the
   designated Verification writer may atomically publish those exact reports,
   no-replace, into the run-local `verification` subtree; only the Checkpoint
   controller may reference their hashes and publish the CP6 seal.

8. Protected experiments4/5/6 and the PDF may be read only inside a newly
   established OS-enforced read-only G0 boundary. They are never repaired,
   rebaselined to hide drift, or modified.

9. Failure at any phase publishes an immutable `BLOCKED` terminal record for
   that run. A retry uses another fresh `sealed_run_id`; paths are never
   recycled and partial runs are never completed in place.

## Consequences

- The current canonical artifacts stay available for forensic comparison but
  cannot support a completion claim.
- G0 through G6 tools must accept an explicit strict run root, enforce
  descriptor-based containment/no-follow/no-replace rules, and reject canonical
  legacy paths in strict mode.
- A reviewed versioned-path plan/test amendment must exist before a strict run
  starts; it must inherit the entire v4 suite and add namespace, chain, and
  legacy-rejection tests from this ADR.
- Until one complete strict run reaches CP6, the project status remains
  BLOCKED, even if facade metadata tests or legacy copy-integrity tests pass.
