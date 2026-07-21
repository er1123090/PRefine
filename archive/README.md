# Archive boundary

`archive-map.json` and [archive-map.md](archive-map.md) are non-destructive
planning indexes for historical sources, runtime evidence, and paper outputs.
They authorize no move, deletion, or externalization. Active modules under
`src/exp7` must never import this boundary or the historical implementation
trees it describes.

Verify that the checked-in plan still matches the tree:

```bash
python -B scripts/archive_map.py --check
```

`paper_outputs/raw` remains blocked from externalization until an independent
off-host immutable backup records object versions and a restore transcript
verifies all 521 content hashes. The current final seal and raw manifest are
admission evidence, not a recoverable off-host backup.

## Read-only backup preflight

With no evidence arguments, the preflight fails closed with exit code 2 and a
JSON `status` of `blocked`; it writes nothing:

```bash
python -B scripts/archive_preflight.py
```

Run the complete check only against provider-exported evidence and a dedicated
restore root that mirrors every canonical `destination_relative_path`:

```bash
python -B scripts/archive_preflight.py \
  --backup-proof /absolute/path/backup-proof.json \
  --object-manifest /absolute/path/object-manifest.jsonl \
  --retention-evidence /absolute/path/retention-evidence.json \
  --restore-transcript /absolute/path/restore-transcript.json \
  --restore-root /absolute/path/restored-copy
```

All objects use strict, versioned schemas: missing or unknown keys, duplicate
JSON keys or paths, unsafe paths, symlinks, non-regular files, stale retention,
placeholder version IDs, and any count, size, SHA-256, or path-set mismatch are
rejected.

- `experiments7-archive-backup-proof/v1` is one JSON object with exactly
  `backup_id`, `canonical_manifest_sha256`, `created_at`,
  `independent_restore`, `object_count`, `object_manifest_sha256`, `off_host`,
  `restore_transcript_sha256`, `retention_evidence_sha256`, `schema`, and
  `total_bytes`.
- `experiments7-archive-object/v1` is one JSONL row per canonical object with
  exactly `object_uri`, `path`, `retention_evidence_id`, `schema`, `sha256`,
  `size`, and the immutable provider `version_id`.
- `experiments7-archive-retention-evidence/v1` is one JSON object with exactly
  `captured_at`, `evidence_id`, `immutability_enabled`, `mode`, `object_count`,
  `provider_evidence_id`, `retention_until`, `schema`, `storage_location`, and
  `versioning_enabled`.
- `experiments7-archive-restore-transcript/v1` is one JSON object with exactly
  `canonical_manifest_sha256`, `files`, `object_count`,
  `object_manifest_sha256`, `restore_id`, `restored_at`,
  `retention_evidence_sha256`, `schema`, and `total_bytes`. Every `files` row
  has exactly `path`, `sha256`, `size`, `status`, and `version_id`, with
  `status` equal to `restored_verified`.

The command binds all four evidence files to the canonical 521-row manifest,
streams and hashes every restored file, and requires an exact restored path
set. Success is named `status=evidence_bundle_structurally_valid` and is scoped
by `validation_domain=archive_evidence_structure`. The deprecated
`preflight_passed=true` alias means structural validation only, as recorded by
`preflight_scope=structural_only`.

A structurally valid report still has `provider_authenticity_verified=false`,
`independent_approval_verified=false`, `cutover_receipt_verified=false`,
`externalization_authorized=false`, and `archive_map_mutated=false`. This offline
check does not contact the storage provider, authorize a move, or alter the
archive plan. No local JSON, digest, HMAC, or self-issued attestation can replace
provider-authenticated off-host proof and independent approval. Those proof and
approval artifacts must be issued outside this repository; the final validator
only admits and verifies them.

## Durable final-completion receipt

`scripts/validate.py --mode final-completion` independently requires
`archive/cutover-receipt.json` with schema
`experiments7-cutover-receipt/v1`. The receipt must SHA-bind these canonical,
non-symlink files and a freshly inspected post-cutover local-state digest:

- actual local state with `data/sources`, `experiments`, and `methods` absent;
  `paper_outputs/raw` may be absent or contain exactly the regular-file metadata
  stubs `README.md` and `schema.json`, whose sizes and hashes are signed;
- `archive/archive-map.json` (`experiments7-archive-map/v1`), with
  `planned_only=false`;
- complete `archive/cutover-source-inventory.jsonl` and
  `archive/cutover-destination-inventory.jsonl` inventories for
  `data/sources`, `experiments`, `methods`, and `paper_outputs/raw`;
- `archive/cutover-restore-report.json`, whose counts, bytes, hashes, and
  manifest bindings prove a verified restore of the complete destination set;
- `archive/provider-offhost-proof.json`, signed by a configured provider key
  and binding immutable off-host object versions and rollback pointers; and
- a named independent approval plus reviewer signature over the complete
  receipt.

Trusted RSA public keys are never admitted from a repository-local file. The
caller must supply an external bundle with schema
`experiments7-cutover-trust/v1` and its exact SHA-256:

```bash
python -B scripts/validate.py --mode final-completion \
  --cutover-trust-bundle /outside/experiments7/cutover-trust.json \
  --cutover-trust-sha256 <64-lowercase-hex>
```

The bundle path must be outside the repository and is read through the same
ancestor-symlink-safe descriptor admission as receipt artifacts. A missing path
or digest, a repo-contained bundle, or a digest mismatch remains blocked.
Provider and reviewer identities and keys must be distinct. Both signatures use `rsa-pkcs1v15-sha256` with RSA keys of at
least 2048 bits. Local/file URIs, placeholder identities or versions, HMACs,
unsigned digests, incomplete inventories, stale/tampered bindings, and
self-approval fail closed.

The canonical receipt, inventory, restore, and provider-proof files do not exist
in the current repository because no evidence was fabricated. The trust bundle
is intentionally external and must never be added to the repository. Their
absence is the `cutover_receipt_missing_or_invalid` completion blocker. Editing
`planned_only` or deleting source trees cannot remove that independent gate. A
verified receipt also cannot authorize final completion while any cutover payload
remains in the actual local tree.

## Phase and final status

```bash
python -B scripts/validate.py --mode phase-readiness
python -B scripts/validate.py --mode final-completion
```

The first command currently returns exit 0 with `status=phase_ready` and
`final_complete=false`. The second currently returns exit 2 with
`status=final_blocked`; its completion blockers are the planning-only
historical archive cleanup and the missing verified
cutover receipt. Only a
blocker-free `final-completion` run may emit
`status=final_complete`. These statuses describe project structure and physical
cutover only. Both reports state `experiment_completion=not_evaluated` and
`actual_experiment_completion_claimed=false`; neither proves that a provider run
completed.
