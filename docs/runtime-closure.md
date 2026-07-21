# Snapshot publication and external runtime closure

The generated G2 control artifacts remain pending and are never rewritten. A real
external run is enabled only when three additive, exact-hash contracts validate:

1. `snapshots/publications/<publication_id>/manifest.json` and `seal.json` bind all
   129 planned destinations to immutable files under `experiments7`.
2. `runtime/closures/<runtime_id>/manifest.json` and `seal.json` bind the Python
   executable and every declared runtime artifact by absolute path, byte count,
   and SHA-256.
3. `configs/external-runs/<run_config_id>.json` binds one profile to those two IDs,
   closes every selected subprocess stage, and types every argument as `literal`,
   exact-hash `input`, or contained `output`.

Publication and closure manifests use canonical JSON. Their sibling seals contain
`manifest_sha256` and `manifest_bytes`; manifests, seals, snapshots, run configs,
and input files must be non-symlink regular files with no write bits. Publication
is no-replace: an existing or partial publication ID is never repaired in place.

Run-config bindings have these forms:

```json
{"kind":"literal","value":"single"}
{"kind":"input","path":"inputs/query.json","sha256":"<64 hex>","bytes":123}
{"kind":"output","path":"metrics.csv"}
```

The subprocess receives an empty inherited environment plus closure defaults,
explicit allowlisted environment variables, and contained `HOME`/`TMPDIR` paths.
Python bytecode and user-site loading are disabled. Stdout, stderr, output hashes,
selection hash, publication hash, runtime hash, and run-config hash are recorded
under the new run directory.

Current state is intentionally fail-closed: no publication, runtime closure, or
external run config is shipped by this change, and no origin experiment tree is
read. Once a separately authorized sealed snapshot operation has populated all
planned destinations, the overlay can be created with:

```bash
python -B scripts/runtime/publish_snapshot_overlay.py \
  --publication-id <new-id> --json
```

Validation never executes origin code:

```bash
python -B scripts/validation/validate_snapshot_publication.py \
  --publication-id <id> --json
python -B scripts/validation/validate_external_contract.py \
  --profile-id <profile> --publication-id <id> \
  --runtime-id <runtime> --run-config-id <config> --json
```
