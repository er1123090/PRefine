# Readable experiment workspace

The top-level `data`, `methods`, `experiments`, `outputs`, and `results`
directories follow the organization used in experiments4, experiments5, and
experiments6. The first three contain actual audited files:

```text
data/
  sources/experiments4|experiments5|experiments6/...
methods/
  rag/experiments4|experiments5|experiments6/...
  mem0/experiments4|experiments5|experiments6/...
  our_memory/experiments4|experiments5|experiments6/...
  ...
experiments/
  experiments4|experiments5|experiments6/evaluation/...
  experiments4|experiments5|experiments6/shared/...
  variants/<semantic_label>.json
outputs/
results/
```

There are 443 byte-identical source copies backed by audited SHA-256 records:
83 paper-relevant files from experiments4, 72 from experiments5, and all 288
regular files in the sealed experiments6 base. The base manifest's 26 symlink
aliases are deliberately not reproduced. The copies are normal files, not
symlinks or hardlinks. The original experiments4/5/6 trees are read-only
inputs to this layout and are not changed. The sealed `environments` tree
remains the canonical runtime.

Each JSON entry has a safe repository-relative `canonical_path`, a
`code_paths` list, a `semantic_label`, and sorted, unique `source_origins`.
The paths point to the readable files above. Output-affecting
experiments4 and experiments5 distinctions keep the audited labels such as
`eval4_single_legacy`, `eval4_multiturn_legacy`, and `eval5_canonical`.
Folder names identify only the source tree. Output-affecting labels such as
`eval4_single_legacy` and `eval5_canonical` remain in variant documents;
the layout does not invent an `eval6` or `infer6` meaning.

`outputs` lists raw/run artifacts. `results` lists metrics, aggregates,
evaluation mappings, and the fail-closed unresolved ledger. Public paper
navigation intentionally exposes only `paper_outputs/final` and
`paper_outputs/raw`; admission, inventory, provenance, and strict-run internals
remain canonical but are not promoted as public paper outputs.

## Commands

```bash
python -B scripts/layout.py list
python -B scripts/layout.py show experiments
python -B scripts/layout.py sync-code --check
python -B scripts/layout.py build-indexes --check
python -B scripts/layout.py validate
```

`sync-code` regenerates or verifies the regular-file copies and their
`readable_code_manifest.json`. `build-indexes` regenerates deterministic JSON from the audited registry,
profile, final seal, raw manifest, immutable dataset snapshot, and current run
directories. `--check` is read-only and fails when a committed catalog is
stale. `validate` additionally rejects traversal, containment escapes,
symlinked navigation files, non-regular navigation files, undeclared files, and
count drift from `1908 / 1696 / 212 / 521 / 84`, as well as readable-code
hash or file-set drift.
