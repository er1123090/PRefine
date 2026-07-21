# Experiments

`experiments4`, `experiments5`, and `experiments6` contain actual evaluation and shared
experiment source files. `variants` contains one JSON document for each of
the 84 audited variants, including every corresponding `code_paths` entry.

`index.json` lists the same 84 variants. Use `semantic_label` to preserve
output-affecting differences and `code_paths` to locate their readable
implementations. The sealed executable interface remains
`scripts/experiment_variants.py`.
