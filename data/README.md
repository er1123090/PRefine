# Data

`datasets_index.json` lists dataset files from the immutable experiments6
runtime snapshot. That snapshot is explicitly recorded as an experiments4
release/path alias, so both provenance roots are retained.

`sources/experiments4`, `sources/experiments5`, and
`sources/experiments6` contain actual
audited data and data-preparation source files copied from the corresponding
experiment trees. The sealed dataset bytes under `environments/` remain the
canonical runtime input.
