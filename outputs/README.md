# Published experiment artifacts

This directory keeps only completed, reusable experiment artifacts in Git.
Large files are stored through Git LFS.

Included:

- completed A-MEM and LangMem memory artifacts for MPT v2 0725;
- completed three-model `ours_memory` artifacts used by inference;
- the completed RAG Chroma artifact;
- evaluated full-set predictions and evaluations;
- evaluated final 4,000-example stratified predictions and evaluations;
- final paper tables, spreadsheets, manifests, and run summaries.

Excluded:

- `archive/`;
- incomplete Mem0 and Mem0_local construction checkpoints;
- 1,000- and 3,000-example intermediate samples;
- raw inference traces, provider requests/responses, and batch state;
- runtime logs, PID files, caches, smoke runs, and aborted runs.

Run `git lfs pull` after cloning to materialize the large artifacts.
