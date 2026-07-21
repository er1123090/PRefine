# Verification

Verified on 2026-06-06 in `/data/minseo`.

## Release Inventory

- Release folder: `/data/minseo/experiments6`
- Release size: 113M
- Python/shell source files in release: 224
- Public wrapper scripts in `scripts/`: 23
- Source files inventoried from `experiments4` outside `e-mem`: 219
- Inventoried source files copied or intentionally documented: 219 / 219

## Checks Run

```bash
python -m compileall -q /data/minseo/experiments6
find /data/minseo/experiments6 -type d -name __pycache__ -prune -exec rm -rf {} +
find /data/minseo/experiments6 -type f \( -name '*.sh' -o -name '*.bash' \) -print0 | xargs -0 -r -n1 bash -n
rg -n --pcre2 'experiments4|experiments5|personal-|personal_|conv_api|/data/minseo/(?!experiments6)' /data/minseo/experiments6 -g '*.py' -g '*.sh' -g '*.bash'
rg -n '/data/minseo/experiments6|temp_queries\.json|session_memory_eval_0312' \
  /data/minseo/experiments6/scripts \
  /data/minseo/experiments6/src/baselines/mem0/mem0-api.py \
  /data/minseo/experiments6/src/extended_schema \
  $(find /data/minseo/experiments6/src/preference_memory/session_memory_eval -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) -print)
for wrapper in \
  /data/minseo/experiments6/scripts/build_langmem_memory.sh \
  /data/minseo/experiments6/scripts/build_mem0_memory.sh \
  /data/minseo/experiments6/scripts/build_preference_memory_api.sh \
  /data/minseo/experiments6/scripts/build_preference_memory_vllm.sh \
  /data/minseo/experiments6/scripts/build_rag_index.sh \
  /data/minseo/experiments6/scripts/evaluate_multiturn_f1.sh \
  /data/minseo/experiments6/scripts/run_langmem_multiturn.sh \
  /data/minseo/experiments6/scripts/run_langmem_singleturn.sh \
  /data/minseo/experiments6/scripts/run_mem0_multiturn.sh \
  /data/minseo/experiments6/scripts/run_mem0_singleturn.sh \
  /data/minseo/experiments6/scripts/run_preference_memory_multiturn_api.sh \
  /data/minseo/experiments6/scripts/run_preference_memory_multiturn_vllm.sh \
  /data/minseo/experiments6/scripts/run_preference_memory_singleturn_api.sh \
  /data/minseo/experiments6/scripts/run_preference_memory_singleturn_vllm.sh \
  /data/minseo/experiments6/scripts/run_rag_multiturn.sh \
  /data/minseo/experiments6/scripts/run_rag_singleturn.sh \
  /data/minseo/experiments6/scripts/run_self_refine_preference_api.sh \
  /data/minseo/experiments6/scripts/run_vanilla_llm_multiturn_api.sh \
  /data/minseo/experiments6/scripts/run_vanilla_llm_multiturn_vllm.sh \
  /data/minseo/experiments6/scripts/run_vanilla_llm_singleturn_api.sh \
  /data/minseo/experiments6/scripts/run_vanilla_llm_singleturn_vllm.sh
do
  rg -q '\$RELEASE_ROOT/(data|query|pref|schema|runs)' "$wrapper" || exit 1
done
python - <<'PY'
import os, re, shlex
root = "/data/minseo/experiments6"
missing = []
for dirpath, _, files in os.walk(root):
    for name in files:
        if not name.endswith(".sh"):
            continue
        path = os.path.join(dirpath, name)
        text = open(path, encoding="utf-8").read()
        match = re.search(r'^PYTHON_SCRIPT=(.+)$', text, re.MULTILINE)
        if not match:
            continue
        value = shlex.split(match.group(1), comments=True)[0]
        if value.startswith("$RELEASE_ROOT/"):
            target = os.path.join(root, value.removeprefix("$RELEASE_ROOT/"))
        elif value.startswith("${RELEASE_ROOT}/"):
            target = os.path.join(root, value.removeprefix("${RELEASE_ROOT}/"))
        elif value.startswith("$PACKAGE_ROOT/"):
            target = os.path.join(dirpath, value.removeprefix("$PACKAGE_ROOT/"))
        elif value.startswith("${PACKAGE_ROOT}/"):
            target = os.path.join(dirpath, value.removeprefix("${PACKAGE_ROOT}/"))
        else:
            target = value if os.path.isabs(value) else os.path.join(dirpath, value)
        if not os.path.exists(target):
            missing.append((path, target))
if missing:
    for path, target in missing:
        print(f"{path} -> {target}")
    raise SystemExit(1)
print("all_shell_python_script_targets_exist")
PY
python - <<'PY'
import re
import shlex
import subprocess
from pathlib import Path

release = Path("/data/minseo/experiments6").resolve()
fixed_scripts = sorted((release / "src" / "extended_schema").glob("**/*.sh"))
failures = []
checked = 0
for physical in fixed_scripts:
    rel_physical = physical.relative_to(release)
    aliases = [release / rel_physical]
    if rel_physical.parts[:2] == ("src", "extended_schema"):
        aliases.append(release / Path(*rel_physical.parts[1:]))
    text = physical.read_text(encoding="utf-8")
    match = re.search(r"^PYTHON_SCRIPT=(.+)$", text, re.MULTILINE)
    if not match:
        continue
    script_value = shlex.split(match.group(1), comments=True)[0]
    for invocation in aliases:
        cmd = """set -euo pipefail
script_path="$1"
SCRIPT_DIR="$(cd -P "$(dirname "$script_path")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
printf '%s\n' "$RELEASE_ROOT"
"""
        result = subprocess.run(
            ["bash", "-c", cmd, "_", str(invocation)],
            text=True,
            capture_output=True,
            check=True,
        )
        derived_root = Path(result.stdout.strip())
        if derived_root != release:
            failures.append(f"{invocation}: derived {derived_root}, expected {release}")
            continue
        target_text = script_value.replace("$RELEASE_ROOT", str(derived_root)).replace(
            "${RELEASE_ROOT}", str(derived_root)
        )
        target = Path(target_text)
        if not target.exists():
            failures.append(f"{invocation}: PYTHON_SCRIPT target missing {target}")
        checked += 1
if failures:
    print("\n".join(failures))
    raise SystemExit(1)
print(f"fixed400_release_root_invocation_check_ok invocations={checked}")
PY
python - <<'PY'
import subprocess
from pathlib import Path

repo = Path("/data/minseo")
release = repo / "experiments6"
manifest = release / "docs" / "source_manifest.md"
ignored = []
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.startswith("| `"):
        continue
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    if len(cells) < 2:
        continue
    release_path = cells[1].strip("`")
    if release_path == "-" or " and " in release_path:
        continue
    target = release / release_path
    if not target.exists() or target.is_dir():
        continue
    rel = target.relative_to(repo).as_posix()
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-v", rel],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        ignored.append(result.stdout.strip())
if ignored:
    print("\n".join(ignored))
    raise SystemExit(1)
print("manifest_public_files_not_git_ignored")
PY
find /data/minseo/experiments6 -path '*e-mem*' -print
find /data/minseo/experiments6 -type d \( -name __pycache__ -o -name .git -o -name logs -o -name output -o -name outputs -o -name inference -o -name inference_single -o -name inference_multi -o -name eval_results -o -name chroma_db_rag -o -name memory_snapshots -o -name indexes \) -print
```

Results:

- Python syntax: passed.
- Shell syntax: passed.
- Stale `experiments4` / `experiments5` / personal local paths in executable code: 0 matches.
- Public wrappers and clone-critical entrypoints have no hard-coded
  `/data/minseo/experiments6`, `temp_queries.json`, or
  `session_memory_eval_0312` defaults.
- The whole `src/extended_schema` tree is included in that clone-critical
  entrypoint scan, including the fixed-400 table builders and method rerun
  wrappers.
- The session-memory eval package is included in that scan so its README,
  shell runner defaults, and Python defaults remain release-relative.
- Public wrappers pass release-root-derived data/config/default output paths
  where the underlying Python entrypoint has clone-sensitive defaults.
- Shell `PYTHON_SCRIPT` targets: all referenced Python scripts exist.
- Fixed-400 shell wrappers derive the release root correctly from both
  `extended_schema/...` symlink invocations and `src/extended_schema/...`
  physical invocations: `fixed400_release_root_invocation_check_ok
  invocations=28`.
- Manifest-listed public release files are not ignored by Git.
- `e-mem` paths in release tree: 0 matches.
- Generated/cache/output directory names in release tree: 0 matches after removing `compileall` caches.
- Manifest source table reconciles as 201 copied source files + 18 omitted
  generated shard launchers + 1 release control file.
- README and source manifest exist.
- Smoke matrix present for evaluation, preference memory, vanilla LLM, RAG,
  Mem0, LangMem, extended schema, dataset tools, self-refine, analysis, and
  wrappers.
- Final rerun marker after code-review cycle 3 fixes:
  `final_verification_after_cycle3_fixes_ok`.
- Final rerun marker after code-review cycle 4 fixes:
  `final_verification_after_cycle4_fixes_ok` with
  `table_builders_checked=2`.
- Final rerun marker after code-review cycle 5 fixes:
  `final_verification_after_cycle5_fixes_ok` with
  `table_builders_checked=2` and `session_memory_eval_files_checked=21`.
- UltraQA cycle 2 rerun marker:
  `ultraqa_cycle2_all_static_adversarial_checks_ok` after retargeting
  clone-sensitive extended-schema fixed-400 scripts to release-root paths.
- Final review-blocker rerun marker:
  `final_ultraqa_static_gate_ok` after validating fixed-400 symlink and
  physical invocation path handling.

## Protected Directory Check

Before and after status snapshots are stored in:

- `.omx/artifacts/autopilot-experiment6/experiments4.before.status`
- `.omx/artifacts/autopilot-experiment6/experiments4.after.status`
- `.omx/artifacts/autopilot-experiment6/experiments5.before.status`
- `.omx/artifacts/autopilot-experiment6/experiments5.after.status`

Comparison results:

- `experiments4`: unchanged.
- `experiments5`: internal status unchanged. The broader parent git status sees
  the new sibling folder `../experiments6/`, so the protected comparison is
  scoped to paths inside `experiments5`.

## Non-Goals

The verification did not run full API, Mem0, RAG, LangMem, or GPU/vLLM
experiments. Those runs require credentials, external services, model serving,
and substantial runtime. The release checks verify that the code is present,
syntax-valid, documented, and retargeted from `experiments4` to `experiments6`.
