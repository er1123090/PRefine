# Non-destructive archive map

Planning index only: it authorizes no move, deletion, or externalization.

| Current path | Files | Bytes | Symlinks | Planned disposition |
| --- | ---: | ---: | ---: | --- |
| `data/sources` | 41 | 79,729,488 | 0 | `archive_after_cutover` |
| `methods` | 307 | 3,599,879 | 0 | `archive_after_cutover` |
| `experiments` | 183 | 1,096,628 | 0 | `archive_after_cutover` |
| `environments` | 517 | 71,220,961 | 26 | `keep_active_audit` |
| `lineage` | 2 | 92,219 | 0 | `keep_active_audit` |
| `manifests` | 71 | 34,370,859 | 0 | `keep_active_audit` |
| `variants` | 5 | 218,624 | 0 | `keep_active_audit` |
| `paper_outputs/final` | 7 | 6,879,419 | 0 | `keep_active_audit` |
| `paper_outputs/raw` | 523 | 7,686,997,692 | 0 | `externalize_after_verified_backup` |
| `paper_outputs/admission` | 7 | 2,926,322 | 0 | `keep_active_audit` |
| `paper_outputs/inventory` | 9 | 2,115,859 | 0 | `keep_active_audit` |
| `paper_outputs/provenance` | 4 | 7,333,133 | 0 | `keep_active_audit` |
| `paper_outputs/strict-runs` | 47 | 31,790,752 | 0 | `keep_active_audit` |
| `outputs` | 3 | 3,221 | 0 | `generated_rebuildable` |
| `results` | 2 | 1,462 | 0 | `generated_rebuildable` |
| `runs` | 46 | 15,969,600 | 0 | `split_before_cleanup` |

`paper_outputs/raw` remains blocked from externalization until an off-host immutable backup and restore transcript exist.

```bash
python -B scripts/archive_map.py
python -B scripts/archive_map.py --check
```
