# Standalone ECPR VLT3 (`prefine-1229-dev6-noncheating`)

이 디렉터리는 `ours_memory`의 PReFine 방법론을 런타임 의존성 없이 복제하고,
1229_dev6에서 singleturn/multiturn personalized tool calling을 개선하기 위한 후보
ECPR(Evidence-Calibrated Preference Routing) VLT3를 구현한다. `/data/minseo/experiments4/ours_memory`
또는 기존 `ours_memory2/ours_memory2`를 import하거나 실행하지 않는다.

## VLT3 개선 메커니즘

- R3 action prompt는 논문의 singleturn/multiturn PReFine 지시를 source-local template로
  고정한다. 같은 mode에서 baseline과 candidate는 정확히 하나의 memory block만 다르다.
- VLT3는 frozen V1/V2 ontology를 수정하지 않고 Table 5의 LOW_COST, HIGH_COST,
  SOLO_USAGE를 append-only로 추가한다. GROUP_USAGE는 명시적으로 제외한다.
- typed evidence는 원본 JSON scalar의 `value_type`과 `source_literal`을 보존해 지원 여부를
  판정하지만, 두 필드는 action prompt에 절대 직렬화하지 않는다.
- 현재 query가 VLT target slot을 명시하면 transfer를 억제한다. PQR ABSTAIN은 항상 memory를
  비우며, SELECT도 schema/query/ontology 재검증을 통과한 closed tuple만 허용한다.
- 평가는 nonce capability, commit-once receipts, same-FD/O_NOFOLLOW input binding,
  finite command timeout, hash-bound strict critic receipt를 쓰는 one-shot 체인이다.
- 두 metric 경로의 정확한 독립성 주장은
  `same-process independent metric implementation sharing frozen parser/row contracts`이다.
  프로세스 또는 parser 독립성을 주장하지 않는다.

## 검증 상태

- 전체 synthetic/static 회귀: 108/108 PASS
- Phase3B one-shot/tamper/replay 회귀: 21/21 PASS
- VLT3 registration chain: 9 nodes
- implementation manifest: schema 3, revision `vlt3`, 60 influential files, mode `0400`
- pre-seal strict critic: integrity PASS / execution NOT_RUN / performance NOT_RUN
- final attempt lock: ABSENT
- target model calls: 0
- gold opens: 0
- target metrics: 0

따라서 singleturn/multiturn 성능 향상은 아직 측정되지 않았으며, PReFine보다 높다는
성능 주장은 하지 않는다.

## Target-free 진단

`reports/TARGET_FREE_VLT3_DIAGNOSTICS.json`은 query, tasks, gold, predictions, latent model
outputs, metrics를 열지 않고 sanitized history 265건만 production typed-evidence 코드로
집계했다.

- LOW_COST: 62 examples / 62 qualified hypotheses
- HIGH_COST: 3 examples / 3 qualified hypotheses
- SOLO_USAGE: 17 examples / 22 qualified hypotheses
- frozen target mappings: 12

이는 후보 transfer가 작동할 구조적 기회가 존재한다는 target-free 근거일 뿐, PQR 선택,
실제 query 제약, 최종 authorization, action output 또는 정확도를 예측하지 않는다.

## 동결 해시

- preregistration: `2ec42394f9157fa0fe690195dc3010afc81c2bb117d0e94a9dfeef0535b5dae0`
- VLT1 ontology: `1d9ffdf29a2af37367bbd63033c927473789e6eddb71926999bb88e56bb198a1`
- VLT2 ontology: `be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b`
- VLT3 ontology: `652febe52842c3ac9d4b5fb4bb1f4e81a8a0c3072610e9717513eca121efc0d3`
- R3 scope: `a5e582293faa22c7e5b1496362a4e0b9c61f03ce66b2aaaa1e2db05a73be09cc`
- R3 action prompt amendment: `a73d15836e20a8e56f87c2c93e1eacf8408cb12b9b6160fe303612efbcd59e98`
- VLT3 audit amendment: `236dc25599924efc94c29297e2444760790070155f8629c6d6915d9f30f68852`
- target-free report: `d32f7f8e85bf043e721004531190f8f506878fd1bed28111c3138f4694b2c7bc`
- VLT3 runtime: `5de17cca8ae3649a3f4fc8e2da46a6615906e6896b5436652b4eeb6d69450855`
- implementation manifest: `04a479a708d362f812a5c4467c9669915d6c2f74de227c3f36abb44c3b5f47b2`

의도적으로 존재하지 않는 `VLT_AUDIT_AMENDMENT.json`과
`manifests/expected_runtime_contract.vlt2.json`은 소급 생성하지 않았다.

## 재현 및 다음 단계

안전한 non-target 검증:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python -c 'from pathlib import Path; from ecpr.integrity import validate_implementation_manifest; print(validate_implementation_manifest(Path("."))[1])'
PYTHONPATH=. python critic.py
```

구현은 이미 봉인되어 `prepare`와 unsealed VLT audit을 의도적으로 거부한다. 실제 final
evaluation은 승인된 GPU 0–3에서 `python run_gpu.py --final`을 한 번만 실행하며, 그 시점에만
final-attempt lock과 gold-open capability를 소비한다. 이 저장소 작업에서는 실행하지 않았다.

## 주요 파일

- `R3_METHOD_SCOPE_CLARIFICATION.json`, `R3_ACTION_PROMPT_AMENDMENT.json`
- `VLT3_AUDIT_AMENDMENT.json`, `configs/latent_trait_ontology.vlt3.json`
- `target_free_vlt3_diagnostics.py`, `reports/TARGET_FREE_VLT3_DIAGNOSTICS.json`
- `ecpr/final_protocol.py`, `ecpr/independent_audit.py`, `critic.py`, `run_gpu.py`
- `manifests/expected_runtime_contract.vlt3.json`
- `manifests/implementation_manifest.json`
