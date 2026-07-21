# Experiment report: R3 / ECPR VLT3

## Outcome

Standalone PReFine baseline과 ECPR VLT3 후보의 구현, append-only 등록, target-free 구조
진단, one-shot 평가 프로토콜 강화 및 synthetic/static 검증을 완료했다. target model,
gold, target metric은 실행하지 않았으므로 singleturn/multiturn 성능은 `NOT_RUN`이다.
PReFine 대비 성능 향상 또는 preregistered gate PASS를 주장하지 않는다.

## Frozen protocol

- candidate: `ecpr_v1`, revision `vlt3`
- baseline: standalone source-faithful PReFine
- model contract: Qwen/Qwen3-8B snapshot
  `b968826d9c46dd6066d109eabc6255188de91218`
- hardware contract: GPUs 0–3, tensor parallel size 4
- primary metric: `0.5 * singleturn pooled micro-F1 + 0.5 * multiturn pooled micro-F1`
- bootstrap: example-id clusters, 10,000 draws, seed 2026071601
- one-sided randomization: example-id clusters, 100,000 draws, seed 2026071602
- full gates: ΔBMF1 ≥ .010, bootstrap lower > 0, p < .05, per-task/slot-slice/parse
  guardrails, 100% coverage, equal action budget
- metric audit boundary:
  `same-process independent metric implementation sharing frozen parser/row contracts`

마지막 문장은 프로세스 또는 parser 독립성을 뜻하지 않는다. 두 경로는 frozen row
contracts와 `ecpr.parsing.slot_value_map`을 공유하고, indexing/count aggregation/metric
formula/bootstrap/randomization/gate 계산을 별도 구현한다.

## Added mechanism

1. R3 source-faithful action prompts
   - exact `(mode, schema_key)` pair로 상세 single/multi template를 고른다.
   - baseline/candidate는 같은 template을 쓰고 memory block만 다르다.
   - current dialogue와 schema가 memory보다 우선한다.
2. Append-only Table 5 VLT3
   - frozen V1/V2 bytes는 유지한다.
   - LOW_COST, HIGH_COST, SOLO_USAGE를 활성화하고 GROUP_USAGE는 제외한다.
   - mapping identity는 `(trait_id, domain)`이며 opposition은 같은 group 안에서만 적용한다.
3. Exact typed-wire support
   - `value_type`과 exact JSON-scalar `source_literal`을 typed evidence에 보존한다.
   - 두 provenance 필드는 candidate action prompt에 직렬화하지 않는다.
4. Query-explicit suppression
   - current query가 mapped target slot을 명시하면 latent transfer를 억제한다.
   - malformed multi-turn/query-mask/schema mismatch는 provider call 전에 fail-closed한다.
5. Phase3B one-shot final protocol
   - stage order: runtime → pre_gold → gold_open → result → critic
   - nonce capabilities, commit-once 0400 receipts, O_NOFOLLOW same-FD reads, finite command
     timeouts, provider journal seal, result/source/axis-bound critic receipt를 요구한다.
   - 이미 `gold_open`이 커밋된 evaluator replay는 prediction이나 gold를 열기 전에 거부된다.

## Validation evidence

- full unittest discovery: 108/108 PASS (27.581 s)
- Phase3B focused suite: 21/21 PASS (0.590 s)
- combined VLT3/VLT2/Phase2/Phase3A/ledger/evaluator gate: 54/54 PASS (2.782 s)
- VLT3 registration/runtime validation: PASS, 9-node chain
- pre-seal `critic.py --strict --unsealed-vlt`: exit 0
  - integrity: PASS
  - execution: NOT_RUN
  - performance: NOT_RUN
  - external inputs opened: 0
  - target tasks opened: 0
  - sanitized history opened by critic: 0
  - gold rows opened: 0
  - metrics computed: 0
  - implementation files audited: 28
  - external method imports: 0
  - symlinks: 0
  - raw source literal prompt exposure: 0
- implementation manifest recomputation: PASS
  - schema: 3
  - candidate revision: `vlt3`
  - influential files: 60
  - file mode: `0400`
  - SHA256: `04a479a708d362f812a5c4467c9669915d6c2f74de227c3f36abb44c3b5f47b2`
- final-attempt lock: ABSENT

Tamper coverage에는 critic source 변경, critic audit-axis digest 변경 후 receipt 재해시,
stage symlink/gap, wrong nonce, same-FD path swap, journal tamper, evaluator replay,
typed-source type/spelling alias, forged query mask와 schema/mode mismatch가 포함된다.

## Target-free structural evidence

`TARGET_FREE_VLT3_DIAGNOSTICS.json`은 production typed-hypothesis builder와 trait matcher를
sanitized history 265건에만 적용했다. query/tasks/gold/predictions/model outputs/metrics는
열지 않았다.

| Trait | Examples with qualified evidence | Qualified hypotheses |
| --- | ---: | ---: |
| LOW_COST | 62 | 62 |
| HIGH_COST | 3 | 3 |
| SOLO_USAGE | 17 | 22 |

12개 frozen target mapping에 대해 typed-only precondition count를 기록했다. 예를 들어
LOW_COST→GetRestaurants는 62건, LOW_COST→GetFlights는 58건, SOLO_USAGE→GetFlights는
15건에서 typed-only precondition을 만족했다. 이 수치는 PQR selection, query constraint,
verified latent authorization, action output 또는 accuracy가 아니다.

Target-free report SHA256:
`d32f7f8e85bf043e721004531190f8f506878fd1bed28111c3138f4694b2c7bc`.

## Append-only hashes

- VLT1 ontology: `1d9ffdf29a2af37367bbd63033c927473789e6eddb71926999bb88e56bb198a1`
- VLT2 ontology: `be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b`
- VLT3 ontology: `652febe52842c3ac9d4b5fb4bb1f4e81a8a0c3072610e9717513eca121efc0d3`
- R3 method scope: `a5e582293faa22c7e5b1496362a4e0b9c61f03ce66b2aaaa1e2db05a73be09cc`
- R3 action prompt amendment: `a73d15836e20a8e56f87c2c93e1eacf8408cb12b9b6160fe303612efbcd59e98`
- VLT3 audit amendment: `236dc25599924efc94c29297e2444760790070155f8629c6d6915d9f30f68852`
- VLT3 expected runtime: `5de17cca8ae3649a3f4fc8e2da46a6615906e6896b5436652b4eeb6d69450855`

실제로 존재하지 않았던 `VLT_AUDIT_AMENDMENT.json`과
`manifests/expected_runtime_contract.vlt2.json`은 소급 생성하지 않았다.

## GPU stop condition

2026-07-16의 기존 `reports/GPU_BLOCKER.json`과 이번 bounded read-only 재확인은 모두
다음을 기록했다.

```text
Failed to initialize NVML: Unknown Error
```

이번 재확인 명령은 `timeout 5 nvidia-smi --query-gpu=...`였고 exit code는 255였다.
vLLM, target memory, action inference, evaluator 또는 final attempt를 시작하지 않았다.

## Remaining risk and stop condition

실제 Qwen3-8B prompt behavior, singleturn/multiturn BMF1, confidence interval, p-value와
guardrail 결과는 미측정이다. GPU 0–3이 회복된 뒤 봉인된 one-shot final pipeline을 한 번
실행하고 hash-bound critic receipt가 PASS일 때만 성능 결론을 낼 수 있다. 현재 작업은
non-cheating 구현·구조 검증과 immutable seal에서 중단한다.
