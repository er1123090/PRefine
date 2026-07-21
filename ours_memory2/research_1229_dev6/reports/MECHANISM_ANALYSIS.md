# Mechanism analysis

## 범위와 근거

이 문서는 `ecpr_v1 / vlt3`의 현재 구현만 설명한다. 근거는 공개된 로컬 소스,
고정 schema/preference-slot 설정, standalone `ours_memory2` prompt, 그리고
PREFINE 논문뿐이다. held-out query 목록, 정답, target metadata, prediction, evaluator
metric은 메커니즘 선택 근거로 사용하지 않는다
(`configs/latent_trait_ontology.vlt3.json:426-474`).

논문 인용의 페이지는 PDF 페이지 기준이다. 주요 근거는 다음과 같다.

- 현재 query의 명시 제약을 모두 만족하고, 남은 preference argument만 history에서
  추론해야 한다는 문제 정의: PDF p.3, lines 75-84.
- latent preference는 반복적인 in-domain 행동 또는 cross-domain regularity로 나타나는
  가설이라는 정의: PDF p.5, §5.2, lines 165-173.
- generate-verify-refine와 verifier의 evidence/abstraction/actionability/temporal 조건:
  PDF pp.5-6, §5.3, lines 181-204.
- MPT의 preference-sensitive slot은 전체 schema의 strict subset이며, Budget/Travel의
  정확한 mapping과 solo 유지/group 제외 근거: PDF p.14, Table 4-5,
  lines 479-500.
- inference prompt의 schema filtering, repeated choice, cross-domain consistency,
  schema 밖 추론 금지: PDF p.25, Figure 10.

## PREFINE baseline

Baseline은 sanitized chronological history와 이전 belief로 latent constraint를
generate/refine하고, 별도 verifier가 후보를 검증한다. action 단계의 memory block은 최종
latent abstraction과 누적 historical API calls이다
(`ecpr/memory.py:112-150`, `ecpr/prompts.py:204-217`).

이 패키지는 외부 `ours_memory`를 import하지 않는다. generator/refiner/verifier와 action
prompt는 패키지 안에 복제되어 있다. history-session당 최대 10 generation slots는 사전에
고정된 실행 예산이지 성능 향상을 확인하고 선택한 값이 아니다. 논문의 main experiment는
3회 cap을 사용하고, 10회로 늘려도 일관된 이득이 없었다고 보고한다
(PDF p.6, lines 219-221).

Singleturn과 multiturn action prompt는 서로 다른 source-local PREFINE template을 쓰며,
`singleturn -> single`, `multiturn -> multi`가 맞지 않으면 모델 호출 전에 실패한다.
같은 mode 안에서는 baseline/candidate가 동일한 3-argument prompt builder와 동일 skeleton을
공유하고 memory block만 다르다 (`ecpr/prompts.py:21-132,321-350`).

## VLT3 candidate: 두 개의 보수적 memory 경로

Candidate prompt에는 두 종류의 memory만 들어갈 수 있다.

1. 현재 public query가 선택한 schema domain에 속하는 routed typed hypothesis.
2. 모든 firewall 검사를 통과한 경우에만 생성되는 하나의 closed VLT tuple
   `{domain, slot, trait_id, value}`.

자유형 raw latent text는 candidate prompt에 직렬화되지 않는다. latent abstraction은 trait
판정의 입력일 뿐이며, prompt에는 승인된 closed tuple만 들어간다. Typed prompt projection도
`domain/slot/value/support/counterevidence/confidence/last_seen`만 포함하고,
`source_literal`, `value_type`, call index/digest provenance는 audit에만 남는다
(`ecpr/prompts.py:220-318`, `ecpr/latent_firewall.py:1677-1744`).
따라서 “candidate prompt에 routed typed evidence와 raw latent abstraction이 함께 들어간다”는
설명은 VLT3에는 맞지 않는다.

### Typed hypothesis와 routing

History call의 preference slot 관측은 normalized value만으로 합쳐지지 않는다.
`(normalized value, JSON scalar type, exact source literal)`이 모두 같은 관측끼리만
hypothesis가 된다. 예를 들어 string `"1"`, integer `1`, float `1.0`, boolean
`true`는 별도 evidence다 (`ecpr/memory.py:22-109`).

기본 typed routing gate는 다음을 모두 요구한다.

- public query gate가 하나의 schema domain을 선택할 것;
- hypothesis domain/slot이 그 schema와 registered preference slots에 속할 것;
- support >= 2, confidence >= 0.67, conflict ratio <= 0.34;
- 같은 slot에 두 개의 qualified value가 남으면 그 slot을 abstain할 것.

최대 6개 routed hypotheses, overlay 384 lexical tokens, 전체 memory block 1,536 lexical
tokens 제한이 적용된다. Overlay cap은 prompt에 실제로 보이는 projection에만 계산되므로
audit-only `source_literal`이나 provenance가 memory 선택량을 바꾸지 않는다
(`ecpr/router.py:35-120`; candidate parameters는 preregistration).

### Table 5 exact typed support와 target grounding

VLT3 taxonomy는 `LOW_COST`, `HIGH_COST`, `SOLO_USAGE` 세 trait만 활성화한다.
논문 Table 5의 source observation을 느슨한 synonym이나 normalized alias로 인정하지 않고,
history의 exact JSON scalar source wire와 일치할 때만 typed support로 인정한다
(`configs/latent_trait_ontology.vlt3.json:801-951`,
`ecpr/latent_firewall.py:1258-1310`).

| Trait | exact typed source support |
|---|---|
| `LOW_COST` | Restaurants `price_range="cheap"`; RentalCars `car_type="Compact"`; Hotels `average_star="1"` or `"2"`; RideSharing `shared_ride="True"`; Travel `free_entry="True"`; Flights `flight_class="Economy"` |
| `HIGH_COST` | Restaurants `price_range="pricey"`; RentalCars `car_type="Full-size"`; Hotels `average_star="4"` or `"5"` |
| `SOLO_USAGE` | Buses `group_size="1"`; Flights `passengers="1"`; RideSharing `number_of_seats="1"`; Events `number_of_tickets="1"`; Restaurants `number_of_seats="1"` |

Source evidence type과 target schema output type은 별도 계약이다. 예를 들어 history의
`shared_ride="True"` string은 LOW_COST evidence지만, target `GetRideSharing` tuple은
schema-native boolean `true`다. Target mapping은 정확히 선언된
`(trait_id, domain)` identity와 schema key/type/enum을 다시 검증한다
(`ecpr/latent_firewall.py:1065-1109`).

현재 target serialization surface는 다음과 같이 더 좁다
(`configs/latent_trait_ontology.vlt3.json:632-799`).

- LOW_COST: Restaurants `cheap`, RentalCars `Compact`, Flights `Economy`,
  RideSharing boolean `true`, Travel boolean `true`.
- HIGH_COST: Restaurants `pricey`, RentalCars `Full-size`.
- SOLO_USAGE: Buses/Flights/RideSharing/Events의 string `"1"`;
  Restaurants `number_of_seats="1"`은 multi schema에서만 허용.
- Hotels의 star observations는 이 revision에서 corroborating evidence일 수 있지만
  target tuple mapping은 아니다.

이는 논문이 schema-agnostic latent memory를 주장한다고 해서 임의의 새 mapping을 허용한다는
뜻이 아니다. VLT3 실행은 frozen public schema와 ontology에 선언된 mapping에만 닫혀 있다.
논문은 abstract memory를 test-time schema에 grounding한다고 설명한다
(PDF p.6, lines 197-204); VLT3는 그 grounding을 보수적인 closed registry로 구현한다.

### SOLO 포함, GROUP 제외, opposition

논문 Table 5에는 solo와 group value가 모두 보이지만, 바로 뒤 설명은
`solo_usage`만 유지하고 `group_usage`는 party size 2의 의미가 couple/group 사이에서
모호해 제외한다고 명시한다 (PDF p.14, lines 491-496). VLT3도 이에 맞춰
`SOLO_USAGE`만 trait/support/target mapping에 넣고 `GROUP_USAGE`는 correspondence
metadata에만 `excluded`로 남긴다
(`configs/latent_trait_ontology.vlt3.json:962-986`).

Opposition은 group-local이다.

- `LOW_COST <-> HIGH_COST`만 서로 반대다.
- cost evidence와 SOLO evidence는 서로 중립이다.
- SOLO에는 positive `GROUP_USAGE` trait도, global group-size veto도 없다.
- 다만 어떤 domain에서든 policy-qualified 반대 cost trait evidence가 하나라도 있으면
  해당 LOW/HIGH transfer는 거부한다.

이 계약은 `configs/latent_trait_ontology.vlt3.json:481-556,953-960`과
`ecpr/latent_firewall.py:1561-1580`에 고정되어 있다.

### 최소 corroboration과 target-slot guard

VLT3의 latent transfer는 모든 trait에 대해 최소 하나의 policy-qualified non-target
hypothesis를 요구한다. 여기서 “하나”는 raw call 한 번이 아니라, 기본 threshold
(support >= 2, confidence/conflict gate)를 통과하고 exact Table 5 source literal에 매칭된
하나의 hypothesis다. Evidence는 target domain과 다른 domain이어야 한다
(`ecpr/latent_firewall.py:1234-1255,1358-1376,1555-1596`).

Target domain의 같은 slot에 policy-qualified typed evidence가 이미 있으면 VLT는 값을
덮어쓰지 않는다. 서로 다른 typed identities가 있으면 `AMBIGUOUS`, 정확한 type/value가
mapped value와 같으면 `REDUNDANT`, 다르면 `CONFLICT`로 fail closed한다
(`ecpr/latent_firewall.py:1597-1629`).

## Current-query constraint mask

Action case는 VLT 판단 전에 exact task `query/mode/schema_key/schema`와 frozen ontology로
constraint mask를 만든다. Selected Table 5 target slot에 대해 다음 신호만 탐지한다.

- mapping-specific explicit value phrases는 직접 mask 신호로 사용;
- party/count slot에서는 digit 또는 frozen zero-through-twelve word가 public slot alias
  가까이에 있을 때만 count 신호로 사용;
- count와 alias 사이 최대 거리 3 tokens. Non-count slot alias만 단독으로 나온 경우에는
  mask하지 않음.

Multiturn에서는 production query gate와 같은 `extract_user_payload`를 사용하므로
Assistant/Tool text를 explicit user constraint로 승격하지 않는다. Role 구조가 malformed여서
User payload를 얻지 못하면 선택 domain의 적용 가능한 mapped slot을 fail closed로 mask한다
(`ecpr/latent_firewall.py:881-970`).

같은 mask가 routed typed path와 VLT path에 모두 적용된다
(`ecpr/action_case.py:74-140`). Decider와 serializer는 mask를 독립적으로 재계산해 forged
empty mask, 다른 query/mode/schema binding을 거부한다
(`ecpr/latent_firewall.py:1471-1490,1677-1710`). Mask된 slot에는 closed tuple도
직렬화하지 않는다.

한계도 명시적이다. 이 detector는 frozen Table 5 mapped slots와 보수적 registry만 다룬다.
임의의 자연어 paraphrase, 비-Table-5 slot, 또는 registry 밖의 explicit constraint를
완전하게 판별한다고 주장하지 않는다. 나머지는 action prompt의
“explicit current dialogue > memory”, “memory는 otherwise-missing preference slot만”
규칙에 의존한다 (`ecpr/prompts.py:30-38`). 따라서 query mask는 일반 목적 semantic
parser가 아니라, 좁고 재현 가능한 추가 안전장치다.

## Action prompt 안전 규칙

Single/multi template 모두 다음을 강제한다.

- 현재 dialogue와 schema가 authoritative이며 explicit user value가 memory보다 우선;
- memory는 누락된 preference slot만 채움;
- 요청 operation 변경, schema 밖 function/slot/value 생성 금지;
- weak/conflicting/ambiguous/insufficient memory는 생략;
- 과거 API call 전체 replay 금지;
- date/time/location/name/identifier/itinerary 같은 transient history 재사용 금지;
- 현재 missing preference slot에 관련된 안정 선호값만 재사용;
- 출력은 설명 없는 단일 `FunctionName(slot="value", ...)` call.

구현 근거는 `ecpr/prompts.py:30-132,321-350`이다. 이 prompt precedence는 query mask가
포착하지 못하는 일반 자연어 제약의 최종 방어선이지만, 모델 기반 지시이므로 deterministic
mask와 같은 완전성을 주장하지 않는다.

## Causal ablations

Ablation은 final candidate 정의가 아니며, 각 flag도 VLT 권한을 넓히지 않는다.

- `no_latent`: routed typed evidence는 유지하되 closed VLT tuple을 금지.
- `no_typed`: typed routing과 typed corroboration이 모두 사라지므로 VLT도 금지.
  즉 raw latent-only prompt가 아니다.
- `no_counterevidence`: direct typed router의 conflict-ratio filter만 제거할 수 있지만
  VLT의 qualified evidence/opposition/target-slot 검사는 완화하지 않음.
- `no_routing`: direct typed selection threshold를 완화하되 public domain/schema/
  preference-slot/query-mask 경계와 VLT qualification은 유지.

## Target-free와 non-cheating 경계

VLT3 ontology와 query registry provenance는 paper Table 5, frozen public schemas,
preference-slot config, target-independent generic lexicon만 authority로 선언한다. Current or
held-out target query를 taxonomy/threshold/alias/distance/test-design에 쓰는 것, raw latent
output을 mapping 근거로 쓰는 것, gold/answer/target metadata/metric을 보는 것은 금지된다
(`configs/latent_trait_ontology.vlt3.json:379-474`).

현재 query 자체는 action inference 입력이며 explicit-constraint mask와 public routing에
사용된다. 이것은 정답을 보는 것이 아니라 논문 문제 정의가 요구하는 query constraint
처리다 (PDF p.3, lines 75-84). 반대로 query에서 정답 preference를 추출해 ontology나
threshold를 사후 변경하는 feedback edge는 허용되지 않는다.

Memory builder는 sanitized history만 읽고, candidate prompt projection은 raw call과 audit
provenance를 제외한다. Evaluator-only gold 경계와 one-shot final action protocol은 이
메커니즘과 별도의 integrity layer에서 검증된다. 이 문서는 아직 실행하지 않은 target
성능이나 PReFine 대비 개선 폭을 주장하지 않는다.
