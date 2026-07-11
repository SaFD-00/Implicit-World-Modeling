# Monkey-Collector Architecture

`Monkey-Collector` 는 Android AccessibilityService 앱과 Python 서버를 조합해 GUI world model 학습용 데이터를 수집하는 파이프라인이다. 현재 코드는 "전환 감지는 App, 액션 선택과 저장은 Server" 구조를 기준으로 구현되어 있다.

## 1. 시스템 개요

### 역할 분리

- Android App
  - foreground 앱 감지
  - 화면 안정화 판단
  - screenshot 및 XML dump 생성
  - TCP 로 신호와 payload 전송
- Python Server
  - TCP 수신
  - XML 파싱
  - 다음 action 선택
  - ADB 실행
  - raw session 저장
  - page map 및 JSONL 변환

### 핵심 설계 포인트

- 전환 감지는 App 의 [`ScreenStabilizer.kt`](./app/app/src/main/java/com/monkey/collector/ScreenStabilizer.kt) 에서 수행한다.
- no-change, first screen, external app recovery 는 server collection loop 에서 처리한다.
- 세션은 패키지명 기반 두 root — `data/{package}/`(영속)·`runtime/{package}/`(휘발성) — 이고, 기본 동작은 resume(page 지식 복원 포함)이다.

## 2. 컴포넌트 구조

### Android App

경로: [`app/app/src/main/java/com/monkey/collector`](./app/app/src/main/java/com/monkey/collector)

- `CollectorService.kt`
  - AccessibilityService 본체
  - foreground package / activity 추적
  - screen change 발생 시 screenshot + XML 전송 (캡처 본체는 `AccessibilityService.takeScreenshot`)
  - external app 감지 및 client-side 복구. `EXCLUDED_PACKAGES` 에 systemui/permissioncontroller/`com.monkey.collector` 와 함께 **`com.google.android.gms`/`gsf`/`com.android.vending`** 를 포함해, Google 로그인·Play 핸드오프 화면을 "외부 앱(타깃 이탈)"으로 처리한다 — 미포함 시 sign-in 핸드오프에서 외부앱 재실행 스톰이 발생했다.
  - 서버 드리븐 standby 루프: TCP 연결을 유지하며 서버의 `START {package}` 수신 시 `startCollection` 트리거, `SESSION_END` 수신 시 `stopCollection` 수행
- `ScreenStabilizer.kt`
  - 저해상도 프레임 비교 / 안정화 대기 / 시각 변화 판정 / first screen 판정
  - MediaProjection 기반 *stabilization* VirtualDisplay 관리. **동의 토큰은 모던 Android 에서 단발성**이라 세션마다 재사용 시 `createVirtualDisplay` 가 `SecurityException` 을 던진다 → `startCaptureSession` 은 ① reuse-guard(이미 projection+display 가 있으면 early-return) ② acquire/`createVirtualDisplay` try/catch 로 감싸 실패 시 imageReader/projection 정리 후 **return(=stabilization 없이 graceful-degrade)**. 과거엔 이 예외가 uncaught 라 2번째 세션에서 **client 프로세스가 사망** → 핸드셰이크 desync + signal timeout 연쇄였다. 캡처 본체는 `AccessibilityService.takeScreenshot` 이라 stabilization 이 degrade 돼도 수집은 계속된다.
- `BitmapComparator.kt`
  - 프레임 diff 계산
- `ScreenCapture.kt`
  - `AccessibilityService.takeScreenshot()` 래퍼
- `XmlDumper.kt`
  - Accessibility tree -> raw XML
- `TcpClient.kt`
  - App -> Server `P/S/X/E/N/F`
  - Server -> App JSON control message 수신
- `MainActivity.kt`
  - IP / Port 설정
  - MediaProjection 권한 브리지
- `MediaProjectionHelper.kt`
  - Activity 와 Service 사이 권한 데이터 전달

### Python Server

경로: [`src/monkey_collector/`](./src/monkey_collector)

- `domain/`
  - [`actions.py`](./src/monkey_collector/domain/actions.py): Action dataclass 들
  - [`activity_coverage.py`](./src/monkey_collector/domain/activity_coverage.py): Activity coverage CSV. ground truth 의 분모(`total_activities`)와 분자 후보 집합 모두 `session_manager._resolve_declared_activities` 가 결정하며, 1차 소스는 [`catalog/activities.json`](./catalog/activities.json) (androguard manifest 추출), 폴백은 `adb dumpsys package`. catalog hit 시 (`allow_dynamic_total=False`) 분모 고정 + 분자(`unique_visited`)는 catalog set 안의 activity 만 normalize 후 카운트 + `coverage = min(1.0, ...)` 클램프. catalog 외 activity 는 `activity` 컬럼에 그대로 기록되지만 coverage 에는 영향 없음. catalog `aliases`(activity-alias→targetActivity) 는 `_alias_map` 으로 로드돼 record/resume/`is_declared` 에서 normalize 직후 target 으로 해석된다 — alias 방문이 target 카운트로 병합되고(예: Calendar `.AllInOneCalendarActivity`→`allinone.AllInOneCalendarActivity`) 분모는 늘지 않는다. catalog miss 폴백 (`allow_dynamic_total=True`) 은 legacy 동작 (target package 의 미선언 activity 발견 시 분모 동적 확장 + 모든 visited 카운트). coverage 기록에 쓰는 activity 는 `collection_loop._resolve_coverage_activity` 가 판별한다 — a11y `activity_name` 이 declared activity 이면(`is_declared` = `_total_set` 멤버십) 그대로 신뢰(adb 미호출), 아니면(다수 프레임에서 generic `.../android.view.ViewGroup` 로 보고돼 coverage 가 고착되던 케이스) `adb.get_current_activity()`(topResumedActivity, bounded 5s)로 실제 foreground activity 를 해석해 그 값을 기록한다. matcher/page_graph/save_observation/event 는 원래 a11y `activity_name` 을 그대로 유지(구조적 dedup 불변).
  - [`cost_tracker.py`](./src/monkey_collector/domain/cost_tracker.py): LLM 비용 추적 CSV. `agent` 컬럼으로 호출 주체(`text_generator` / `screen_grouper`) 구분.
  - [`page_graph.py`](./src/monkey_collector/domain/page_graph.py): 페이지 그래프 생성. live 수집은 `get_or_create_page_by_match`(ScreenMatcher 의 `page_key` → 노드)로, 오프라인 재구성은 두 갈래다 — `pages/` 가 있는 세션은 `build_graph_from_new_layout`(events.jsonl 의 `page_key`/`observation_num` 을 직접 읽는 **정확한** 재구성, 구조 근사 불필요)로, `pages/` 없이 구형 flat `xml/` 만 있는 마이그레이션 이전 세션은 기존 `build_graph_from_session`(activity+Jaccard 구조 근사)으로 degrade 한다. `PageNode` 에 `page_key`/`element_names`/`observation_count` 필드(구 `page_graph.json` 은 `.get` 기본값으로 로드). `next_observation_num`(legacy no-matcher 경로 전용 observation 할당, 매번 새로 할당)과 `record_observation`(두 경로가 공유하는 `observation_count` 증가 지점)이 있다. matcher 가 `pending` 으로 거부한 로딩/스플래시 화면은 노드를 만들지 않는다(collection_loop 에서 `get_or_create_page_by_match` 호출 자체를 건너뜀).
- `llm/` — 모든 LLM 소비자가 공유하는 단일 클라이언트
  - [`client.py`](./src/monkey_collector/llm/client.py): `LLMClient` — env 기반(`OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` / `OPENROUTER_MODEL`, 기본 `qwen/qwen3.7-plus`) OpenRouter **Chat Completions** 래퍼. `chat()` 한 경로로 호출하며 cost_tracker 에 `prompt_tokens`/`completion_tokens` 기록. `chat()` 은 per-call `timeout`(`.create(timeout=)`)·`max_retries`(`.with_options(max_retries=)`) override 를 받는다 — 공유 클라이언트의 전역 기본값(`self._timeout`, SDK 기본 retries)은 건드리지 않아 다른 소비자(text_generator 등)에 영향 없음. `create_llm_client()` 는 키 없으면 `None` 반환.
  - [`element_extractor.py`](./src/monkey_collector/llm/element_extractor.py): `ElementExtractor` — **단일 호출 element 추출**(MobileGPT-V2 Node-Clustering 포팅). `encode_to_html_xml` 표현을 LLM 에 보내, 각 element 마다 `name`/`description`/`parameters` 와 `element_index`(같은 기능 family 전체)·`key_element_index`(대표 anchor 1~3)를 **한 번에** 받는다(MobileGPT-V2 가 SubtaskExtractor·TriggerUIAgent 두 호출로 나눠 뽑는 것과 달리 1호출로 합침). `description`/`parameters` 는 `ElementFamily` 를 거쳐 `{step}_elements.json` 까지 그대로 영속된다. `known_elements` 가 주어지면 그와 겹치지 않는 것만 반환(expand). 이 추출은 matcher 의 per-frame hot path 위에서 돌므로 in-loop 호출을 bound 한다 — `max_tokens=6000`(정상 화면 1000~4600 토큰; runaway 생성이 13151 토큰으로 메인 루프를 ~4분 블록하던 사건을 상한, 정상 화면은 truncate 안 함) + `timeout=60s`(hung provider connection bound) + `max_retries=1`(타이트한 timeout 이 기본 retries 로 wall-clock 을 배가하지 않게). truncation/timeout 은 예외로 잡혀 `[]` 로 graceful degrade(수집 흐름 무영향). `create_element_extractor()` 는 클라이언트 없으면 `None`.
  - [`prompts/element_extractor_prompt.py`](./src/monkey_collector/llm/prompts/element_extractor_prompt.py): MobileGPT-V2 의 subtask_extractor(추출 규칙·SAME-FUNCTION GROUP·KNOWN 제외) + trigger_ui(대표 선택 기준)를 병합한 단일 프롬프트.
- 인프라 모듈에 인접
  - [`catalog_activities.py`](./src/monkey_collector/catalog_activities.py): `catalog/activities.json` 의 process-lifetime 캐시 (`ActivityCatalog`). activity coverage ground truth 1차 소스.
- `pipeline/`
  - [`collector.py`](./src/monkey_collector/pipeline/collector.py): collector facade
  - [`session_manager.py`](./src/monkey_collector/pipeline/session_manager.py): session init/resume/finalize. `_resolve_declared_activities` 헬퍼가 catalog 우선, dumpsys 폴백 정책을 적용한다. `init_or_resume_session`(3-tuple `(session_id, resume_step, is_resumed)` 반환; `--new-session` 은 `data/`+`runtime/` 두 root 를 모두 삭제)와 `rehydrate_session`(재개일 때만: `page_graph.json` 을 `state.page_graph` 로 로드 — 안 하면 `finalize_session` 이 이번 세션에서 재방문한 page 만으로 그래프를 덮어써 나머지 page 이력을 날린다 — 그리고 `ScreenMatcher` 가 있으면 `rehydrate_screen_matcher` 호출)로 재개 시 지식 복원을 담당.
  - [`collection_loop.py`](./src/monkey_collector/pipeline/collection_loop.py): 메인 루프. 세션 종료 조건은 `_has_budget` 이 판정한다 — `budget_mode="time"` 이면 벽시계 deadline(`clock() + max_duration_sec`, `run_collection_loop` 진입마다 재계산 = per-run 예산; injectable `now` 로 테스트) 초과까지, `"steps"` 면 `state.step < state.max_step` 까지(둘 다 idle backstop `idle_iterations < max_idle` 와 AND). 시간 만료든 스텝 소진이든 루프는 **정상 return** 하므로 `_run_session` 의 `finally: finalize_session` 이 항상 실행돼 `metadata.json` 의 `completed_at`/`total_steps` 가 채워진다(과거 `completed_at:null` 은 mid-run kill 증상 — 시간 예산은 kill 없이 clean-stop 을 보장). resume 는 매 실행 fresh full duration(누적 경과시간 persist 안 함).
  - [`recovery.py`](./src/monkey_collector/pipeline/recovery.py): retry / recovery 상수와 helper
  - [`screen_guard.py`](./src/monkey_collector/pipeline/screen_guard.py): 화면 분류 가드(키보드/권한 다이얼로그/시스템·런처 화면). `SYSTEM_PACKAGES` 에 `gms`/`gsf`/`vending`/launcher 를 포함해, 타깃 앱이 Google 로그인·Play 화면으로 drift 한 것을 "앱 이탈"로 판정(클라이언트 `EXCLUDED_PACKAGES` 와 이중 방어). 권한 다이얼로그는 grant 우선 버튼 탐색으로 자동 처리.
  - [`screen_matching/`](./src/monkey_collector/pipeline/screen_matching): **BM25 unique-page matching** 엔진 (Mobile3M 메커니즘). `ScreenMatcher` 가 화면마다 page 식별을 수행하고 그 `page_key` 가 page_graph 노드와 탐색 abstract page 를 모두 결정한다 — **LLM-free**. LLM element 추출은 옵션 enrichment(새 page 의 `families` 채우기)일 뿐 식별 자체는 LLM 을 쓰지 않는다. `element_extraction`·`luminance_prefilter` 가 **모두 off** 일 때만 matcher 가 없어 구조 지문 식별(`structure_str`)로 degrade(BM25 는 무의존이라 이 gating 은 보수적 유지일 뿐 필수는 아니다).
    - [`ui_attributes.py`](./src/monkey_collector/pipeline/screen_matching/ui_attributes.py): `UIAttributes`(self+parent+children 구조 지문)와 매칭 primitives(`find_matching_node` ancestor-walk, `text_blind_requirements`, `get_ui_key_attrib`, `extract_interactable_indexes`). MobileGPT-V2 xml_parser 를 MC encoded 스키마(`tag/aria-label/alt/text/type/value`, `id`/`class` 없음)로 적응 포팅, distinctive 판정은 `aria-label/alt`.
    - [`screen_matcher.py`](./src/monkey_collector/pipeline/screen_matching/screen_matcher.py): `ScreenMatcher.match` — ① 구조 지문 pre-filter(exact 재방문 short-circuit, 캐시된 `(page_key, observation_num)` 그대로 반환·재사용) + interactable(button/input) 0개 화면 거부(`pending` — 로딩/스플래시가 빈 page 로 등록돼 이후 화면을 모두 흡수하는 blackhole 차단; 첫 유효 화면의 page_key 가 `0`) ② **serialize**: encoded XML → 정규화 element-line 문서(`element_lines.serialize_element_lines`; leaf/scroll 노드 1줄씩, `index`/`bounds` 제거) ③ **BM25 retrieve**: `bm25.Bm25Index` 로 저장 page 들의 element-line 문서 중 top-K 후보 ④ **conjunctive verify**: 각 후보를 점수순으로 element 기준(대칭차 `|A △ B| < element_diff_max`, 또는 Jaccard `> element_jaccard_min`) **AND** pixel 게이트(luminance 차이 비율 `< page_pixel_diff_threshold`, 기본 0.3)로 확인 — 둘 다 통과하는 첫 후보를 `BM25_MERGE`(stored page_key), 모두 실패면 `NEW`(새 page_key, element-line 문서를 BM25 코퍼스에 add). 이 엔진은 **LLM-free**다. **PAGE 식별이 끝나면 별도로 OBSERVATION 식별**을 한다: merge 라면 `_record_observation`이 그 page 만의 luminance 지문(`_page_luminance_lookup`, 더 빡빡한 `screenshot_diff_threshold` 기준 — PAGE 게이트와 별개)으로 기존 observation 재사용 여부를 판단하고, 없으면(또는 prefilter 비활성/screenshot 없음) 새 observation 을 할당한다; 새 page 는 항상 observation 0. 결과는 `ScreenMatch.observation_num`/`is_new_observation` 에 담기며, 호출자(`collection_loop.py`)는 `is_new_observation` 일 때만 `data/{package}/pages/{page_key}/{observation_num}/` 에 새 파일을 쓴다. **`persist_filtered`(config `screen_matching.persist_filtered`, 기본 on)면 구조 prefilter 재방문·BM25 merge dedup 도 그 page 아래 새 `observation_num` 을 받아 `is_new_observation=true` 로 저장된다**(방문마다 `0,1,2,…` per-visit 체인; page 정체성 불변, 구조 prefilter 히트는 near-dup 이라 luminance 지문 재-append 생략) — off 면 재사용이라 쓰기 없음(기존 절약 동작). `families`(현재 인덱스 element family, `description`/`parameters` 포함)는 extractor 있을 때 새 페이지면 추출 family 를, merge·구조 재방문이면 매칭 page 의 저장 element 를 현재 화면 anchor 로 재매핑해 채워 반환한다(extractor 없으면 빈 값; 재방문 family 는 `elements.json` 에만 쓰이고 same-function 압축에는 미반영 — page_key 가드). pixel 게이트는 no-screenshot/luminance off/후보 지문 없음이면 abstain(True) 해 element 기준 단독 판정, size 불일치면 `luminance_diff`=1.0 → 게이트 실패 = 다른 page(Mobile3M dimension-mismatch=different). (한계: element-line 은 button/input 없는 scroll/long-click 전용 화면을 pending 처리.) **enrichment-off 모드**(extractor=None): BM25 matching 은 그대로 돌고 `families` 만 빈 값 — `element_extraction` off 여도 page/observation dedup 이 유지된다. 세션 재개 시 `pipeline/screen_matching/rehydrate.py` 가 `reset()` 직후 `_registry`/`_fp_to_key`/`_counter`/BM25 코퍼스를 디스크에서 복원한다(아래 참조).
    - [`element_lines.py`](./src/monkey_collector/pipeline/screen_matching/element_lines.py): encoded XML → 정규화 element-line 문서(`serialize_element_lines`; 참고 `xml_to_html.any_tree_to_html` 를 MC encoded 스키마로 이식 — leaf/scroll 노드 1줄씩, `index`/`bounds` 제거, 속성 고정순서 `aria-label/alt/type/value/checked/role/data-scroll`+text)와 set 기준(`element_diff_count`=`|A △ B|`, `element_jaccard`). 순수 파이썬, LLM/numpy/PIL 무의존.
    - [`bm25.py`](./src/monkey_collector/pipeline/screen_matching/bm25.py): `Bm25Index` — Okapi BM25(`k1=1.5, b=0.75`) 이식(참고 대비 **top-1→top-K**, numpy 제거). 토큰 = element-line 문자열 전체(참고 `readlines()` 방식). `add_document` 마다 df/idf/avgdl 전체 재계산(참고 `appendItem` 의 stale-IDF 버그 회피), `+1` IDF 변형(1-doc 비음수)·`avgdl==0` 가드. `add_document`/`clear`/`top_k(query,k)`(score-desc, 동점 page_key 오름차순).
    - [`luminance.py`](./src/monkey_collector/pipeline/screen_matching/luminance.py): luminance 게이트/observation-dedup primitives (MobileGPT-V2 `memory_manager` 포팅). `extract_luminance_features`(screenshot bytes → `convert("RGB")` → width-100 LANCZOS 리사이즈 → `convert("L")` = ITU-R BT.601 luma)·`luminance_diff`(`ImageChops.difference` 히스토그램의 strict `|ΔY| > threshold` 픽셀 비율). **순수 Pillow**(numpy 미사용; `Pillow>=10.0` 의존성). 지문 자체(`PageKnowledge.luminance_features`, `(observation_num, PIL Image)` 쌍, page 당 `_MAX_LUMINANCE_OBS=10` cap)는 세션 인메모리로만 보관해 별도 캐시 파일로 영속하지는 않지만, 재개 시 `rehydrate.py` 가 각 observation 의 저장된 `screenshot.png` 로부터 이 함수로 다시 뽑아 복원한다 — 원본 스크린샷이 이미 디스크에 있으므로 별도 지문 캐시 포맷을 새로 만들지 않는다는 원래 설계 방침 유지.
    - [`page_knowledge.py`](./src/monkey_collector/pipeline/screen_matching/page_knowledge.py): `PageKnowledge`(page_key·elements·`key_elements`(name→anchor 지문)·`extra_uis`·**`element_lines`**[BM25 문서, 생성 시 고정]·`luminance_features`[`(observation_num, PIL Image)` 쌍, 인메모리 capped]·`next_observation_num`[다음 관측 번호, 인메모리]) + `KnowledgeRegistry`(세션별 in-memory 저장). `to_dict()`/`from_dict()` 로 `page.json`(고정 anchor + `element_lines`; luminance/next_observation_num 은 제외 — 재개 시 재파생) 왕복 직렬화. `element_lines` 직렬화는 additive/back-compat.
    - [`rehydrate.py`](./src/monkey_collector/pipeline/screen_matching/rehydrate.py): `rehydrate_screen_matcher(matcher, writer)` — 재개 시 `data/{package}/pages/` 를 순회하며 각 page 의 `page.json` 을 로드하고, 각 observation 의 `raw.xml`(→ `_fp_to_key` 구조 캐시 재구성)과 (luminance 활성 시) `screenshot.png`(→ luminance 지문 재추출)로 등록. `element_lines` 가 없는 legacy `page.json` 은 첫 observation 의 `raw.xml` 로부터 재계산(fallback). `matcher.rehydrate(...)` 가 BM25 코퍼스를 각 page 의 `element_lines` 로 재구축한다(page_key 순). `_counter` 는 발견된 최대 page 번호(0-based 정수) 다음 값. `matcher.reset()` 직후, `session_manager.rehydrate_session` 이 호출한다.
  - [`exploration/`](./src/monkey_collector/pipeline/exploration): LLM-guided 탐색 엔진 (LLM-Explorer 포팅). `Explorer` Protocol 을 구현하는 `LLMGuidedExplorer` 가 coverage-driven unexplored-first 선택 + same-function 압축 + 최단경로 navigation 을 수행한다. `set_match_context(page_key, families)` 로 ScreenMatcher 결과를 받는다(없으면 `structure_str` fallback).
    - [`state.py`](./src/monkey_collector/pipeline/exploration/state.py): `SemanticState` — raw XML → `state_str`(내용 포함)·`structure_str`(구조만)·`page_key`(element-set 식별, matcher 없으면 structure_str)·`SemanticElement` 목록. element 는 encoded index 기준이라 extractor family 와 1:1 정렬, scroll 컨테이너는 UITree 에서 음수 index 로 보강.
    - [`memory.py`](./src/monkey_collector/pipeline/exploration/memory.py): `Memory` — `(page_key, element_signature, action_type)` 단위 커버리지 추적. extractor 의 element family(`element_index`)로 동등 element 를 한 번에 explored 처리(탐색 공간 압축); same-function group 은 `page_key` 첫 sighting 에만 계산하고(matcher 없으면 degrade), merge 재방문은 family 가 전달돼도 첫 sighting group 을 재사용한다(page_key 가드).
    - [`transition_graph.py`](./src/monkey_collector/pipeline/exploration/transition_graph.py): `TransitionGraph` — navigation 용 `page_key` 그래프(networkx). `shortest_nav_steps` 가 미탐색 화면까지 최단경로 산출.
    - [`navigator.py`](./src/monkey_collector/pipeline/exploration/navigator.py): `Navigator` — `_nav_steps` 큐를 매 step App 신호를 받으며 순차 소비(signature 재매칭). 무한루프 가드 포함.
    - [`action_mapper.py`](./src/monkey_collector/pipeline/exploration/action_mapper.py): semantic action(`touch/select/long_touch/set_text/scroll`) → domain `Action` 변환. `set_text` 는 `TextGenerator` 위임.
    - [`constants.py`](./src/monkey_collector/pipeline/exploration/constants.py): 엔진 튜닝 상수(LLM-Explorer 원본 값 보존).
  - [`text_generator.py`](./src/monkey_collector/pipeline/text_generator.py): random 또는 공용 `LLMClient` 기반 입력 텍스트 생성. `set_app_context()` 로 세션마다 현재 앱 설명을 받아 LLM 프롬프트에 `App under test:` 줄로 주입(`Collector._run_session` 이 호출, csv 미등록 앱은 package_id 폴백). random 전략에선 no-op.
- 인프라 모듈 (monkey_collector/ 직속)
  - [`adb.py`](./src/monkey_collector/adb.py): ADB wrapper. 상단 상수 `REQUIRED_AVD_NAME = "Pixel6-2"` 에 맞춰 `adb devices` + `emu avd name` 으로 해당 AVD 의 emulator serial 을 해석하고, 이후 모든 명령에 `-s <serial>` 을 prefix 한다. 다중 디바이스 환경에서도 단일 AVD 만 쓰도록 강제.
  - [`tcp_server.py`](./src/monkey_collector/tcp_server.py): TCP 서버와 signal queue (`CollectionServer`)
  - [`storage.py`](./src/monkey_collector/storage.py): `DataWriter` — 두 root(`data_dir`/`runtime_dir`) 로 분리된 세션 저장. `save_observation`(observation 폴더 1개의 screenshot+XML 파생 4종+`elements.json` 을 한 번에 씀, 재사용 시 호출 안 함)·`save_page_knowledge`/`load_page_knowledge`(`page.json` 왕복)·`list_pages`/`list_observations`(디스크 리스팅이 ground truth — 재개 카운터의 근거)·`next_frame_index`(events.jsonl 정렬용 단조 카운터, 파일 쓰기와 분리)·`regenerate_xml_variants`(새/구형 레이아웃 자동 판별).
- `xml/`
  - [`ui_tree.py`](./src/monkey_collector/xml/ui_tree.py): action selection 용 UI tree
  - [`structured_parser.py`](./src/monkey_collector/xml/structured_parser.py): 구조적 XML parser
  - [`parser_base.py`](./src/monkey_collector/xml/parser_base.py): `Parser` ABC
- `export/`
  - [`converter.py`](./src/monkey_collector/export/converter.py): raw session -> ShareGPT JSONL. 각 action 을 이벤트의 `page_key`/`observation_num` 으로 `data/{package}/pages/{page_key}/{observation_num}/` 에 조인하고, after 는 다음 action 의 before 화면으로 잡아 중간 로딩 프레임을 건너뛴다(연속 이벤트가 같은 observation 을 가리키면 — 재사용 — 시각적 변화가 없으므로 스킵). `pages/` 없이 구형 flat `xml/` 만 있으면 `_convert_session_legacy`(기존 `frame_index` 조인)로 degrade.
  - [`graph_visualizer.py`](./src/monkey_collector/export/graph_visualizer.py): page graph HTML 시각화

### 앱 카탈로그 & 설치 상태

`run` 서브커맨드가 수집할 앱 목록을 결정할 때 사용하는 두 모듈.

- `src/monkey_collector/pipeline/app_catalog.py`
  - `AppCatalog`: stdlib csv 로 `catalog/apps.csv` 파싱, BOM/대소문자 정규화.
  - `AppJob`: frozen dataclass (category, sub_category, app_name, package_id, source, priority, notes, installed). `description` 프로퍼티가 `app_name (category/sub_category) — notes` 한 줄을 만들어 input-text LLM 프롬프트의 앱 컨텍스트로 쓰인다(CLI `_resolve_app_contexts` → `Collector(app_contexts=...)`).
  - `filter(categories, priorities, installed)`: case-insensitive 필터. `installed=True` 는 `catalog/apps.csv` 의 `installed` 컬럼이 `true` 인 앱만 반환.
  - `installed_apps()` / `find_by_package(pkg)` 헬퍼로 `run --apps all` 과 명시적 패키지 목록 해소를 지원.
- `src/monkey_collector/pipeline/installed_sync.py`
  - `adb shell pm list packages` 결과로 `catalog/apps.csv` 의 `installed` 컬럼을 in-place (임시파일 + `os.replace`) 로 갱신. `sync-installed` 서브커맨드의 백엔드.

실행 흐름 (`run --apps …`):

```
AppCatalog.load(catalog/apps.csv)
  -> installed_apps()  (--apps all)
     or resolve tokens (--apps com.X com.Y)
  -> candidate packages

filter completed:
  -> scan {output}/{pkg}/metadata.json
  -> drop pkg where completed_at is set (unless --force)
  -> packages: list[str]

Collector.run_queue(packages)
  for pkg in packages:
    wait_for_client_connection
    server.send_start(pkg)
    client replies with P message
    adb.force_stop(pkg) + adb.launch_app(pkg)
    run_collection_loop   # 단일 세션 — 중단된 세션은 여기서 resume
    finalize_session      # completed_at 기록 + SESSION_END 송신
```

## 3. 데이터 흐름

### 수집 루프

```
Android AccessibilityEvent
  -> ScreenStabilizer 안정화 판단
  -> no-change 이면 N signal
  -> 외부 앱 감지면 E signal
  -> 변화가 있으면 screenshot + XML + metadata 전송
  -> Python server 가 latest signal 소비
  -> XML parse + ScreenMatcher 가 page 식별 + observation 식별 (BM25 unique-page matching, LLM-free; element_extraction on 이면 새 page 에서 families enrichment 만 추가)
       (pre-filter[+interactable 0개 화면은 pending 거부] → element-line serialize → BM25 top-K → (element diff AND pixel gate) 검증 → page_key
        → (page 확정 후) 그 page 소유 observation 중 픽셀 일치하는 게 있으면 재사용, 없으면 새 observation 할당)
       → pending 이면 page 노드·observation 파일 생성 안 함(로딩/스플래시 대기)
       → 아니면 page_graph 노드 = page_key, 새 페이지 element family(description/parameters 포함) 는 탐색에 전달;
         observation 이 새로 할당됐을 때만 data/{package}/pages/{page_key}/{observation_num}/ 에 screenshot+XML+elements.json 저장(persist_filtered on 이면 필터된 재방문도 새 observation 으로 저장, off 면 재사용은 쓰기 없음), 새 page 면 page.json 도 저장
  -> LLMGuidedExplorer 가 action 선택 (미탐색 우선 + element family 압축 + 미탐색 화면 navigation; input_text 필요 시 공용 LLMClient 로 텍스트 생성)
  -> ADB 실행
  -> events.jsonl 에 frame_index/page_key/observation_num 기록 (새 관측·재사용 관계없이 매 처리 화면)
  -> 다음 step 반복
```

세션 재개(resume) 시에는 루프 시작 전에 `session_manager.rehydrate_session` 이 `data/{package}/pages/` 를 다시 읽어 `state.page_graph` 와 `ScreenMatcher` 지식을 모두 복원한다 — 그렇지 않으면 이미 알고 있는 page 를 다시 "새 page" 로 등록하고, `finalize_session` 이 이번 세션에서 재방문한 page 만으로 `page_graph.json` 을 덮어써 나머지 이력을 잃는다.

### TCP 프로토콜

App -> Server:

- `P`: target package
- `S`: screenshot payload
- `X`: XML + activity + package metadata
- `E`: external app signal
- `N`: no-change signal
- `F`: finish signal

Server -> App (newline-delimited JSON):

- `{"type":"START","package":"<pkg>"}`: 서버가 다음으로 수집할 앱을 지정. 클라이언트는 `startCollection` 을 트리거하고 `P` 메시지로 같은 pkg 를 회신한다.
- `{"type":"SESSION_END"}`: 현재 세션 종료 요청. 클라이언트는 `stopCollection` 을 수행하고 `F` 회신 후 소켓을 닫은 뒤 즉시 새 소켓으로 자동 재접속한다.

`CollectionServer` 는 signal queue 를 사용해 최신 signal 기준으로 collection loop 를 진행한다.

#### 세션 전환 핸드셰이크

`run_queue` 가 다음 앱으로 넘어갈 때:

1. 이전 세션 `finalize_session` 이 `SESSION_END` 송신
2. Android 가 `F` 회신 → 서버 `_handle_client` 의 `F` 핸들러가 break 후 `_client = None`
3. Android 가 fresh 소켓으로 재접속 → `_run` 의 accept 루프가 새 `_client` 등록
4. Python 측은 `reset_for_new_session()` 으로 큐/이벤트 상태만 초기화하고 **소켓은 그대로 유지** — 닫으면 클라이언트가 한 번 더 재접속하지 않아 다음 `wait_for_connection` 이 타임아웃한다.

### Action Space 와 탐색 전략

[`src/monkey_collector/domain/actions.py`](./src/monkey_collector/domain/actions.py) 의 6종 domain action 을 [`exploration/`](./src/monkey_collector/pipeline/exploration) 의 `LLMGuidedExplorer` 가 선택한다. 엔진 내부에서는 `(touch / select / long_touch / set_text / scroll)` 의 semantic action 으로 추론하고, `ActionMapper` 가 domain action 으로 변환한다.

| 엔진 action | domain action | 설명 |
| ----------- | ------------- | ---- |
| `touch`     | `Tap`         | clickable element 탭 |
| `select`    | `Tap`         | checkbox/toggle 탭 |
| `long_touch`| `LongPress`   | 롱프레스 (해당 element 의 `touch` 가 explored 된 뒤에만 후보) |
| `set_text`  | `InputText`   | 포커스 후 텍스트 입력 (`TextGenerator` 가 생성) |
| `scroll`    | `Swipe`       | scroll 컨테이너 수직 스와이프 |
| (fallback)  | `PressBack`   | 더 이상 탐색·도달할 게 없을 때 후퇴 |

이 6종 외에 **`OpenApp`(`action_type: "open_app"`)** 이 있다. 단 탐색 액션이 아니라 **record-only** 다 — `select_action` 이 생성하지도, `execute_action` 이 실행하지도 않는다. external app 이탈에서 복구 루프가 타깃 앱을 재실행할 때(아래 §4) events.jsonl 에만 기록되며, **navigation 전이로는 절대 쓰이지 않는다**(live page graph·explorer routing memory·offline page-graph 재빌드가 모두 배제; 이벤트에 `transition: false` 표식). `ACTION_REGISTRY` 에는 등록돼 있어 `action_from_dict` 라운드트립은 가능하다.

매 step `select_action` 오케스트레이션:

1. **진행 중 navigation** 이 있으면 큐의 다음 step 을 현재 화면에서 signature 로 재매칭해 실행.
2. **현재 화면 미탐색** action 이 있으면 그중 하나를 선택(`long_touch` 후순위).
3. 없으면 **전역 미탐색** action 을 target 으로 골라 `TransitionGraph` 최단경로를 큐에 적재하고 첫 step 실행. **어느 target 을 고를지는 `exploration.strategy` (DFS/BFS/GREEDY) 가 결정한다** (아래 참조).
4. 그래도 없으면 **back** 으로 후퇴 (첫/루트 화면에서는 앱 종료 방지를 위해 back 대신 화면 내 tap).

#### navigate-target 선택 전략 (`exploration.strategy`)

위 3단계의 **전역 미탐색 target 선택**은 `exploration.strategy` 값에 따라 달라진다([`navigator.py`](./src/monkey_collector/pipeline/exploration/navigator.py) `plan_to_unexplored`). 세 전략 모두 미탐색 target 을 하나 고른 뒤 현재 화면에서 그 target 까지 `TransitionGraph` 최단경로로 라우팅하는 점은 같고, **어떤 target 을 고르는가**만 다르다.

- **GREEDY** — 현재 화면에서 navigation path 가 가장 짧은 (= 이동 비용 최소) target. 기존 동작이며 `LLMGuidedExplorer.__init__` 의 backward-compat 기본값.
- **BFS** — 세션 root page 로부터 **BFS depth 가 가장 얕은** target (root 에 가까운 shallow 화면 우선). builtin/`config/run.yaml` 의 canonical 기본값(= production 기본값).
- **DFS** — 세션 root page 로부터 **BFS depth 가 가장 깊은** target (deep 화면 우선).

depth 는 `TransitionGraph.bfs_distances(root_page_key)` 로 산출하며 root 는 `Memory.root_page_key`(이번 세션에서 가장 먼저 관측한 state)다. root 가 아직 없으면(세션 첫 step) BFS/DFS 도 GREEDY 로 폴백하고, depth 를 알 수 없는 후보는 BFS 에선 가장 깊게(=후순위)·DFS 에선 가장 얕게 취급한다. 동률은 navigation path 가 짧은 쪽으로 깬다. CLI/config 가 해소한 strategy 를 `cli.py` 가 `LLMGuidedExplorer` 에 명시적으로 주입하므로 production 기본값은 BFS 가 된다.

핵심 특성:

- 좌표가 아닌 **element signature**(content 기반)로 커버리지를 추적해 스크롤/리렌더에 강건.
- `ElementExtractor` 의 element family(`element_index`)를 **탐색에 반영** — 동등 element 하나를 탐색하면 family 전체가 explored 되어 탐색 공간이 압축된다. LLM 키가 없으면(matcher 미생성) family 없이 순수 unexplored-first 로 degrade 한다.
- 세션마다 `explorer.reset()` 으로 메모리(transition graph·커버리지)를 격리해 앱 간 오염을 막는다.
- abstract page 식별은 **`page_key`**(ScreenMatcher 의 BM25 unique-page 식별; matcher 없으면 `structure_str` fallback)로 하며, 이 동일한 `page_key` 가 `page_graph.json` 산출물(live `get_or_create_page_by_match`)의 노드도 결정한다 — 과거 구조 지문 기반 디커플은 BM25 element-line matching 으로 커플링되었다(Mobile3M 메커니즘).

실행은 `LLMGuidedExplorer.execute_action` 이 `AdbClient` ([`src/monkey_collector/adb.py`](./src/monkey_collector/adb.py)) 메서드로 위임. `AdbClient` 는 CLI 진입점에서 단일 인스턴스로 생성되어 `LLMGuidedExplorer` 와 `Collector` 에 주입된다. 생성 시점에 `Pixel6-2` AVD 의 emulator serial 을 해석해 저장하므로, 해당 AVD 가 실행 중이어야 한다.

## 4. 세션 관리와 복구

### 세션 라이프사이클

- 저장 위치는 두 root: `data/{package}/`(영속)·`runtime/{package}/`(휘발성) — timestamp 기반 새 디렉터리는 만들지 않는다.
- 동일 패키지에 `runtime/{package}/metadata.json` 이 있으면 resume. 이때 `completed_at` 은 `None` 으로 되돌아가고, `session_manager.rehydrate_session` 이 `data/{package}/pages/` 로부터 `state.page_graph` 와 `ScreenMatcher` 지식을 복원한다(재개해도 기존 page 를 다시 "새 page" 로 등록하지 않음).
- `run` 은 큐 구성 단계에서 `completed_at` 이 채워진 앱을 자동 skip 한다. `--force` 로 우회.
- `run --new-session` 은 해당 앱의 `data/{package}/`·`runtime/{package}/` 를 **모두** 삭제하고 새로 시작 — 한쪽만 지우면 남은 쪽에서 지식이 다시 rehydrate 돼 "새 세션"이 되지 않는다.
- `reset` 서브커맨드로 범위 단위 (all / apps) 일괄 삭제 가능 (`--data-dir`/`--runtime-dir` 두 root 모두 대상)
- 세션 정상 종료(스텝 소진 또는 시간 예산 만료, `budget_mode` 에 따라) 시 `completed_at` 기록, page graph 재빌드, HTML 시각화 생성. 다음 `run` 부터는 이 앱이 큐에서 자동 제외.

### 주요 복구 규칙

[`src/monkey_collector/pipeline/recovery.py`](./src/monkey_collector/pipeline/recovery.py) 기준 상수:

- `MAX_NO_CHANGE_RETRIES = 3`
- `MAX_EXTERNAL_APP_RETRIES = 10`
- `MAX_SAME_PAGE_STEPS = 5`
- `MAX_EMPTY_UI_RETRIES = 2`

주요 동작:

- no-change 시 이전에 실패한 element 를 exclusion 하고 재선택
- first screen 에서는 back 을 금지하고 tap fallback 사용
- external app 시 `return_to_app()` 후 필요하면 `recover()` 수행. gms/Play 패키지는 `screen_guard.SYSTEM_PACKAGES`(+클라이언트 `EXCLUDED_PACKAGES`)로 drift 판정.
  - **open_app 기록**: 복구가 실제로 타깃 앱을 **재실행**하면(`return_to_app`/`recover` 가 launch 했으면 `True` 반환) 그 재실행을 `open_app` 액션으로 events.jsonl 에 **excursion 당 1회**(`state.open_app_logged` dedup; 다음 in-app 프레임에서 재무장) 기록한다 — open_app 학습 데이터 확보용. 단 이 open_app 은 **의도된 화면 전이가 아니므로 navigation 으로 쓰지 않는다**: 복구 직후 `state.last_action`/`last_ui_tree` 를 비워 live page graph 엣지를 막고, `return_to_app`/`recover` 가 explorer 의 `_last_record` 를 비워 routing memory 전이를 막고, 이벤트의 `transition: false` 가 offline 재빌드(`_load_events`)와 world-modeling 변환에서 배제시킨다. `from_package`(이탈한 외부 앱), `app_name`(apps.csv 조인, 없으면 빈 문자열)도 함께 남긴다.
  - 한계: 서버측 기록은 **서버가 재실행한 경우**만 포착한다. 클라이언트(`CollectorService.kt`)가 `consecutiveBackCount>=3`/launcher 에서 자체 force-launch 하면 서버는 모르므로 그 경우 open_app 이 과소기록될 수 있다.
- 빈 UI tree 가 반복되면 대기 후 재시도. 빈 판정은 raw count 가 아니라 `get_interactable_elements()` 기준(clickable/scrollable/editable 등이 0일 때) — 비상호작용 노드만 있는 화면을 "콘텐츠 있음"으로 오판하지 않는다.
- **런타임 권한 다이얼로그 자동 허용**: XML 신호가 온 경우 `is_permission_dialog` → `_handle_permission_dialog` 가 grant 버튼(`PERMISSION_BUTTON_KEYWORDS`, "while using the app" 최우선)을 탭한다. 단 permissioncontroller `GrantPermissionsActivity` 는 **accessibility 이벤트를 안 내보내** 신호 없이 timeout 만 발생하므로, timeout 경로에서 `_try_grant_permission_via_adb` 가 adb 로 foreground 를 확인하고 `uiautomator dump` 한 뒤 **clickable 버튼만** 스캔해 "While using the app"(없으면 "Allow") 를 탭한다("Only this time"/"Don't allow" 는 deny-guard 로 회피). 권한 미허용 화면을 relaunch 로 건너뛰지 않고 허용 후 탐색을 계속한다.
- 세션 핸드셰이크에서 START 후 client `P`(package ACK)가 5초 내 없으면 `session_manager.receive_target_package` 가 abort(`None`) — 죽은 세션(클라이언트 크래시/stale 소켓)에 step 예산을 blind 소진하지 않고 다음 앱으로 넘어간다.

## 5. 저장 포맷

세션은 **두 root** 로 나뉜다 — `data/{package}/` (영속 코퍼스: page/observation 지식, page_graph)와 `runtime/{package}/` (휘발성: 재개 상태, cost/coverage, 액션 타임라인). `DataWriter(data_dir="data", runtime_dir="runtime")` 가 이 split 을 관리한다.

```
data/{package}/
├── pages/
│   ├── 0/                            # page_key (0-based 정수, page_ 접두사·zero-pad 없음)
│   │   ├── page.json                 # PageKnowledge.to_dict() — elements/key_elements/extra_uis. 새 page 생성 시 1회만 기록(merge 는 anchor 를 바꾸지 않으므로 재기록 없음)
│   │   ├── 0/                        # observation 0 (이 page 의 첫 관측)
│   │   │   ├── screenshot.png
│   │   │   ├── raw.xml
│   │   │   ├── parsed.xml
│   │   │   ├── hierarchy.xml
│   │   │   ├── encoded.xml
│   │   │   ├── pretty.xml
│   │   │   └── elements.json         # --element-extraction on 일 때만 (element-set match)
│   │   └── 1/                        # observation 1 (다른 렌더 상태, 또는 persist_filtered on 이면 필터된 재방문)
│   │       └── ... (동일 7개 파일)
│   └── 1/
│       └── 0/...
├── page_graph.json
└── page_graph.html

runtime/{package}/
├── metadata.json
├── events.jsonl
├── activity_coverage.csv     # ground truth: catalog/activities.json (fallback: dumpsys)
└── cost.csv
```

page 디렉터리명은 `page_key`(0-based 정수, 예 `0`/`1` — `page_` 접두사 없음), observation 디렉터리명은 `observation_num`(0-based 정수) 이며 **zero-pad 를 쓰지 않는다**(예전엔 `page_0`/`0000` 였다). 이 이름은 곧 `events.jsonl`/`elements.json`/`page_graph.json` 에 조인 키로 박히는 `page_key` 값과 동일하다. **필터된 재방문(구조적 exact-match 또는 luminance 픽셀-매치)의 저장 여부는 `screen_matching.persist_filtered`(기본 on)가 결정한다** — 기본값에선 각 재방문이 그 page 아래 새 `observation_num`(방문마다 `0,1,2,…` per-visit 체인)으로 저장되고, off 면 재사용 관계 화면은 새 파일을 전혀 쓰지 않는다. 어느 쪽이든 `DataWriter.save_observation`은 `ScreenMatch.is_new_observation` 이 참일 때만 호출된다(`collection_loop.py`). 마이그레이션 이전(flat `screenshots/`/`xml/`, `pages/` 없음) 세션은 그대로 남으며, `regenerate`/`page-map`/`convert` 는 그런 세션을 감지하면 자동으로 구형 로직으로 degrade 한다(flat→pages 마이그레이션 스크립트는 없음 — 재수집). 구형 `page_`/zero-pad 이름으로 저장된 세션을 새 bare 이름으로 옮기려면 `python -m monkey_collector.migrate_layout`(디렉터리 rename + JSON `page_key` 재작성 + `logs/`→`runtime/logs/` 이동, 기본 dry-run·`--apply` 로 실행·idempotent)을 쓴다.

`events.jsonl` 은 한 줄에 하나의 이벤트다. 대부분은 처리된 화면 하나당(pending 제외) 실행된 domain action(`action.to_dict()` + `step` + `frame_index` + `page_key` + `observation_num` [+`activity_name`/`no_change_retry`])이고, 그 외 두 종류의 비-탐색 마커가 섞인다:

- `{"type": "external_app", "step", ...payload}` — external 이탈 신호(TCP 수신 스레드에서 기록).
- `{"action_type": "open_app", "element_index": -1, "package", "app_name", "step", "transition": false, "trigger": "external_recovery", "from_package"}` — external 복구의 타깃 앱 재실행(메인 루프에서 excursion 당 1회). `transition: false` 는 **navigation 소비자(offline page-graph 재빌드 `_load_events`, world-modeling converter)가 이 이벤트를 건너뛰게** 하는 표식이다. events.jsonl/metadata.json 은 두 스레드에서 동시 기록되므로 `DataWriter` 가 `threading.Lock` 으로 append·counter 를 보호한다.

**`frame_index`/`page_key`/`observation_num` (정렬·조인 키).** `frame_index` 는 `DataWriter.next_frame_index()` 가 할당하는 단조 증가 카운터로, **처리된 모든 화면**(pending 포함)에 대해 정확히 한 번 호출된다 — 하지만 pending 화면은 이 함수 호출 이후 조기 반환돼 이벤트 자체가 로그되지 않으므로, `frame_index` 시퀀스에는 "소비됐지만 로그되지 않은" gap 이 있을 수 있다(무해함 — 그 gap 에 해당하는 관측 파일 자체가 없다). `page_key`/`observation_num` 이 실제 파일 위치(`data/{package}/pages/{page_key}/{observation_num}/`)를 가리키는 **진짜 조인 키**다 — converter 와 `build_graph_from_new_layout` 은 이 둘로 화면을 찾는다. `step` 은 루프 카운터 라벨이며 조인 키가 아니다 — signal timeout·no_change·empty-UI 대기·keyboard/permission/system 처리 같은 비-action 반복에서는 증가하지 않고 **실제 action 을 실행한 경우에만 증가**한다. `no_change_retry` 이벤트와 `transition: false` 마커, 그리고 `page_key`/`observation_num` 이 없는 이벤트(마이그레이션 이전 세션)는 converter·page-graph 가 건너뛴다.

`DataWriter.save_observation()` 와 `regenerate_xml_variants()` 는 raw XML 에서 아래 파생 파일을 만든다(관측 폴더 안에서는 접두사 없이, 구형 flat 레이아웃에서는 `{step}_` 접두사로).

- `parsed.xml` (`{step}_parsed.xml`): semantic HTML tags + bounds + index
- `hierarchy.xml` (`{step}_hierarchy.xml`): text / bounds / index 제거
- `encoded.xml` (`{step}_encoded.xml`): bounds 제거, index 유지
- `pretty.xml` (`{step}_pretty.xml`): encoded XML pretty-print

`elements.json` 은 `--element-extraction on` 일 때 `ScreenMatcher` 가 만든 element-set match annotation 이다: `page_key`/`match_type`/`is_new_page`/`activity`(재개 시 `_fp_to_key` 복원에 필요)와 element family 목록(`name`/`description`/`parameters`/`element_index`/`key_element_index`; index 는 그 observation 의 `encoded.xml` 기준, `description`/`parameters` 는 LLM 이 추출한 의미·인자). 새 페이지면 그 화면에서 추출된 family 가, merge·구조 재방문·luminance 재방문이면 매칭된 page 의 저장 element 를 현재 화면 anchor 에 재매칭해 채운다. `match_type` 은 `NEW`/`STRUCTURAL_IDENTICAL`/`BM25_MERGE`/`PENDING_EMPTY` 이며, `BM25_MERGE` 는 BM25 후보가 element diff + pixel 게이트를 통과해 기존 page 로 병합된 경우다(extractor off 면 `families` 는 빈 목록). interactable(button/input) 이 없는 로딩/스플래시 화면은 matcher 가 등록을 거부(`pending`)해 page 노드도 관측도 만들지 않는다(첫 유효 화면의 page_key 가 `0`). LLM 호출 결과이므로 `regenerate_xml_variants` 의 결정적 파생 대상이 아니다.

**세션 재개(resume)**: `DataWriter.resume_session()` 은 `events.jsonl` 의 최댓값 `frame_index + 1` 로 `step_count` 를 복원한다(파일 개수를 세지 않음 — pending 프레임이나 `persist_filtered` off 재사용 관측처럼 파일 없는 frame_index 가 있어 파일 수와 어긋나므로; rehydrate 는 각 page 의 `next_observation_num` 을 디스크의 최대 observation +1 로 복원해 per-visit 체인을 이어간다). `session_manager.rehydrate_session` 이 `data/{package}/pages/` 를 순회해 `state.page_graph`(← `page_graph.json`)와 `ScreenMatcher` 지식(← 각 page 의 `page.json` + 관측들의 `raw.xml`/`screenshot.png`)을 모두 복원한다.

## 6. CLI 와 공개 API

### CLI

[`src/monkey_collector/cli.py`](./src/monkey_collector/cli.py) 가 아래 서브커맨드를 제공한다.

- `run` — server-driven 수집 (`--apps all` 또는 `--apps PKG [PKG ...]`).
- `sync-installed` — `adb pm list packages` 결과로 `catalog/apps.csv` 의 `installed` 컬럼 갱신.
- `reset`
- `convert`
- `convert-all`
- `page-map`
- `page-map-all`
- `regenerate`

### 공개 API

[`src/monkey_collector/__init__.py`](./src/monkey_collector/__init__.py) 는 아래 주요 타입을 export 한다.

- `Collector`
- `AppCatalog`, `AppJob`
- `LLMGuidedExplorer`, `Explorer`
- `TextGenerator`
- `RandomTextGenerator`
- `LLMTextGenerator`
- `LLMClient`, `ElementExtractor` (+ `create_llm_client`, `create_element_extractor`)
- `ScreenMatcher` (+ `create_screen_matcher`)
- `CollectionServer`
- `AdbClient`
- `DataWriter`
- `Converter`
- `PageGraph`
- `build_graph_from_session`

## 7. 설정 시스템

수집 파라미터는 [`config/run.yaml`](./config/run.yaml) 과 [`src/monkey_collector/config.py`](./src/monkey_collector/config.py) 로 중앙화되어 있다. 같은 값을 여러 경로로 지정할 수 있고, **나중 레이어가 앞 레이어를 덮어쓴다**.

### 해석 순서 (4단계, later wins)

```
builtin defaults  (config.py `_BUILTIN_DEFAULTS`)
  → config/run.yaml
  → MC_* 환경변수
  → CLI 플래그
```

builtin canonical default 의 `exploration.strategy` 는 **BFS** 이며 `config/run.yaml` 의 canonical 값과 일치한다. 알 수 없는 strategy 값은 경고 로그 후 **GREEDY** 로 폴백한다. 운영 플래그 `--apps`/`--force`/`--new-session` 은 YAML/env 로 설정할 수 없는 **CLI 전용**이다. CLI 의 YAML-커버 파라미터는 기본값이 `None`(= "CLI 에서 지정 안 함" sentinel)이라, 플래그를 주지 않으면 config 값이 그대로 쓰인다.

### `config/run.yaml` 섹션

| 섹션 | 키 | 설명 |
| ---- | -- | ---- |
| `exploration` | `strategy` | 탐색 전략 `DFS`\|`BFS`\|`GREEDY` (canonical 기본 `BFS`). navigate-target 선택 의미는 §3 참조 |
| `collection` | `budget_mode` | 세션 종료 조건 `time`\|`steps` (제품 기본 `time`). `time` = `max_duration` 경과까지, `steps` = `max_steps` action 까지(legacy). CLI `--budget-mode`, 또는 `--steps`/`--duration` 중 준 쪽으로 추론 |
| `collection` | `max_duration` | `budget_mode=time` 일 때 벽시계 예산. 형식 `2h`\|`120m`\|`7200s`\|`7200`(맨숫자=초). 파싱 실패/음수는 경고 후 7200s 폴백 (기본 `2h`) |
| `collection` | `max_steps` | `budget_mode=steps` 일 때 앱 세션당 최대 step 수 (기본 1500) |
| `collection` | `seed` | explorer random seed (기본 42) |
| `collection` | `action_delay_ms` | action 간 지연(ms) (기본 1500) |
| `collection` | `port` | TCP 서버 포트 (기본 12345) |
| `collection` | `data_dir` | 영속 데이터 루트 — pages/observations, page_graph (기본 `data`) |
| `collection` | `runtime_dir` | 휘발성 런타임 루트 — metadata, events, cost/coverage (기본 `runtime`) |
| `llm` | `input_mode` | 입력 텍스트 생성 모드 `api`\|`random` (기본 `api`) |
| `llm` | `element_extraction` | LLM element 추출 + element-set screen matching on/off (기본 `false` — 기본적으로 LLM 은 입력 텍스트 생성에만 쓰고 element 추출은 opt-in; `on` 이어도 `OPENROUTER_API_KEY` 없으면 구조 지문으로 자동 degrade) |
| `screen_matching` | `luminance_prefilter` | luminance on/off (기본 `true`; OBSERVATION dedup + PAGE pixel 게이트 지문 공급; off 면 pixel 게이트 abstain → element 기준 단독 판정) |
| `screen_matching` | `luminance_threshold` | 픽셀 밝기 `|ΔY|` 변화 임계값 0–255 (기본 10) |
| `screen_matching` | `screenshot_diff_threshold` | 같은 **OBSERVATION** 으로 판정할 차이 픽셀 비율 (기본 0.02) |
| `screen_matching` | `luminance_low_res_width` | luminance 지문 다운스케일 너비 px (기본 100) |
| `screen_matching` | `persist_filtered` | 필터된(prefilter/dedup) 재방문을 그 page 아래 자체 observation(방문마다 `0,1,2,…`)으로 저장 on/off (기본 `true`; `false` 면 재사용 관측은 파일 미기록 = 기존 절약 동작) |
| `screen_matching` | `bm25_top_k` | BM25 로 화면당 검증할 후보 page 수 (논문 top-5; 기본 5) |
| `screen_matching` | `element_criterion` | element 동일-page 기준 `diff`(`|A△B|<element_diff_max`) \| `jaccard`(`>element_jaccard_min`) (기본 `diff`) |
| `screen_matching` | `element_diff_max` | 같은 page 로 볼 최대 상이 element-line 수 (`diff` 모드, 기본 5) |
| `screen_matching` | `element_jaccard_min` | 같은 page 로 볼 최소 element-line Jaccard (`jaccard` 모드, 기본 0.5) |
| `screen_matching` | `page_pixel_diff_threshold` | PAGE 병합을 확정하는 pixel 게이트 차이 픽셀 비율 (기본 0.3) |

### MC_* 환경변수

YAML 위, CLI 아래 레이어. 각 변수는 대응 키를 타입 변환해 덮어쓴다.

| 환경변수 | 대상 키 | 타입 |
| -------- | ------- | ---- |
| `MC_EXPLORATION_STRATEGY` | `exploration.strategy` | str (upper-case 정규화) |
| `MC_COLLECTION_BUDGET_MODE` | `collection.budget_mode` | str (`time`\|`steps`) |
| `MC_COLLECTION_MAX_DURATION` | `collection.max_duration` | str (`2h`\|`120m`\|`7200s`\|`7200`; 파싱은 config 에서) |
| `MC_COLLECTION_MAX_STEPS` | `collection.max_steps` | int |
| `MC_COLLECTION_SEED` | `collection.seed` | int |
| `MC_COLLECTION_ACTION_DELAY_MS` | `collection.action_delay_ms` | int |
| `MC_COLLECTION_PORT` | `collection.port` | int |
| `MC_COLLECTION_DATA_DIR` | `collection.data_dir` | str |
| `MC_COLLECTION_RUNTIME_DIR` | `collection.runtime_dir` | str |
| `MC_LLM_INPUT_MODE` | `llm.input_mode` | str |
| `MC_LLM_ELEMENT_EXTRACTION` | `llm.element_extraction` | bool (`true/1/yes/on`) |
| `MC_SCREEN_MATCHING_LUMINANCE_PREFILTER` | `screen_matching.luminance_prefilter` | bool (`true/1/yes/on`) |
| `MC_SCREEN_MATCHING_LUMINANCE_THRESHOLD` | `screen_matching.luminance_threshold` | int |
| `MC_SCREEN_MATCHING_SCREENSHOT_DIFF_THRESHOLD` | `screen_matching.screenshot_diff_threshold` | float |
| `MC_SCREEN_MATCHING_LUMINANCE_LOW_RES_WIDTH` | `screen_matching.luminance_low_res_width` | int |
| `MC_SCREEN_MATCHING_PERSIST_FILTERED` | `screen_matching.persist_filtered` | bool (`true/1/yes/on`) |
| `MC_SCREEN_MATCHING_BM25_TOP_K` | `screen_matching.bm25_top_k` | int |
| `MC_SCREEN_MATCHING_ELEMENT_CRITERION` | `screen_matching.element_criterion` | str (`diff`\|`jaccard`) |
| `MC_SCREEN_MATCHING_ELEMENT_DIFF_MAX` | `screen_matching.element_diff_max` | int |
| `MC_SCREEN_MATCHING_ELEMENT_JACCARD_MIN` | `screen_matching.element_jaccard_min` | float |
| `MC_SCREEN_MATCHING_PAGE_PIXEL_DIFF_THRESHOLD` | `screen_matching.page_pixel_diff_threshold` | float |
| `MC_CONFIG_PATH` | (YAML 파일 경로 자체) | path — 대체 yaml 위치 지정 |

### `config.py` API

타입드 dataclass: `RunConfig`(`exploration`/`collection`/`llm`/`screen_matching` = `ExplorationConfig`/`CollectionConfig`/`LlmConfig`/`ScreenMatchingConfig`).

- `load_run_config(path: Path | str | None = None) -> RunConfig`
  - builtin defaults → YAML → MC_* env 를 순서대로 병합해 `RunConfig` 반환.
  - `path` 가 주어지면 그 YAML 을 사용, 없으면 `MC_CONFIG_PATH`, 그래도 없으면 패키지 루트의 `config/run.yaml`. 존재하지 않는 파일은 무시(빈 dict 취급)된다 — 테스트에서 `Path("/nonexistent")` 로 파일 레이어를 건너뛸 수 있다.
- `merge_with_cli_args(config: RunConfig, args: argparse.Namespace) -> RunConfig`
  - CLI 플래그 중 `None` 이 아닌 것만 `config` 위에 덮어써 새 `RunConfig` 반환(`dataclasses.replace`). `--element-extraction on/off`, 폐기 예정 `--screen-grouping off` 도 여기서 `element_extraction` 으로 매핑된다. boolean 운영 플래그(`force`/`new_session`)는 `RunConfig` 에 들어가지 않는 CLI 전용이다.

`cli.py` 의 `run` 서브커맨드는 `load_run_config(args.config)` → `merge_with_cli_args(cfg, args)` 로 최종 설정을 만든 뒤, 해소된 strategy 를 `LLMGuidedExplorer` 에 명시적으로 주입한다.
