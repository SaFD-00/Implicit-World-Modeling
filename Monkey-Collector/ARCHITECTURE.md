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
- 세션 디렉토리는 패키지명 기반 `data/raw/{package}/` 이고, 기본 동작은 resume 이다.

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
  - [`activity_coverage.py`](./src/monkey_collector/domain/activity_coverage.py): Activity coverage CSV. ground truth 의 분모(`total_activities`)와 분자 후보 집합 모두 `session_manager._resolve_declared_activities` 가 결정하며, 1차 소스는 [`catalog/activities.json`](./catalog/activities.json) (androguard manifest 추출), 폴백은 `adb dumpsys package`. catalog hit 시 (`allow_dynamic_total=False`) 분모 고정 + 분자(`unique_visited`)는 catalog set 안의 activity 만 normalize 후 카운트 + `coverage = min(1.0, ...)` 클램프. catalog 외 activity 는 `activity` 컬럼에 그대로 기록되지만 coverage 에는 영향 없음. catalog miss 폴백 (`allow_dynamic_total=True`) 은 legacy 동작 (target package 의 미선언 activity 발견 시 분모 동적 확장 + 모든 visited 카운트).
  - [`cost_tracker.py`](./src/monkey_collector/domain/cost_tracker.py): LLM 비용 추적 CSV. `agent` 컬럼으로 호출 주체(`text_generator` / `screen_grouper`) 구분.
  - [`page_graph.py`](./src/monkey_collector/domain/page_graph.py): 페이지 그래프 생성. live 수집은 `get_or_create_page_by_match`(ScreenMatcher 의 `page_key` → 노드)로, 오프라인 `build_graph_from_session`/`page-map` 는 기존 구조 지문 `get_or_create_page`(activity+Jaccard)로 식별한다(live=element-set, post-hoc=구조 근사 분기). `PageNode` 에 `page_key`/`element_names` 필드 추가(구 `page_graph.json` 은 `.get` 기본값으로 로드). matcher 가 `pending` 으로 거부한 로딩/스플래시 화면은 노드를 만들지 않는다(collection_loop 에서 `get_or_create_page_by_match` 호출 자체를 건너뜀).
- `llm/` — 모든 LLM 소비자가 공유하는 단일 클라이언트
  - [`client.py`](./src/monkey_collector/llm/client.py): `LLMClient` — env 기반(`OPENROUTER_API_KEY` / `OPENROUTER_BASE_URL` / `OPENROUTER_MODEL`, 기본 `qwen/qwen3.7-plus`) OpenRouter **Chat Completions** 래퍼. `chat()` 한 경로로 호출하며 cost_tracker 에 `prompt_tokens`/`completion_tokens` 기록. `create_llm_client()` 는 키 없으면 `None` 반환.
  - [`element_extractor.py`](./src/monkey_collector/llm/element_extractor.py): `ElementExtractor` — **단일 호출 element 추출**(MobileGPT-V2 Node-Clustering 포팅). `encode_to_html_xml` 표현을 LLM 에 보내, 각 element 마다 `name`/`description`/`parameters` 와 `element_index`(같은 기능 family 전체)·`key_element_index`(대표 anchor 1~3)를 **한 번에** 받는다(MobileGPT-V2 가 SubtaskExtractor·TriggerUIAgent 두 호출로 나눠 뽑는 것과 달리 1호출로 합침). `description`/`parameters` 는 `ElementFamily` 를 거쳐 `{step}_elements.json` 까지 그대로 영속된다. `known_elements` 가 주어지면 그와 겹치지 않는 것만 반환(expand). 실패 시 `[]`(수집 흐름 무영향). `create_element_extractor()` 는 클라이언트 없으면 `None`.
  - [`prompts/element_extractor_prompt.py`](./src/monkey_collector/llm/prompts/element_extractor_prompt.py): MobileGPT-V2 의 subtask_extractor(추출 규칙·SAME-FUNCTION GROUP·KNOWN 제외) + trigger_ui(대표 선택 기준)를 병합한 단일 프롬프트.
- 인프라 모듈에 인접
  - [`catalog_activities.py`](./src/monkey_collector/catalog_activities.py): `catalog/activities.json` 의 process-lifetime 캐시 (`ActivityCatalog`). activity coverage ground truth 1차 소스.
- `pipeline/`
  - [`collector.py`](./src/monkey_collector/pipeline/collector.py): collector facade
  - [`session_manager.py`](./src/monkey_collector/pipeline/session_manager.py): session init/resume/finalize. `_resolve_declared_activities` 헬퍼가 catalog 우선, dumpsys 폴백 정책을 적용한다.
  - [`collection_loop.py`](./src/monkey_collector/pipeline/collection_loop.py): 메인 루프
  - [`recovery.py`](./src/monkey_collector/pipeline/recovery.py): retry / recovery 상수와 helper
  - [`screen_guard.py`](./src/monkey_collector/pipeline/screen_guard.py): 화면 분류 가드(키보드/권한 다이얼로그/시스템·런처 화면). `SYSTEM_PACKAGES` 에 `gms`/`gsf`/`vending`/launcher 를 포함해, 타깃 앱이 Google 로그인·Play 화면으로 drift 한 것을 "앱 이탈"로 판정(클라이언트 `EXCLUDED_PACKAGES` 와 이중 방어). 권한 다이얼로그는 grant 우선 버튼 탐색으로 자동 처리.
  - [`screen_matching/`](./src/monkey_collector/pipeline/screen_matching): **element-set screen matching** 엔진 (MobileGPT-V2 Node-Clustering 포팅). `ScreenMatcher` 가 화면마다 page 식별을 수행하고 그 `page_key` 가 page_graph 노드와 탐색 abstract page 를 모두 결정한다. 키 없으면(extractor=None) 생성되지 않아 구조 지문 식별로 degrade.
    - [`ui_attributes.py`](./src/monkey_collector/pipeline/screen_matching/ui_attributes.py): `UIAttributes`(self+parent+children 구조 지문)와 매칭 primitives(`find_matching_node` ancestor-walk, `text_blind_requirements`, `get_ui_key_attrib`, `extract_interactable_indexes`, `mask_xml_to_indexes`). MobileGPT-V2 xml_parser 를 MC encoded 스키마(`tag/aria-label/alt/text/type/value`, `id`/`class` 없음)로 적응 포팅, distinctive 판정은 `aria-label/alt`.
    - [`screen_matcher.py`](./src/monkey_collector/pipeline/screen_matching/screen_matcher.py): `ScreenMatcher.match` — ① 구조 지문 pre-filter(exact 재방문 short-circuit, LLM 0회) + interactable(button/input) 0개 화면 거부(`pending`, LLM 0회 — 로딩/스플래시가 빈 page 로 등록돼 이후 화면을 모두 흡수하는 blackhole 차단; 첫 유효 화면이 `page_0`) ② step-1 text-blind ALL-match(저장 anchor) → supported + remaining ③ expand(remaining 마스킹 후 재추출, dry/cap 까지) ④ set-classify(A vs B) ⑤ dispatch(MERGE=stored page_key frozen / NEW=새 page_key, anchor 를 현재 화면에서 fingerprint). `families`(현재 인덱스 element family, `description`/`parameters` 포함)는 새 페이지면 추출 family 를, merge·구조 재방문이면 매칭 page 의 저장 element 를 현재 화면 anchor 로 재매핑해 채워 반환한다(재방문 family 는 `{step}_elements.json` 에만 쓰이고 same-function 압축에는 미반영 — page_key 가드). (한계: element-set 은 scroll 컨테이너를 제외하므로 button/input 없이 scroll/long-click 만 있는 드문 화면도 pending 처리된다.)
    - [`set_classifier.py`](./src/monkey_collector/pipeline/screen_matching/set_classifier.py): A vs B set 분류(EQSET/SUPERSET_MERGE/SUBSET_MERGE/OVERLAP_MERGE/OVERLAP_NEW/DISJOINT). containment 는 무조건 merge(scroll-reveal 수렴)하되 **B 가 비어있지 않을 때만** — 빈 저장 page(B=∅)는 merge 대상이 될 수 없어(→ DISJOINT) 어쩌다 등록된 빈 page 도 blackhole 이 되지 않는다. OVERLAP 만 two-sided tolerance band 게이트.
    - [`page_knowledge.py`](./src/monkey_collector/pipeline/screen_matching/page_knowledge.py): `PageKnowledge`(page_key·elements·`key_elements`(name→anchor 지문)·`extra_uis`) + `KnowledgeRegistry`(세션별 in-memory 저장).
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
  - [`storage.py`](./src/monkey_collector/storage.py): raw session 저장 및 XML variant 재생성 (`DataWriter`)
- `xml/`
  - [`ui_tree.py`](./src/monkey_collector/xml/ui_tree.py): action selection 용 UI tree
  - [`structured_parser.py`](./src/monkey_collector/xml/structured_parser.py): 구조적 XML parser
  - [`parser_base.py`](./src/monkey_collector/xml/parser_base.py): `Parser` ABC
- `export/`
  - [`converter.py`](./src/monkey_collector/export/converter.py): raw session -> ShareGPT JSONL
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
  -> XML parse + (element-extraction on 이면) ScreenMatcher 가 element-set 으로 page 식별
       (pre-filter[+interactable 0개 화면은 pending 거부] → step-1 ALL-match → expand 재추출 → set-classify → page_key)
       → pending 이면 page 노드·{step}_elements.json 생성 안 함(로딩/스플래시 대기)
       → 아니면 page_graph 노드 = page_key, 새 페이지 element family(description/parameters 포함) 는 탐색에 전달, {step}_elements.json 저장(merge·재방문이면 매칭 page element 를 현재 화면 기준으로 재매핑해 저장)
  -> LLMGuidedExplorer 가 action 선택 (미탐색 우선 + element family 압축 + 미탐색 화면 navigation; input_text 필요 시 공용 LLMClient 로 텍스트 생성)
  -> ADB 실행
  -> screenshot/XML/event 저장
  -> 다음 step 반복
```

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
- abstract page 식별은 **`page_key`**(ScreenMatcher 의 element-set 식별; matcher 없으면 `structure_str` fallback)로 하며, 이 동일한 `page_key` 가 `page_graph.json` 산출물(live `get_or_create_page_by_match`)의 노드도 결정한다 — 과거 구조 지문 기반 디커플은 element-set matching 으로 커플링되었다(MobileGPT-V2 포팅).

실행은 `LLMGuidedExplorer.execute_action` 이 `AdbClient` ([`src/monkey_collector/adb.py`](./src/monkey_collector/adb.py)) 메서드로 위임. `AdbClient` 는 CLI 진입점에서 단일 인스턴스로 생성되어 `LLMGuidedExplorer` 와 `Collector` 에 주입된다. 생성 시점에 `Pixel6-2` AVD 의 emulator serial 을 해석해 저장하므로, 해당 AVD 가 실행 중이어야 한다.

## 4. 세션 관리와 복구

### 세션 라이프사이클

- 저장 위치는 `data/raw/{package}/`
- 동일 패키지에 `metadata.json` 이 있으면 resume. 이때 `completed_at` 은 `None` 으로 되돌아간다 (진행 중 상태).
- `run` 은 큐 구성 단계에서 `completed_at` 이 채워진 앱을 자동 skip 한다. `--force` 로 우회.
- `run --new-session` 은 해당 앱 세션을 삭제하고 새로 시작
- `reset` 서브커맨드로 범위 단위 (all / apps) 일괄 삭제 가능
- 세션 정상 종료 시 `completed_at` 기록, page graph 재빌드, HTML 시각화 생성. 다음 `run` 부터는 이 앱이 큐에서 자동 제외.

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
- 빈 UI tree 가 반복되면 대기 후 재시도. 빈 판정은 raw count 가 아니라 `get_interactable_elements()` 기준(clickable/scrollable/editable 등이 0일 때) — 비상호작용 노드만 있는 화면을 "콘텐츠 있음"으로 오판하지 않는다.
- **런타임 권한 다이얼로그 자동 허용**: XML 신호가 온 경우 `is_permission_dialog` → `_handle_permission_dialog` 가 grant 버튼(`PERMISSION_BUTTON_KEYWORDS`, "while using the app" 최우선)을 탭한다. 단 permissioncontroller `GrantPermissionsActivity` 는 **accessibility 이벤트를 안 내보내** 신호 없이 timeout 만 발생하므로, timeout 경로에서 `_try_grant_permission_via_adb` 가 adb 로 foreground 를 확인하고 `uiautomator dump` 한 뒤 **clickable 버튼만** 스캔해 "While using the app"(없으면 "Allow") 를 탭한다("Only this time"/"Don't allow" 는 deny-guard 로 회피). 권한 미허용 화면을 relaunch 로 건너뛰지 않고 허용 후 탐색을 계속한다.
- 세션 핸드셰이크에서 START 후 client `P`(package ACK)가 5초 내 없으면 `session_manager.receive_target_package` 가 abort(`None`) — 죽은 세션(클라이언트 크래시/stale 소켓)에 step 예산을 blind 소진하지 않고 다음 앱으로 넘어간다.

## 5. 저장 포맷

세션별 기본 구조:

```
data/raw/{package}/
├── metadata.json
├── screenshots/
│   └── 0000.png
├── xml/
│   ├── 0000.xml
│   ├── 0000_parsed.xml
│   ├── 0000_hierarchy.xml
│   ├── 0000_encoded.xml
│   ├── 0000_pretty.xml
│   └── 0000_elements.json   # --element-extraction on 일 때만 (element-set match)
├── events.jsonl
├── activity_coverage.csv     # ground truth: catalog/activities.json (fallback: dumpsys)
├── cost.csv
├── page_graph.json
└── page_graph.html
```

`DataWriter.save_xml()` 와 `regenerate_xml_variants()` 는 raw XML 에서 아래 파생 파일을 만든다.

- `_parsed.xml`: semantic HTML tags + bounds + index
- `_hierarchy.xml`: text / bounds / index 제거
- `_encoded.xml`: bounds 제거, index 유지
- `_pretty.xml`: encoded XML pretty-print

`{step}_elements.json` 은 `--element-extraction on` 일 때 `ScreenMatcher` 가 만든 element-set match annotation 이다: `page_key`/`match_type`/`is_new_page` 와 element family 목록(`name`/`description`/`parameters`/`element_index`/`key_element_index`; index 는 `{step}_encoded.xml` 기준, `description`/`parameters` 는 LLM 이 추출한 의미·인자). 새 페이지면 그 화면에서 추출된 family 가, merge·구조 재방문이면 매칭된 page 의 저장 element 를 현재 화면 anchor 에 재매칭해 채운다(anchor 기준이라 `element_index` 는 anchor index 로 근사되고, 현재 화면에 렌더되지 않은 element 는 제외 — 모두 현재 step 인덱스 공간). interactable(button/input) 이 없는 로딩/스플래시 화면은 matcher 가 등록을 거부(`pending`)해 page 노드도 `_elements.json` 도 만들지 않는다(첫 유효 화면이 `page_0`). LLM 호출 결과이므로 `save_xml`/`regenerate_xml_variants` 의 결정적 파생 대상이 아니다.

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
| `collection` | `max_steps` | 앱 세션당 최대 step 수 (기본 1500) |
| `collection` | `seed` | explorer random seed (기본 42) |
| `collection` | `action_delay_ms` | action 간 지연(ms) (기본 1500) |
| `collection` | `port` | TCP 서버 포트 (기본 12345) |
| `collection` | `output_dir` | raw session 출력 루트 (기본 `data/raw`) |
| `llm` | `input_mode` | 입력 텍스트 생성 모드 `api`\|`random` (기본 `api`) |
| `llm` | `element_extraction` | LLM element 추출 + element-set screen matching on/off (기본 `true`; `OPENROUTER_API_KEY` 없으면 구조 지문으로 자동 degrade) |
| `screen_matching` | `cluster_merge_tolerance` | OVERLAP element-set merge 의 two-sided tolerance band 0.0–1.0 (기본 0.2) |
| `screen_matching` | `max_expand_iters` | 화면당 expand(잔여 UI 재추출) 최대 반복 (기본 3) |

### MC_* 환경변수

YAML 위, CLI 아래 레이어. 각 변수는 대응 키를 타입 변환해 덮어쓴다.

| 환경변수 | 대상 키 | 타입 |
| -------- | ------- | ---- |
| `MC_EXPLORATION_STRATEGY` | `exploration.strategy` | str (upper-case 정규화) |
| `MC_COLLECTION_MAX_STEPS` | `collection.max_steps` | int |
| `MC_COLLECTION_SEED` | `collection.seed` | int |
| `MC_COLLECTION_ACTION_DELAY_MS` | `collection.action_delay_ms` | int |
| `MC_COLLECTION_PORT` | `collection.port` | int |
| `MC_COLLECTION_OUTPUT_DIR` | `collection.output_dir` | str |
| `MC_LLM_INPUT_MODE` | `llm.input_mode` | str |
| `MC_LLM_ELEMENT_EXTRACTION` | `llm.element_extraction` | bool (`true/1/yes/on`) |
| `MC_SCREEN_MATCHING_CLUSTER_MERGE_TOLERANCE` | `screen_matching.cluster_merge_tolerance` | float |
| `MC_SCREEN_MATCHING_MAX_EXPAND_ITERS` | `screen_matching.max_expand_iters` | int |
| `MC_CONFIG_PATH` | (YAML 파일 경로 자체) | path — 대체 yaml 위치 지정 |

### `config.py` API

타입드 dataclass: `RunConfig`(`exploration`/`collection`/`llm`/`screen_matching` = `ExplorationConfig`/`CollectionConfig`/`LlmConfig`/`ScreenMatchingConfig`).

- `load_run_config(path: Path | str | None = None) -> RunConfig`
  - builtin defaults → YAML → MC_* env 를 순서대로 병합해 `RunConfig` 반환.
  - `path` 가 주어지면 그 YAML 을 사용, 없으면 `MC_CONFIG_PATH`, 그래도 없으면 패키지 루트의 `config/run.yaml`. 존재하지 않는 파일은 무시(빈 dict 취급)된다 — 테스트에서 `Path("/nonexistent")` 로 파일 레이어를 건너뛸 수 있다.
- `merge_with_cli_args(config: RunConfig, args: argparse.Namespace) -> RunConfig`
  - CLI 플래그 중 `None` 이 아닌 것만 `config` 위에 덮어써 새 `RunConfig` 반환(`dataclasses.replace`). `--element-extraction on/off`, 폐기 예정 `--screen-grouping off` 도 여기서 `element_extraction` 으로 매핑된다. boolean 운영 플래그(`force`/`new_session`)는 `RunConfig` 에 들어가지 않는 CLI 전용이다.

`cli.py` 의 `run` 서브커맨드는 `load_run_config(args.config)` → `merge_with_cli_args(cfg, args)` 로 최종 설정을 만든 뒤, 해소된 strategy 를 `LLMGuidedExplorer` 에 명시적으로 주입한다.
