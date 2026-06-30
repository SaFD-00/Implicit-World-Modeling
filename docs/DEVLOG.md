# Dev Log

시점성 진행 로그 (append-only). 최신 엔트리를 위에 추가한다. 과거 엔트리는 수정·삭제하지 않는다.
상세 결과는 Notion Dev Log / Experiments DB, 계획은 [ROADMAP.md](./ROADMAP.md) 참조.

## 2026-06-30 — Monkey-Collector converter 프레임-이벤트 정렬 버그 수정 (frame_index 도입)

world-modeling 학습데이터 변환에서 **action 라벨이 엉뚱한 프레임에 붙는 정렬 버그**를 발견·수정했다. `input_text` 처럼 페이지가 안 바뀌는 action 도 `_encoded.xml` 의 `value` 변화로 before≠after 학습쌍이 되는데, 실제 변환 결과는 `input_text` 가 중복되고 Discard 다이얼로그 프레임에 input 라벨이 붙는 등 어긋나 있었다. 근본 원인은 events.jsonl 의 `step`(루프 카운터, 비-저장 반복에서도 증가)과 프레임 파일 인덱스(`step_count`, 저장 시에만 증가)가 다른 체계인데 둘을 잇는 조인 키가 디스크에 없던 것. 코드 변경(저장 포맷 변경 포함), 작업트리 미커밋.

- Fix: 수집 시점에 정상 action 이벤트에 `frame_index = writer.step_count - 1`(before 프레임 파일 인덱스)을 기록(`collection_loop._process_xml_signal`). converter(`export/converter.py`)와 offline page-graph 재빌드(`domain/page_graph.py _load_events`)를 `events.get(step_idx)` 추측·`_find_event_by_index` 폴백에서 **`frame_index` 직접 조인**으로 교체. converter 의 after 프레임은 "다음 action 의 before 프레임"으로 잡아 중간 로딩 프레임을 건너뛴다.
- step 의미 재정의: `state.step += 1` 을 정상 action 경로 1곳만 남기고 **13곳 제거**(signal timeout·permission·system·stale·no_change·keyboard·same-page-stuck·empty-UI·예외). 이제 `step` 은 실제 action 에서만 증가 → 사용자 요청대로 signal timeout 등이 step 을 소비하지 않음. 부수: step 이 상한 역할을 하던 경로 보호용 `idle_iterations` 절대 가드(`max_step*4`) 추가 + stale 경로 `clear_signal_queue` 보강. activity_coverage 기록을 `step_count`(파일 인덱스)로 키잉해 page-graph CSV 폴백 정합 유지.
- 하위호환: `frame_index` 없는 구(舊) 세션은 converter 가 변환 스킵, page-graph 재빌드는 토폴로지만 복원하고 엣지 라벨 `unknown` 으로 degrade → 재수집/regenerate 필요(옛 휴리스틱 폴백 제거).
- 검증: 합성 세션 sanity 로 same-page `input_text` 가 (before=빈 필드 → Input → after=채워진 필드, element_index 정확) 학습쌍으로 정렬됨을 실증. 회귀 테스트 6종(step≠frame_index 조인, empty-UI 건너뜀, no_change_retry/transition:false/frame_index 부재 제외, 마지막 action after 부재) + page-graph 엣지 라벨 회귀 추가. 기존 fixture 가 `step==파일인덱스` 를 우연히 1:1 로 맞춰 버그를 못 잡던 것도 `step=frame_index+100` 으로 어긋나게 수정.
- 변경(작업트리, 11 files): src 3(`pipeline/collection_loop.py`, `export/converter.py`, `domain/page_graph.py`) + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 3(`tests/unit/{test_converter,test_page_graph}.py`, `tests/fixtures/session_fixtures.py`) + repo docs 2(`docs/{DEVLOG,CHANGELOG}.md`).
- 검증: 전체 **pytest passed**, ruff 신규 위반 0(변경 파일 clean; 선재 `test_structured_parser.py` F401 은 그대로).
- 커밋: 미커밋(작업트리). 직후 project-sync + git-push 예정.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` 갱신. 루트 `docs/{README,ARCHITECTURE}.md` 는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-30 — Monkey-Collector external 복구 시 open_app 기록 (navigation 격리)

external app 이탈에서 복구 루프가 타깃 앱을 재실행하는 동작(=사실상의 `open_app`)을 events.jsonl 에
액션으로 기록하기 시작했다 — open_app 학습 데이터 확보용. 단 external 이탈은 의도된 화면 전이가 아니므로
이 open_app 은 **navigation 으로 절대 쓰이지 않게 3중 격리**했다. 코드 변경(기록 동작 추가), 작업트리 미커밋.

- 기록: `return_to_app()`/`recover()` 를 `-> bool`(실제 `launch_app` 여부)로 바꾸고, `collection_loop._handle_external_app` 가 launch 시 `_record_open_app`→`DataWriter.log_open_app` 로 `{action_type:"open_app", package, app_name, step, transition:false, trigger:"external_recovery", from_package}` 를 **excursion 당 1회**(`state.open_app_logged` dedup, 다음 in-app 프레임에서 재무장) 기록. `app_name` 은 apps.csv 조인(`cli._resolve_app_names`→`Collector(app_names=...)`), 없으면 빈 문자열.
- navigation 격리(3중): (1) 복구 후 `state.last_action`/`last_ui_tree` 클리어 → live page graph 엣지 차단, (2) `return_to_app`/`recover` 진입 시 `explorer._last_record` 클리어 → routing memory 전이 차단(모든 복구 호출처의 선재 stale-전이 버그도 동시 수정), (3) 이벤트 `transition:false` + `page_graph._load_events` 스킵 가드 → offline 재빌드(`page-map`)·world-modeling converter 배제.
- 스레드 안전: open_app 은 메인 루프, external_app 콜백은 TCP 수신 스레드에서 events.jsonl/metadata.json 에 동시 기록 → `DataWriter` 에 `threading.Lock` 추가해 append·counter 보호. `OpenApp` dataclass 는 record-only(execute_action·select_action 미경유)지만 `ACTION_REGISTRY` 등록으로 라운드트립 가능.
- 한계: 서버측 기록은 서버 주도 재실행만 포착. 클라이언트(`CollectorService.kt`)가 자체 force-launch 하는 경우는 과소기록될 수 있음. open_app 이벤트는 before/after XML 쌍이 없어 기존 world-modeling 변환으로는 소비되지 않음(의도된 분리) — 학습은 별도 변환에서 `step`/`package` 로 직전 in-app 프레임과 페어링해 파생.
- 변경(작업트리, 18 files): src 7(`domain/actions.py`, `storage.py`, `pipeline/exploration/explorer.py`, `pipeline/collection_loop.py`, `domain/page_graph.py`, `pipeline/collector.py`, `cli.py`) + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 6 + repo docs 2(`docs/{DEVLOG,CHANGELOG}.md`).
- 검증: 전체 **pytest passed**, ruff 신규 위반 0(선재 위반은 그대로). 부수: external 카운터 테스트 2건의 stale 기댓값(`recover==6`)을 실제 동작(reinit 분기도 recover 호출 → 7)에 맞게 수정 — HEAD 에서도 이미 실패하던 선재 red.
- 커밋: 미커밋(작업트리). 직후 project-sync + git-push 예정.
- 카테고리: devlog

## 2026-06-30 — Monkey-Collector 빈 page_0 blackhole 버그 수정 + LLM element description/parameters 디스크 보존

element-set screen matching에서 두 결함을 함께 고쳤다: (1) 세션 첫 로딩/스플래시처럼 interactable이 0개인
화면이 빈 page_0로 등록돼 이후 모든 화면을 흡수하던 **blackhole** 버그, (2) LLM이 추출한 element의
`description`/`parameters`가 디스크 저장 시 누락되던 문제. 코드 변경만(기록 전용 아님), 작업트리 미커밋.

- Fix 1 (blackhole): `ScreenMatcher.match()`에 entry guard 추가 — interactable(button/input) 0개 화면은 LLM 호출 없이 `pending`으로 거부해 page로 등록하지 않음(첫 유효 화면이 page_0이 됨). `set_classifier`에 안전망 — 저장 page B=∅이면 SUPERSET_MERGE 불가→DISJOINT라 어쩌다 등록된 빈 page도 sink가 되지 않음. `collection_loop`는 pending 시 page 노드 생성·`save_elements`를 스킵. 부수: `extract_interactable_indexes`를 root-inclusive(`tree.iter()`)로 수정해 단일 루트 interactable 누락 차단.
- Task 2 (description/parameters): `ElementFamily`에 `description`/`parameters` 필드 추가(끝에 — 하위호환), families 생성부에서 ExtractedElement의 5필드 전부 채움, `DataWriter.save_elements`가 `{step}_elements.json`의 각 element에 description/parameters 키를 직렬화. 최종 element 형태: name/description/parameters/element_index/key_element_index.
- 범위 결정: 버그리포트의 "merge 시 scroll-reveal element 누적"(권장)은 page-identity/tie-break 드리프트 위험으로 이번에서 제외(별도 변경으로 분리). 라이브 스모크도 제외 — 단위/오프라인 검증으로 대체.
- 변경(작업트리, 12 files): src 5 + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 4. 관련 단위테스트 갱신/추가.
- 커밋: 미커밋(작업트리). 직전 커밋 b0ac999, 직후 git-push 예정.
- 결과/검증: 전체 **545 passed**. 기존 3 failures는 직전 커밋 5cc02c4(max_steps 100→1500)/10d9eee(reinit on timeout)發로 본 변경과 무관.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`만 갱신(작업트리). 루트 `docs/{README,ARCHITECTURE}.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-29 — 화면 그룹핑(ScreenGrouper)을 element-set screen matching으로 교체 (MobileGPT-V2 Node-Clustering 포팅)

LLM "화면 그룹핑"(`llm/screen_grouper.py`, annotation 전용이라 탐색에 미반영)을 MobileGPT-V2의 Node-Clustering을
포팅한 **element-set screen matching**(`pipeline/screen_matching/`)으로 교체했다. 화면당 단일 LLM 호출로 각
element의 `element_index`(같은 기능 family 전체)와 `key_element_index`(대표 anchor)를 함께 추출하고,
set-classification으로 산출한 `page_key`가 page_graph 노드와 탐색 abstract page를 같은 키로 결정한다 — 과거
grouping↔page matching 디커플을 의도적으로 커플링했다. (동시 진행한 app-context 입력 텍스트 생성은 아래 엔트리 참조.)

- 신규 패키지 `pipeline/screen_matching/`: `ScreenMatcher` / `ui_attributes`(UIAttributes 지문 + find_matching_node ancestor-walk + text_blind + get_ui_key_attrib + mask_xml_to_indexes) / `set_classifier` / `page_knowledge`(PageKnowledge·KnowledgeRegistry). 신규 `llm/element_extractor.py` + `llm/prompts/element_extractor_prompt.py`로 MobileGPT-V2의 SubtaskExtractor·TriggerUIAgent 2호출을 1호출로 병합.
- `ScreenMatcher.match` 흐름: ① 구조 지문 pre-filter(exact 재방문 short-circuit, LLM 0회) → ② step-1 text-blind ALL-match(저장 anchor) → supported+remaining → ③ expand(remaining 마스킹 후 재추출, dry/`--max-expand-iters` cap) → ④ set-classify(EQSET/SUPERSET_MERGE/SUBSET_MERGE는 containment-always-merge, OVERLAP만 two-sided tolerance band) → ⑤ dispatch(MERGE=stored page_key frozen / NEW=새 page_key + anchor를 현재 화면에서 fingerprint).
- linchpin은 인덱스 공간 일치(extractor encoded XML = `{step}_encoded.xml` = `SemanticElement.index`, 모두 parse→_renumber→_clear_bounds): V2 파서는 포팅하지 않고 매칭 함수만 MC encoded 스키마(tag/aria-label/alt/text/type/value; id/class 없음)로 적응, distinctive 판정은 aria-label/alt. 산출 `page_key`가 `PageGraph.get_or_create_page_by_match`와 `SemanticState.page_key`(Memory/TransitionGraph/Navigator 키)를 모두 결정. `finalize_session`은 matcher 활성 시 라이브 `state.page_graph`(page_key/element_names)를 그대로 저장, 오프라인 page-map만 구조 지문 재구성.
- CLI: `--screen-grouping` → `--element-extraction {on,off}`(기본 on, `--screen-grouping`은 deprecated alias 유지) + `--cluster-merge-tolerance`(0.2) + `--max-expand-iters`(3). 산출물 `{step}_groups.json` → `{step}_elements.json`, cost.csv agent 라벨 `screen_grouper` → `element_extractor`. degrade: `OPENROUTER_API_KEY` 없거나 off면 matcher 미생성 → `page_key=structure_str` fallback + Memory 압축 없음 = 기존 파이프라인 byte-for-byte.
- 변경(작업트리): 신규 `pipeline/screen_matching/{__init__,ui_attributes,screen_matcher,set_classifier,page_knowledge}.py`·`llm/element_extractor.py`·`llm/prompts/element_extractor_prompt.py`·`tests/unit/test_{ui_attributes,set_classifier,screen_matcher,element_extractor}.py`; 수정 `domain/page_graph.py`·`pipeline/{collection_loop,collector,session_manager,exploration/{explorer,memory,navigator,state,transition_graph}}.py`·`storage.py`·`cli.py`·`llm/__init__.py`·`__init__.py`·`Monkey-Collector/{ARCHITECTURE,README,AGENTS}.md` + 다수 기존 테스트; 삭제 `llm/screen_grouper.py`·`tests/unit/test_screen_grouper.py`.
- 커밋: 미커밋(작업트리). last_sync(2026-06-29T17:56:42+09:00) 이후 신규 커밋 없음 — 직전 커밋 beb6a8c, 곧 커밋·푸시 예정.
- 결과/검증: 전체 테스트 green(신규 test_ui_attributes/test_set_classifier/test_screen_matcher/test_element_extractor + 기존 test_memory_unexplored/test_navigator/test_transition_graph_nav/test_storage/test_page_graph/test_semantic_state 갱신), 신규 파일 ruff 0/mypy clean, 전체 mypy 회귀 0(baseline 28 유지)·ruff 16→10. 라이브 org.tasks E2E PASS(Pixel6-2): element-extraction on → page_graph.json `page_key`(`page_0`,`page_1`)+element_names 저장, `{step}_elements.json` families(element_index+key_element_index), match_type ladder 전부 관측(NEW/SUBSET_MERGE/STRUCTURAL_IDENTICAL×20/DISJOINT/EQSET), pre-filter로 24스텝 중 LLM 2회($0.003), 크래시 0; degrade off → 구조경로 완주(page_key 0·elements.json 0·LLM 0).
- 카테고리: devlog

## 2026-06-29 — input-text LLM 생성에 앱 설명 주입

탐색 중 텍스트 입력값을 LLM(`--input-mode api`)으로 생성할 때 프롬프트에 "현재 어떤 앱을 탐색 중인지"가
전혀 없어 앱 도메인에 안 맞는 입력(쇼핑앱 검색창에 일반 단어 등)이 나올 수 있던 문제를 보완했다.
`catalog/apps.csv`의 사람이 읽을 수 있는 메타데이터를 input-text 프롬프트에 주입하도록 배선했다.

- 설계: 공유 `TextGenerator` 인스턴스(cli에서 1회 생성→explorer의 `ActionMapper`+`Collector` 양쪽 공유)에 세션마다 setter로 앱 설명을 박는 방식 채택. `generate()`에 인자를 스레딩하는 대안은 `TextGenerator`/`ActionMapper`/`Explorer` Protocol 시그니처가 줄줄이 바뀌어 기각.
- 변경(작업트리, src 4): `pipeline/app_catalog.py`(`AppJob.description` 프로퍼티: `app_name (category/sub_category) — notes`, app_name 없으면 package_id 폴백); `pipeline/text_generator.py`(`TextGenerator` 베이스에 no-op `set_app_context` 훅 + `LLMTextGenerator` 오버라이드, `generate()`가 `App under test:` 줄을 프롬프트 앞에 조건부 prepend, SYSTEM_PROMPT에 도메인 맞춤 규칙 1줄); `pipeline/collector.py`(`__init__(app_contexts=...)` + `_run_session`이 패키지 확정 후 **무조건** `set_app_context(self._app_contexts.get(pkg, pkg))` — 공유 generator의 이전 앱 설명 누수 방지, `text_generator=None` 가드); `cli.py`(신규 `_resolve_app_contexts(packages)→dict` 헬퍼: AppCatalog 로드 best-effort, 미등록/누락은 dict 제외→Collector가 package_id 폴백; `Collector(app_contexts=...)` 배선). `_resolve_run_packages`는 테스트가 시그니처 고정이라 미변경.
- 테스트(신규만, 기존 0 수정): `test_app_catalog`(description 4종: full/no-notes/no-category/package_id 폴백), `test_text_generator`(app_context 포함·미설정/공백 시 줄 생략·random no-op), `test_run_resume`(`_resolve_app_contexts` 정상/csv없음→{}/미등록 제외), `tests/integration/test_collector`(map→설명 호출·빈map→package_id 폴백 호출).
- 결과/검증: 전체 **545 passed**(타깃 96 포함), 변경 src+신규 단위테스트 ruff clean(`set_app_context` 베이스 no-op은 의도적 선택훅이라 `# noqa: B027`). 라이브 스모크는 미실행(정적 검증까지).
- 커밋: 미커밋(작업트리).
- 문서: `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` + `docs/CHANGELOG.md` 갱신.
- 카테고리: devlog

## 2026-06-29 — setup-collector 스킬 전면 갱신 + 라이브 재검증 + 런타임 권한 자동허용

Monkey-Collector `setup-collector` 스킬을 MobileGPT-V2 `setup-emulator` 구조(SKILL.md 오케스트레이션 +
`references/` deep-dive)로 재구성하고, 이전 디버깅 세션의 소스 패치들을 커밋 가능한 상태로 정리·문서화한 뒤
emulator-5554/Pixel6-2에서 end-to-end 라이브 재검증했다. 검증 중 사용자 피드백으로 런타임 권한 다이얼로그
자동허용("While using the app")을 추가 보강했다.

- 변경(작업트리): `.claude/skills/setup-collector/SKILL.md`(AVD MobileGPT-V2-2→Pixel6-2 전수 교체, 빌드 JDK8→JDK17/AGP8.2, APK 경로 `app/app/build/outputs`, `local.properties` 노트, 신규 §6-c MediaProjection 재동의·§6-d Google 로그인·§6-e 더미데이터 시드·§9 라이브 검증, 전 단계 멱등); `.claude/skills/setup-collector/references/` 8종 신설(client-build, mediaprojection-accessibility, google-login, run-and-verify, seed-helpers, seed-pim, seed-notes-tasks, seed-media-misc); `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`(Pixel6-2·JDK17·MediaProjection 단발성/graceful-degrade·EXCLUDED gms·screen_guard·no-ACK abort·권한 자동허용 반영); `.gitignore`(`**/*.secrets.local` 추가).
- 소스 패치(5파일): `app/.../ScreenStabilizer.kt`(MediaProjection 토큰 단발성 reuse-guard + `createVirtualDisplay` try/catch graceful-degrade), `CollectorService.kt`(`EXCLUDED_PACKAGES += gms/gsf/vending`), `pipeline/screen_guard.py`(`SYSTEM_PACKAGES` 확장), `pipeline/session_manager.py`(no-ACK 시 abort), `pipeline/collection_loop.py`(빈-UI 가드를 `get_interactable_elements` 기준으로 + 신규 `_try_grant_permission_via_adb`: permissioncontroller `GrantPermissionsActivity`는 a11y 이벤트 부재로 timeout만 발생 → `adb uiautomator dump`로 clickable 버튼만 스캔해 "While using the app" 우선 탭, deny-guard).
- 커밋: 미커밋(작업트리). last_sync(2026-06-29T15:48:25+09:00) 이후 신규 커밋 없음 — 위 변경은 모두 워킹트리(8 modified + `setup-collector/references/` untracked), 직전 커밋 d7ca522.
- 결과/검증(emulator-5554/Pixel6-2, 패치 APK 재설치): Google 로그인 성공(Accounts:1, seungwoo896); 더미데이터 7앱 주입(연락처5·Simple Calendar4·Markor4·org.tasks4·RetroMusic3·OpenTracks3·Joplin3, API33 재검증); smoke run 2/2 PASS — org.tasks **21 pages/29 transitions**, Drive(`com.google.android.apps.docs`) **7 pages/9 transitions**, client 크래시 0, gms 외부앱 스톰 0(Drive top=100% apps.docs), signal timeout는 탐색 소진 시 자연 종료; 권한 자동허용 단위검증(3-button→"While using the app", 2-button→"Allow" 제목 회피, deny-only→무탭) + 실다이얼로그 수동 탭 CAMERA granted=true 확인, 관련 단위테스트 57+건 PASS.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`만 갱신(작업트리). 루트 `docs/ARCHITECTURE.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-29 — Monkey-Collector 탐색 엔진을 LLM-Explorer 방식으로 전면 교체

Monkey-Collector의 탐색 엔진을 기존 `SmartExplorer`(화면 단위 weighted-random)에서
`LLMGuidedExplorer`(참조 구현 LLM-Explorer 포팅: coverage-driven unexplored-first 선택 +
LLM same-function 압축 + UI transition graph 최단경로 navigation)로 완전 교체했다.
App(Kotlin)/Server TCP 아키텍처·데이터 수집 파이프라인·저장 포맷(`data/raw/{pkg}/`, `page_graph.json`,
`events.jsonl`, xml variants, `cost.csv`, `activity_coverage.csv`)은 그대로 유지한다. 이전엔
annotation 전용이라 탐색에 반영되지 않던 same-function grouping(ScreenGrouper)을 이제 탐색에 직접 반영하며,
LLM 키가 없으면 순수 unexplored-first로 graceful degrade한다.

- 변경: working tree 14 modified + `pipeline/exploration/` 신규 8파일(`state`/`memory`/`transition_graph`/`navigator`/`action_mapper`/`explorer`/`constants`/`__init__`) + 단위·통합 테스트 6 신규(`test_semantic_state`/`test_action_mapper`/`test_transition_graph_nav`/`test_memory_unexplored`/`test_navigator`/`test_llm_guided_explorer`); `pipeline/explorer.py`(SmartExplorer) + 전용 테스트 3개 삭제. DI 배선 교체(`cli.py`/`collector.py`/`recovery.py`/`pipeline/__init__.py`/`__init__.py`), `collection_loop`는 `set_raw_xml`→`set_screen_context` 2곳만 수정(계약 유지), 세션마다 `explorer.reset()`로 메모리 격리. `networkx` 의존성 명시 추가(numpy/pandas 불필요).
- 설계: `SemanticState`(state_str/structure_str/encoded-index 기반 SemanticElement), `Memory`((structure_str, element_signature, action_type) 단위 커버리지 + same-function 압축), `TransitionGraph`(networkx structure 그래프 + `shortest_nav_steps`), `Navigator`(`_nav_steps` 큐, 매 step signature 재매칭, 무한루프 가드), `action_mapper`(semantic action→domain Action), `explorer.py`(LLMGuidedExplorer + Explorer Protocol). navigation은 Server 순차 실행(Kotlin App 무변경).
- 커밋: 미커밋(작업트리). last_sync(2026-06-28T22:52:55+09:00) 이후 신규 커밋 없음. 현재 작업트리: explorer.py 등 4 삭제(D) + 15 modified + `exploration/`·신규 테스트 6 untracked.
- 결과/검증: 전체 테스트 green, 새 모듈 ruff/mypy 완전 clean(전체 mypy 부채 baseline 29→28로 감소). Pixel6-2 E2E VERDICT PASS — org.tasks 수집 25 steps 정상 종료(completed_at 기록), page_graph **12 nodes / 15 edges**(이전 SmartExplorer 세션 7 pages 대비 증가), action 다양성 tap15/long_press5/swipe4/press_back1, screen_grouper LLM 10회 호출(same-function 압축 동작), xml/screenshots/groups 25씩 저장, activity coverage 2/48(LocationPicker 지도 화면 도달; 지도 정적화면 timeout은 공통 recovery 경로).
- 문서: 패키지 정본 `Monkey-Collector/{ARCHITECTURE,README,AGENTS}.md` 갱신(Action Space/exploration 패키지 서술) — 본 루트 `docs/ARCHITECTURE.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-28 — OpenRouter LLM 통합 라이브 검증 (qwen/qwen3.7-plus, API Key + AVD)

이전에 구현·커밋한 Monkey-Collector OpenRouter 공용 LLM 통합(입력 텍스트 생성 + 화면 의미 그룹핑)을
실제 API Key와 AVD(Pixel6-2)로 라이브 검증했다. 정적 검증과 실제 모델·실제 디바이스 화면 기반
동작 검증을 모두 통과(VERDICT PASS). 이 엔트리는 검증 기록 전용이며 코드 변경은 없다.

- 정적: `uv run pytest -q` 504 passed; 변경 파일(`llm/`, `text_generator.py`, 신규 테스트) ruff clean(전체 트리 16건은 기존 baseline `test_structured_parser.py` F401/I001), 신규 LLM 코드 mypy 0건(`save_groups`는 `is None` 가드로 narrow; 전체 29건은 기존 baseline storage `session_dir str|None` 등), AGENTS.md 게이트=pytest 통과. CLI `run --help`에 `--screen-grouping {on,off}` + `--input-mode {api,random}` 노출 확인.
- 모델 슬러그: `qwen/qwen3.7-plus` 실제 OpenRouter chat 호출 성공('OK' 반환)으로 유효 확정. 잘못된 슬러그는 OpenRouter가 400 "not a valid model ID" 반환 확인.
- 화면 의미 그룹핑(핵심 신규): 실제 org.tasks task-edit 화면 → 의미 그룹 3개(Task name input+save / Info banner / Task attribute selectors) → `xml/0000_groups.json` 저장; 별도 task-list 화면은 독립 grouping(`xml/0001_groups.json`).
- 구조 캐시: 동일 화면 재호출 시 LLM 미호출(screen_grouper cost row 불변, 동일 객체 반환) — 비용 절감 동작 확인.
- 입력 생성: 실제 EditText(`org.tasks:id/title`, "Task name")에 문맥 인지 텍스트 'Buy groceries' 생성(LLMTextGenerator 경로).
- 비용 귀속: cost.csv agent 컬럼 정확 분리(screen_grouper×2 + text_generator×1), 단가 계산 정확(예: 533 in·1851 out → $0.0024344 @ qwen/qwen3.7-plus $0.40/$1.20 per 1M).
- Graceful fallback: 잘못된 `OPENROUTER_MODEL` override → 화면 그룹핑 빈 결과 + 입력 텍스트 random fallback, 둘 다 예외 전파 없음.
- 검증 방식: collection_loop/`cli.cmd_run`과 동일하게 DataWriter·CostTracker·공용 LLMClient·ScreenGrouper·LLMTextGenerator를 wiring한 harness로 실구동(scratchpad 격리, 프로젝트 data/raw 무손상).
- 미검증(정직 보고): 풀 `monkey-collect run` TCP 전 구간은 미실행 — 동반 Android 앱 `com.monkey.collector`가 AVD에 미설치라 TCP 핸드셰이크 상대 부재(해당 transport 계층은 이번 LLM 통합 변경과 무관).
- 변경: 코드 변경 없음(기록 전용). last_sync(2026-06-28T22:24:32+09:00) 이후 신규 커밋 없음; 관련 코드 커밋은 이전 동기화에 반영(af40196·13d1a48·a284c72·879f130·a22b030). 작업트리 미커밋: `.project-sync.json`, `docs/CHANGELOG.md`, `docs/DEVLOG.md`(이전 동기화 산출물).
- 카테고리: devlog

## 2026-06-28 — Monkey-Collector LLM 통합 재구성 (OpenRouter 공용 클라이언트 + 화면 의미 그룹핑)

`.claude/references/LLM-Explorer`(droidbot 기반 탐색기)의 `GPT` 클래스 패턴을 참고해 Monkey-Collector의
LLM 통합을 재구성했다. env 기반 OpenRouter OpenAI-호환 Chat Completions 래퍼인 **공용 클라이언트**
(`llm/client.py`, 기본 모델 `qwen/qwen3.7-plus`, 키 없으면 None → graceful fallback)를 신설하고 두 사용처가
이를 공유한다: (1) 입력 텍스트 생성을 OpenAI Responses API(gpt-5-nano)에서 공용 `LLMClient.chat()`으로
이전(OpenRouter가 Responses API 미지원 → Chat Completions 전환), (2) LLM-Explorer `_gen_state_semantic_info`를
이식한 **화면 의미 그룹핑**(`llm/screen_grouper.py`, `--screen-grouping {on,off}` 기본 on, 요소를 기능별로
묶어 `xml/{step}_groups.json` annotation 저장, 동일 구조 캐시로 비용 절감, 실패 시 수집 흐름 무영향). 비용은
cost.csv의 agent 컬럼(text_generator/screen_grouper)으로 구분하고 cost_tracker에 qwen 단가를 추가했다.

- 변경: `llm/{client,screen_grouper,__init__}.py`(신규), `pipeline/text_generator.py`·`collection_loop.py`·`collector.py`, `storage.py`, `cli.py`, `domain/cost_tracker.py`, `.env.example` + Monkey-Collector README/ARCHITECTURE/AGENTS 동기화 (19 files, +835 −175 @ HEAD~5..HEAD)
- 커밋: af40196 (공용 LLM 패키지), 13d1a48 (text 생성 이전), a284c72 (파이프라인 배선), 879f130 (문서 동기화), a22b030 (루트 .claude gitignore) — main push 완료
- 결과/검증: 전체 504 tests passed (신규 test_llm_client/test_screen_grouper + test_text_generator/test_storage 갱신 포함); 도입분 ruff/mypy 이슈 해결(잔여는 baseline 기존 이슈); `run --help`에 `--screen-grouping` 노출 확인. 라이브 스모크 미실행(Pixel6-2 AVD + OPENROUTER_API_KEY 필요).
- 후속: 모델 슬러그 `qwen/qwen3.7-plus`는 하드-고정 안 함(`OPENROUTER_MODEL` override, 카탈로그 실제 슬러그 확인 권장); 그룹핑은 annotation 저장만이며 explorer 탐색 편향 미적용(UITree index ↔ encoded index 공간 불일치) → 후속 분리 가능.
- 카테고리: devlog

## 2026-06-28 — `/project-sync` 초기 설정 (docs·memory·notion)

`/project-sync init`으로 프로젝트 기록 동기화를 설정했다. 모노레포 루트에 `docs/` 허브를 만들고,
Notion 워크스페이스의 5개 카테고리 DB + Timeline 허브 ID를 랜딩 페이지에서 자동 추출해 config에 등록했다.

- 변경: `.project-sync.json`(신규), `docs/{README,ARCHITECTURE,ROADMAP,CHANGELOG,DEVLOG}.md`(신규)
- 활성 플랫폼: `docs`, `memory`, `notion` — Obsidian은 이 Linux 머신에 Vault가 없어 제외
- 결과/검증: Notion 랜딩 페이지 읽기 접근 검증 완료(5 DB + 허브 ID 추출, config.md 캐시와 일치)
- 카테고리: devlog
