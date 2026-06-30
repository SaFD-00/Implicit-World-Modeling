# Changelog

[Keep a Changelog](https://keepachangelog.com/) 스타일. 시점성 진행은 [DEVLOG.md](./DEVLOG.md),
계획은 [ROADMAP.md](./ROADMAP.md) 참조.

## [Unreleased]

### Added
- Monkey-Collector: external 복구의 타깃 앱 재실행을 `open_app` 액션으로 events.jsonl 에 기록(`DataWriter.log_open_app`, excursion 당 1회) — open_app 학습용. `return_to_app`/`recover` 가 launch 여부를 `bool` 반환, `cli._resolve_app_names`→`Collector(app_names=...)` 로 `app_name` 조인. navigation 격리 3중(`state.last_action` 클리어·`explorer._last_record` 클리어·`transition:false` + `page_graph._load_events` 스킵) + `DataWriter` 멀티스레드 `threading.Lock`  (2026-06-30)
- `docs/` 루트 문서 허브 + `/project-sync` 설정(`.project-sync.json`) 도입  (2026-06-28)
- Monkey-Collector: OpenRouter 공용 LLM 클라이언트(`llm/client.py`, 기본 `qwen/qwen3.7-plus`) + 화면 의미 그룹핑(`llm/screen_grouper.py`, `--screen-grouping` 플래그)  (2026-06-28)
- Monkey-Collector: `setup-collector` 스킬 `references/` deep-dive 8종(client-build, mediaprojection-accessibility, google-login, run-and-verify, seed-helpers, seed-pim, seed-notes-tasks, seed-media-misc) + 런타임 권한 다이얼로그 adb 자동허용(`collection_loop._try_grant_permission_via_adb`, "While using the app" 우선 탭·deny-guard)  (2026-06-29)
- Monkey-Collector: element-set screen matching 패키지(`pipeline/screen_matching/` — ScreenMatcher·ui_attributes·set_classifier·page_knowledge) + 화면당 단일 LLM 호출 `llm/element_extractor.py`/`llm/prompts/element_extractor_prompt.py`(element_index family + key_element_index anchor 동시 추출, MobileGPT-V2 Node-Clustering 포팅); CLI `--cluster-merge-tolerance`·`--max-expand-iters` 플래그  (2026-06-29)

### Changed
- Monkey-Collector: 입력 텍스트 생성을 OpenAI Responses API(gpt-5-nano)에서 공용 OpenRouter `LLMClient`(Chat Completions)로 이전  (2026-06-28)
- Monkey-Collector: 탐색 엔진을 `SmartExplorer`(화면 단위 weighted-random)에서 `LLMGuidedExplorer`(coverage-driven unexplored-first + LLM same-function 압축 + transition-graph 최단경로 navigation, 신규 `pipeline/exploration/` 패키지)로 전면 교체; App/Server TCP·저장 포맷 유지, `networkx` 의존성 추가  (2026-06-29)
- Monkey-Collector: `setup-collector` 스킬을 SKILL.md 오케스트레이션 + `references/` 구조로 재구성(AVD Pixel6-2, 빌드 JDK17/AGP8.2, MediaProjection 재동의·Google 로그인·더미데이터 시드·라이브 검증 단계 추가, 전 단계 멱등)  (2026-06-29)
- Monkey-Collector: input-text LLM 생성 프롬프트에 현재 앱 설명 주입 — `AppJob.description`(`app_name (category/sub_category) — notes`, csv 미등록 앱은 package_id 폴백)을 CLI `_resolve_app_contexts`→`Collector(app_contexts=...)`→세션마다 `TextGenerator.set_app_context()`로 전달해 앱 도메인에 맞는 입력값 생성; 공유 generator 누수 방지 위해 매 세션 무조건 set, random 전략은 no-op  (2026-06-29)
- Monkey-Collector: 화면 의미 그룹핑(annotation 전용)을 element-set screen matching으로 교체 — 산출 `page_key`가 page_graph 노드(`PageGraph.get_or_create_page_by_match`)와 탐색 abstract page(`SemanticState.page_key`, Memory/TransitionGraph/Navigator 키)를 동시 결정(과거 grouping↔matching 디커플을 커플링); CLI `--screen-grouping` → `--element-extraction {on,off}`(deprecated alias 유지), 산출물 `{step}_groups.json`→`{step}_elements.json`, cost.csv 라벨 `screen_grouper`→`element_extractor`; 키 없거나 off면 `page_key=structure_str` fallback(기존 파이프라인 byte-for-byte)  (2026-06-29)

### Removed
- Monkey-Collector: `llm/screen_grouper.py` + `tests/unit/test_screen_grouper.py` — element-set screen matching(`pipeline/screen_matching/`)으로 대체  (2026-06-29)

### Fixed
- Monkey-Collector: MediaProjection 토큰 단발성 reuse-guard + `createVirtualDisplay` graceful-degrade(`ScreenStabilizer`); `EXCLUDED_PACKAGES`(`CollectorService`)·`SYSTEM_PACKAGES`(`screen_guard`)에 gms/gsf/vending 추가로 외부앱 스톰 차단; no-ACK 세션 abort(`session_manager`)  (2026-06-29)
- Monkey-Collector: 빈 page_0 blackhole 수정 — interactable 0개 화면(첫 로딩/스플래시)을 `ScreenMatcher.match()` entry guard로 `pending` 거부해 page 미등록(첫 유효 화면이 page_0), `collection_loop`가 pending 시 page 노드·`save_elements` 스킵; `set_classifier` 안전망(저장 page B=∅이면 SUPERSET_MERGE 불가→DISJOINT라 빈 page가 sink 안 됨); `extract_interactable_indexes`를 root-inclusive(`tree.iter()`)로 수정해 단일 루트 interactable 누락 차단  (2026-06-30)
- Monkey-Collector: LLM element의 `description`/`parameters`가 디스크 저장 시 누락되던 문제 수정 — `ElementFamily`에 두 필드 추가(끝에, 하위호환), families 생성부가 ExtractedElement 5필드 전부 채움, `DataWriter.save_elements`가 `{step}_elements.json` 각 element에 직렬화(최종: name/description/parameters/element_index/key_element_index)  (2026-06-30)

> 검증(2026-06-28): 위 OpenRouter LLM 통합을 실제 API Key + AVD(Pixel6-2)로 라이브 검증 — 정적 504 passed, 모델 슬러그(`qwen/qwen3.7-plus`) 실호출·화면 의미 그룹핑·문맥 입력 생성·비용 귀속·graceful fallback 동작 확인 (VERDICT PASS). 상세는 [DEVLOG.md](./DEVLOG.md).

<!-- project-sync: 릴리스/버전 변경 요약을 Added/Changed/Fixed로 한 줄씩 추가. -->
