# Dev Log

시점성 진행 로그 (append-only). 최신 엔트리를 위에 추가한다. 과거 엔트리는 수정·삭제하지 않는다.
상세 결과는 Notion Dev Log / Experiments DB, 계획은 [ROADMAP.md](./ROADMAP.md) 참조.

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
