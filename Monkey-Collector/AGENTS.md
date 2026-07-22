# AGENTS.md

`Monkey-Collector/` 하위 프로젝트에서 작업하는 에이전트를 위한 가이드다.

## 현재 코드 기준 요약

- Python 쪽 진입점은 [`src/monkey_collector/cli.py`](./src/monkey_collector/cli.py) 의 `monkey-collect` CLI 다.
- 공개 API 는 [`src/monkey_collector/__init__.py`](./src/monkey_collector/__init__.py) 에서 export 된다.
- Android 앱 코드는 [`app/app/src/main/java/com/monkey/collector`](./app/app/src/main/java/com/monkey/collector) 아래에 있다.
- 서버 구조는 `domain`, `llm`, `pipeline`, `export`, `xml` 5개 서브패키지 + 인프라 모듈 (`adb.py`, `tcp_server.py`, `storage.py`) 로 구성된다.

## 환경 셋업 & 평가 방법

### 1. 셋업 (초기화 시 1회, 멱등)

- `/setup-emulator` 스킬([`/.claude/skills/setup-emulator/`](../.claude/skills/setup-emulator))이 전 과정을 자동화한다 — AVD·APK·client 빌드·접근성·prefs·MediaProjection·Google 로그인·더미데이터 시드·검증.
- **client 빌드는 AGP 8.2 → JDK 17 필수**: `JAVA_HOME=$(/usr/libexec/java_home -v 17) ./gradlew :app:assembleDebug` (APK 는 `app/app/build/outputs/...`).
- **클라이언트(.kt) 수정은 APK 재빌드·재설치해야 디바이스에 반영된다** — 재설치를 빼먹으면 옛 APK 가 계속 돈다.
- Google 자격증명은 `Monkey-Collector/.secrets.local`(gitignore)에서만 읽는다 — **커밋 금지**.

### 2. 게이트 (코드 변경 시)

```bash
./.venv/bin/python -m pytest tests    # bare `python` 금지 (venv 밖 인터프리터를 잡는다)
```

현재 기준선은 **858 passed** 다(2026-07-22). 이 수가 줄면 회귀로 본다 — 단 **테스트를 의도적으로 삭제한 변경은 예외**이고, 그때는 삭제 개수까지 세어 새 기준선을 여기에 갱신한다(직전 869 − canvas 테스트 17 + converter dedup 신규 6 = 858).

#### 빠른 검증 포인트

- `pytest -q` (전체)
- `pytest -q tests/unit` (외부 의존 없음, 빠름)
- `pytest -q tests/integration` (ADB / TCP / subprocess mock)
- `pytest -q tests/unit/test_cli.py tests/integration/test_collector.py tests/unit/test_storage.py`
- `pytest -q tests/unit/test_llm_client.py tests/unit/test_text_generator.py` (LLM 클라이언트 / 입력 텍스트 생성)
- `pytest -q tests/unit/test_bm25.py tests/unit/test_element_lines.py tests/unit/test_screen_matcher.py tests/unit/test_rehydrate.py tests/unit/test_luminance.py tests/unit/test_config.py` (BM25 unique-page matching + 재개 rehydration)
- `pytest -q tests/unit/test_app_catalog.py tests/integration/test_installed_sync.py tests/unit/test_run_resume.py tests/unit/test_session_manager.py tests/unit/test_reset.py`
- `pytest -q tests/unit/test_page_graph.py tests/unit/test_page_graph_matching.py tests/unit/test_converter.py` (page graph 재구성 + world-modeling 변환)
- `python -m monkey_collector.cli run --help`
- `python -m monkey_collector.cli sync-installed --help`
- `python -m monkey_collector.cli reset --help`
- `python -m monkey_collector.cli page-map --help`

### 3. 수치 SSoT — 어디서 읽는가

수집 결과 수치는 **아래 출처에서만** 읽는다. 로그를 grep 해 세지 마라.

- **steps** = `runtime/apps/<pkg>/metadata.json` 의 `total_steps`. 서버 로그의 `Session complete: … steps=N` 과 일치함이 검증됐다.
- ⚠️ **로그를 `grep -cE 'Step [0-9]+:'` 로 세지 마라** — 재시도·타임아웃·키보드 처리 라인까지 세는 **오염 카운터**다. 실측 196/181/186 vs 진짜 145/113/130.
- **pages/edges** = `data/raw/<pkg>/page_graph.json` 의 `len(nodes)`/`len(edges)`. pages 는 로그의 `max_page_id + 1` 로도 재도출 가능하다.
- ⚠️ **`page_graph.json` 은 run 간에 덮어써진다** — 반드시 **아카이브본**에서 읽어라. 이 함정이 실제로 데이터를 두 번 죽였다.
- **학습 examples** = `data/processed/gui-model_stage1.jsonl` 의 라인 수. **현재 24앱 9,943 examples**(2026-07-22 재생성; 중복 0, 이미지 1:1). 게이트별 감소는 pre-dedup **13,299** → XML 게이트만 **11,376** → \+ page 게이트 **9,943** 이다(게이트 계약은 불변식 10).
- ⚠️ **`convert-all` 은 append-only 라 재실행하면 누적된다** — examples 수를 재측정하려면 `data/processed/` 의 기존 `gui-model_stage1.jsonl` 과 `images/` 를 먼저 비워라. 파일 라인 수만 믿고 세면 누적본을 "현재 산출" 로 오독한다.

### 4. 효과 측정 (arm 비교)

프로토콜 본문은 아래 `## 효과 측정 프로토콜` 섹션이 정본이다. 헬퍼 계약:

- `../.claude/handoff/reset_app.sh` — 앱 리셋 + seed 보존 검증
- `../.claude/handoff/measure.sh` — 리셋 + 수집 1회 (steps SSoT 출력)
- `../.claude/handoff/instrument_p1.sh` — 위 + logcat + **산출물 아카이브**. ⚠️ **아카이브 없이 돌리면 다음 run 이 `data/raw/<pkg>/` 를 덮어쓴다.**

### 5. F2 on/off arm 구성 (server pull poke)

- `collection.poke_delay_sec` 가 `0` 이하이면 poke 가 비활성된다 — [`collection_loop.py`](./src/monkey_collector/pipeline/collection_loop.py) 의 `if delay <= 0 or delay >= timeout:`.
- 따라서 `MC_COLLECTION_POKE_DELAY_SEC=0` 으로 **같은 코드에서 baseline(poke off) arm** 을 만들 수 있다 — worktree 불필요.
- 이 경우 env 가 arm 정의의 일부다(`## 효과 측정 프로토콜` 의 "`env | grep MC_` 가 비어 있어야 한다" 는 그 외 모든 env 에 적용된다). **arm 출처(어느 토글로 만든 arm 인지)를 로그·META 에 반드시 기록하라.**

## 불변식 — 깨뜨리면 안 되는 것

아래 10항은 이 수집기의 **계약**이다. 이 목록이 인덱스 겸 계약 선언이고, 세부는 각 항이 가리키는 파일/섹션에 있다.
깨져도 대개 **크래시가 아니라 조용한 데이터 오염**으로 나타나므로, 건드리기 전에 여기부터 읽어라.

1. **외부앱 제외 목록은 양쪽 짝이다**
   - 계약: 클라이언트 `CollectorService.EXCLUDED_PACKAGES` 의 `gms`/`gsf`/`vending` ↔ 서버 [`pipeline/screen_guard.py`](./src/monkey_collector/pipeline/screen_guard.py) 의 `SYSTEM_PACKAGES` 를 **함께** 유지한다.
   - 깨지면: Google 로그인·Play 핸드오프 화면에서 **외부앱 재실행 스톰**.
2. **MediaProjection 토큰은 단발성이다**
   - 계약: `ScreenStabilizer.startCaptureSession` 의 reuse-guard + `createVirtualDisplay` try/catch graceful-degrade 를 유지한다.
   - 깨지면: 2세션째 `SecurityException` → client 프로세스 사망 → signal timeout 연쇄.
3. **신호 이름 계약**
   - 계약: App→Server `P`/`S`/`X`/`E`/`N`/`F`, Server→App `{"type":"START"}`/`{"type":"SESSION_END"}`/`{"type":"CAPTURE"}` (`CAPTURE` 는 F2 서버 pull 로 추가됨).
   - 깨지면: `CollectorService.beginStandby` 가 이 계약에 의존하므로 standby·수집이 멈춘다.
4. **세션 전환 핸드셰이크는 fresh 소켓을 보존한다**
   - 계약: `CollectionServer.reset_for_new_session()` 은 큐/이벤트만 초기화하고 **소켓은 그대로 둔다**.
   - 깨지면: 여기서 소켓을 닫으면 클라이언트가 두 번째 재접속을 하지 않아 **두 번째 세션부터 `wait_for_connection` 이 전부 타임아웃**.
5. **signal 큐는 제어신호를 잃지 않는다**
   - 계약: `get_latest_signal` 은 **연속된 `xml` 프레임만** collapse 하고 `external_app`/`finish`/`no_change` 는 **절대 드롭하지 않는다**.
   - 깨지면: 외부앱 복구·세션 종료 신호가 뒤이은 `xml` 에 덮여 유실된다. 테스트: `tests/integration/test_server.py::TestSignalQueue`.
6. **`state.step` 은 정상 action 경로에서만 `+1`**
   - 계약: signal timeout·no_change·empty-UI 대기·keyboard/permission/system/stale 같은 **비-action 반복에서는 올리지 않는다**.
   - 깨지면: `step` 이 `frame_index` 와 어긋나 정렬이 깨진다 — 과거 정렬 버그의 원인이 정확히 이것이다.
7. **`--new-session`/`reset` 은 두 root 를 함께 지운다 — 단 수집 세션 상태만**
   - 계약: `data/raw/{pkg}/` 와 `runtime/apps/{pkg}/` 를 **동시에** 삭제한다.
   - 계약: **`data/processed/` 는 절대 지우지 않는다.** 파생 학습 코퍼스는 수집 세션 상태가 아니다. `data/processed` 는 리셋 루트 `data/raw` 의 **형제**라 어떤 스코프도 닿지 않는다 — 가드가 아니라 레이아웃으로 성립한다(`pipeline/reset.py`, `tests/unit/test_reset.py::TestProcessedCorpusPreserved`).
   - 깨지면: 한쪽만 지우면 남은 쪽에서 지식이 rehydrate 되어 **"새 세션"이 되지 않는다**. 반대로 리셋 루트를 `data/` 로 올려 잡으면 수집과 무관한 학습 코퍼스까지 날린다.
8. **`open_app` 은 navigation 전이가 아니다 (3중 격리)**
   - 계약: 복구 시 `state.last_action` 클리어 + `explorer._last_record` 클리어 + 이벤트 `transition: false` — 셋을 모두 유지한다.
   - 깨지면: 하나라도 바꾸면 **가짜 전이**가 live page graph·routing memory·offline 재빌드/world-modeling 변환으로 샌다.
9. **측정 불변식**
   - 계약: 매 run 앱 리셋 + arm 간 차이는 **코드(또는 문서화된 config 토글) 하나뿐** + seed 보존.
   - 깨지면: confound 로 오염돼 인과 해석이 불가하다 — 과거 "다양성 +78%" 오귀속 사례. 헬퍼: `../.claude/handoff/reset_app.sh`.
10. **converter dedup 게이트는 2종이고 스코프가 서로 다르다**
    - 계약: **XML 3튜플 게이트**(`(before_encoded_xml, action_json, after_encoded_xml)`)는 **전역**, **page 3튜플 게이트**(`(package, before_page_key, action_json, after_page_key)`)는 **패키지(앱) 스코프**로 유지한다. 둘 다 끄는 플래그가 없고, 하나라도 히트하면 write·이미지·count 가 전부 없다.
    - 깨지면: page 게이트에서 `package` 를 빼 전역으로 만들면 `page_key` 가 **앱마다 0 부터 재시작하는 카운터**라 서로 무관한 앱의 전이가 우연한 카운터 일치로 합쳐진다 — **실측 452건이 조용히 사라졌다**. 반대로 `action_json` 을 예제의 human turn 문자열에서 뽑으면 before-XML 이 통째로 섞여 게이트가 사실상 no-op 이 된다. 어느 쪽도 크래시 없이 코퍼스만 틀어진다. 세부는 [ARCHITECTURE.md](./ARCHITECTURE.md) §`converter.py`.

## 어디를 수정해야 하는가

- CLI 옵션이나 서브커맨드를 바꾸면 [`src/monkey_collector/cli.py`](./src/monkey_collector/cli.py) 와 [`tests/unit/test_cli.py`](./tests/unit/test_cli.py) 를 함께 수정한다.
  - ADB 는 `AdbClient()` 를 인자 없이 생성하며, 내부에서 `Pixel6-2` 라는 이름의 AVD 를 자동 탐색해 해당 emulator serial 로 모든 명령을 고정한다 (상수 `REQUIRED_AVD_NAME` 은 [`src/monkey_collector/adb.py`](./src/monkey_collector/adb.py) 상단, 기본값 `Pixel6-2` 이며 env `MC_AVD` 로 오버라이드 가능).
  - AVD 이름을 바꿔야 한다면 이 상수와 관련 문서 / 테스트를 함께 수정한다.
- 수집 루프 동작은 [`src/monkey_collector/pipeline/collector.py`](./src/monkey_collector/pipeline/collector.py), [`src/monkey_collector/pipeline/collection_loop.py`](./src/monkey_collector/pipeline/collection_loop.py), [`src/monkey_collector/pipeline/session_manager.py`](./src/monkey_collector/pipeline/session_manager.py) 가 기준이다.
  - activity coverage 분모와 분자 후보 집합은 `session_manager._resolve_declared_activities` 가 결정한다 — catalog hit 이면 (`allow_dynamic_total=False`) 분모 고정 + `unique_visited` 는 catalog set 안의 activity 만 카운트, miss 면 dumpsys 폴백 + WARNING 로그 + legacy 동적 확장.
  - backfill 은 별도 스크립트 없이 `tracker.resume()` 이 같은 정책으로 처리한다.
  - coverage 에 기록하는 activity 는 `collection_loop._resolve_coverage_activity` 가 판별한다 — a11y `activity_name` 이 declared activity(`ActivityCoverageTracker.is_declared`)이면 그대로 쓰고, 아니면(a11y 가 generic `.../android.view.ViewGroup` 로 보고하는 프레임) `adb.get_current_activity()`(topResumedActivity, bounded 5s)로 실제 foreground
    activity 를 해석한다.
  - matcher/page_graph/save/event 는 원래 a11y `activity_name` 을 유지하니(blast radius 억제) 이 분기를 바꾸면 `tests/unit/test_coverage_activity.py`·`test_activity_coverage.py` 를 함께 본다.
  - 세션 종료 조건은 `collection_loop._has_budget` 이 판정한다 — `budget_mode`(config `collection.budget_mode`, 제품 기본 `time`)가 `time` 이면 벽시계 `max_duration`(기본 `2h`, `run_collection_loop` 진입마다 재계산) 까지, `steps` 면 `max_steps` 까지 돈다.
  - 어느 쪽이든 루프가 정상 return 해 `finally` 의 `finalize_session` 이 `completed_at` 을 채운다(시간 예산은 mid-run kill 없이 clean-stop 을 보장).
  - 이 로직/기본값을 바꾸면 `tests/unit/test_loop_budget.py`·`test_config.py`(`parse_duration`/`_normalize_budget_mode`)·`test_cli.py` 를 함께 본다.
- 앱 목록 / 설치 상태 처리는 두 모듈로 분리되어 있다:
  - [`src/monkey_collector/pipeline/app_catalog.py`](./src/monkey_collector/pipeline/app_catalog.py): `catalog/apps.csv` 파싱과 category/priority/installed 필터. 새 필수 컬럼 추가는 `_REQUIRED_COLUMNS` 와 `AppJob` 을 동시에 수정. `installed` 는 optional 컬럼 — 누락된 CSV 는 자동으로 모두 `false` 로 해석된다.
  - [`src/monkey_collector/pipeline/installed_sync.py`](./src/monkey_collector/pipeline/installed_sync.py): `sync-installed` 서브커맨드의 백엔드. `catalog/apps.csv` 의 `installed` 컬럼만 in-place 로 덮어쓰므로 다른 필드를 건드리지 마라.
  - [`catalog/`](./catalog): 앱 카탈로그 자원. `apps.csv`, `apks/{pkg}.apk`, `download_apks.py`, `extract_activities.py`, `activities.json` 이 한곳에 모여 있다. CLI 기본값은 `catalog/apps.csv` 를 가리키므로 CWD 가 프로젝트 루트라는 전제로 동작한다.
  - [`catalog/activities.json`](./catalog/activities.json): androguard 가 `apks/*.apk` 의 `AndroidManifest.xml` 에서 뽑아낸 declared activity 사전(원본, 불변).
    - `activity_coverage.csv` 의 분모(`total_activities`)와 분자(`unique_visited`) 는 이 파일의 `activities` 리스트에서 [`src/monkey_collector/catalog_activities.py`](./src/monkey_collector/catalog_activities.py) 의 `ActivityCatalog.get_declared`/`get_aliases` 가 `NON_NAVIGABLE_CLASS_PREFIXES`(3p framework/SDK namespace:
      `androidx.car.app.`/`com.android.billingclient.`/`com.google.android.gms.`/`com.google.android.play.core.`)를 제외한 결과 기준이다 — 앱 자체 namespace 의 activity(예: notification permission 화면)는 이름이 non-navigable 해 보여도 제외하지 않는다(의도적 보수주의).
    - `ActivityCatalog` 가 이 필터를 적용한 채로 process-lifetime 캐시한다.
    - 각 엔트리는 `activities` 외에 `aliases`(`<activity-alias>` name→targetActivity 맵; androguard `get_activities()` 는 alias 를 반환하지 않으므로 manifest lxml 트리에서 element 단위로 추출)를 담는다 — alias 방문은 target 으로 해석돼 분자에만 반영되고 분모(`activities`)는 고정, target 이 필터 대상이면 alias 도 함께 드롭된다.
    - 새 APK 가 추가되면 `python -m catalog.extract_activities` 로 갱신한다(이 재생성은 필터를 거치지 않은 raw 사전을 다시 만들 뿐, 필터는 항상 `ActivityCatalog` 조회 시점에 적용된다).
  - [`src/monkey_collector/pipeline/reset.py`](./src/monkey_collector/pipeline/reset.py): 수집 데이터 삭제 스코프 해소(`all` / `packages`)와 `shutil.rmtree` 실행. 순수 함수 (`resolve_targets`, `delete_targets`).
- 완료 앱 스킵 로직은 [`src/monkey_collector/cli.py`](./src/monkey_collector/cli.py) 의 `_resolve_run_packages` / `_load_completed_packages` 에 있다. `metadata.completed_at` 이 채워진 앱은 기본적으로 큐에서 제외되고, `--force` 로 우회한다. 이 규약이 바뀌면 `tests/unit/test_run_resume.py` 를 함께 업데이트한다.
- 액션 선택 로직은 [`src/monkey_collector/pipeline/exploration/`](./src/monkey_collector/pipeline/exploration) (LLM-Explorer 알고리즘 포팅, 런타임 LLM 미호출: `CoverageGuidedExplorer` + `Explorer` Protocol, `SemanticState`/`Memory`/`TransitionGraph`/`Navigator`/`ActionMapper`) 와 그 단위
  테스트(`tests/unit/test_semantic_state.py`·`test_memory_unexplored.py`·`test_transition_graph_nav.py`·`test_navigator.py`·`test_action_mapper.py`, `tests/integration/test_coverage_guided_explorer.py`) 를 함께 본다.
  - 커버리지는 `(page_key, element_signature, action_type)` 단위로 추적하고, `TransitionGraph` 최단경로로 미탐색 화면까지 navigation 한다.
  - page 식별(`page_key`)은 [`pipeline/screen_matching/`](./src/monkey_collector/pipeline/screen_matching) 의 `ScreenMatcher`(**BM25 unique-page matching**, Mobile3M 메커니즘, **LLM-free**)가 결정하며, 이 `page_key` 가 page_graph 노드와 탐색 abstract page 를 모두 좌우한다 — 매칭 흐름(pre-filter → element-line serialize → BM25 top-K → conjunctive verify)·pixel 게이트·`persist_filtered`·rehydrate 메커니즘은 [ARCHITECTURE.md](./ARCHITECTURE.md) §2(screen_matching) 참조.
  - screen_matching 을 바꾸면 `tests/unit/test_bm25.py`·`test_element_lines.py`·`test_screen_matcher.py`·`test_rehydrate.py`·`test_luminance.py`·`test_config.py`(+`test_ui_attributes.py`) 를 함께 본다.
  - 하이퍼파라미터(`bm25_top_k`/`element_criterion`/`element_diff_max`/`element_jaccard_min`/`page_pixel_diff_threshold` 등)는 `screen_matching` config(builtin+yaml+env+CLI 4단계)로 조정한다 — 키 목록은 [ARCHITECTURE.md](./ARCHITECTURE.md) §7 참조.
- LLM 통합은 [`src/monkey_collector/llm/client.py`](./src/monkey_collector/llm/client.py) 의 공용 `LLMClient` (OpenRouter Chat Completions, 기본 `qwen/qwen3.7-plus`, env `OPENROUTER_API_KEY`/`OPENROUTER_BASE_URL`/`OPENROUTER_MODEL`) 하나로 모인다.
  - 유일한 런타임 소비자는 텍스트 입력 생성 [`text_generator.py`](./src/monkey_collector/pipeline/text_generator.py) 이다 (random fallback 유지; `Collector._run_session` 이 세션마다 `set_app_context()` 로 현재 앱 설명 — `catalog/apps.csv` 의 `AppJob.description`, 미등록 앱은 package_id 폴백 — 을 프롬프트에 `App under test:` 줄로 주입).
  - provider/모델을 바꾸려면 `client.py`, `.env.example`, [`cost_tracker.py`](./src/monkey_collector/domain/cost_tracker.py) 의 `MODEL_PRICING` 을 함께 본다.
  - `OPENROUTER_API_KEY` 가 없으면 입력 텍스트 생성은 random 으로 degrade 하며 수집 흐름을 깨지 않는다. page 식별(`ScreenMatcher`)은 LLM 무관이라 그대로 동작한다.
  - 비용은 `cost.csv` 의 `agent` 컬럼(현재 `text_generator`)에 기록된다.
- 세션 저장 형식은 [`src/monkey_collector/storage.py`](./src/monkey_collector/storage.py) 가 기준이다.
  - `DataWriter` 는 두 root 로 나뉜다 — `data/raw/{package}/pages/{page_key}/{observation_num}/` (영속, `save_observation`/`save_page_knowledge` 는 `is_new_observation` 일 때만 새로 씀) 와 `runtime/apps/{package}/` (휘발성: metadata/events/cost/coverage — `runtime/logs/` 가 그 형제).
  - 세션 재개(resume)는 `session_manager.rehydrate_session` → `screen_matching/rehydrate.py` 가 `data/raw/{package}/pages/` 를 다시 읽어 `ScreenMatcher` 지식과 `state.page_graph` 를 모두 복원한다.
  - **코퍼스는 `data/` 아래에서 이원화돼 있다** — `data/raw/`(수집 원본, `collection.data_dir` 기본값)와 `data/processed/`(`convert-all` 이 만드는 학습 변환 산출물: `gui-model_stage1.jsonl` + `images/`). 수집기는 `data/raw` 만 쓰고, `convert-all` 은 `data/raw` 를 읽어 `data/processed` 에 쓴다.
- world-modeling 변환은 [`src/monkey_collector/export/converter.py`](./src/monkey_collector/export/converter.py) 가 기준이다.
  - **완전중복 예제는 항상 1건만 남는다 — 끄는 플래그가 없다.** 판정 게이트는 **2종**이며(불변식 10), 기존 필터(`transition:false`·`no_change_retry`·조인키 null·같은 observation 연속·`before==after`)를 모두 통과한 뒤의 마지막 관문이다. 둘 중 **하나라도 히트하면** JSONL 미기록 + 이미지 미복사 + count 미증가.
    - **XML 3튜플 게이트** — `(before_encoded_xml, action_json, after_encoded_xml)`. 스코프 **전역**.
    - **page 3튜플 게이트** — `(package, before_page_key, action_json, after_page_key)`. 스코프 **패키지(앱)**. matcher 가 확정한 `page_key` 를 신뢰하므로 encoded XML 의 바이트 흔들림에 안 흔들린다. 구형 평면 레이아웃(`_convert_session_legacy`)은 `page_key` 가 없어 이 게이트가 적용되지 않는다.
  - dedup 은 **`Converter` 인스턴스 단위**다. `convert-all` 은 `Converter` 하나를 모든 세션에 재사용하므로 **XML 게이트가 앱(세션) 경계를 넘어 전역으로** 걸린다 — 두 앱에서 같은 encoded XML 전이가 나오면 1건만 남는다. 의도된 동작이다.
  - ⚠️ **page 게이트는 전역이면 안 된다** — `page_key` 는 앱마다 0 부터 재시작하는 카운터라 `package` 를 빼면 무관한 앱의 전이가 합쳐진다(실측 452건 오제거). 상세는 불변식 10.
  - ⚠️ **`convert-all` 은 append-only 다** — output JSONL 을 `"a"` 로만 열고 truncate 하지 않으며, dedup seen-set 은 인스턴스 로컬이라 **파일에 이미 있는 줄과는 대조하지 않는다**. 재실행 전 `data/processed/` 의 기존 `gui-model_stage1.jsonl` 과 `images/` 를 비우지 않으면 중복이 그대로 누적된다.
- XML 파싱 규약은 [`src/monkey_collector/xml/ui_tree.py`](./src/monkey_collector/xml/ui_tree.py), [`src/monkey_collector/xml/structured_parser.py`](./src/monkey_collector/xml/structured_parser.py) 를 본다.
- Android 측 전환 감지와 TCP 프로토콜은 [`CollectorService.kt`](./app/app/src/main/java/com/monkey/collector/CollectorService.kt), [`ScreenStabilizer.kt`](./app/app/src/main/java/com/monkey/collector/ScreenStabilizer.kt), [`TcpClient.kt`](./app/app/src/main/java/com/monkey/collector/TcpClient.kt) 에 있다 — 안정성 규약(외부앱 제외 목록 짝맞춤, MediaProjection 토큰 단발성)은 불변식 1·2 참조.
  - **클라이언트(.kt) 수정은 APK 재빌드(JDK 17)·재설치해야 디바이스에 반영**된다.
- 환경 셋업/검증(`/setup-emulator` 스킬, JDK17 client 빌드, Google 자격증명 정책)은 위 「환경 셋업 & 평가 방법 → 1. 셋업」과 동일 — 세부는 그쪽을 본다.

## 작업 시 주의점

- 세션은 두 root 로 나뉜다 — `data/raw/{package}/`(영속)와 `runtime/apps/{package}/`(휘발성). `--new-session`/`reset` 은 반드시 두 root 를 함께 지우되 `data/processed/` 는 건드리지 않는다(불변식 7) — 메커니즘은 [ARCHITECTURE.md](./ARCHITECTURE.md) §4~§5 참조.
- 기본 동작은 같은 앱 패키지의 기존 세션을 이어서 저장하는 것이다. `run` 커맨드의 `--new-session` 은 해당 앱 한 개만 초기화한다. 더 넓은 범위 삭제는 `monkey-collect reset` 을 사용한다.
- App -> Server signal 이름 `P`, `S`, `X`, `E`, `N`, `F` 와 Server -> App 제어 메시지 (`{"type":"START","package":...}`, `{"type":"SESSION_END"}`) 계약을 깨지 마라. Android 측 `CollectorService.beginStandby` 루프가 이 계약에 의존한다.
- 세션 전환 핸드셰이크: `SESSION_END` → 클라이언트 `F` 회신 + 소켓 close → 클라이언트 한 번 자동 재접속.
  - Python 의 `CollectionServer.reset_for_new_session()` 은 큐/이벤트만 초기화하고 fresh 소켓은 보존해야 한다.
  - 여기서 소켓을 닫으면 클라이언트가 두 번째 재접속을 하지 않아 두 번째 세션부터 `wait_for_connection` 이 전부 타임아웃한다.
- signal 큐 소비(`CollectionServer.get_latest_signal`)는 **연속된 `xml` 프레임만** latest 로 collapse 한다.
  - `external_app`/`finish`/`no_change` 같은 제어 신호는 루프 시맨틱(외부앱 복구·세션 종료)을 나르므로 **절대 조용히 드롭하지 않는다** — drain 중 제어 신호를 만나면 멈추고 그걸 즉시 반환하며, 그 뒤에 큐에 쌓인 신호는 다음 호출을 위해 남긴다.
  - 예전 "모두 drain 하고 latest 만 반환" 은 `finish`/`external_app` 이 뒤이은 `xml` 에 덮여 유실됐다.
  - 이 시맨틱을 바꾸면 `tests/integration/test_server.py::TestSignalQueue` 를 함께 본다.
- first screen 보호, no-change retry, external app recovery 는 collector 의 핵심 동작이다. 관련 상수는 [`src/monkey_collector/pipeline/recovery.py`](./src/monkey_collector/pipeline/recovery.py) 에 있다.
- signal timeout escalation 은 6-place 로 노출된 `collection.signal_timeout_sec`(config.py builtin defaults + `CollectionConfig` + `_from_raw`(비양수 폴백) + env `MC_COLLECTION_SIGNAL_TIMEOUT_SEC` + run.yaml + cli `--signal-timeout`→`Collector(xml_timeout=...)`)로 대기 시간을, `recovery.MAX_SIGNAL_TIMEOUTS`(collection_loop 사용)로 relaunch 전 nudge 횟수를 정한다.
  - 6-place 중 하나만 바꾸면 canonical 주석/기본값 계약(`_BUILTIN_DEFAULTS`↔run.yaml)이 어긋나므로 함께 갱신하라.
- back-exit page 학습: `press_back` 이 런처(`screen_guard.is_launcher`)로 앱을 이탈시킨 페이지를 `CollectionState.back_exit_page_ids` 에 세션 내 기록하고(두 경로: `safe_press_back` 반환값 + `_handle_external_app` 진입부 detected_package 판정), `_back_would_exit(state)`(root ∪ 학습된 페이지)로 그 페이지에서 back 을
  억제(relaunch/tap)한다 — `select_action(is_root_screen=...)` 도 이 값을 받는다.
  - 복구 경량화로 `explorer.return_to_app` 은 이미 런처에 있으면 back 을 생략하고 곧장 relaunch 한다.
  - tap 이탈은 미마킹(D4).
  - 키보드 dismiss 는 예외적으로 마킹된다 — back-exit 페이지 위 키보드는 `adb.hide_keyboard()`(KEYCODE_ESCAPE, back-stack 을 pop 하지 않음)로 닫되 연속 `MAX_KEYBOARD_ESCAPES`(2)회 상한 후 back 폴백하고, back 으로 닫다 런처 이탈이 확인되면 그 페이지를 마킹한다(`_mark_if_back_exited`, D-B1).
- 무성과 반복 액션 가드(R2 volume-not-diversity 제거): ① `TransitionGraph.add`(transition_graph.py)는 한 액션 `(signature, action_type)` 을 기록하기 전에 src 의 다른 out-edge 에서 같은 pair 를 제거(빈 엣지 삭제, self-loop 관측 시에도)해 stale 엣지로 인한 Navigator 결정론적 livelock 을 끊는다.
  - ② D2 반복-액션 서킷브레이커(`collection_loop._process_xml_signal`, `CollectionState.action_repeat_counts`): 같은 `(page_key, action_type, element_index)` 가 신규 page 없이 `collector.max_action_repeats`(기본8)회 실행되면 다음 시도는 execute 대신 stuck-on-page 와 동일하게 back/relaunch 로 탈출(step 미증가, 카운터 리셋).
  - ③ D3 plateau 조기 종료(`CollectionState.steps_since_new_page`/`no_progress_stop`): 실제-액션 스텝이 `collector.max_steps_without_new_page`(기본98 = 아카이브 최대 productive gap 49 의 2 배, U3a) 동안 신규 page 0 이면 앱 포화로 clean-stop → finalize 가 `completed_at` 을 채워 예산을 다음 앱으로 넘긴다(재수집은 --force).
  - 두 임계값은 `collection.max_action_repeats`/`collection.max_steps_without_new_page` 4-place(config.py builtin defaults + `CollectionConfig` + `_from_raw` + env) + run.yaml + `Collector.__init__`(cli.py 전달)로 노출되며 0 이하면 가드 비활성.
  - 두 카운터는 신규 page 시 리셋(진행 시 용서, 5e2254e).
- external 복구가 타깃 앱을 재실행하면 `open_app` 액션을 events.jsonl 에 excursion 당 1회 기록한다(`collection_loop._record_open_app`, `DataWriter.log_open_app`) — 3중 격리 메커니즘은 불변식 8 및 [ARCHITECTURE.md](./ARCHITECTURE.md) §4 참조.
  - `return_to_app`/`recover` 의 `bool` 반환(launch 여부)·이 격리·`transition` 표식 중 하나라도 바꾸면 open_app 이 가짜 전이로 샐 수 있으니 함께 검토하라.
- `src/monkey_collector/__init__.py` 의 공개 export 를 바꾸면 패키지 사용 코드와 문서도 같이 갱신한다.
- 저장 포맷을 바꾸면 converter, page-map, regenerate, `rehydrate.py`, `config.py`(`data_dir`/`runtime_dir`), `paths.py`(`runtime/` sub-root), `pipeline/reset.py`, 테스트를 함께 갱신해야 한다.
- action 이벤트의 `page_key`/`observation_num` 이 실제 화면 파일 위치를 가리키는 **조인 키**다(`frame_index`/`step` 은 조인 키가 아님, `step` 증가 규칙은 불변식 6) — 스키마는 [ARCHITECTURE.md](./ARCHITECTURE.md) §5 참조.
  - step 증가 지점이나 page_key/observation_num 주입·조인을 바꾸면 converter·page-map·`build_graph_from_new_layout`·테스트를 함께 검토하라.

## 알려진 한계 (의도적으로 수용)

- **F2 poke 의 spurious `N` — 라벨 오귀속 (수용됨, 2026-07-14)**: 서버 pull(F2)의 CAPTURE poke 는 액션 후
  1.5s 에 발화한다. 그 시점까지 화면 변화가 아직 렌더링되지 않았으면 클라이언트는 XML 해시가 같아
  `N`(no_change)을 보내고, 서버는 이를 "액션 무효"로 확정 해석한다 — element 를 exclude 하고 retry 를 쏘며
  `state.last_action` 을 덮어쓴다. 그 결과 뒤늦게 도착한 원래 액션의 효과가 retry 액션에 **오귀속**된다.
  실측상 명백한 오판은 스텝의 1% 미만이다 — 전체 수치는
  [`../.claude/handoff/f2-server-pull-results.md`](../.claude/handoff/f2-server-pull-results.md) 의 "결함 6" 절을 보라.
  2-poke 합의(판정 시점 1.5s→3.0s, 관측 최대 렌더 지연 2.779s 초과)로 고칠 수 있으나 매 run 에 약
  +65~75s(900s 의 7~8%) 의 **상시 예산세**가 붙어 <1% 라벨 노이즈보다 비싸다고 판단, **고치지 않기로
  결정했다**. 이 결정을 뒤집으려면 예산세 실측부터 다시 하라. 이 코퍼스를 IWM 학습에 쓸 때 이 라벨
  노이즈가 있음을 인지할 것.

- **지도류 연속 캔버스의 page 지문 파편화 (S-9 — 진단 완료, 안전한 수정 없음. 2026-07-14)**:
  osmand 의 지도 화면이 팬/줌마다 다른 page 로 등록된다 — 한 run 에서 71 pages 중 **46개가 MapActivity**
  였다(지도 방문 수는 다른 arm 과 같은데도). **raw page count 는 지도류 앱에서 신뢰할 수 없다.**
  IWM Stage-1 이 diff-weighted loss 라 학습 신호를 직접 오염시킨다.

  **원인은 규명됐다 (`confirmed`, `single-case`)**: 오프라인 리플레이(라이브를 event-for-event 재현)로
  page mint 이벤트를 전수 분류한 결과 **element-blocked 가 armA 95.6% / armB 84.6%** 다 — 픽셀 게이트가
  아니라 **element-line 문서**가 원인이다. 지도의 축척(`100 ft`/`200 ft`)·거리·주소 텍스트와 상태별
  aria-label(`3D mode`↔`2D mode`)이 재방문마다 바뀌어 `|A△B|` 가 `element_diff_max=5` 를 크게 넘는다
  (median 13). **임계값 상향으로는 못 고친다**(max 41~45).

  **시도했던 후보 수정(canvas-gated text-blind 병합, 구 `screen_matching.canvas_merge`)은 2026-07-22 에
  코드베이스에서 제거됐다** — 기본값이 계속 OFF 였고 안전해지지 않았다. 아래는 그 실패의 실측 기록이다.
  파편화는 절반으로 줄었다(46→27, 26→14, 오프라인 리플레이). 그러나 **지도가 아닌 화면까지 병합했다**:
  osmand 의 **내비게이션 드로어**(메뉴 15개 — 앱의 메인 허브)와 **턴바이턴 추적 모드**가 지도 page 로
  들어갔다. 이유 둘 다 실측됐다 — (1) 드로어·내비 오버레이는 그 밑 지도 surface leaf 를 a11y 트리에
  **그대로 남기므로** canvas 판정이 함께 발화하고, a11y 가 그 화면들을 **`MapActivity` 로 보고**하기 때문에
  activity 가드로도 못 막는다. (2) 드로어 메뉴 항목은 **aria-label 없는 순수 TextView** 라 텍스트를 가리면
  전부 동일한 `<p></p>` 로 뭉개지고, **집합 기반** element 기준이 카디널리티를 버린다(15개와 3개가
  같아 보인다). **카디널리티 복원(multiset diff)은 그 화면들을 분리하지만, 수정이 수행한 병합 23건 중
  20건을 함께 깨뜨려 수정 자체를 무효화한다** — 임계값을 다시 맞추는 것은 사후 끼워맞추기다.
  → **S-9 는 미해결로 남아 있고, 현재 코드에는 어떤 완화 경로도 없다.** 다음 시도는 이 두 실측을
  출발점으로 삼아라(다시 재보지 마라). 기존 아카이브는 영향이 없다 — 기본값이 계속 OFF 였으므로
  이 경로로 병합된 page 가 애초에 없고, 남아 있는 `page.json` 의 `is_canvas`/`element_lines_blind` 키는
  로드 시 무시된다(마이그레이션 불필요).

- **BM25 병합의 패키지 교차 오염 (수정됨 — `screen_matching.package_guard`, 기본 ON. 2026-07-14)**:
  BM25 병합 경로는 화면이 **어느 앱의 것인지 검사하지 않았다**(구조 prefilter 만 activity 로 키를 잡는다).
  그래서 런처 홈 프레임이 일반적인 element-line 몇 개를 공유하는 것만으로 앱의 page 로 병합될 수 있었다 —
  **수정 전 라이브 코퍼스에서 실측됨**(osmand armB 2건, broccoli 1건). 가드는 **후보 page 가 현재 화면과
  같은 package 에서 mint 됐을 것**을 추가로 요구한다(어느 한쪽 package 를 모르면 abstain — fail-open).
  ⚠️ **이 가드는 소급 적용되지 않는다**: 기존 아카이브의 오염된 page_graph 는 **재수집해야 사라진다.**

## 효과 측정 프로토콜 (수집기 변경의 효과를 판정할 때)

수집기 변경(가드·탐색정책·임계값)의 효과를 수치로 판정하려면 **반드시** 아래를 따른다. 이 프로토콜 없이 뽑은 비교는 confound 로 오염돼 인과 해석이 불가하다 — 과거에 실제로 "다양성 +78%" 를 fix 효과로 오귀속했다가 전면 정정한 사례가 있다.

- **앱 상태 리셋 (매 run 마다)**: 측정 run 시작 전 대상 앱을 **동일한 clean state 로 되돌린다**. 이전 run 이 만든 변경(생성된 레시피·바뀐 설정·캐시된 뷰)이 다음 run 으로 흘러 confound 가 된다. 헬퍼: [`../.claude/handoff/reset_app.sh`](../.claude/handoff/reset_app.sh).
  - **user app**(musicplayer/broccoli/osmand): `adb uninstall` **후** `install -r -g catalog/apks/<pkg>.apk`. `install -r` 단독은 앱 데이터를 보존하므로 리셋이 되지 않는다 — uninstall 이 필수다.
  - **system app**(`com.google.android.calendar` = `/product/app/CalendarGooglePrebuilt`): uninstall 불가 → `pm clear` 가 동등한 데이터 리셋이다.
  - **seed 코퍼스는 리셋 후에도 동일해야 한다**(검증됨): musicplayer 의 데모 mp3 3곡은 공유 저장소(`/sdcard/Music`)라 uninstall 에 생존하고, calendar 의 seed 이벤트 25건은 **별도 priv-app** 인 `com.android.providers.calendar` DB 에 있어 앱 `pm clear` 에 생존한다.
    - broccoli 레시피는 앱 자체 DB 라 uninstall 시 소멸 → 재시드 필요.
    - `reset_app.sh` 는 리셋 전후 seed 개수를 비교해 달라지면 실패한다.
- **arm 짝맞춤**: baseline 과 treatment 를 **같은 프로토콜로 각각 수집**한다. 과거 데이터(리셋 없이 수집된 iter3~5 아카이브)를 새 프로토콜 수치와 직접 비교하지 마라 — apples-to-oranges 다. arm 사이에 달라지는 것은 **수집기 코드 하나뿐**이어야 한다(같은 앱 리셋·같은 duration·같은 디바이스 `emulator-5556`).
- **treatment 오염 금지**: 측정 도중 working tree 의 수집기 소스를 수정하지 마라.
  - 앱마다 새 프로세스가 뜨므로 중간에 코드가 바뀌면 뒤 앱이 다른 코드로 돈다.
  - baseline arm 은 `git worktree` 로 격리해 돌린다.
  - `env | grep MC_` 가 비어 있어야 한다(env override 가 treatment 를 덮는다).
  - `data_dir`/`runtime_dir` 는 CWD-상대(`storage.py`)라 worktree arm 과 메인 arm 은 서로 다른 트리에 쓴다 — cross-arm 덮어쓰기는 없다.
- **디바이스**: `emulator-5556` 고정 (`adb.py` 의 `REQUIRED_AVD_NAME` 기본값 `'Pixel6-2'`, env `MC_AVD` 로 오버라이드 가능하나 측정 arm 간 AVD 혼용 금지). 다른 AVD 를 쓰면 그 자체가 새 confound 다.

### ⚠️ 함정 1 — provider-backed 앱은 앱 리셋으로 오염이 안 지워진다 (iter6 실측)

**seed 가 리셋에 생존하는 바로 그 성질이, 수집기가 만든 오염도 생존시킨다.** calendar 이벤트는 별도 priv-app `com.android.providers.calendar` DB 에 살아서 `pm clear com.google.android.calendar` 가 닿지 않는다 — seed 25건이 살아남는 이유이자, **수집기가 탐색 중 만든 이벤트도 살아남는 이유**다.

iter6 실측: calendar armA 의 900s 수집이 이벤트를 58건 생성(**25 → 83**) → 다음 armB 가 **3.3배 데이터**에서 출발 → **calendar arm 쌍 전체가 비교 불가**가 됐다.

`reset_app.sh` 는 이걸 **못 잡는다**: run *내부*의 `before == after` 만 검사하고 **canonical baseline 으로의 복원**은 검사하지 않는다. 두 run 모두 자체 검증을 통과했다(25→25, 83→83).

→ provider-backed 앱을 arm 에 넣으려면 **매 run 전 provider DB 를 canonical seed 로 복원**하고, 검사를 `after == canonical` 로 바꿔야 한다.
**`pm clear com.android.providers.calendar` 는 쓰지 마라** — 계정 sync 상태까지 날린다.
musicplayer 는 안전하다(수집기가 mp3 를 만들 수 없다) — **그래서 musicplayer 가 load-bearing clean isolator 다.**

### ⚠️ 함정 2 — 노이즈 바닥이 크다. arm 당 n=1 로는 판정할 수 없다 (iter6 실측)

musicplayer 를 900s·리셋 프로토콜·동일 디바이스에서 **똑같은 코드로 두 번** 돌린 결과: **15p/33e/113steps vs 12p/23e/130steps** (pages −3, edges −10, steps +17).

같은 실험에서 측정한 **fix 효과**(pre-fix → fix)는 pages +1, edges +0, steps −32.

→ **동일 코드의 run 간 변동이 측정하려는 효과보다 크다.** "표본이 적으니 조심하라"가 아니라 실측된 노이즈 추정치다. 이 크기의 효과를 판정하려면 **arm 당 최소 3 run + 분산 병기**가 필요하다. **단일 run 델타를 효과로 주장하지 마라.**

## 문서 동기화 원칙

- README 는 실제 운영 절차와 CLI 예시 중심으로 유지한다.
- ARCHITECTURE 는 현재 파일 구조와 TCP / storage 계약 중심으로 유지한다.
- CLI, 저장 구조, Android 서비스 흐름이 바뀌면 README, ARCHITECTURE, AGENTS 를 함께 수정한다.
