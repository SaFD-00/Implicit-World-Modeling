# Monkey-Collector

Android GUI world model 학습용 데이터를 수집하는 App + Server 파이프라인이다. Android AccessibilityService 앱이 화면 전환을 감지해 screenshot 과 XML 을 보내고, Python 서버가 다음 action 을 선택해 ADB 로 실행한다.

## 개요

구성 요소:

- Android app: [`app/`](./app)
- Python server package: [`src/monkey_collector/`](./src/monkey_collector)
- 앱 카탈로그 / APK 보관: [`catalog/`](./catalog) (`apps.csv`, `apks/`, `activities.json`, `download_apks.py`, `extract_activities.py`)
- CLI entrypoint: `monkey-collect`
- 테스트: [`tests/`](./tests)

현재 코드 기준 핵심 동작:

- 서버 드리븐 파이프라인: Python 서버가 TCP 로 `START {package}` 메시지를 보내면 Android 앱이 해당 앱을 수집한다. 사용자가 앱 측에서 버튼을 누르는 단계는 없다.
- 수집 대상은 `catalog/apps.csv` 의 `installed=true` 로 표시된 앱들 중에서 고른다. `sync-installed` 서브커맨드가 `adb pm list packages` 기반으로 이 컬럼을 자동 갱신한다.
- App 이 screen stabilization 과 visual change 판정을 담당하고, Server 가 `LLMGuidedExplorer` (LLM-Explorer 포팅: 미탐색 우선 + same-function 압축 + 미탐색 화면 navigation) 로 action 을 선택하고 raw session 을 저장한다.
- 세션 디렉토리는 `data/raw/{package}/` 형식이다. `metadata.json` 의 `completed_at` 이 채워진 앱은 다음 `run` 에서 **자동으로 건너뛴다** (중단된 세션은 resume). `--force` 로 완료된 앱도 다시 수집 가능.
- LLM 사용처는 두 곳이며 모두 하나의 **OpenRouter Chat Completions** 공용 클라이언트(기본 모델 `qwen/qwen3.7-plus`)를 쓴다: (1) **input text 생성** (`--input-mode api`, 없으면 hardcoded `random`) — 프롬프트에 현재 탐색 중인 앱 설명(`catalog/apps.csv` 의 `app_name (category/sub_category) — notes`, csv 미등록 앱은 package_id 로 폴백)을 함께 넣어 앱 도메인에 맞는 입력값(쇼핑앱 검색창→상품 검색어, 메모앱→메모 내용 등)을 생성한다, (2) **element 추출 + element-set screen matching** (`--element-extraction on`, MobileGPT-V2 Node-Clustering 포팅): 화면마다 element 를 한 번의 호출로 추출(각 element 에 `element_index`(같은 기능 family)와 `key_element_index`(대표 anchor)를 함께)하고, 그 element-set 으로 page 를 식별한다 — 이 page 식별이 page_graph 와 탐색 abstract page 를 모두 결정하고 결과는 `xml/{step}_elements.json` 으로 저장된다. `OPENROUTER_API_KEY` 가 없으면 자동 비활성화된다(입력은 random fallback, page 식별은 구조 지문으로 degrade).

## 설치

### Python server

Monkey-Collector 는 uv 가 관리하는 `.venv` 를 사용한다 (`uv sync` 가 `.python-version` 의 3.10 을 자동 설치). 형제 프로젝트 `../GUI-Model/` 도 동일한 uv 흐름이다.

```bash
# 1) uv 설치 (한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Monkey-Collector 설치
cd /path/to/Monkey-Collector
uv sync                    # runtime 만
uv sync --extra dev        # 개발 도구 (pytest, ruff, mypy) 포함

# 3) 활성화
source .venv/bin/activate
# (또는 명령마다 'uv run <cmd>' 형태로 호출)
```

`uv.lock` 은 함께 커밋한다. CI / 다른 머신에서는 `uv sync --frozen` 으로 동일 환경을 복구한다.

주요 런타임 의존성: `loguru`, `Pillow`, `openai` (OpenAI-호환 SDK 로 OpenRouter 호출), `python-dotenv`, `pyvis`, `requests`, `gplaydl`. 정확한 버전은 [`pyproject.toml`](./pyproject.toml) 의 `[project] dependencies` 섹션을 참조한다.

선택 사항:

```bash
cp .env.example .env
```

`.env` 또는 환경변수에 `OPENROUTER_API_KEY` 를 넣으면 `--input-mode api` (입력 텍스트 생성)와 `--element-extraction on` (element 추출 + element-set screen matching)에서 LLM 을 사용한다. 모델과 엔드포인트는 `OPENROUTER_MODEL` (기본 `qwen/qwen3.7-plus`), `OPENROUTER_BASE_URL` (기본 `https://openrouter.ai/api/v1`) 로 덮어쓸 수 있다.

추가 전제:

- Python 3.10+
- ADB 가 PATH 에 있거나 `ANDROID_HOME` 이 설정되어 있어야 한다
- **`Pixel6-2` 이름의 AVD 가 실행 중이어야 한다** (`emulator -avd Pixel6-2`). `AdbClient` 가 `adb devices` 에서 해당 AVD 의 emulator serial 을 자동 탐색해 모든 ADB 명령에 `-s <serial>` 로 고정한다. 다른 AVD 나 실기기가 함께 붙어 있어도 무방하며, AVD 가 없으면 `AdbClient()` 생성 시 즉시 `RuntimeError` 로 실패한다.

### Android app

gradle 프로젝트 루트는 `app/`, app 모듈은 `app/app/`, 산출 APK 는 **중첩 경로 `app/app/build/outputs/...`** 다. **AGP 8.2 빌드 실행에는 JDK 17 이 필요**하다(컴파일 bytecode 타깃은 jvmTarget=1.8).

```bash
cd app
[ -f local.properties ] || printf 'sdk.dir=%s\n' "$HOME/Library/Android/sdk" > local.properties   # gitignore — 없으면 생성
JAVA_HOME="$(/usr/libexec/java_home -v 17)" ./gradlew :app:assembleDebug
adb install -r -g app/app/build/outputs/apk/debug/app-debug.apk
```

설치 후 디바이스에서 AccessibilityService 를 활성화하고(§ MainActivity 의 "Open Accessibility Settings"), "Save & Ready" 로 MediaProjection 동의를 한다. **재설치할 때마다 MediaProjection 동의를 다시 받아야 한다**(토큰 단발성). 단 캡처 본체는 AccessibilityService.takeScreenshot 라 미동의 시에도 크래시 없이 동작한다.

> **전체 환경 셋업은 `/setup-collector` 스킬이 자동화**한다 — AVD 부팅·APK 다운로드/설치·client 빌드(JDK17)/설치·접근성·prefs·MediaProjection·Google 로그인·더미데이터 시드·검증을 멱등(초기화 시 1회)으로 수행한다.

## 빠른 시작

### 1. Android 앱 준비

1. Monkey-Collector 앱에서 server IP / port 입력
2. Save & Ready → Accessibility 권한 + MediaProjection 권한 허용
3. 이후 앱은 백그라운드에서 서버 연결을 유지한다. 사용자가 수집 시작 버튼을 누를 필요가 없다.

### 2. 디바이스 설치 앱을 `catalog/apps.csv` 에 반영

```bash
monkey-collect sync-installed
```

`adb pm list packages` 결과를 읽어 `catalog/apps.csv` 의 `installed` 컬럼(`true`/`false`)을 in-place 로 갱신한다.

### 3. APK manifest 에서 activity catalog 갱신

```bash
python -m catalog.extract_activities      # .venv 활성화 상태에서
# 또는 비활성 상태에서: uv run python -m catalog.extract_activities
```

`catalog/apks/*.apk` 의 `AndroidManifest.xml` 을 androguard 로 파싱해 `catalog/activities.json` 을 만든다. 이 파일이 `activity_coverage.csv` 의 ground truth (분모 + 분자 후보 집합) 다 — 누락되면 자동으로 `adb dumpsys package` 로 폴백한다 (legacy 동작).

이미 한 번 생성한 뒤에는 `catalog/apks/` 에 새 APK 를 추가하거나 기존 APK 를 교체할 때만 다시 실행하면 된다.

### 4. 수집 실행

```bash
# catalog/apps.csv 의 installed=true 인 앱 전부 순차 수집 (이미 완료된 앱은 자동 skip)
monkey-collect run --apps all --steps 1500

# 원하는 앱만 지정 (완료 여부는 동일하게 체크)
monkey-collect run --apps com.google.android.deskclock com.google.android.calculator --steps 1500

# 완료된 앱도 다시 수집
monkey-collect run --apps all --force

# 특정 앱의 기존 세션을 폐기하고 새로 시작
monkey-collect run --apps com.google.android.deskclock --new-session

# 입력 텍스트를 hardcoded 로 (API 비용 없음)
monkey-collect run --apps all --input-mode random
```

동작:

- 서버가 각 앱마다 `adb shell am start` 로 앱을 실행하고, TCP 로 `{"type": "START", "package": "com.X"}` 를 보낸다. Android 앱은 standby 연결을 유지하다가 START 를 받아 자동으로 수집을 시작한다.
- 한 세션이 끝나면 서버가 `SESSION_END` 를 보내 클라이언트를 정리하고, Android 는 `F` 회신 후 새 소켓으로 자동 재접속한다. 서버는 이 fresh 소켓을 그대로 유지하고 다음 앱으로 `START` 를 송신한다. 이때 이전 세션의 `F`(finish) 회신이 신호 큐에 남을 수 있어, `run_collection_loop` 는 시작 시 큐를 비운다 — 비우지 않으면 새 세션이 stale finish 를 읽고 0-step 으로 끝나며 큐의 다음 앱들까지 연쇄로 0-step 된다.
- 큐 구성 시 `data/raw/{pkg}/metadata.json` 의 `completed_at` 이 채워진 앱은 **완료로 판정되어 스킵**. `--force` 로 우회하거나, 중단된(미완료) 세션은 `completed_at` 이 `null` 이라 자동으로 resume 된다.
- **재초기화(reinit)**: signal timeout 5연속 또는 external app 10회 도달 시 세션을 종료하는 대신, target app 을 force-stop + relaunch 하고 카운터를 초기화해 탐색을 재개한다. 재초기화는 각각 최대 3회까지 허용되며, 초과 시 세션이 종료된다.
- **open_app 기록**: external app 복구가 타깃 앱을 실제로 재실행하면 그 재실행을 `open_app` 액션으로 `events.jsonl` 에 **이탈(excursion) 당 1회** 기록한다(`{"action_type":"open_app", "package", "app_name", "transition":false, "trigger":"external_recovery", "from_package"}`). 이 open_app 은 복구 동작이지 의도된 화면 전이가 아니므로 page graph·탐색 전이·world-modeling 변환 어디에도 navigation 으로 쓰이지 않는다(`transition:false`).

## 설정 (`config/run.yaml`)

수집 파라미터는 `config/run.yaml` 로 관리한다. 값은 **4단계**로 해석되며 뒤쪽이 앞쪽을 덮어쓴다:

1. 빌트인 기본값 (canonical strategy = `BFS`)
2. `config/run.yaml` (다른 경로는 `MC_CONFIG_PATH` 또는 `--config` 로 지정)
3. `MC_*` 환경변수
4. CLI 플래그

즉 **빌트인 → `run.yaml` → `MC_*` env → CLI** 순으로 우선순위가 높아진다. YAML 이 커버하는 argparse 파라미터는 CLI 기본값이 `None`(미지정 sentinel)이라, 플래그를 주지 않으면 config 값이 그대로 쓰인다. `--apps` / `--force` / `--new-session` 같은 운영 플래그는 CLI 전용이다.

`config/run.yaml` 섹션:

- `exploration.strategy`: `DFS` | `BFS` | `GREEDY` (canonical 기본 `BFS`)
- `collection.{max_steps, seed, action_delay_ms, port, output_dir}`
- `llm.{input_mode, element_extraction}`
- `screen_matching.{cluster_merge_tolerance, max_expand_iters}`

대응 환경변수: `MC_EXPLORATION_STRATEGY`, `MC_COLLECTION_MAX_STEPS`, `MC_COLLECTION_SEED`, `MC_COLLECTION_ACTION_DELAY_MS`, `MC_COLLECTION_PORT`, `MC_COLLECTION_OUTPUT_DIR`, `MC_LLM_INPUT_MODE`, `MC_LLM_ELEMENT_EXTRACTION`, `MC_SCREEN_MATCHING_CLUSTER_MERGE_TOLERANCE`, `MC_SCREEN_MATCHING_MAX_EXPAND_ITERS`, 그리고 대체 yaml 을 가리키는 `MC_CONFIG_PATH`.

### 탐색 전략 (DFS / BFS / GREEDY)

세 전략 모두 미탐색 target 을 고른 뒤 shortest path 로 route 하며, target 선택 기준만 다르다:

- `BFS`: 세션 root page 기준 BFS depth 가 가장 얕은 target (shallow first) — **canonical 기본값**
- `DFS`: BFS depth 가 가장 깊은 target (deep first)
- `GREEDY`: 현재 화면에서 navigation path 가 가장 짧은 target (기존 동작)

depth 는 이번 세션에서 처음 관측한 root page 를 기준으로 계산한다. 전략은 다음 중 하나로 지정하며 우선순위는 위 4단계와 동일하다:

- `config/run.yaml` 의 `exploration.strategy`
- `--strategy {DFS,BFS,GREEDY}` CLI 플래그
- `MC_EXPLORATION_STRATEGY` 환경변수

```bash
monkey-collect run --apps all --strategy DFS
monkey-collect run --apps all --config config/run.yaml
```

잘못된 값은 경고를 출력한 뒤 `GREEDY` 로 폴백한다.

## CLI

### `run`

서버 드리븐 수집. `catalog/apps.csv` 의 `installed=true` 앱 전부 또는 지정한 패키지 목록을 순차 수집한다.

```bash
monkey-collect run --apps all --steps 1500
monkey-collect run --apps com.google.android.deskclock --steps 1500
```

주요 옵션:

- `--apps` (필수): `all` 이면 `catalog/apps.csv` 의 `installed=true` 전부. 아니면 하나 이상의 package_id.
- `--strategy`: 탐색 전략 `DFS` / `BFS` / `GREEDY` 선택 (canonical 기본 `BFS`). 미지정 시 `config/run.yaml` 의 `exploration.strategy` 를 따른다 (의미는 「설정」 섹션 참조).
- `--config`: 사용할 config YAML 경로 (기본 `config/run.yaml`).
- `--steps`: 세션당 최대 step 수 (기본 1500)
- `--seed`: explorer 랜덤 시드 (기본 42)
- `--delay`: action 사이 대기 시간(ms, 기본 1500)
- `--port`: TCP server port (기본 12345)
- `--output`: raw session 저장 루트 (기본 `data/raw`)
- `--input-mode`: 입력 텍스트 생성 모드 `api` (LLM, 현재 앱 설명을 프롬프트에 포함) 또는 `random` (hardcoded). 기본 `api`
- `--element-extraction`: element 추출 + element-set screen matching `on` / `off`. 기본 `on` (LLM 클라이언트 없으면 자동 비활성, 구조 지문 식별로 degrade). `on` 이면 화면마다 element(family + anchor)를 추출해 page 식별에 쓰고 `xml/{step}_elements.json` 으로 저장. (`--screen-grouping` 은 deprecated alias)
- `--cluster-merge-tolerance`: OVERLAP element-set merge 의 two-sided tolerance band (기본 0.2)
- `--max-expand-iters`: 화면당 expand(남은 UI 재추출) 최대 반복 (기본 3)
- `--new-session`: 해당 패키지의 기존 세션을 삭제하고 새로 시작
- `--force`: `completed_at` 이 채워진 앱도 다시 수집 (기본은 완료 앱 skip)

### `sync-installed`

디바이스에서 `pm list packages` 를 조회해 `catalog/apps.csv` 의 `installed` 컬럼을 갱신한다. `run --apps all` 이전에 한 번 실행해두면 대상 큐가 최신 상태에서 구성된다.

```bash
monkey-collect sync-installed
monkey-collect sync-installed --apps-csv custom_apps.csv
```

주요 옵션:

- `--apps-csv`: 갱신할 apps.csv 경로 (기본 `catalog/apps.csv`)

### `reset`

수집된 세션 데이터를 범위 단위로 삭제한다. 특정 패키지만 재수집하거나 전체 결과를 날리고 싶을 때 사용한다.

```bash
# 전체 삭제
monkey-collect reset --all --yes

# 특정 패키지만
monkey-collect reset --apps com.example.foo,com.example.bar --yes

# 미리 보기
monkey-collect reset --apps com.example.foo --dry-run
```

주요 옵션:

- `--all`: `--output` 전체를 삭제 (다른 스코프 플래그와 상호 배타)
- `--apps`: 삭제할 package_id 리스트
- `--output`: 데이터 루트 (기본 `data/raw`)
- `--dry-run`: 삭제 없이 대상 경로만 출력
- `--yes`: 확인 프롬프트 스킵

### `convert`

```bash
monkey-collect convert \
  --session data/raw/com.example.app \
  --output data/processed/gui-model_stage1.jsonl \
  --images-dir data/processed/images \
  --label 1
```

주요 옵션:

- `--label`: 이미지 파일명에 사용되는 세션 라벨 정수 (기본 `1`).

### `convert-all`

```bash
monkey-collect convert-all \
  --raw-dir data/raw \
  --output data/processed/gui-model_stage1.jsonl \
  --images-dir data/processed/images
```

### `page-map`

```bash
monkey-collect page-map --session data/raw/com.example.app
monkey-collect page-map --session data/raw/com.example.app --threshold 0.9 --no-open
```

### `page-map-all`

```bash
monkey-collect page-map-all --raw-dir data/raw --no-open
```

### `regenerate`

```bash
monkey-collect regenerate --raw-dir data/raw
```

raw XML 을 기준으로 `_parsed.xml`, `_hierarchy.xml`, `_encoded.xml`, `_pretty.xml` 를 다시 만든다.

## 저장 구조

기본 raw session 구조:

```
data/raw/{package}/
├── metadata.json
├── screenshots/
├── xml/
├── events.jsonl
├── activity_coverage.csv     # ground truth: catalog/activities.json (fallback: dumpsys)
├── cost.csv
├── page_graph.json
└── page_graph.html
```

`events.jsonl` 은 실행된 action 당 한 줄이며, 각 action 은 `frame_index`(그 action 이 결정된 before 프레임의 파일 인덱스)로 저장된 프레임에 정렬된다 — world-modeling converter 와 offline page-graph 재빌드의 조인 키다. `step` 은 루프 카운터 라벨로 정상 action 에서만 증가한다. 자세한 스키마는 [ARCHITECTURE.md](./ARCHITECTURE.md) 참조.

`xml/` 아래에는 raw XML 과 함께 다음 파생 파일이 저장된다.

- `{step}_parsed.xml`
- `{step}_hierarchy.xml`
- `{step}_encoded.xml`
- `{step}_pretty.xml`
- `{step}_elements.json` — `--element-extraction on` 일 때만. `ScreenMatcher` 의 element-set match: `page_key`/`match_type`/`is_new_page` + element family(`name`/`description`/`parameters`/`element_index`/`key_element_index`, index 는 `{step}_encoded.xml` 기준). 새 페이지면 그 화면에서 추출된 family 가, merge·재방문(`is_new_page=false`)이면 매칭된 page 의 저장 element 를 현재 화면 anchor 에 재매핑해 채운다(anchor 기준이라 `element_index` 는 anchor index 로 근사, 현재 화면에 없는 element 는 제외). interactable 없는 로딩/스플래시 화면은 page 로 등록하지 않으므로(첫 유효 화면이 `page_0`) 그 step 에는 `_elements.json` 이 생성되지 않는다. LLM 호출 결과라 `regenerate` 로는 재생성되지 않는다.

## 프로젝트 구조

```
Monkey-Collector/
├── README.md, ARCHITECTURE.md, AGENTS.md
├── pyproject.toml, .env.example
│
├── app/                              # Android Gradle 프로젝트
│
├── src/monkey_collector/             # Python 패키지
│   ├── cli.py
│   ├── adb.py, tcp_server.py, storage.py, catalog_activities.py
│   ├── domain/        (actions, page_graph, activity_coverage, cost_tracker)
│   ├── llm/           (client, element_extractor, prompts)  # 공용 OpenRouter 클라이언트 + element 추출
│   ├── pipeline/      (collector, collection_loop, recovery,
│   │                   session_manager, app_catalog, installed_sync,
│   │                   reset, text_generator)
│   │     ├── screen_matching/  (screen_matcher, ui_attributes, set_classifier, page_knowledge)
│   │     └── exploration/      (explorer, memory, navigator, transition_graph, state, ...)
│   ├── export/        (converter, graph_visualizer)
│   └── xml/           (ui_tree, structured_parser, parser_base)
│
├── catalog/                          # 앱 카탈로그
│   ├── apps.csv
│   ├── apks/                         # *.apk + MISSING.md
│   ├── activities.json               # androguard 추출 manifest activity (coverage ground truth)
│   ├── download_apks.py
│   └── extract_activities.py         # apks/*.apk → activities.json
│
├── tests/
│   ├── conftest.py, fixtures/
│   ├── unit/                         # 13 files
│   └── integration/                  # 6 files
│
└── (gitignored) data/raw/, logs/, *.egg-info/
```

## 코드 읽기 시작점

- [`src/monkey_collector/cli.py`](./src/monkey_collector/cli.py): 실제 CLI
- [`src/monkey_collector/pipeline/collector.py`](./src/monkey_collector/pipeline/collector.py): 수집 진입점
- [`src/monkey_collector/pipeline/exploration/`](./src/monkey_collector/pipeline/exploration): action selection (LLM-guided 탐색 엔진)
- [`src/monkey_collector/storage.py`](./src/monkey_collector/storage.py): 세션 포맷
- [`app/app/src/main/java/com/monkey/collector`](./app/app/src/main/java/com/monkey/collector): Android 앱

구조 설명은 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md) 를 본다.
