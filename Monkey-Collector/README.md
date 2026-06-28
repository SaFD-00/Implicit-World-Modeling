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
- App 이 screen stabilization 과 visual change 판정을 담당하고, Server 가 SmartExplorer 로 action 을 선택하고 raw session 을 저장한다.
- 세션 디렉토리는 `data/raw/{package}/` 형식이다. `metadata.json` 의 `completed_at` 이 채워진 앱은 다음 `run` 에서 **자동으로 건너뛴다** (중단된 세션은 resume). `--force` 로 완료된 앱도 다시 수집 가능.
- LLM 사용처는 두 곳이며 모두 하나의 **OpenRouter Chat Completions** 공용 클라이언트(기본 모델 `qwen/qwen3.7-plus`)를 쓴다: (1) **input text 생성** (`--input-mode api`, 없으면 hardcoded `random`), (2) **화면 요소 의미 그룹핑** (`--screen-grouping on`, 같은 기능 요소를 묶어 한 화면을 의미 단위로 "나눈다" — 결과는 `xml/{step}_groups.json` annotation 으로 저장). `OPENROUTER_API_KEY` 가 없으면 자동 비활성화된다(입력은 random fallback, 그룹핑은 skip).

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

`.env` 또는 환경변수에 `OPENROUTER_API_KEY` 를 넣으면 `--input-mode api` (입력 텍스트 생성)와 `--screen-grouping on` (화면 의미 그룹핑)에서 LLM 을 사용한다. 모델과 엔드포인트는 `OPENROUTER_MODEL` (기본 `qwen/qwen3.7-plus`), `OPENROUTER_BASE_URL` (기본 `https://openrouter.ai/api/v1`) 로 덮어쓸 수 있다.

추가 전제:

- Python 3.10+
- ADB 가 PATH 에 있거나 `ANDROID_HOME` 이 설정되어 있어야 한다
- **`MobileGPT-V2-2` 이름의 AVD 가 실행 중이어야 한다** (`emulator -avd MobileGPT-V2-2`). `AdbClient` 가 `adb devices` 에서 해당 AVD 의 emulator serial 을 자동 탐색해 모든 ADB 명령에 `-s <serial>` 로 고정한다. 다른 AVD 나 실기기가 함께 붙어 있어도 무방하며, AVD 가 없으면 `AdbClient()` 생성 시 즉시 `RuntimeError` 로 실패한다.

### Android app

```bash
cd app
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

설치 후 디바이스에서 AccessibilityService 를 활성화해야 한다.

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
monkey-collect run --apps all --steps 100

# 원하는 앱만 지정 (완료 여부는 동일하게 체크)
monkey-collect run --apps com.google.android.deskclock com.google.android.calculator --steps 50

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

## CLI

### `run`

서버 드리븐 수집. `catalog/apps.csv` 의 `installed=true` 앱 전부 또는 지정한 패키지 목록을 순차 수집한다.

```bash
monkey-collect run --apps all --steps 100
monkey-collect run --apps com.google.android.deskclock --steps 50
```

주요 옵션:

- `--apps` (필수): `all` 이면 `catalog/apps.csv` 의 `installed=true` 전부. 아니면 하나 이상의 package_id.
- `--steps`: 세션당 최대 step 수 (기본 100)
- `--seed`: explorer 랜덤 시드 (기본 42)
- `--delay`: action 사이 대기 시간(ms, 기본 1500)
- `--port`: TCP server port (기본 12345)
- `--output`: raw session 저장 루트 (기본 `data/raw`)
- `--input-mode`: 입력 텍스트 생성 모드 `api` (LLM) 또는 `random` (hardcoded). 기본 `api`
- `--screen-grouping`: 화면 요소 의미 그룹핑 `on` / `off`. 기본 `on` (LLM 클라이언트 없으면 자동 비활성). `on` 이면 화면마다 같은 기능 요소 그룹을 `xml/{step}_groups.json` 으로 저장
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

`xml/` 아래에는 raw XML 과 함께 다음 파생 파일이 저장된다.

- `{step}_parsed.xml`
- `{step}_hierarchy.xml`
- `{step}_encoded.xml`
- `{step}_pretty.xml`
- `{step}_groups.json` — `--screen-grouping on` 일 때만. LLM 이 묶은 같은 기능 요소 그룹(`{step}_encoded.xml` 의 `index` 기준) annotation. LLM 호출 결과라 `regenerate` 로는 재생성되지 않는다.

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
│   ├── llm/           (client, screen_grouper)   # 공용 OpenRouter 클라이언트 + 화면 그룹핑
│   ├── pipeline/      (collector, collection_loop, explorer, recovery,
│   │                   session_manager, app_catalog, installed_sync,
│   │                   reset, text_generator)
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
- [`src/monkey_collector/pipeline/explorer.py`](./src/monkey_collector/pipeline/explorer.py): action selection
- [`src/monkey_collector/storage.py`](./src/monkey_collector/storage.py): 세션 포맷
- [`app/app/src/main/java/com/monkey/collector`](./app/app/src/main/java/com/monkey/collector): Android 앱

구조 설명은 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md) 를 본다.
