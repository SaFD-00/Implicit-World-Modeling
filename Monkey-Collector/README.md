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
- App 이 screen stabilization 과 visual change 판정을 담당하고, Server 가 `CoverageGuidedExplorer` (LLM-Explorer 알고리즘 포팅, 런타임에 LLM 미호출: 미탐색 우선 + 미탐색 화면 navigation) 로 action 을 선택하고 raw session 을 저장한다.
- 세션은 두 root 로 나뉜다 — `data/{package}/`(영속: pages/observations, page_graph), `runtime/{package}/`(휘발성: metadata, events, cost/coverage).
  - `metadata.json` 의 `completed_at` 이 채워진 앱은 다음 `run` 에서 **자동으로 건너뛴다** (중단된 세션은 resume — 재개 시 기존 page 지식도 함께 복원).
  - `--force` 로 완료된 앱도 다시 수집 가능.
- LLM 은 런타임에서 **입력 텍스트 생성 단 하나**에만 쓰인다 — page 식별은 항상 LLM-free 다.
  - **OpenRouter Chat Completions** 공용 클라이언트(기본 모델 `qwen/qwen3.7-plus`)로 **input text 생성** (`--input-mode api`, 없으면 hardcoded `random`) 을 수행한다 — 프롬프트에 현재 탐색 중인 앱 설명(`catalog/apps.csv` 의 `app_name (category/sub_category) — notes`, csv 미등록 앱은 package_id 로 폴백)을 함께
    넣어 앱 도메인에 맞는 입력값(쇼핑앱 검색창→상품 검색어, 메모앱→메모 내용 등)을 생성한다.
  - page 식별은 **BM25 unique-page matching**(Mobile3M 메커니즘, LLM-free): encoded XML 을 element-line 문서로 직렬화 → BM25 로 후보 page 검색 → element diff(`|A△B|<element_diff_max` 또는 Jaccard) **AND** pixel 게이트(luminance 차이 `< page_pixel_diff_threshold`)로 확인 → 기존 page 병합(`BM25_MERGE`)하거나 새 page 발급.
  - 이 `page_key` 가 page_graph 와 탐색 abstract page 를 모두 결정한다.
  - `OPENROUTER_API_KEY` 가 없으면 입력 텍스트 생성은 자동으로 `random` 폴백하지만 BM25 page 식별은 그대로 동작한다.
  - 필터된(구조 prefilter/BM25 dedup) 재방문의 **저장 여부**는 `screen_matching.persist_filtered`(기본 ON)가 결정한다 — 기본값에선 필터된 재방문도 그 page 아래 **자체 observation**(방문마다 `0,1,2,…` 체인)으로 저장하고, off 로 두면 재사용 관측은 파일을 전혀 쓰지 않는다.

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

`.env` 또는 환경변수에 `OPENROUTER_API_KEY` 를 넣으면 `--input-mode api` (입력 텍스트 생성)에서 LLM 을 사용한다.
page 식별(BM25 unique-page matching)은 LLM 없이 동작한다.
모델과 엔드포인트는 `OPENROUTER_MODEL` (기본 `qwen/qwen3.7-plus`), `OPENROUTER_BASE_URL` (기본 `https://openrouter.ai/api/v1`) 로 덮어쓸 수 있다.

추가 전제:

- Python 3.10+
- ADB 가 PATH 에 있거나 `ANDROID_HOME` 이 설정되어 있어야 한다
- **`Pixel6-2` 이름의 AVD 가 실행 중이어야 한다** (`emulator -avd Pixel6-2`). `AdbClient` 가 `adb devices` 에서 해당 AVD 의 emulator serial 을 자동 탐색해 모든 ADB 명령에 `-s <serial>` 로 고정한다(기본값 `Pixel6-2`, env `MC_AVD` 로 오버라이드 가능). 다른 AVD 나 실기기가 함께 붙어 있어도 무방하며, AVD 가 없으면 `AdbClient()` 생성 시 즉시 `RuntimeError` 로 실패한다.

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
# 기본 종료 조건은 시간 예산 — 앱당 2h (config/run.yaml 의 budget_mode=time, max_duration=2h)
monkey-collect run --apps all

# 시간 예산 명시 (앱당 벽시계 예산; 형식 2h/120m/7200s)
monkey-collect run --apps all --budget-mode time --duration 2h

# 스텝 예산으로 전환 (앱당 최대 step 수까지)
monkey-collect run --apps all --budget-mode steps --steps 1500

# 원하는 앱만 지정 (완료 여부는 동일하게 체크)
monkey-collect run --apps com.google.android.deskclock com.google.android.calculator

# 완료된 앱도 다시 수집
monkey-collect run --apps all --force

# 특정 앱의 기존 세션을 폐기하고 새로 시작
monkey-collect run --apps com.google.android.deskclock --new-session

# 입력 텍스트를 hardcoded 로 (API 비용 없음)
monkey-collect run --apps all --input-mode random
```

동작:

- 서버가 각 앱마다 `adb shell am start` 로 앱을 실행하고, TCP 로 `{"type": "START", "package": "com.X"}` 를 보낸다. Android 앱은 standby 연결을 유지하다가 START 를 받아 자동으로 수집을 시작한다.
- 한 세션이 끝나면 서버가 `SESSION_END` 를 보내 클라이언트를 정리하고, Android 는 `F` 회신 후 새 소켓으로 자동 재접속한다.
  - 서버는 이 fresh 소켓을 그대로 유지하고 다음 앱으로 `START` 를 송신한다.
  - 이때 이전 세션의 `F`(finish) 회신이 신호 큐에 남을 수 있어, `run_collection_loop` 는 시작 시 큐를 비운다 — 비우지 않으면 새 세션이 stale finish 를 읽고 0-step 으로 끝나며 큐의 다음 앱들까지 연쇄로 0-step 된다.
  - 큐 소비 시 연속된 `xml` 프레임만 collapse 되고 제어 신호(`external_app`/`finish`/`no_change`)는 유실되지 않는다 — 계약은 `AGENTS.md` 불변식 5 참조.
- **세션 종료 조건**은 `budget_mode`(기본 `time`)로 정한다 — `time` 이면 앱당 벽시계 `max_duration`(기본 `2h`) 경과까지, `steps` 면 `max_steps`(기본 1500) action 까지 탐색한다. 모드 추론 규칙은 「CLI」 절의 `--budget-mode` 참조, clean-stop 보장·resume 예산 시맨틱은 [ARCHITECTURE.md](./ARCHITECTURE.md) §4 참조.
- 큐 구성 시 `runtime/{pkg}/metadata.json` 의 `completed_at` 이 채워진 앱은 **완료로 판정되어 스킵**. `--force` 로 우회하거나, 중단된(미완료) 세션은 `completed_at` 이 `null` 이라 자동으로 resume 된다(재개 시 `data/{pkg}/pages/` 로부터 page 지식도 함께 복원).
- **서버 pull (`CAPTURE` poke)**: 클라이언트는 접근성 이벤트가 떴을 때만 프레임을 push 하므로, 이벤트가 안 뜨면 signal 대기 창 내내 침묵한다 — 서버가 `collection.poke_delay_sec`(기본 1.5s) 침묵 후 `CAPTURE` 로 프레임을 당겨온다(대기당 최대 2회).
  - 클라이언트 응답 프로토콜(XML 해시 비교 → 프레임/`N`/`E`)과 poke 의 시간 예산 시맨틱(`signal_timeout_sec` 분할, escalation 시점 불변)은 [ARCHITECTURE.md](./ARCHITECTURE.md) §2~§3 참조.
- **재초기화(reinit)**: signal timeout `MAX_SIGNAL_TIMEOUTS`(3) 연속 또는 external app `MAX_EXTERNAL_APP_RETRIES`(10) 회 도달 시 세션을 종료하는 대신, target app 을 force-stop + relaunch 하고 카운터를 초기화해 탐색을 재개한다.
  - 재초기화 한도는 `MAX_TIMEOUT_REINITS`(20) / `MAX_EXTERNAL_REINITS`(10) 이며, 초과 시 세션이 종료된다.
  - 상수 정본은 `src/monkey_collector/pipeline/recovery.py`.
- **open_app 기록**: external app 복구가 타깃 앱을 재실행하면 `open_app` 액션을 `events.jsonl` 에 이탈(excursion) 당 1회 기록한다 — 이는 navigation 전이가 아니다(3중 격리 메커니즘·이벤트 스키마는 [ARCHITECTURE.md](./ARCHITECTURE.md) §4~§5 참조).

## 설정 (`config/run.yaml`)

수집 파라미터는 `config/run.yaml` 로 관리한다. 값은 **4단계**로 해석되며 뒤쪽이 앞쪽을 덮어쓴다:

1. 빌트인 기본값 (canonical strategy = `BFS`)
2. `config/run.yaml` (다른 경로는 `MC_CONFIG_PATH` 또는 `--config` 로 지정)
3. `MC_*` 환경변수
4. CLI 플래그

즉 **빌트인 → `run.yaml` → `MC_*` env → CLI** 순으로 우선순위가 높아진다. YAML 이 커버하는 argparse 파라미터는 CLI 기본값이 `None`(미지정 sentinel)이라, 플래그를 주지 않으면 config 값이 그대로 쓰인다. `--apps` / `--force` / `--new-session` 같은 운영 플래그는 CLI 전용이다.

`config/run.yaml` 섹션:

- `exploration.strategy`: `DFS` | `BFS` | `GREEDY` (canonical 기본 `BFS`)
- `collection.{budget_mode, max_duration, max_steps, seed, action_delay_ms, poke_delay_sec, port, data_dir, runtime_dir}` (`budget_mode` 기본 `time`, `max_duration` 기본 `2h`, `poke_delay_sec` 기본 `1.5`)
- `llm.{input_mode}`
- `screen_matching.{luminance_prefilter, luminance_threshold, screenshot_diff_threshold, luminance_low_res_width, persist_filtered, bm25_top_k, element_criterion, element_diff_max, element_jaccard_min, page_pixel_diff_threshold, canvas_merge, canvas_min_area_frac, package_guard}`

각 키에 대응하는 `MC_*` 환경변수(+ 타입)의 전체 목록은 [ARCHITECTURE.md](./ARCHITECTURE.md) 의 「MC_* 환경변수」 표가 정본이다. 대체 yaml 경로는 `MC_CONFIG_PATH` 로 지정한다.

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
monkey-collect run --apps all                                   # 기본: 시간 예산 앱당 2h
monkey-collect run --apps all --budget-mode time --duration 2h  # 시간 예산 명시
monkey-collect run --apps com.google.android.deskclock --budget-mode steps --steps 1500
```

주요 옵션:

- `--apps` (필수): `all` 이면 `catalog/apps.csv` 의 `installed=true` 전부. 아니면 하나 이상의 package_id.
- `--strategy`: 탐색 전략 `DFS` / `BFS` / `GREEDY` 선택 (canonical 기본 `BFS`). 미지정 시 `config/run.yaml` 의 `exploration.strategy` 를 따른다 (의미는 「설정」 섹션 참조).
- `--config`: 사용할 config YAML 경로 (기본 `config/run.yaml`).
- `--budget-mode {time,steps}`: 세션 종료 조건. `time` = `--duration` 벽시계 예산, `steps` = `--steps` action 수. 미지정 시 `--steps`/`--duration` 중 준 쪽으로 추론(둘 다 있고 모드 미지정이면 config 값 유지 + 경고). 기본 `config/run.yaml` 의 `budget_mode`(제품 기본 `time`)
- `--duration`: `budget_mode=time` 일 때 앱 세션당 벽시계 예산. 형식 `2h`/`120m`/`7200s`/`7200`(맨숫자=초). 기본 `config` 의 `max_duration`(`2h`)
- `--steps`: `budget_mode=steps` 일 때 세션당 최대 step 수 (기본 1500)
- `--seed`: explorer 랜덤 시드 (기본 42)
- `--delay`: action 사이 대기 시간(ms, 기본 1500)
- `--port`: TCP server port (기본 12345)
- `--data-dir`: 영속 데이터 루트 — pages/observations, page_graph (기본 `data`)
- `--runtime-dir`: 휘발성 런타임 루트 — metadata, events, cost/coverage (기본 `runtime`)
- `--input-mode`: 입력 텍스트 생성 모드 `api` (LLM, 현재 앱 설명을 프롬프트에 포함) 또는 `random` (hardcoded). 기본 `api`
- `--luminance-prefilter`: luminance `on` / `off`. 기본 `on` — OBSERVATION dedup + PAGE pixel 게이트 지문 공급(off 면 pixel 게이트 abstain → element 기준 단독)
- `--luminance-threshold`: 픽셀 밝기 차이 `|ΔY|` 임계값 0–255 (기본 10)
- `--screenshot-diff-threshold`: 같은 **observation** 으로 볼 차이 픽셀 비율 (기본 0.02)
- `--luminance-low-res-width`: luminance 지문 다운스케일 너비 px (기본 100)
- `--persist-filtered`: 필터된(prefilter/dedup) 재방문을 그 page 아래 자체 observation(방문마다 `0,1,2,…`)으로 저장 `on` / `off`. 기본 `on` — `off` 면 재사용 관측은 파일을 안 쓰는 기존 절약 동작
- `--bm25-top-k`: BM25 로 화면당 검증할 후보 page 수 (기본 5)
- `--element-criterion`: element 동일-page 기준 `diff`(`|A△B|<--element-diff-max`) / `jaccard`(`>--element-jaccard-min`) (기본 `diff`)
- `--element-diff-max`: 같은 page 로 볼 최대 상이 element-line 수 (기본 5)
- `--element-jaccard-min`: 같은 page 로 볼 최소 element-line Jaccard (`jaccard` 모드, 기본 0.5)
- `--page-pixel-diff-threshold`: PAGE 병합을 확정하는 pixel 게이트 차이 픽셀 비율 (기본 0.3)
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

- `--all`: `--data-dir`/`--runtime-dir` 전체를 삭제 (다른 스코프 플래그와 상호 배타). 두 root 를 함께 지워야 한다 — `data/` 만 남으면 다음 실행이 그 페이지 지식을 다시 rehydrate 해버려 "새 세션"이 되지 않는다.
- `--apps`: 삭제할 package_id 리스트
- `--data-dir`: 영속 데이터 루트 (기본 `data`)
- `--runtime-dir`: 휘발성 런타임 루트 (기본 `runtime`)
- `--dry-run`: 삭제 없이 대상 경로만 출력
- `--yes`: 확인 프롬프트 스킵

### `convert`

```bash
monkey-collect convert \
  --package com.example.app \
  --output data/processed/gui-model_stage1.jsonl \
  --images-dir data/processed/images \
  --label 1
```

주요 옵션:

- `--data-dir` / `--runtime-dir`: 두 root (기본 `data`/`runtime`)
- `--package`: 변환할 세션의 package id (= 두 root 아래 디렉터리명)
- `--label`: 이미지 파일명에 사용되는 세션 라벨 정수 (기본 `1`).

`pages/` 가 있는 세션은 `events.jsonl` 의 `page_key`/`observation_num` 으로 정확히 조인한다. `pages/` 없이 구형 flat `xml/` 만 있는 세션(마이그레이션 이전 백업)은 자동으로 legacy 변환으로 degrade 되므로, 그런 백업을 변환할 땐 `--data-dir`/`--runtime-dir` 을 같은 경로로 지정한다(원래 한 디렉터리에 다 있었으므로).

### `convert-all`

```bash
monkey-collect convert-all \
  --data-dir data \
  --runtime-dir runtime \
  --output data/processed/gui-model_stage1.jsonl \
  --images-dir data/processed/images
```

`--data-dir` 아래 각 세션 디렉터리명으로 `--runtime-dir` 의 대응 세션을 찾아 짝짓는다. 존재하지 않는 `--data-dir` 은 크래시 없이 경고 후 0건으로 종료한다.

### `page-map`

```bash
monkey-collect page-map --package com.example.app
monkey-collect page-map --package com.example.app --threshold 0.9 --no-open
```

`pages/` 가 있으면 `events.jsonl` 의 `page_key`/`observation_num` 으로 정확히 재구성한다(구조적 근사 불필요). 없고 구형 flat `xml/` 뿐이면 구조적/Jaccard 근사로 degrade.

### `page-map-all`

```bash
monkey-collect page-map-all --data-dir data --runtime-dir runtime --no-open
```

### `regenerate`

```bash
monkey-collect regenerate --data-dir data
```

raw XML 을 기준으로 파생 파일(`parsed.xml`, `hierarchy.xml`, `encoded.xml`, `pretty.xml`)을 다시 만든다. `pages/{page_key}/{obs}/raw.xml` 각각을, 구형 세션이면 `xml/{step}_*.xml` 을 대상으로 한다.

## 저장 구조

두 root 로 분리된 세션 구조 — `data/` 는 영속 코퍼스(page/observation 지식), `runtime/` 는 세션 진행 상태:

```
data/{package}/                            # 영속 — page/observation 코퍼스
├── pages/
│   ├── 0/                                 # page_key (0-based 정수, page_ 접두사·zero-pad 없음)
│   │   ├── page.json                      # page 의 고정 identity(element_lines[BM25 문서]·is_canvas·element_lines_blind·first_activity), 최초 생성 시 1회만 기록
│   │   ├── 0/                             # observation (persist_filtered on 이면 방문마다 0,1,2… 체인)
│   │   │   ├── screenshot.png, raw.xml, parsed.xml, hierarchy.xml, encoded.xml, pretty.xml
│   │   │   └── elements.json              # matched observation 마다 (page_key/match_type/is_new_page/activity)
│   │   └── 1/...                          # 필터된 재방문(persist_filtered on)
│   └── 1/
│       └── 0/...
├── page_graph.json
└── page_graph.html

runtime/{package}/                         # 휘발성 — 재개 상태·타임라인
├── metadata.json
├── events.jsonl
├── activity_coverage.csv     # ground truth: catalog/activities.json (fallback: dumpsys)
└── cost.csv
```

이미 저장된 page 와 구조적으로 동일하거나(structural exact-match) BM25+element/pixel 로 병합되는(`BM25_MERGE`) 화면을 재방문하면 `screen_matcher` 가 같은 `page_key` 로 단락한다(page 정체성 재사용, LLM 0회). 필터된 재방문의 저장 여부는 `screen_matching.persist_filtered`(기본 ON)가 결정한다 — 메커니즘은 [ARCHITECTURE.md](./ARCHITECTURE.md) §5(저장 포맷) 참조.

`events.jsonl` 은 처리된 화면마다 한 줄이며, `page_key`/`observation_num` 이 실제 화면 파일을 가리키는 **조인 키**다(`frame_index`/`step` 은 조인 키가 아님) — 스키마는 [ARCHITECTURE.md](./ARCHITECTURE.md) §5 참조.

`elements.json` 은 matched observation 마다 `ScreenMatcher` 가 만든 match annotation(`page_key`/`match_type`/`is_new_page`/`activity` 4개 키)을 담는다 — 스키마는 [ARCHITECTURE.md](./ARCHITECTURE.md) §5 참조.

세션 재개(resume) 시 `data/{package}/pages/` 로부터 `ScreenMatcher` 지식과 `page_graph.json` 을 모두 복원한다(기존 page 를 "새 page" 로 재발견하지 않음) — 메커니즘은 [ARCHITECTURE.md](./ARCHITECTURE.md) §4~§5 참조.

마이그레이션 이전(flat `screenshots/`/`xml/`) 세션은 그대로 남겨두며 마이그레이션 스크립트는 없다 — `regenerate`/`page-map`/`convert` 는 `pages/` 없이 `xml/` 만 있는 세션을 감지하면 자동으로 구형 로직으로 degrade 한다.

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
│   ├── llm/           (client)  # 공용 OpenRouter 클라이언트 (입력 텍스트 생성)
│   ├── pipeline/      (collector, collection_loop, recovery,
│   │                   session_manager, app_catalog, installed_sync,
│   │                   reset, text_generator)
│   │     ├── screen_matching/  (screen_matcher, bm25, element_lines, luminance, ui_attributes, page_knowledge, rehydrate)
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
│   ├── unit/                         # 컴포넌트별 유닛 테스트
│   └── integration/                  # adb·서버·수집 통합 테스트
│
└── (gitignored) data/, runtime/ (logs/ 포함), *.egg-info/
```

## 코드 읽기 시작점

- [`src/monkey_collector/cli.py`](./src/monkey_collector/cli.py): 실제 CLI
- [`src/monkey_collector/pipeline/collector.py`](./src/monkey_collector/pipeline/collector.py): 수집 진입점
- [`src/monkey_collector/pipeline/exploration/`](./src/monkey_collector/pipeline/exploration): action selection (LLM-guided 탐색 엔진)
- [`src/monkey_collector/storage.py`](./src/monkey_collector/storage.py): 세션 포맷
- [`app/app/src/main/java/com/monkey/collector`](./app/app/src/main/java/com/monkey/collector): Android 앱

구조 설명은 [`ARCHITECTURE.md`](./ARCHITECTURE.md), 작업 규칙은 [`AGENTS.md`](./AGENTS.md) 를 본다.
