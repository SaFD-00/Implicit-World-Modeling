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
  - screen change 발생 시 screenshot + XML 전송
  - external app 감지 및 client-side 복구
  - 서버 드리븐 standby 루프: TCP 연결을 유지하며 서버의 `START {package}` 수신 시 `startCollection` 트리거, `SESSION_END` 수신 시 `stopCollection` 수행
- `ScreenStabilizer.kt`
  - 저해상도 프레임 비교
  - 안정화 대기와 시각 변화 판정
  - first screen 판정
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
  - [`cost_tracker.py`](./src/monkey_collector/domain/cost_tracker.py): LLM 비용 추적 CSV
  - [`page_graph.py`](./src/monkey_collector/domain/page_graph.py): 페이지 그래프 생성
- 인프라 모듈에 인접
  - [`catalog_activities.py`](./src/monkey_collector/catalog_activities.py): `catalog/activities.json` 의 process-lifetime 캐시 (`ActivityCatalog`). activity coverage ground truth 1차 소스.
- `pipeline/`
  - [`collector.py`](./src/monkey_collector/pipeline/collector.py): collector facade
  - [`session_manager.py`](./src/monkey_collector/pipeline/session_manager.py): session init/resume/finalize. `_resolve_declared_activities` 헬퍼가 catalog 우선, dumpsys 폴백 정책을 적용한다.
  - [`collection_loop.py`](./src/monkey_collector/pipeline/collection_loop.py): 메인 루프
  - [`recovery.py`](./src/monkey_collector/pipeline/recovery.py): retry / recovery 상수와 helper
  - [`explorer.py`](./src/monkey_collector/pipeline/explorer.py): SmartExplorer
  - [`text_generator.py`](./src/monkey_collector/pipeline/text_generator.py): random 또는 OpenAI 기반 입력 텍스트 생성
- 인프라 모듈 (monkey_collector/ 직속)
  - [`adb.py`](./src/monkey_collector/adb.py): ADB wrapper. 상단 상수 `REQUIRED_AVD_NAME = "MobileGPT-V2-2"` 에 맞춰 `adb devices` + `emu avd name` 으로 해당 AVD 의 emulator serial 을 해석하고, 이후 모든 명령에 `-s <serial>` 을 prefix 한다. 다중 디바이스 환경에서도 단일 AVD 만 쓰도록 강제.
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
  - `AppJob`: frozen dataclass (category, sub_category, app_name, package_id, source, priority, notes, installed).
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
  -> XML parse
  -> SmartExplorer 가 action 선택
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

### Action Space

[`src/monkey_collector/domain/actions.py`](./src/monkey_collector/domain/actions.py) 에 정의된 6종 action 을 [`src/monkey_collector/pipeline/explorer.py`](./src/monkey_collector/pipeline/explorer.py) 의 `SmartExplorer` 가 가중치 기반으로 선택한다.

| action_type  | 파라미터                                | 설명                            |
| ------------ | --------------------------------------- | ------------------------------- |
| `tap`        | `x, y, element_index`                   | 좌표 또는 UI 요소 탭            |
| `swipe`      | `x1, y1, x2, y2, duration_ms=300`       | 스와이프 제스처                 |
| `input_text` | `text, x, y, element_index`             | 포커스 후 텍스트 입력           |
| `long_press` | `x, y, duration_ms=1000, element_index` | 롱프레스                        |
| `press_back` | —                                       | 안드로이드 Back                 |
| `press_home` | —                                       | 안드로이드 Home                 |

기본 가중치 (`DEFAULT_WEIGHTS`):

| action       | weight |
| ------------ | -----: |
| `tap`        |   0.40 |
| `press_back` |   0.20 |
| `swipe`      |   0.20 |
| `input_text` |   0.10 |
| `long_press` |   0.10 |
| `press_home` |   0.00 |

상황별 가중치 보정:

- 첫 화면에서는 `press_back = 0` (앱 종료 방지)
- editable 요소가 있으면 `input_text ≥ 0.25` 로 부스트
- clickable 이 없으면 `tap = 0.05`
- scrollable 이 없으면 `swipe = 0.05`
- 모든 가중치 합이 0 이면 PressBack 으로 fallback (첫 화면이면 random tap)

실행은 `SmartExplorer.execute_action` 이 `AdbClient` ([`src/monkey_collector/adb.py`](./src/monkey_collector/adb.py)) 메서드로 위임. `AdbClient` 는 CLI 진입점에서 단일 인스턴스로 생성되어 `SmartExplorer` 와 `Collector` 에 주입된다. 생성 시점에 `MobileGPT-V2-2` AVD 의 emulator serial 을 해석해 저장하므로, 해당 AVD 가 실행 중이어야 한다.

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
- external app 시 `return_to_app()` 후 필요하면 `recover()` 수행
- 빈 UI tree 가 반복되면 대기 후 재시도

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
│   └── 0000_pretty.xml
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
- `SmartExplorer`
- `TextGenerator`
- `RandomTextGenerator`
- `LLMTextGenerator`
- `CollectionServer`
- `AdbClient`
- `DataWriter`
- `Converter`
- `PageGraph`
- `build_graph_from_session`
