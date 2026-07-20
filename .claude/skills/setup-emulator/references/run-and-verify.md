# 수집 실행 & 라이브 검증 (§9)

`monkey-collect run` 으로 수집이 end-to-end 동작하는지 확인하는 상세. SKILL.md §9 에서 위임.

## 실행 명령

```bash
cd "$ROOT"
uv run monkey-collect run --apps <pkg...|all> [--steps N | --duration 10m] --port 12345 \
  [--new-session] [--force] [--input-mode api|random]
```

| 플래그 | 의미 |
|--------|------|
| `--apps all` | `catalog/apps.csv` 의 `installed=true` 앱 전체. 또는 `--apps org.tasks com.X` 로 명시 |
| `--steps N` | 세션당 최대 스텝. 주면 `budget_mode` 가 `steps` 로 추론됨 |
| `--duration 10m` | 세션당 wall-clock 예산(`2h`/`120m`/`7200s`). `budget_mode=time`(기본)에서 실제로 쓰이는 값 |
| `--new-session` | 해당 앱의 기존 세션 삭제 후 새로 시작 |
| `--force` | 이미 완료(`completed_at`)된 앱도 재수집 |
| `--input-mode api` | LLM 으로 입력텍스트 생성(기본). `.env` 의 `OPENROUTER_API_KEY` 필요. 없으면 `random` |
| `--port 12345` | TCP 서버 포트 — client prefs `server_port` 와 일치해야 함 |

> **`--screen-grouping` 은 존재하지 않는 플래그다** — 과거 문서에 있었으나 현재 CLI 에는 없어, 그대로 실행하면 `error: unrecognized arguments: --screen-grouping on` 으로 **즉시 실패**한다(2026-07-20 실측). 실제 옵션은 `monkey-collect run --help` 로 확인할 것.
>
> **⚠️ 기본 예산 = 앱당 2시간**: canonical 기본은 `budget_mode: time` + `max_duration: 2h` 이며 `max_steps: 1500` 은 `budget_mode=steps` 일 때만 적용된다. `--apps all` 로 30~50개를 돌리면 **총 60~100시간**이다. 전수 수집 전에 `--duration` 으로 현실적 캡을 줄 것. 실측 처리량(Pixel6-2/API33) 약 **10~22 스텝/분**.
>
> **⚠️ zsh word-split 함정**: `TARGETS=$(cat list.txt); … --apps $TARGETS` 는 zsh 에서 분리되지 않아 전체 문자열이 **패키지 1개**로 전달되고, 그 문자열이 세션 디렉터리명이 되어 `OSError: [Errno 63] File name too long` 으로 죽는다(2026-07-20 실측). `bash -c 'T=$(cat list.txt); exec uv run monkey-collect run --apps $T …'` 또는 zsh `${=TARGETS}` 를 쓴다.

## 핸드셰이크 & 프로토콜

서버가 TCP(0.0.0.0:12345) 리슨 → client(CollectorService standby)가 `10.0.2.2:12345` 로 연결 → 앱별로:

1. 서버 `wait_for_connection` → `send_start({package})` → client `P`(package ACK).
   - **no-ACK(5s)** 면 `session_manager.receive_target_package` 가 `None` 반환 → 세션 abort(blind 수집 방지). 죽은 세션에 step 예산을 안 태운다.
2. 서버가 `adb force-stop` + `am start`(launch) → 앱 전경화.
3. client AccessibilityService 가 `TYPE_WINDOW_*` 이벤트 → 스크린샷(`S`) + XML(`X`) push.
4. 신호: `P` package · `S` screenshot · `X` xml(+top_package/activity) · `N` no-change · `E` external-app · `F` finish.
5. 끝나면 서버 `SESSION_END` → page_graph 빌드/시각화.

## ⚠️ 죽은 소켓 — 서버 재시작 후 client 가 재연결하지 않음 (전수 실행을 통째로 날리는 실패)

**증상**: 모든 세션이 아래를 반복하고, 큐 끝까지 조용히 헛돈다 — 앱당 120초씩 소진.

```
INFO  … session_manager:wait_for_connection - Waiting for device to connect …
ERROR … session_manager:wait_for_connection - Device did not connect within 120 seconds
WARN  … collector:run_queue - Session N ended without result for <pkg>
```

**원인**: 이전 서버 프로세스가 죽으면(크래시·`pkill`·인자 오류로 인한 예외 등) client 의 `CollectorService` 가 **죽은 소켓을 계속 붙들고 있어** 새로 뜬 서버에 재연결하지 않는다. 서버는 정상적으로 리슨 중이라 서버 쪽 로그만 보면 원인이 안 보인다.

**복구 — accessibility off→on 토글**(client `force-stop` 불필요, 2026-07-20 실측 성공):

```bash
SVC="com.monkey.collector/com.monkey.collector.CollectorService"
adb -s "$SERIAL" shell settings put secure accessibility_enabled 0
adb -s "$SERIAL" shell settings delete secure enabled_accessibility_services   # put "" 는 `Bad arguments` 로 실패한다
sleep 3
adb -s "$SERIAL" shell settings put secure enabled_accessibility_services "$SVC"
adb -s "$SERIAL" shell settings put secure accessibility_enabled 1
```

그 다음 **1개 앱 · 소량 스텝으로 재연결을 먼저 검증**하고(`--apps <one> --steps 12`), 로그에 `Device connected` 가 뜨고 `data/<pkg>/page_graph.json` 이 생기는 것을 확인한 뒤에야 전수 실행을 재시작한다.

> **모니터링 시 필수**: 장시간 실행을 감시할 때 필터에 **`did not connect` 와 `ended without result` 를 반드시 포함**할 것. 성공 신호(`Session complete:`)만 grep 하면 이 실패는 **완전한 침묵**으로 나타나 "진행 중"과 구분되지 않는다(2026-07-20 실측: 이 누락으로 33앱 사이클을 통째로 날렸다).

## 세션 종료 조건 & 재초기화

| 조건 | 임계값 | 동작 |
|------|--------|------|
| signal timeout 연속 | `MAX_SIGNAL_TIMEOUTS=3` | → force-stop + relaunch (재초기화). 재초기화 **20**회 초과 시 세션 종료 |
| external app 연속 | `MAX_EXTERNAL_APP_RETRIES=10` | → `recover()` + 카운터 리셋 (재초기화). 재초기화 **10**회 초과 시 세션 종료 |
| no-change 연속 | `MAX_NO_CHANGE_RETRIES=3` | → back press (세션 유지) |
| same page 연속 | `MAX_SAME_PAGE_STEPS=5` | → back press (세션 유지) |
| empty UI 연속 | `MAX_EMPTY_UI_RETRIES=2` | → back press (세션 유지) |
| max_steps 도달 | `--steps` (기본 1500) | → 세션 종료 (정상) |
| `F` (finish) 신호 | client 가 보냄 | → 세션 종료 (정상) |

> **⚠️ 실제 코드 상수(`recovery.py`)는 20/10 — "3회 초과 시 종료"는 과거 값**: `MAX_TIMEOUT_REINITS=20`, `MAX_EXTERNAL_REINITS=10` 이다(코드 주석에 과거 `MAX_TIMEOUT_REINITS=3` budget이 2~4분 만에 소진돼 정상 세션까지 죽이던 이력이 남아 있음 — budget-loop-fix 의 일부로 상향됐고 문서만 갱신 안 된 상태였다). 로그의 `external app reinit (N/10)` 표시는 버그가 아니라 이 실제 상수를 그대로 찍은 것.

**재초기화 로직** (`recovery.py`/`collection_loop.py`):
- timeout 3연속(`MAX_SIGNAL_TIMEOUTS`) → `force_stop(pkg)` + `launch_app(pkg)` + `timeout_count=0` → `reinit_timeout_count += 1`
  - `reinit_timeout_count > MAX_TIMEOUT_REINITS(20)` 이면 세션 종료
- external app 10회 → `explorer.recover(pkg)` + `external_app_count=0` → `reinit_external_count += 1`
  - `reinit_external_count > MAX_EXTERNAL_REINITS(10)` 이면 세션 종료
- `REINIT_FORGIVE_STEPS=15`: reinit 이후 진짜 진전이 이 스텝만큼 쌓이면 해당 reinit 카운터를 용서(리셋) — 정상 세션이 반복적인 일시 정지로 예산을 다 쓰지 않게 함.

## 알려진 실패모드 ↔ 적용된 수정

| 실패모드(과거 `--apps all`) | 근본원인 | 수정 위치 |
|---|---|---|
| **signal timeout 연쇄 / near-empty XML** | MediaProjection 토큰 재사용 → 2세션째 `createVirtualDisplay` SecurityException 으로 **client 프로세스 사망** → 핸드셰이크 desync | `ScreenStabilizer.kt` reuse-guard + try/catch graceful-degrade (client 재빌드 필요) |
| **timeout 5회 후 즉시 세션 종료** | 임시 AccessibilityService 단절·앱 자체 stuck 등 일시적 원인에도 세션 종료 | `collection_loop.py` 재초기화 — force-stop+relaunch 후 `timeout_count=0`, 3회 초과 시 종료 |
| **외부앱(docs) 스톰** | Google 로그인/Play 핸드오프 화면을 타깃 이탈로 인식 못 해 재실행 폭주 | client `CollectorService.kt` `EXCLUDED_PACKAGES`+= gms/gsf/vending; 서버 `screen_guard.py` `SYSTEM_PACKAGES`+= 동일(이중 방어) |
| **external app 10회 후 즉시 세션 종료** | 앱 내 링크(URL scheme/share intent)가 항상 외부를 열어 10회 소진 | `collection_loop.py` 재초기화 — `recover()` + `external_app_count=0`, 3회 초과 시 종료 |
| **빈 UI 인데 계속 진행** | empty 가드가 raw count 만 봄 | `collection_loop.py` 가드를 `get_interactable_elements()` 기준으로 |
| **죽은 세션에 step 예산 소진** | no-ACK 시 서버가 요청 패키지로 blind 진행 | `session_manager.py` no-ACK 면 abort(None) |

> step 0 직후 `external app: nexuslauncher` 1~2회는 **force-stop→launch 전이의 정상 신호**(런처가 잠깐 전경)이며 collection_loop 가 재실행으로 복구한다 — 연속 10회(`external app (10/10)`)로 쌓이며 못 빠져나오는 게 진짜 스톰이다. 이제 10회 도달 시에도 `recover()` + 카운터 리셋으로 세션을 이어간다.

## 라이브 스모크 검증 (대표 2앱)

```bash
cd "$ROOT"
adb -s "$SERIAL" logcat -b crash -c
uv run monkey-collect run --apps org.tasks com.google.android.apps.docs \
  --steps 100 --new-session --input-mode api --port 12345
```

### 성공판정
- 두 앱 모두 `data/<pkg>/page_graph*`(또는 세션 dir 내 page_graph json) **노드 ≥ 2** 생성.
- 서버 로그에 **연속 signal-timeout 없음**, `external app (10/10)` 스톰 **없음**.
- client 프로세스 사망 없음:
  ```bash
  adb -s "$SERIAL" logcat -b crash -d | grep -i "com.monkey.collector" || echo "no client crash (good)"
  ```
- Drive 세션이 gms 로그인 핸드오프에서 **무한 재실행으로 빠지지 않음**(로그인돼 있으면 정상 탐색, 로그아웃이면 drift 처리 후 깔끔히 종료).

### 결과 위치
- 영속 데이터: `data/<pkg>/pages/` (page/observation: screenshot, xml, elements.json), `data/<pkg>/page_graph*`.
- 휘발성 상태: `runtime/<pkg>/` (metadata.json, events.jsonl, cost/coverage csv).
- page map 시각화: `monkey-collect page-map`(또는 세션 finalize 시 자동 생성).
