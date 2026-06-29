# 수집 실행 & 라이브 검증 (§9)

`monkey-collect run` 으로 수집이 end-to-end 동작하는지 확인하는 상세. SKILL.md §9 에서 위임.

## 실행 명령

```bash
cd "$ROOT"
uv run monkey-collect run --apps <pkg...|all> --steps N --port 12345 \
  [--new-session] [--force] [--input-mode api|random] [--screen-grouping on|off]
```

| 플래그 | 의미 |
|--------|------|
| `--apps all` | `catalog/apps.csv` 의 `installed=true` 앱 전체. 또는 `--apps org.tasks com.X` 로 명시 |
| `--steps N` | 세션당 최대 스텝 |
| `--new-session` | 해당 앱의 기존 세션 삭제 후 새로 시작 |
| `--force` | 이미 완료(`completed_at`)된 앱도 재수집 |
| `--input-mode api` | LLM 으로 입력텍스트 생성(기본). `.env` 의 `OPENROUTER_API_KEY` 필요. 없으면 `random` |
| `--screen-grouping on` | LLM 화면요소 그룹핑(기본). annotation 전용 |
| `--port 12345` | TCP 서버 포트 — client prefs `server_port` 와 일치해야 함 |

## 핸드셰이크 & 프로토콜

서버가 TCP(0.0.0.0:12345) 리슨 → client(CollectorService standby)가 `10.0.2.2:12345` 로 연결 → 앱별로:

1. 서버 `wait_for_connection` → `send_start({package})` → client `P`(package ACK).
   - **no-ACK(5s)** 면 `session_manager.receive_target_package` 가 `None` 반환 → 세션 abort(blind 수집 방지). 죽은 세션에 step 예산을 안 태운다.
2. 서버가 `adb force-stop` + `am start`(launch) → 앱 전경화.
3. client AccessibilityService 가 `TYPE_WINDOW_*` 이벤트 → 스크린샷(`S`) + XML(`X`) push.
4. 신호: `P` package · `S` screenshot · `X` xml(+top_package/activity) · `N` no-change · `E` external-app · `F` finish.
5. 끝나면 서버 `SESSION_END` → page_graph 빌드/시각화.

## 알려진 실패모드 ↔ 적용된 수정

| 실패모드(과거 `--apps all`) | 근본원인 | 수정 위치 |
|---|---|---|
| **signal timeout 연쇄 / near-empty XML** | MediaProjection 토큰 재사용 → 2세션째 `createVirtualDisplay` SecurityException 으로 **client 프로세스 사망** → 핸드셰이크 desync | `ScreenStabilizer.kt` reuse-guard + try/catch graceful-degrade (client 재빌드 필요) |
| **외부앱(docs) 스톰** | Google 로그인/Play 핸드오프 화면을 타깃 이탈로 인식 못 해 재실행 폭주 | client `CollectorService.kt` `EXCLUDED_PACKAGES`+= gms/gsf/vending; 서버 `screen_guard.py` `SYSTEM_PACKAGES`+= 동일(이중 방어) |
| **빈 UI 인데 계속 진행** | empty 가드가 raw count 만 봄 | `collection_loop.py` 가드를 `get_interactable_elements()` 기준으로 |
| **죽은 세션에 step 예산 소진** | no-ACK 시 서버가 요청 패키지로 blind 진행 | `session_manager.py` no-ACK 면 abort(None) |

> step 0 직후 `external app: nexuslauncher` 1~2회는 **force-stop→launch 전이의 정상 신호**(런처가 잠깐 전경)이며 collection_loop 가 재실행으로 복구한다 — 연속 10회(`external app (10/10)`)로 쌓이며 못 빠져나오는 게 진짜 스톰이다.

## 라이브 스모크 검증 (대표 2앱)

```bash
cd "$ROOT"
adb -s "$SERIAL" logcat -b crash -c
uv run monkey-collect run --apps org.tasks com.google.android.apps.docs \
  --steps 40 --new-session --input-mode api --screen-grouping on --port 12345
```

### 성공판정
- 두 앱 모두 `data/raw/<pkg>/page_graph*`(또는 세션 dir 내 page_graph json) **노드 ≥ 2** 생성.
- 서버 로그에 **연속 signal-timeout 없음**, `external app (10/10)` 스톰 **없음**.
- client 프로세스 사망 없음:
  ```bash
  adb -s "$SERIAL" logcat -b crash -d | grep -i "com.monkey.collector" || echo "no client crash (good)"
  ```
- Drive 세션이 gms 로그인 핸드오프에서 **무한 재실행으로 빠지지 않음**(로그인돼 있으면 정상 탐색, 로그아웃이면 drift 처리 후 깔끔히 종료).

### 결과 위치
- 세션 데이터: `data/raw/<pkg>/` (screenshots/, xml, metadata.json, page_graph).
- page map 시각화: `monkey-collect page-map`(또는 세션 finalize 시 자동 생성).
