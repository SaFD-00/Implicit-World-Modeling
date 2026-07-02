# MediaProjection 동의 & AccessibilityService (§6-a / §6-c)

화면 캡처(MediaProjection)와 접근성 서비스 셋업의 상세. SKILL.md §6-a/§6-c 에서 위임.

## 캡처 아키텍처 — 두 경로

Monkey-Collector client 의 화면 캡처는 **두 메커니즘**을 쓴다:

1. **AccessibilityService.takeScreenshot** (`canTakeScreenshot=true`, `accessibility_config.xml`) — 캡처 본체. 접근성만 켜져 있으면 동작.
2. **MediaProjection (`ScreenStabilizer`)** — *stabilization* 보조. 화면이 안정될 때까지 프레임을 비교해 캡처 타이밍을 잡는 용도.

→ **MediaProjection 이 없거나 토큰이 만료돼도 캡처는 (1)로 계속된다.** MediaProjection 은 품질 보조이지 필수 아님.

## 동의 플로우

`MainActivity` → 서버 IP/포트 입력(prefs 저장) → **"Save & Ready"** 탭 → `mediaProjectionManager.createScreenCaptureIntent()` → 시스템 동의 다이얼로그("Start now") → 결과(resultCode/resultData)를 `MediaProjectionHelper` 싱글톤에 저장 → `CollectorService` 가 `startCaptureSession(resultCode, resultData)` 로 사용.

```bash
adb -s "$SERIAL" shell appops set com.monkey.collector PROJECT_MEDIA allow 2>/dev/null
adb -s "$SERIAL" shell am start -n com.monkey.collector/.MainActivity; sleep 3
# "SAVE & READY" 좌표 잡아 탭
B=$(adb -s "$SERIAL" exec-out uiautomator dump /dev/tty 2>/dev/null | tr '>' '>\n' | grep -i 'SAVE & READY' \
    | sed -E 's/.*bounds="\[([0-9]+),([0-9]+)\]\[([0-9]+),([0-9]+)\]".*/\1 \2 \3 \4/' | head -1)
[ -n "$B" ] && { set -- $B; adb -s "$SERIAL" shell input tap $(( ($1+$3)/2 )) $(( ($2+$4)/2 )); }
sleep 2
# 시스템 "Start now"(또는 "시작") 다이얼로그가 뜨면 탭 (appop 자동승인 시 안 뜸)
B=$(adb -s "$SERIAL" exec-out uiautomator dump /dev/tty 2>/dev/null | tr '>' '>\n' | grep -iE 'text="(Start now|시작)"' \
    | sed -E 's/.*bounds="\[([0-9]+),([0-9]+)\]\[([0-9]+),([0-9]+)\]".*/\1 \2 \3 \4/' | head -1)
[ -n "$B" ] && { set -- $B; adb -s "$SERIAL" shell input tap $(( ($1+$3)/2 )) $(( ($2+$4)/2 )); }
adb -s "$SERIAL" shell input keyevent KEYCODE_HOME
```

> **실측(Pixel6-2/API33)**: `appops … PROJECT_MEDIA allow` 가 설정돼 있으면 "Save & Ready" 후 **동의 다이얼로그 없이 자동 승인**되고 `"Ready. Server will launch target apps automatically."` 토스트 뜬 뒤 standby 진입. 다이얼로그가 뜨는 이미지/버전에선 "Start now" 탭이 필요.

## 토큰 단발성 & graceful-degrade (이전 crash 원인·수정)

- **토큰은 단발성**: 모던 Android 에서 동의 토큰(resultCode/resultData)은 1회용이다. **재설치/재시작 후 같은 토큰으로 `getMediaProjection`/`createVirtualDisplay` 재호출 시 `SecurityException`**.
- **과거 버그**: `ScreenStabilizer.startCaptureSession` 가 세션마다 같은 토큰을 재사용 → 2번째 세션의 `createVirtualDisplay` 가 uncaught `SecurityException` 으로 **client 프로세스 사망** → 핸드셰이크 desync + signal timeout 연쇄.
- **수정(`ScreenStabilizer.kt`)**: ① reuse-guard(`mediaProjection != null && virtualDisplay != null` 이면 early-return) ② acquire/`createVirtualDisplay` 를 try/catch 로 감싸 실패 시 imageReader/projection 정리 후 **return(=stabilization 없이 degrade)**. 덕분에 토큰이 만료/무효여도 크래시 없이 (1) takeScreenshot 캡처로 계속.

→ 검증 시 2세션 이상(예 org.tasks→Drive) 돌려 **2번째 세션 시작에서 프로세스 사망이 없는지** `adb logcat -b crash` 로 확인.

## AccessibilityService (§6-a)

- 컴포넌트: `com.monkey.collector/com.monkey.collector.CollectorService` (manifest `.CollectorService` 를 패키지 namespace 로 푼 것).
- adb 활성(콜론 구분 리스트 — 기존 보존 merge, 멱등):
  ```bash
  CUR=$(adb -s "$SERIAL" shell settings get secure enabled_accessibility_services | tr -d '\r'); [ "$CUR" = null ] && CUR=""
  J=$(printf "%s:%s" "$CUR" "com.monkey.collector/com.monkey.collector.CollectorService" | sed 's/^://;s/:$//' | tr ':' '\n' | awk 'NF&&!s[$0]++' | paste -sd: -)
  adb -s "$SERIAL" shell settings put secure enabled_accessibility_services "$J"
  adb -s "$SERIAL" shell settings put secure accessibility_enabled 1
  ```
- `install -r`(replace) 은 접근성 토글을 보존한다. 전체 uninstall 후 재설치하면 리셋되므로 다시 켠다.
- 컴포넌트명 오타가 있으면 토글이 **조용히 무시**된다 — 활성 후 `settings get secure enabled_accessibility_services | tr ':' '\n' | grep CollectorService` 로 확인.

## EXCLUDED_PACKAGES (gms 핸드오프, 관련 수정)

`CollectorService.kt` 의 `EXCLUDED_PACKAGES` 에 `com.google.android.gms`/`gsf`/`com.android.vending` 추가 — Google 로그인/Play 핸드오프 화면을 "외부 앱(타깃 이탈)"으로 처리해 **외부앱 재실행 스톰**을 막는다. 서버측은 `screen_guard.py` 의 `SYSTEM_PACKAGES` 가 동일 패키지를 drift 로 처리(이중 방어). 상세는 [run-and-verify.md](run-and-verify.md).
