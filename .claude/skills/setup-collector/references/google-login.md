# Google 계정 로그인 (§6-d)

Google 로그인이 필요한 앱을 수집하려면 디바이스에 Google 계정이 있어야 한다. SKILL.md §6-d 에서 위임.

## 자격증명 규약 — 커밋 금지

**평문 비밀번호를 스킬/문서/repo 에 절대 넣지 않는다.** `.gitignore` 에 이미 `**/.secrets.local`·`**/*.secrets.local` 룰이 있으나(`git check-ignore -v Monkey-Collector/.secrets.local` 로 확인됨), 그래도 계정/비밀번호는 **로컬 비밀파일에만** 두고 커밋 전 반드시 추적 여부를 재확인한다.

- 문서/스킬에는 **플레이스홀더** `$GOOGLE_ACCOUNT` / `$GOOGLE_PASSWORD` 만 쓴다.
- 실제 값은 `Monkey-Collector/.secrets.local` 에 둔다(`.gitignore` 의 `**/*.secrets.local` 로 무시 — `git check-ignore` 로 확인):
  ```
  GOOGLE_ACCOUNT=you@gmail.com
  GOOGLE_PASSWORD=...
  ```
- 런타임 로드: `set -a; . "$ROOT/.secrets.local"; set +a`
- 푸시 전 `git status` 로 `.secrets.local` 이 추적되지 않는지 반드시 확인.

## 멱등 — 계정 있으면 skip

```bash
HAVE=$(adb -s "$SERIAL" shell dumpsys account 2>/dev/null | grep -c "Account {.*type=com.google}")
[ "$HAVE" != "0" ] && echo "이미 Google 계정 있음 — 로그인 skip"
```
`dumpsys account` 의 `Accounts: N` / `Account {name=…, type=com.google}` 로 확인. **계정 0개일 때만** 신규 등록.

> **naive 패턴 주의**: `type=com.google` 문자열만 단순 카운트하던 이전 패턴은 `AuthenticatorDescription {type=com.google.android.gm.pop3, ...}` 류 서비스 등록 라인까지 세어 계정이 0개인 디바이스에서도 항상 ≥1(emulator-5558 실측: 계정 0개 → 4, 1개 → 5)을 반환한다 — 위 멱등 가드가 절대 발화하지 않아 로그인이 필요해도 조용히 skip된다. `Account {.*type=com.google}` 가 실측 검증된 정확한 패턴이다(계정 0개 → 0, 1개 → 1) — 행 시작의 `Account {` 접두가 `AuthenticatorDescription {…}` 서비스 라인을 배제하고, 닫는 `}` 는 `type=com.google.android.gm.*` 같은 하위타입 오탐(`type=com.google` 뒤에 `}` 가 곧바로 와야 매치)까지 막는다.

## 로그인 절차 (Settings → Add account → Google)

UI 자동화는 `uiautomator dump` 로 각 화면의 필드/버튼 좌표를 잡아 `input text`/`input tap`. 화면 흐름(API33 실측):

1. `adb -s "$SERIAL" shell am start -a android.settings.ADD_ACCOUNT_SETTINGS` → "Add an account" 목록.
2. **Google** 행 탭 → "Checking info…" 로딩 화면(~3초) 경유 → `com.google.android.gms/.auth.uiflows.minutemaid.MinuteMaidActivity`(Pixel6-3/API33 실측).
3. "Sign in with ease" 인트로 → **SKIP**(전화번호 계정검색 생략). **이 화면은 계정/gms 상태에 따라 생략될 수 있음 — Pixel6-3/API33 실측에서는 미등장, 곧바로 이메일 입력 화면으로 진입.**
4. "Email or phone" 필드 탭 → `input text "$GOOGLE_ACCOUNT"` → **Next**.
5. "Welcome" 비번 화면 → 필드 탭 → `input text "$GOOGLE_PASSWORD"` → **Next**.
6. "Google Terms of Service" → **I agree**.
7. "Make sure you can always sign in"(복구 전화/이메일) → **Cancel**(선택사항, 계정은 이미 등록됨). **이 화면도 등장하지 않을 수 있음 — Pixel6-3/API33 실측에서는 ToS 직후 바로 다음 화면으로 넘어가 미등장.**
8. **"Google services"(기기 백업 동의) 화면** — "Use basic device backup" 스위치(기본 on, 건드리지 않음) + **ACCEPT** 버튼 탭. ACCEPT 후 자동으로 런처 홈으로 복귀한다(`KEYCODE_HOME` 불필요). ⚠️ **ACCEPT 버튼이 화면 진입 직후 `clickable=true` 지만 `text` 라벨이 비어 렌더링 미완료 상태일 수 있어 첫 탭이 씹힐 수 있다 — `uiautomator dump` 로 `text="ACCEPT"` 확인 후 탭한다.** (기존 문서의 "Set a home address" → Skip / restore opt-in 화면은 Pixel6-3/API33 실측에서 미등장.)
9. 확인: `adb -s "$SERIAL" shell dumpsys account 2>/dev/null | grep -E "Accounts:|Account \{.*type=com\.google\}"` → `Accounts: 1`.

> 좌표는 해상도(`pixel_6` 디바이스 프로파일 = 1080×2400 — Pixel6-2/Pixel6-3 공통, Pixel6-3/API33 실측)·언어에 따라 다르니, 하드코딩 말고 매 화면 `uiautomator dump` 로 버튼 텍스트("SKIP"/"Next"/"I agree"/"Cancel"/"ACCEPT")의 bounds 를 잡아 중심을 탭한다.

## 차단 시 폴백

- **2FA / 리캡차 / "이 기기에서 로그인할 수 없음"**: 자동화로 못 뚫으면 사용자에게 **수동 로그인 1회**를 요청한다(`! adb -s emulator-5554 shell am start -a android.settings.ADD_ACCOUNT_SETTINGS` 후 직접 입력).
- `google_apis`(비-playstore) 이미지는 일부 케이스에서 sign-in 을 차단할 수 있다. (실측: Pixel6-2/API33 에서는 정상 로그인됨. **Pixel6-3/API33 에서도 `.secrets.local` 자격증명으로 §6-d 자동화가 1회 시도로 완주됨(2026-07-18 실측, 2FA 미발생).**)
- **gms-storm 수정만 검증**하려면 로그인 불필요 — 로그아웃 상태로 Drive 를 돌려 gms 로 bounce 될 때 스톰/크래시 없이 처리되는지로도 검증된다([run-and-verify.md](run-and-verify.md)).

## Google 로그인 필요 앱 (MC 카탈로그, installed)

`com.google.android.apps.docs`(Drive) · `com.google.android.gm`(Gmail) · `com.google.android.youtube` · `com.google.android.apps.photos` · `com.google.android.calendar` · `com.google.android.contacts`. (Google Docs/Sheets/Slides 는 현재 미설치.)
