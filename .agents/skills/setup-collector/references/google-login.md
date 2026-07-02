# Google 계정 로그인 (§6-d)

Google 로그인이 필요한 앱을 수집하려면 디바이스에 Google 계정이 있어야 한다. SKILL.md §6-d 에서 위임.

## 자격증명 규약 — 커밋 금지

**평문 비밀번호를 스킬/문서/repo 에 절대 넣지 않는다.** 이 repo 는 GitHub 로 push 되고 `.gitignore` 에 광범위한 secret 룰이 없을 수 있으므로, 계정/비밀번호는 **로컬 비밀파일에만** 둔다.

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
HAVE=$(adb -s "$SERIAL" shell dumpsys account 2>/dev/null | grep -c "type=com.google")
[ "$HAVE" != "0" ] && echo "이미 Google 계정 있음 — 로그인 skip"
```
`dumpsys account` 의 `Accounts: N` / `Account {name=…, type=com.google}` 로 확인. **계정 0개일 때만** 신규 등록.

## 로그인 절차 (Settings → Add account → Google)

UI 자동화는 `uiautomator dump` 로 각 화면의 필드/버튼 좌표를 잡아 `input text`/`input tap`. 화면 흐름(API33 실측):

1. `adb -s "$SERIAL" shell am start -a android.settings.ADD_ACCOUNT_SETTINGS` → "Add an account" 목록.
2. **Google** 행 탭 → `com.google.android.gms/.auth.uiflows.minutemaid.MinuteMaidActivity`.
3. "Sign in with ease" 인트로 → **SKIP**(전화번호 계정검색 생략).
4. "Email or phone" 필드 탭 → `input text "$GOOGLE_ACCOUNT"` → **NEXT**.
5. "Welcome" 비번 화면 → 필드 탭 → `input text "$GOOGLE_PASSWORD"` → **NEXT**.
6. "Google Terms of Service" → **I agree**.
7. "Make sure you can always sign in"(복구 전화/이메일) → **Cancel**(선택사항, 계정은 이미 등록됨).
8. "Set a home address" → **Skip**. restore opt-in → `KEYCODE_HOME`.
9. 확인: `adb -s "$SERIAL" shell dumpsys account | grep "Accounts:"` → `Accounts: 1`.

> 좌표는 해상도(Pixel6-2 = 1080×2424)·언어에 따라 다르니, 하드코딩 말고 매 화면 `uiautomator dump` 로 버튼 텍스트("SKIP"/"NEXT"/"I agree"/"Cancel"/"Skip")의 bounds 를 잡아 중심을 탭한다.

## 차단 시 폴백

- **2FA / 리캡차 / "이 기기에서 로그인할 수 없음"**: 자동화로 못 뚫으면 사용자에게 **수동 로그인 1회**를 요청한다(`! adb -s emulator-5554 shell am start -a android.settings.ADD_ACCOUNT_SETTINGS` 후 직접 입력).
- `google_apis`(비-playstore) 이미지는 일부 케이스에서 sign-in 을 차단할 수 있다. (실측: Pixel6-2/API33 에서는 정상 로그인됨.)
- **gms-storm 수정만 검증**하려면 로그인 불필요 — 로그아웃 상태로 Drive 를 돌려 gms 로 bounce 될 때 스톰/크래시 없이 처리되는지로도 검증된다([run-and-verify.md](run-and-verify.md)).

## Google 로그인 필요 앱 (MC 카탈로그, installed)

`com.google.android.apps.docs`(Drive) · `com.google.android.gm`(Gmail) · `com.google.android.youtube` · `com.google.android.apps.photos` · `com.google.android.calendar` · `com.google.android.contacts`. (Google Docs/Sheets/Slides 는 현재 미설치.)
