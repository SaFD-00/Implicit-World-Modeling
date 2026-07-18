---
name: setup-collector
description: Monkey-Collector 수집 환경 풀 셋업 — Pixel6-2 AVD 부팅(없으면 Pixel6/API33 생성, --recreate 재생성) + Python 환경/androguard preflight(uv sync·uv add androguard) + 카탈로그 APK 다운로드(F-Droid+PlayStore, 호스트 아키텍처에 맞는 ABI) + activities.json 추출 + 타깃 앱 디바이스 일괄 설치 + 첫 실행 온보딩 통과 + collector 앱(com.monkey.collector) 빌드/설치(JDK 17) + AccessibilityService 활성화 + server prefs(10.0.2.2:12345) 설정 + MediaProjection 동의 + Google 계정 로그인(필요 시, 로컬 비밀파일) + 더미데이터 시드(연락처·일정·노트·할일·음악 등) + sync-installed + 검증. **초기화(wipe) 시 1회 풀 셋업이며 모든 단계가 멱등** — 이미 셋업된 디바이스 재실행은 sentinel 가드로 no-op. --skip-* 로 단계 제외, --only 로 특정 패키지만, --force 로 강제
argument-hint: "[--avd NAME] [--recreate] [--no-wipe] [--source all|fdroid|playstore] [--abi ABI] [--only PKG,…] [--server-ip IP] [--server-port PORT] [--skip-download] [--skip-activities] [--skip-install-apps] [--skip-client] [--skip-login] [--skip-seed] [--force]"
---

# Setup Collector

> **⚠️ 호스트 전제: macOS + HW 가속(Apple HVF) 전용.** SDK 경로(`~/Library/Android/sdk`)·`emulator`/`avdmanager`/`adb` 직접 호출이 macOS 기준이다. **Apple Silicon 은 `arm64-v8a` system image + `arm64-v8a` APK** 가 필요하고, Intel Mac 은 `x86_64` 다. 본문의 `<root>` 는 Monkey-Collector 디렉터리(`/Users/bsw/Desktop/Projects/Implicit-World-Modeling/Monkey-Collector`)를 가리키며, 모든 `uv`/`python -m catalog.*`/`monkey-collect` 명령은 거기서 실행한다.

Monkey-Collector 데이터 수집을 시작할 수 있는 상태까지 **환경을 한 번에 준비**한다. **기본 동작**은 `Pixel6-2` AVD 부팅(없으면 생성) → Python 환경 preflight(`androguard` 보정 포함) → 카탈로그 APK 다운로드(F-Droid + PlayStore) → `activities.json` 추출 → 타깃 앱 디바이스 일괄 설치 → collector 앱 빌드/설치(JDK 17) → AccessibilityService 활성화 → server prefs(`10.0.2.2:12345`) 설정 → MediaProjection 동의 → Google 로그인(필요 시) → 더미데이터 시드 → `sync-installed` → 검증이다. `--skip-*` 로 단계를 빼고, `--only` 로 특정 패키지만 다룬다.

> **⭐ 멱등성 — "초기화(wipe)했을 때만 풀 셋업"**: 이 스킬은 **새로 초기화된 디바이스에서 1회 풀 셋업**을 수행하도록 설계됐다. 이미 셋업된 디바이스에서 다시 실행해도 **각 단계가 sentinel 가드로 no-op** 이 되어야 한다 — 앱 설치는 `pm list packages` 에 있으면 skip, client 는 미설치/구버전일 때만 빌드, Google 로그인은 `dumpsys account` 에 `type=com.google` 계정이 0개일 때만, 더미데이터는 앱별 sentinel(예: org.tasks `remoteId LIKE 'mg-seed-%'`) 부재 시에만 주입한다. 누락분만 채우고 나머지는 건너뛴다. 강제 재실행은 `--force`.

> **MobileGPT-V2 의 `setup-emulator` 와의 관계**: 구조(AVD 생성 §0-a / 부팅 §1 / 설치 / 접근성 / **더미데이터 시드** / 검증 / Troubleshooting)와 **더미데이터 시드 레시피**(연락처·일정·노트·할일·음악 등)를 차용하되, 대상이 다르다 — DroidTask/android_world 벤치마크가 아니라 **`catalog/apps.csv` 의 앱들**을, MobileGPT client 2종이 아니라 **Monkey-Collector client 1종(`com.monkey.collector`)** 을 다룬다. AVD `Pixel6-2` 는 setup-emulator 가 만든 Pixel6/API33 AVD 와 동일 스펙이며, 이미 존재하면 재사용한다. server IP 주입은 `update-client-ip` 와, 기본값 복원은 `reset-defaults` 와 메커니즘을 공유한다.

> **상세 참조(references/)** — 본 SKILL.md 는 오케스트레이션이고, 단계별 깊은 내용은 아래로 위임한다:
> - [references/install-sources.md](references/install-sources.md) — F-Droid/PlayStore 다운로드 소스별 절차·인덱스 이탈 대응·로컬 캐시 규약 (§3)
> - [references/app-first-run.md](references/app-first-run.md) — 12앱 첫 실행 온보딩 관찰·통과 커맨드·로그인 판정 (§5-a)
> - [references/app-first-run-batch2.md](references/app-first-run-batch2.md) — 신규 27앱(F-Droid 20+PlayStore standalone 7) 첫 실행 온보딩 (§5-a)
> - [references/install-matrix.md](references/install-matrix.md) — 카탈로그 앱별 설치가능/로그인/온보딩 커버리지 표 (§5-a)
> - [references/client-build.md](references/client-build.md) — gradle 프로젝트 레이아웃·`local.properties`·**JDK 17 빌드**·APK 경로·빌드 트러블슈팅 (§6)
> - [references/mediaprojection-accessibility.md](references/mediaprojection-accessibility.md) — MediaProjection 동의 플로우·**토큰 단발성**·ScreenStabilizer graceful-degrade·접근성 활성화 (§6-a/§6-c)
> - [references/google-login.md](references/google-login.md) — Google 로그인 필요 앱·**로컬 비밀파일 규약**·디바이스 로그인 절차 (§6-d)
> - [references/seed-helpers.md](references/seed-helpers.md) · [seed-pim.md](references/seed-pim.md) · [seed-notes-tasks.md](references/seed-notes-tasks.md) · [seed-media-misc.md](references/seed-media-misc.md) — 앱별 더미데이터 시드 (§6-e)
> - [references/run-and-verify.md](references/run-and-verify.md) — `monkey-collect run` 플래그·프로토콜·**알려진 실패모드↔수정 매핑**·smoke 검증 (§9)

## Options

`$ARGUMENTS` 를 파싱하여 아래 옵션을 처리:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--avd NAME` | `Pixel6-2` | 대상 AVD. Monkey-Collector `AdbClient` 는 기본 `Pixel6-2` 를 쓰지만 env `MC_AVD` 로 오버라이드 가능(`adb.py:13-14` — `REQUIRED_AVD_NAME = os.environ.get("MC_AVD", "Pixel6-2")`). 비기본 AVD 로 바꾸면 **수집 시(`monkey-collect run` 등)에도 `MC_AVD` 를 같은 이름으로 짝맞춰야** 한다 — 안 맞추면 AdbClient 가 해당 AVD 를 못 찾아 실패한다. 단 §6-d 로그인 검증 자체는 AdbClient 를 거치지 않으므로 `--avd` 만으로 충분하다 |
| `--recreate` | false | AVD 재생성 — 기존 AVD 를 `emu kill` 후 `avdmanager delete` 하고 canonical 스펙(Pixel6/API33/arm64-v8a)으로 다시 만든다. ⚠️ 데이터 전소. 이미지/디바이스 스펙 변경 시에만 |
| `--no-wipe` | false | 부팅 시 `-wipe-data` 생략. 미지정 시 기본으로 초기화한다 |
| `--source all\|fdroid\|playstore` | `all` | 카탈로그 APK 다운로드 소스(§3). `download_apks --source` 로 전달 |
| `--abi ABI` | (자동) | F-Droid 다운로드 ABI 필터(§3). 미지정 시 **호스트 아키텍처로 자동 결정**(Apple Silicon→`arm64-v8a`, Intel→`x86_64`). `download_apks --abi` 로 전달 |
| `--only PKG,…` | (없음) | 콤마 구분 package_id allowlist. §3 다운로드·§5 설치를 이 패키지들로 한정 |
| `--server-ip IP` | `10.0.2.2` | collector 앱 prefs 의 `server_ip`(§6-b). 원격 서버면 서버 LAN IP |
| `--server-port PORT` | `12345` | collector 앱 prefs 의 `server_port`(§6-b). 서버(`monkey-collect run --port`)와 일치해야 함 |
| `--skip-download` | false | §3 APK 다운로드 생략(기존 `catalog/apks/*.apk` 재사용) |
| `--skip-activities` | false | §4 `activities.json` 추출 생략 |
| `--skip-install-apps` | false | §5 타깃 앱 디바이스 설치 생략 |
| `--skip-client` | false | §6 collector 앱 빌드/설치 생략(이미 설치돼 있을 때) |
| `--skip-login` | false | §6-d Google 계정 로그인 생략(로그인 불필요하거나 이미 로그인됨) |
| `--skip-seed` | false | §6-e 더미데이터 시드 생략 |
| `--force` | false | §3 다운로드에서 이미 받은 APK 도 재다운로드(`download_apks --force`) 및 멱등 가드 무시 강제 재실행 |

> **단계 의존성**: §4(activities 추출)는 `catalog/apks/*.apk` 가 있어야 의미가 있고 §2 의 `androguard` 를 요구한다. §5(디바이스 설치)는 `catalog/apks/*.apk` 를 깐다. §6-a/§6-b 는 §6(또는 기존 설치)된 client 가 있어야 한다. `--skip-*` 로 일부만 돌릴 때 이 의존성을 고려한다.

## Process

### 0. 인자 파싱 — AVD·serial·ABI 결정

```bash
ROOT="/Users/bsw/Desktop/Projects/Implicit-World-Modeling/Monkey-Collector"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
EMU="$SDK/emulator/emulator"
AVDM="$SDK/cmdline-tools/latest/bin/avdmanager"
PKG="com.monkey.collector"
SVC="$PKG/$PKG.CollectorService"           # AccessibilityService 컴포넌트
APK_OUT="app/app/build/outputs/apk/debug/app-debug.apk"   # 중첩 app/app 주의

AVD_NAME="${AVD_ARG:-Pixel6-2}"
SERVER_IP="${SERVER_IP_ARG:-10.0.2.2}"; SERVER_PORT="${SERVER_PORT_ARG:-12345}"
SOURCE="${SOURCE_ARG:-all}"

# 호스트 아키텍처 → 기본 system image / ABI
case "$(uname -m)" in
  arm64) HOST_ABI="arm64-v8a"; SYS_IMG="system-images;android-33;google_apis;arm64-v8a"; DEV="pixel_6" ;;
  *)     HOST_ABI="x86_64";    SYS_IMG="system-images;android-33;google_apis;x86_64";    DEV="pixel_6" ;;
esac
ABI="${ABI_ARG:-$HOST_ABI}"   # --abi override
```

serial 은 §0-a/§1 이후 `Pixel6-2` 로부터 역추출한다(AdbClient 와 동일 규칙):

```bash
resolve_serial(){
  local s name
  while read -r s state; do
    case "$s" in emulator-*) ;; *) continue;; esac
    [ "$state" = "device" ] || continue
    name="$(adb -s "$s" emu avd name 2>/dev/null | tr -d '\r' | head -1)"
    [ "$name" = "$AVD_NAME" ] && { echo "$s"; return 0; }
  done < <(adb devices | sed '1d;/^$/d')
  return 1
}
```

### 0-a. AVD 생성/재생성 — Pixel 6 / API 33

`Pixel6-2` 를 canonical 스펙(device `pixel_6`, image `$SYS_IMG`)으로 보장한다. 기본은 **없을 때만 생성**, `--recreate` 면 삭제 후 재생성(⚠️ 데이터 전소). 이미 setup-emulator 로 만들어 둔 동명 AVD 가 있으면 그대로 쓴다.

```bash
ini="$HOME/.android/avd/$AVD_NAME.ini"
if [ "$RECREATE" = "true" ] && [ -e "$ini" ]; then
  s="$(resolve_serial)" && adb -s "$s" emu kill 2>/dev/null
  "$AVDM" delete avd -n "$AVD_NAME"; echo "deleted AVD $AVD_NAME (recreate)"
fi
if [ ! -e "$ini" ]; then
  # 이미지가 없으면 먼저: "$SDK/cmdline-tools/latest/bin/sdkmanager" "$SYS_IMG"
  echo no | "$AVDM" create avd -n "$AVD_NAME" -k "$SYS_IMG" -d "$DEV" --force
  echo "created AVD $AVD_NAME ($DEV / android-33 / $HOST_ABI)"
fi
```

> device `pixel_6`(1080×2400) / **API 33(google_apis)** 은 setup-emulator 의 canonical 과 동일하며, `google_apis` 이미지는 `adb root`/`run-as` 가 가능해 §6-b prefs 기록·폴백에 유리하다.

### 1. 에뮬레이터 기동

이미 떠 있으면(자동 감지) 재기동을 건너뛴다. 아니면 `-no-snapshot`(+기본 `-wipe-data`, `--no-wipe` 시 생략)로 부팅한 뒤 boot + PM 준비를 대기한다:

```bash
SERIAL="$(resolve_serial || true)"
if [ -z "$SERIAL" ]; then
  WIPE="-wipe-data"; [ "$NO_WIPE" = "true" ] && WIPE=""
  "$EMU" -avd "$AVD_NAME" -no-snapshot $WIPE &
  # serial 이 붙을 때까지 대기 후 재탐색
  for _ in $(seq 1 60); do SERIAL="$(resolve_serial || true)"; [ -n "$SERIAL" ] && break; sleep 2; done
fi
[ -z "$SERIAL" ] && { echo "ERROR: failed to boot $AVD_NAME"; exit 1; }

adb -s "$SERIAL" wait-for-device
while [ "$(adb -s "$SERIAL" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" != "1" ]; do sleep 2; done
# PM 준비 대기 — 생략하면 §5 첫 설치가 대량 실패한다
until adb -s "$SERIAL" shell pm list packages android 2>/dev/null | grep -q "package:android"; do sleep 2; done
echo "booted: $AVD_NAME ($SERIAL)"
```

### 2. Python 환경 preflight

`<root>` 에서 의존성을 보장한다. **`extract_activities.py` 가 쓰는 `androguard` 가 `pyproject.toml` 에 없어** 따로 추가해야 한다(멱등):

```bash
cd "$ROOT"
uv sync                                  # editable 설치 — download_apks 의 monkey_collector import + monkey-collect CLI
uv pip show androguard >/dev/null 2>&1 || uv add androguard   # 누락 의존성 보정 (§4 필요)
command -v adb >/dev/null || echo "WARN: adb not on PATH (using $SDK/platform-tools/adb 권장)"
uv run gplaydl --help >/dev/null 2>&1 || echo "WARN: gplaydl 미동작 — PlayStore 다운로드(§3) 실패 가능"
grep -q '^OPENROUTER_API_KEY=' .env 2>/dev/null \
  && echo "OPENROUTER_API_KEY 설정됨 (LLM 입력/그룹핑 사용 가능)" \
  || echo "INFO: OPENROUTER_API_KEY 미설정 — 수집은 동작하나 LLM 기능(--input-mode api / --screen-grouping on) 비활성"
```

> `uv add androguard` 는 `pyproject.toml`/`uv.lock` 에 의존성을 영구 추가한다. 일회성으로만 쓰려면 `uv pip install androguard` 로 대체할 수 있다(lock 미변경).

### 3. 카탈로그 APK 다운로드 (`--skip-download` 면 생략)

`catalog/download_apks.py` 로 `catalog/apks/{pkg}.apk` 를 받고 `catalog/apks/MISSING.md` 에 실패/시스템 앱을 기록한다. **F-Droid 필터 ABI 는 호스트와 일치**(`$ABI`)해야 설치 가능한 빌드를 받는다:

```bash
cd "$ROOT"
DL_ARGS=(--source "$SOURCE" --abi "$ABI")   # 배열 — zsh 기본 옵션(no SH_WORD_SPLIT)에서도 bash 와 동일하게 확장됨
[ "$HOST_ABI" = "arm64-v8a" ] && DL_ARGS+=(--playstore-arch arm64)
[ -n "$ONLY" ] && DL_ARGS+=(--only "$ONLY")
[ "$FORCE" = "true" ] && DL_ARGS+=(--force)
uv run python -m catalog.download_apks "${DL_ARGS[@]}"
echo "--- MISSING.md ---"; sed -n '1,20p' catalog/apks/MISSING.md 2>/dev/null
```

> **ABI 가 핵심**: `download_apks` 의 기본 `--abi` 는 `x86_64` 라, Apple Silicon 에서 그대로 두면 arm64 에뮬레이터에 못 까는 빌드를 받는다. §0 에서 `$ABI` 를 `arm64-v8a` 로 자동 설정한다(Intel 은 x86_64). PlayStore 는 base APK 만 저장(split 있으면 MISSING.md 에 경고). **System 소스 앱**(`com.android.settings`, `com.google.android.dialer`)은 다운로드 대상이 아니며(플랫폼 내장) MISSING.md 의 System 섹션에 기록된다.
>
> **F-Droid 인덱스 이탈 주의**: 카탈로그 등록 앱 일부(예 simplemobiletools 3종)가 F-Droid 인덱스에서 완전히 빠져 로컬 캐시도 디바이스도 없는 상태로 굳을 수 있다 — 이 경우 로컬 `catalog/apks/*.apk` 캐시가 유일한 방어선이다. 소스별 절차·실패유형·복원 옵션은 [references/install-sources.md](references/install-sources.md).

### 4. activity 카탈로그 추출 (`--skip-activities` 면 생략)

`catalog/apks/*.apk` 의 `AndroidManifest.xml` 을 androguard 로 파싱해 `catalog/activities.json` 을 만든다(coverage ground truth). §2 의 androguard 가 선행돼야 한다:

```bash
cd "$ROOT"
uv run python -m catalog.extract_activities
uv run python -c "import json;d=json.load(open('catalog/activities.json'));print(f'activities.json: {len(d)} apps / '+str(sum(len(v[\"activities\"]) for v in d.values()))+' activities')"
```

### 5. 타깃 앱 디바이스 일괄 설치 (`--skip-install-apps` 면 생략)

**일괄 설치 CLI 가 없으므로** `catalog/apks/*.apk` 를 직접 `adb install -r -g`(런타임 권한 grant 포함)로 깐다. 실패(ABI 불일치/Play split base-only/서명 충돌)는 카운트·로그만 하고 계속한다. `--only` 면 해당 패키지만. **멱등**: 이미 설치된 패키지는 skip(`--force` 면 재설치):

```bash
cd "$ROOT"
ok=0; fail=0; skip=0; failed=()
for apk in catalog/apks/*.apk; do
  [ -e "$apk" ] || { echo "no APKs in catalog/apks — run §3 first"; break; }
  base="$(basename "$apk" .apk)"
  if [ -n "$ONLY" ]; then case ",$ONLY," in *",$base,"*) ;; *) continue;; esac; fi
  # 멱등 가드: 이미 설치돼 있으면 skip (초기화 시에만 풀 설치)
  if [ "$FORCE" != "true" ] && adb -s "$SERIAL" shell pm list packages 2>/dev/null | grep -q "package:$base"; then
    skip=$((skip+1)); continue
  fi
  if adb -s "$SERIAL" install -r -g "$apk" >/tmp/_inst.log 2>&1; then
    ok=$((ok+1)); echo "[+] $base"
  else
    fail=$((fail+1)); failed+=("$base"); echo "[-] $base — $(tail -1 /tmp/_inst.log)"
  fi
done
echo "installed: $ok ok / $skip skip(already) / $fail failed${failed:+ (${failed[*]})}"
```

> **알려진 실패 유형**: `INSTALL_FAILED_NO_MATCHING_ABIS`(APK ABI ≠ 에뮬레이터 ABI — §3 의 `--abi` 가 호스트와 다르면 발생), Play split base-only APK 의 일부 미동작, 서명 충돌(`adb uninstall <pkg>` 후 재설치). System 앱은 APK 자체가 없어 자연히 대상에서 빠진다.

#### 5-a. 첫 실행 온보딩 관찰 (선택, 권장)

설치된 앱을 §6-e 시드보다 **먼저** 1회씩 실행해 온보딩(캐러셀/로그인유도/특수퍼미션)을 통과시켜 둔다 — 시드 후 첫 실행이면 온보딩 다이얼로그가 시드 데이터 위에 겹쳐 explorer 의 첫 스텝을 낭비시킬 수 있다. 각 앱: `am start -n <resolved-activity>`(§0-a 방식으로 resolve) → `uiautomator dump` 로 버튼 bounds 를 잡아 tap → `am force-stop` 후 재실행으로 멱등(재출현 없음) 확인. 12앱 전부의 실측 화면 시퀀스·통과 커맨드·판정(no-login-usable/partial)은 [references/app-first-run.md](references/app-first-run.md) — Google Maps 는 재검증으로 no-login-usable, YouTube Music 은 매 실행 재로그인유도(비멱등) 인 점에 주의. 신규 앱 온보딩은 [references/app-first-run-batch2.md](references/app-first-run-batch2.md), 앱별 설치자동화 가능 여부 한눈 요약은 [references/install-matrix.md](references/install-matrix.md) 참조. PlayStore 앱 대부분은 split-APK(App Bundle)라 base-APK 단독 설치 불가([install-matrix.md](install-matrix.md) 참조).

### 6. collector 앱 빌드/설치 (`--skip-client` 면 생략)

Monkey-Collector client(`com.monkey.collector`)를 `app/`(gradle 프로젝트 루트)에서 빌드해 설치한다. gradle 프로젝트는 `$ROOT/app`, app 모듈은 `$ROOT/app/app`, **APK 산출 경로는 중첩 `app/app/build/outputs/...`** 다. 상세·트러블슈팅은 [references/client-build.md](references/client-build.md). **멱등**: 미설치이거나 `--force` 일 때만 빌드/설치(빌드 캐시가 있으면 Gradle 이 up-to-date 처리):

```bash
# local.properties 보장 (gitignore 처리 — 추적 안 됨, 없으면 생성)
[ -f "$ROOT/app/local.properties" ] || printf 'sdk.dir=%s\n' "$SDK" > "$ROOT/app/local.properties"

if [ "$FORCE" = "true" ] || ! adb -s "$SERIAL" shell pm list packages 2>/dev/null | grep -q "package:$PKG"; then
  cd "$ROOT/app"
  chmod +x gradlew 2>/dev/null
  JAVA_HOME="$(/usr/libexec/java_home -v 17)" ./gradlew :app:assembleDebug   # AGP 8.2 → JDK 17 필수
  adb -s "$SERIAL" install -r -g "$ROOT/app/$APK_OUT"   # app/app/build/outputs/apk/debug/app-debug.apk
fi
adb -s "$SERIAL" shell pm list packages | grep -q "package:$PKG" && echo "installed client: $PKG"
```

> **전제(빌드)**: **AGP 8.2 / Gradle 8.x 는 빌드 실행에 JDK 17 이 필요**하다(`JAVA_HOME=$(/usr/libexec/java_home -v 17)`). 컴파일 산출 bytecode 타깃은 `jvmTarget=1.8`(compileSdk 34 / minSdk 28)이지만, **gradle 자체를 JDK 8 로 돌리면 빌드가 실패**한다. 빌드 중 `Warning: SDK ... XML versions up to 3 but ... version 4` 경고는 무해(cmdline-tools/스튜디오 버전차) — 무시한다. SDK 경로는 `local.properties` 의 `sdk.dir`(또는 `ANDROID_SDK_ROOT`/`ANDROID_HOME`)로 잡는다.

#### 6-a. AccessibilityService 활성화

client 의 `CollectorService` 토글을 켠다. `enabled_accessibility_services` 는 콜론 구분 리스트로 **덮어쓰기**되므로, 기존 값을 보존하며 우리 컴포넌트를 합친다(멱등):

```bash
adb -s "$SERIAL" shell pm list packages | grep -q "package:$PKG" || { echo "client 미설치 — §6 먼저"; }
CUR=$(adb -s "$SERIAL" shell settings get secure enabled_accessibility_services 2>/dev/null | tr -d '\r'); [ "$CUR" = "null" ] && CUR=""
JOINED=$(printf "%s:%s" "$CUR" "$SVC" | sed 's/^://; s/:$//' | tr ':' '\n' | awk 'NF && !seen[$0]++' | paste -sd: -)
adb -s "$SERIAL" shell settings put secure enabled_accessibility_services "$JOINED"
adb -s "$SERIAL" shell settings put secure accessibility_enabled 1
echo "enabled_accessibility_services -> $JOINED"
```

> 컴포넌트 = `com.monkey.collector/com.monkey.collector.CollectorService` (manifest `.CollectorService` 를 namespace 로 푼 것). 오타가 있으면 토글이 조용히 무시된다.

#### 6-b. server prefs 설정

`update-client-ip` 와 동일 메커니즘으로 `collector_settings` prefs 에 `server_ip`/`server_port` 를 기록한다(런타임 prefs — 재빌드 없음):

```bash
PREFS_DIR="/data/data/$PKG/shared_prefs"; PREFS="$PREFS_DIR/collector_settings.xml"
XML="<?xml version='1.0' encoding='utf-8' standalone='yes' ?>
<map>
    <string name=\"server_ip\">$SERVER_IP</string>
    <int name=\"server_port\" value=\"$SERVER_PORT\" />
</map>"
# 전체 run-as 절을 단일 로컬 문자열로 감싼다 — adb shell 은 원격으로 넘기기 전 argv 를 한 줄로 재조합하므로,
# 로컬에서 따옴표를 나눠 전달하면(예: sh -c "...") 그 quote 문자 자체가 살아남지 못해 mkdir 가 인자 없이 실행된다.
printf '%s\n' "$XML" | adb -s "$SERIAL" shell "run-as $PKG sh -c \"mkdir -p '$PREFS_DIR' && cat > '$PREFS'\""
adb -s "$SERIAL" shell am force-stop "$PKG"
# force-stop 은 client 소유 AccessibilityService 를 플랫폼이 강제로 disable 시킨다(§6-a 가 방금 켠 걸 되돌림) — §6-a 를 재적용해 되살린다
CUR=$(adb -s "$SERIAL" shell settings get secure enabled_accessibility_services 2>/dev/null | tr -d '\r'); [ "$CUR" = "null" ] && CUR=""
JOINED=$(printf "%s:%s" "$CUR" "$SVC" | sed 's/^://; s/:$//' | tr ':' '\n' | awk 'NF && !seen[$0]++' | paste -sd: -)
adb -s "$SERIAL" shell settings put secure enabled_accessibility_services "$JOINED"
adb -s "$SERIAL" shell settings put secure accessibility_enabled 1
echo "prefs -> server_ip=$SERVER_IP server_port=$SERVER_PORT (accessibility 재적용: $JOINED)"
```

> prefs 덮어쓰기는 멱등(항상 같은 값). `run-as` 가 안 되는 환경(release client)은 `adb root` 후 직접 기록으로 폴백. **순서 주의**: 이 force-stop 이후 §6-a 를 재적용하지 않으면 §6-c/§8 시점에 accessibility 가 꺼져 있는 것으로 관측된다 — 위 재적용 라인이 그 자기모순을 이 자리에서 바로 닫는다.

#### 6-c. MediaProjection 동의

화면 캡처(MediaProjection) 동의를 준다. 상세는 [references/mediaprojection-accessibility.md](references/mediaprojection-accessibility.md). 핵심:

```bash
# PROJECT_MEDIA appop 선허용 — google_apis 에뮬레이터에선 이걸로 동의 다이얼로그 없이 자동 승인되는 경우가 많다
adb -s "$SERIAL" shell appops set "$PKG" PROJECT_MEDIA allow 2>/dev/null
# MainActivity 열어 "Save & Ready" 1회 (멱등 — 이미 standby 면 토스트만 뜨고 끝)
adb -s "$SERIAL" shell am start -n "$PKG/.MainActivity" >/dev/null 2>&1; sleep 3
SAVE=$(adb -s "$SERIAL" exec-out uiautomator dump /dev/tty 2>/dev/null | tr '>' '>\n' | grep -iE 'SAVE (&|&amp;) READY' | sed -E 's/.*bounds="\[([0-9]+),([0-9]+)\]\[([0-9]+),([0-9]+)\]".*/\1 \2 \3 \4/' | head -1)
if [ -n "$SAVE" ]; then set -- $SAVE; adb -s "$SERIAL" shell input tap $(( ($1+$3)/2 )) $(( ($2+$4)/2 )); sleep 2; fi
# 시스템 "Start now" 다이얼로그가 뜨면 탭(appop 자동승인 시 안 뜸):
ST=$(adb -s "$SERIAL" exec-out uiautomator dump /dev/tty 2>/dev/null | tr '>' '>\n' | grep -iE 'text="(Start now|시작)"' | sed -E 's/.*bounds="\[([0-9]+),([0-9]+)\]\[([0-9]+),([0-9]+)\]".*/\1 \2 \3 \4/' | head -1)
if [ -n "$ST" ]; then set -- $ST; adb -s "$SERIAL" shell input tap $(( ($1+$3)/2 )) $(( ($2+$4)/2 )); fi
adb -s "$SERIAL" shell input keyevent KEYCODE_HOME
```

> **토큰 단발성 & graceful-degrade**: MediaProjection 동의 토큰은 모던 Android 에서 **단발성**이라 재설치/재시작마다 다시 받아야 한다. 단, 캡처 본체는 `AccessibilityService.takeScreenshot`(`canTakeScreenshot=true`)이고 MediaProjection 은 *stabilization* 용이라, 동의가 없거나 토큰이 만료돼도 **client 는 크래시 없이 degrade**(ScreenStabilizer 수정)하고 스크린샷은 계속 들어온다. `appops PROJECT_MEDIA allow` 가 먹으면 다이얼로그 없이 자동 승인된다.

#### 6-d. Google 계정 로그인 (`--skip-login` 면 생략)

Google 로그인이 필요한 앱(Drive/Docs/Gmail/YouTube/Photos 등)을 수집하려면 디바이스에 Google 계정이 있어야 한다. **멱등**: `dumpsys account` 에 `type=com.google` 계정이 이미 있으면 skip. 자격증명은 **커밋 금지 로컬 비밀파일**에서만 읽는다. 절차·UI 자동화 상세는 [references/google-login.md](references/google-login.md).

```bash
HAVE=$(adb -s "$SERIAL" shell dumpsys account 2>/dev/null | grep -c "Account {.*type=com.google}")
if [ "$SKIP_LOGIN" != "true" ] && [ "$HAVE" = "0" ] && [ -f "$ROOT/.secrets.local" ]; then
  set -a; . "$ROOT/.secrets.local"; set +a    # GOOGLE_ACCOUNT / GOOGLE_PASSWORD
  # Settings → Add account → Google → 이메일 → 비번 → I agree → Google services 백업 동의(ACCEPT) → 홈 복귀
  #   (인트로 SKIP·복구정보 Cancel 등 일부 화면은 계정/gms 상태에 따라 생략 — 조건부 흐름 상세는 references/google-login.md)
  # uiautomator dump 로 각 화면의 필드/버튼 좌표를 잡아 input text/tap (자세히는 references/google-login.md)
  echo "Google 로그인 시도: $GOOGLE_ACCOUNT (계정 0개 → 신규 등록)"
fi
adb -s "$SERIAL" shell dumpsys account 2>/dev/null | grep -E "Accounts:|Account \{.*type=com\.google\}" | head -3
```

> 자격증명은 `references/google-login.md` 의 `$GOOGLE_ACCOUNT`/`$GOOGLE_PASSWORD` **플레이스홀더**로만 문서화하고, 실제 값은 `Monkey-Collector/.secrets.local`(`.gitignore` 의 `**/*.secrets.local`)에만 둔다. **평문 비밀번호를 스킬/문서/repo 에 커밋하지 않는다.** 2FA/리캡차로 막히면 사용자에게 1회 수동 로그인을 요청한다.

#### 6-e. 더미데이터 시드 (`--skip-seed` 면 생략)

explorer 가 리스트/상세/편집 화면을 탐색할 콘텐츠가 있도록, 설치된 카탈로그 앱에 소량(앱당 3~5건)의 더미데이터를 넣는다. setup-emulator 의 시드 레시피를 재사용하며 **모두 멱등(앱별 sentinel)** 이다. 앱별 명령·스키마·gotcha 는:

- [references/seed-helpers.md](references/seed-helpers.md) — 공용 규약(heredoc sqlite3 · prelaunch+WAL · 멱등 sentinel · MediaStore rescan)
- [references/seed-pim.md](references/seed-pim.md) — 연락처 · 캘린더(Simple Calendar) · 시계 알람 · SMS
- [references/seed-notes-tasks.md](references/seed-notes-tasks.md) — Markor · Simple Notes · **org.tasks** · Joplin
- [references/seed-media-misc.md](references/seed-media-misc.md) — RetroMusic · OpenTracks · Broccoli · Expense

```bash
# 전제: adb root (google_apis 이미지). 각 앱은 has <pkg> && sentinel-미존재일 때만 시드.
adb -s "$SERIAL" root >/dev/null 2>&1
# 예: org.tasks (DB=…/databases/database, remoteId UNIQUE 로 멱등). 전체 레시피는 references/seed-*.md
# (Monkey-Collector 카탈로그 설치 교집합: Joplin·Markor·org.tasks·Simple Calendar·Simple Contacts·RetroMusic·OpenTracks)
```

> 시드 레시피는 setup-emulator 에서 Pixel6/API36 으로 실삽입·UI 검증된 것을 가져왔고, **Pixel6-2/API33 에서도 동일하게 동작 재확인**(연락처 5·일정 4·Markor 4·org.tasks 4·RetroMusic 3·OpenTracks 3·Joplin 3). RetroMusic 음악은 호스트 `ffmpeg` 필요(없으면 그 앱만 skip). Simple Notes·Simple Clock·SMS·Broccoli·Expense 는 현재 MC 카탈로그 미포함이라 자연 skip.

### 7. installed 동기화

디바이스 설치 상태를 `catalog/apps.csv` 의 `installed` 컬럼에 반영한다(이후 `monkey-collect run --apps all` 의 대상 큐 구성):

```bash
cd "$ROOT"
uv run monkey-collect sync-installed
```

### 8. 검증

```bash
cd "$ROOT"
echo "=== AVD ($SERIAL) ==="
adb -s "$SERIAL" emu avd name 2>/dev/null | tr -d '\r' | head -1

echo "=== 타깃 앱 설치 수 ==="
adb -s "$SERIAL" shell pm list packages | wc -l | tr -d ' '

echo "=== activities.json ==="
uv run python -c "import json;d=json.load(open('catalog/activities.json'));print(f'{len(d)} apps / '+str(sum(len(v[\"activities\"]) for v in d.values()))+' activities')" 2>/dev/null || echo "(없음)"

echo "=== client + Accessibility ==="
adb -s "$SERIAL" shell pm list packages | grep "package:com.monkey.collector" || echo "client 미설치"
adb -s "$SERIAL" shell settings get secure enabled_accessibility_services | tr ':' '\n' | grep CollectorService || echo "Accessibility 미활성"
echo -n "accessibility_enabled="; adb -s "$SERIAL" shell settings get secure accessibility_enabled | tr -d '\r'

echo "=== server prefs ==="
adb -s "$SERIAL" shell run-as com.monkey.collector cat /data/data/com.monkey.collector/shared_prefs/collector_settings.xml 2>/dev/null | tr -d '\r'

echo "=== Google 계정 (§6-d) ==="
adb -s "$SERIAL" shell dumpsys account 2>/dev/null | grep -E "Accounts:|Account \{.*type=com\.google\}" | head -3

echo "=== 더미데이터 시드 (§6-e, 설치된 것만) ==="
adb -s "$SERIAL" shell "sqlite3 /data/data/org.tasks/databases/database \"SELECT COUNT(*) FROM tasks WHERE remoteId LIKE 'mg-seed-%';\"" 2>/dev/null | tr -d '\r' | sed 's/^/  org.tasks tasks: /'
adb -s "$SERIAL" shell "ls /sdcard/Documents/markor/ 2>/dev/null | wc -l" | tr -d '\r' | sed 's/^/  markor files: /'

echo "=== installed 동기화 (apps.csv) ==="
awk -F, 'NR>1{print $NF}' catalog/apps.csv | sort | uniq -c

echo "=== 다운로드 실패/시스템 앱 ==="
sed -n '1,12p' catalog/apks/MISSING.md 2>/dev/null
```

### 9. 라이브 스모크 검증 (선택, 권장)

셋업이 실제 수집까지 동작하는지 대표 앱으로 1회 end-to-end 확인한다. 상세·성공판정은 [references/run-and-verify.md](references/run-and-verify.md):

```bash
cd "$ROOT"
adb -s "$SERIAL" logcat -b crash -c   # crash 버퍼 비우기
uv run monkey-collect run --apps net.gsantner.markor com.google.android.apps.docs \
  --steps 100 --new-session --input-mode api --screen-grouping on --port 12345
# 성공판정: 두 앱 모두 data/<pkg>/page_graph 노드≥2, 서버 로그에 연속 signal-timeout/외부앱 스톰 없음,
#           crash 버퍼에 com.monkey.collector 프로세스 사망 없음(아래)
adb -s "$SERIAL" logcat -b crash -d | grep -i "com.monkey.collector" || echo "no client crash (good)"
ls data/net.gsantner.markor/page_graph* data/com.google.android.apps.docs/page_graph* 2>/dev/null
```

> **org.tasks 는 대표 예시에서 제외**: 이 환경에서 `org.tasks` 를 `monkey-collect run` 으로 수집하면 시드된 `TaskListActivity` 화면에서도 Step 0 무한 정체(interactable-element 판정 실패 → 외부앱 스톰)가 실측됐다 — 온보딩/시드 자체는 정상이라 collector 코드(screen_matching) 레벨 이슈로 추정. 미해결 상태이며 아래 Troubleshooting 표 참조.

> **다음 단계**: 셋업 후 `uv run monkey-collect run --apps all` 으로 전체 수집을 시작한다 (기본 1500스텝). LLM 입력/그룹핑을 쓰려면 `.env` 에 `OPENROUTER_API_KEY` 를 채우고 `--input-mode api --screen-grouping on` 을 준다.

## Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| `RuntimeError: AVD Pixel6-2 not found` (수집 시) | AVD 미부팅 또는 이름 불일치 | §0-a/§1 로 `Pixel6-2` 부팅. `AdbClient` 는 기본 `Pixel6-2` 를 쓰되 env `MC_AVD` 로 오버라이드 가능(`adb.py:13-14`) — `--avd` 로 비기본 AVD 를 썼다면 수집 시에도 `MC_AVD` 를 그 이름으로 맞춰야 한다 |
| `ModuleNotFoundError: androguard` (§4) | androguard 미설치 | §2 의 `uv add androguard`(또는 `uv pip install androguard`) 실행 |
| `ModuleNotFoundError: monkey_collector` (§3) | editable 미설치 | `<root>` 에서 `uv sync` 후 `uv run python -m catalog.download_apks` |
| `INSTALL_FAILED_NO_MATCHING_ABIS` (§5) | APK ABI ≠ 에뮬레이터 ABI | §3 을 호스트 ABI(`--abi $ABI`)로 다시 다운로드. Apple Silicon=arm64-v8a, Intel=x86_64 |
| PlayStore 다운로드 실패 (§3) | gplaydl 토큰/네트워크 | `uv run gplaydl --help` 동작 확인, 재시도. F-Droid 만 받으려면 `--source fdroid` |
| `run-as: unknown package` (§6-b) | client 미설치 | §6 먼저(또는 `--skip-client` 해제). 폴백 `adb root` 로 직접 기록 |
| gradle 빌드 실패 `Unsupported class file major version` / toolchain (§6) | JDK 8/11 로 빌드 시도 | **JDK 17 필수**: `JAVA_HOME=$(/usr/libexec/java_home -v 17) ./gradlew :app:assembleDebug`. AGP 8.2 요구사항 |
| gradle `SDK ... XML version 4` 경고 (§6) | cmdline-tools/스튜디오 버전차 | **무해 — 무시**. 빌드는 정상 진행됨 |
| 빌드는 됐는데 디바이스에 옛 동작/크래시 (§6) | 패치 소스로 **재빌드/재설치 안 함** | `install -r` 로 최신 APK 재설치 후 §6-c MediaProjection 재동의(토큰 단발성) |
| 수집이 서버에 연결 안 됨 | prefs IP/port ≠ 서버, 또는 접근성/캡처 미허용 | §6-b prefs 와 `monkey-collect run --port` 일치 확인, §6-a 토글 확인, §6-c 캡처 허용. **재연결 시 client 가 죽은 서버 소켓을 붙든 채 새 서버에 자동 재연결하지 않는 경우**: 기존 해법은 client `force-stop`(§6-b) 후 재연결이나, **force-stop 없이 accessibility off→on 토글만으로도 재연결 성공**이 1회 관측됨(추가 확인된 대안 — force-stop 이 §6-a 를 지우는 부작용을 피할 수 있어 우선 시도 가치 있음) |
| Google 로그인 단계가 막힘 (§6-d) | 2FA/리캡차/`google_apis` 인증 차단 | `references/google-login.md` 절차로 재시도. 막히면 사용자 수동 로그인 1회. gms-storm 수정만 검증할 땐 로그아웃 상태 bounce 로도 가능 |
| `org.tasks` 수집 시 Step 0 에서 영구 정체(§9) | interactable-element 판정이 시드된 `TaskListActivity` 화면을 "no interactable" 로 declining → back press 가 앱 종료로 오인돼 외부앱 스톰 반복 | **미해결** — collector UI 파서/screen_matching 코드 레벨 이슈로 추정, 별도 트래킹 필요. §9 대표 스모크 예시는 `net.gsantner.markor` 로 대체됨(실측 통과: steps=90/actions=110/nodes=16/edges=20) |
| 더미데이터가 앱에 안 보임 (§6-e) | sentinel 충돌/스키마차/권한 | `adb root` 확인, `references/seed-*.md` 의 앱별 gotcha(예 Clock `SCHEDULE_EXACT_ALARM`, RetroMusic 곡 길이≥30s) 확인 |
| 부팅 직후 §5 대량 설치 실패 | PM 미준비 | §1 의 `pm list packages android` 대기 루프 통과 후 재실행 |

## 호스트 아키텍처 주의 (Apple Silicon)

QEMU2 는 aarch64 호스트(Apple M-series)에서 x86_64 system image 를 못 돌린다. **Apple Silicon = `arm64-v8a` 이미지 + `arm64-v8a` APK**, Intel Mac = `x86_64`. §0 이 `uname -m` 으로 `$SYS_IMG`/`$ABI` 를 자동 결정하므로, AVD 생성(§0-a)·F-Droid 다운로드(§3)·디바이스 설치(§5)가 한 ABI 로 정렬된다. `--abi` 로 강제 override 할 수 있으나, 에뮬레이터 이미지와 APK ABI 가 어긋나면 §5 가 `NO_MATCHING_ABIS` 로 실패한다.

> **연관 스킬**: server IP/port 변경은 `update-client-ip <ip> [--port P]`, 기본값(`10.0.2.2:12345`) 복원은 `reset-defaults`. 둘 다 §6-b 와 동일한 on-device prefs 기록 메커니즘을 쓴다.
