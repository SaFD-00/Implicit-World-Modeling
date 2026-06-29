---
name: setup-collector
description: Monkey-Collector 수집 환경 풀 셋업 — MobileGPT-V2-2 AVD 부팅(없으면 Pixel6/API33 생성, --recreate 재생성) + Python 환경/androguard preflight(uv sync·uv add androguard) + 카탈로그 APK 다운로드(F-Droid+PlayStore, 호스트 아키텍처에 맞는 ABI) + activities.json 추출 + 타깃 앱 디바이스 일괄 설치 + collector 앱(com.monkey.collector) 빌드/설치 + AccessibilityService 활성화 + server prefs(10.0.2.2:12345) 설정 + sync-installed + 검증. 모두 기본, --skip-* 로 단계 제외, --only 로 특정 패키지만
argument-hint: "[--avd NAME] [--recreate] [--no-wipe] [--source all|fdroid|playstore] [--abi ABI] [--only PKG,…] [--server-ip IP] [--server-port PORT] [--skip-download] [--skip-activities] [--skip-install-apps] [--skip-client] [--force]"
---

# Setup Collector

> **⚠️ 호스트 전제: macOS + HW 가속(Apple HVF) 전용.** SDK 경로(`~/Library/Android/sdk`)·`emulator`/`avdmanager`/`adb` 직접 호출이 macOS 기준이다. **Apple Silicon 은 `arm64-v8a` system image + `arm64-v8a` APK** 가 필요하고, Intel Mac 은 `x86_64` 다. 본문의 `<root>` 는 Monkey-Collector 디렉터리(`/Users/bsw/Desktop/Project/Implicit-World-Modeling/Monkey-Collector`)를 가리키며, 모든 `uv`/`python -m catalog.*`/`monkey-collect` 명령은 거기서 실행한다.

Monkey-Collector 데이터 수집을 시작할 수 있는 상태까지 **환경을 한 번에 준비**한다. **기본 동작**은 `MobileGPT-V2-2` AVD 부팅(없으면 생성) → Python 환경 preflight(`androguard` 보정 포함) → 카탈로그 APK 다운로드(F-Droid + PlayStore) → `activities.json` 추출 → 타깃 앱 디바이스 일괄 설치 → collector 앱 빌드/설치 → AccessibilityService 활성화 → server prefs(`10.0.2.2:12345`) 설정 → `sync-installed` → 검증이다. `--skip-*` 로 단계를 빼고, `--only` 로 특정 패키지만 다룬다.

> **MobileGPT-V2 의 `setup-emulator` 와의 관계**: 구조(AVD 생성 §0-a / 부팅 §1 / 설치 / 접근성 / 검증 / Troubleshooting)를 그대로 차용하되, 대상이 다르다 — DroidTask/android_world 벤치마크가 아니라 **`catalog/apps.csv` 의 앱들**을, MobileGPT client 2종이 아니라 **Monkey-Collector client 1종(`com.monkey.collector`)** 을 다룬다. AVD `MobileGPT-V2-2` 는 setup-emulator 가 만든 Pixel6/API33 AVD 와 동일 스펙이며, 이미 존재하면 재사용한다. server IP 주입은 `update-client-ip` 와, 기본값 복원은 `reset-defaults` 와 메커니즘을 공유한다.

## Options

`$ARGUMENTS` 를 파싱하여 아래 옵션을 처리:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--avd NAME` | `MobileGPT-V2-2` | 대상 AVD. **Monkey-Collector `AdbClient` 가 이 이름을 하드코딩**(`adb.py:15`)하므로 바꾸면 수집이 깨진다 — 특별한 이유 없으면 그대로 둔다 |
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
| `--force` | false | §3 다운로드에서 이미 받은 APK 도 재다운로드(`download_apks --force`) |

> **단계 의존성**: §4(activities 추출)는 `catalog/apks/*.apk` 가 있어야 의미가 있고 §2 의 `androguard` 를 요구한다. §5(디바이스 설치)는 `catalog/apks/*.apk` 를 깐다. §6-a/§6-b 는 §6(또는 기존 설치)된 client 가 있어야 한다. `--skip-*` 로 일부만 돌릴 때 이 의존성을 고려한다.

## Process

### 0. 인자 파싱 — AVD·serial·ABI 결정

```bash
ROOT="/Users/bsw/Desktop/Project/Implicit-World-Modeling/Monkey-Collector"
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
EMU="$SDK/emulator/emulator"
AVDM="$SDK/cmdline-tools/latest/bin/avdmanager"
PKG="com.monkey.collector"
SVC="$PKG/$PKG.CollectorService"           # AccessibilityService 컴포넌트
APK_OUT="app/app/build/outputs/apk/debug/app-debug.apk"   # 중첩 app/app 주의

AVD_NAME="${AVD_ARG:-MobileGPT-V2-2}"
SERVER_IP="${SERVER_IP_ARG:-10.0.2.2}"; SERVER_PORT="${SERVER_PORT_ARG:-12345}"
SOURCE="${SOURCE_ARG:-all}"

# 호스트 아키텍처 → 기본 system image / ABI
case "$(uname -m)" in
  arm64) HOST_ABI="arm64-v8a"; SYS_IMG="system-images;android-33;google_apis;arm64-v8a"; DEV="pixel_6" ;;
  *)     HOST_ABI="x86_64";    SYS_IMG="system-images;android-33;google_apis;x86_64";    DEV="pixel_6" ;;
esac
ABI="${ABI_ARG:-$HOST_ABI}"   # --abi override
```

serial 은 §0-a/§1 이후 `MobileGPT-V2-2` 로부터 역추출한다(AdbClient 와 동일 규칙):

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

`MobileGPT-V2-2` 를 canonical 스펙(device `pixel_6`, image `$SYS_IMG`)으로 보장한다. 기본은 **없을 때만 생성**, `--recreate` 면 삭제 후 재생성(⚠️ 데이터 전소). 이미 setup-emulator 로 만들어 둔 동명 AVD 가 있으면 그대로 쓴다.

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
DL_ARGS="--source $SOURCE --abi $ABI"
[ "$HOST_ABI" = "arm64-v8a" ] && DL_ARGS="$DL_ARGS --playstore-arch arm64"
[ -n "$ONLY" ] && DL_ARGS="$DL_ARGS --only $ONLY"
[ "$FORCE" = "true" ] && DL_ARGS="$DL_ARGS --force"
uv run python -m catalog.download_apks $DL_ARGS
echo "--- MISSING.md ---"; sed -n '1,20p' catalog/apks/MISSING.md 2>/dev/null
```

> **ABI 가 핵심**: `download_apks` 의 기본 `--abi` 는 `x86_64` 라, Apple Silicon 에서 그대로 두면 arm64 에뮬레이터에 못 까는 빌드를 받는다. §0 에서 `$ABI` 를 `arm64-v8a` 로 자동 설정한다(Intel 은 x86_64). PlayStore 는 base APK 만 저장(split 있으면 MISSING.md 에 경고). **System 소스 앱**(`com.android.settings`, `com.google.android.dialer`)은 다운로드 대상이 아니며(플랫폼 내장) MISSING.md 의 System 섹션에 기록된다.

### 4. activity 카탈로그 추출 (`--skip-activities` 면 생략)

`catalog/apks/*.apk` 의 `AndroidManifest.xml` 을 androguard 로 파싱해 `catalog/activities.json` 을 만든다(coverage ground truth). §2 의 androguard 가 선행돼야 한다:

```bash
cd "$ROOT"
uv run python -m catalog.extract_activities
uv run python -c "import json;d=json.load(open('catalog/activities.json'));print(f'activities.json: {len(d)} apps / '+str(sum(len(v[\"activities\"]) for v in d.values()))+' activities')"
```

### 5. 타깃 앱 디바이스 일괄 설치 (`--skip-install-apps` 면 생략)

**일괄 설치 CLI 가 없으므로** `catalog/apks/*.apk` 를 직접 `adb install -r -g`(런타임 권한 grant 포함)로 깐다. 실패(ABI 불일치/Play split base-only/서명 충돌)는 카운트·로그만 하고 계속한다. `--only` 면 해당 패키지만:

```bash
cd "$ROOT"
ok=0; fail=0; failed=()
for apk in catalog/apks/*.apk; do
  [ -e "$apk" ] || { echo "no APKs in catalog/apks — run §3 first"; break; }
  base="$(basename "$apk" .apk)"
  if [ -n "$ONLY" ]; then case ",$ONLY," in *",$base,"*) ;; *) continue;; esac; fi
  if adb -s "$SERIAL" install -r -g "$apk" >/tmp/_inst.log 2>&1; then
    ok=$((ok+1)); echo "[+] $base"
  else
    fail=$((fail+1)); failed+=("$base"); echo "[-] $base — $(tail -1 /tmp/_inst.log)"
  fi
done
echo "installed: $ok ok / $fail failed${failed:+ (${failed[*]})}"
```

> **알려진 실패 유형**: `INSTALL_FAILED_NO_MATCHING_ABIS`(APK ABI ≠ 에뮬레이터 ABI — §3 의 `--abi` 가 호스트와 다르면 발생), Play split base-only APK 의 일부 미동작, 서명 충돌(`adb uninstall <pkg>` 후 재설치). System 앱은 APK 자체가 없어 자연히 대상에서 빠진다.

### 6. collector 앱 빌드/설치 (`--skip-client` 면 생략)

Monkey-Collector client(`com.monkey.collector`)를 `app/` 에서 빌드해 설치한다. **APK 산출 경로는 중첩 `app/app/...`** 다:

```bash
cd "$ROOT/app"
chmod +x gradlew 2>/dev/null
./gradlew assembleDebug
adb -s "$SERIAL" install -r -g "$ROOT/app/$APK_OUT"   # app/app/build/outputs/apk/debug/app-debug.apk
adb -s "$SERIAL" shell pm list packages | grep -q "package:$PKG" && echo "installed client: $PKG"
```

> **전제**: `JAVA_HOME` + Android SDK 로 `gradlew` 가 동작해야 한다(compileSdk 34 / minSdk 28 / JDK 8). 빌드 캐시가 있으면 Gradle 이 up-to-date 처리한다.

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
printf '%s\n' "$XML" | adb -s "$SERIAL" shell run-as "$PKG" sh -c "mkdir -p '$PREFS_DIR' && cat > '$PREFS'"
adb -s "$SERIAL" shell am force-stop "$PKG"
echo "prefs -> server_ip=$SERVER_IP server_port=$SERVER_PORT"
# MediaProjection 동의 best-effort 선허용 (버전에 따라 무시될 수 있음)
adb -s "$SERIAL" shell appops set "$PKG" PROJECT_MEDIA allow 2>/dev/null
```

> **MediaProjection 한계**: 화면 캡처 동의는 본질적으로 **세션마다 뜨는 UI 다이얼로그**라 adb 로 완전 자동화되지 않는다. `appops … PROJECT_MEDIA allow` 는 best-effort 이며, 안 먹으면 **수집 첫 세션에서 앱을 열어 "Save & Ready" → 화면 캡처 허용**을 1회 수동으로 해줘야 한다(이후 `CollectorService` 가 standby 로 서버 연결 유지). 서버 IP/포트는 이미 prefs 에 박혀 있어 입력은 불필요.

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

echo "=== installed 동기화 (apps.csv) ==="
awk -F, 'NR>1{print $NF}' catalog/apps.csv | sort | uniq -c

echo "=== 다운로드 실패/시스템 앱 ==="
sed -n '1,12p' catalog/apks/MISSING.md 2>/dev/null
```

> **다음 단계**: 셋업 후 `uv run monkey-collect run --apps all --steps 100` 으로 수집을 시작한다. LLM 입력/그룹핑을 쓰려면 `.env` 에 `OPENROUTER_API_KEY` 를 채우고 `--input-mode api --screen-grouping on` 을 준다.

## Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| `RuntimeError: AVD MobileGPT-V2-2 not found` (수집 시) | AVD 미부팅 또는 이름 불일치 | §0-a/§1 로 `MobileGPT-V2-2` 부팅. `AdbClient` 는 이 이름을 하드코딩(`adb.py:15`) |
| `ModuleNotFoundError: androguard` (§4) | androguard 미설치 | §2 의 `uv add androguard`(또는 `uv pip install androguard`) 실행 |
| `ModuleNotFoundError: monkey_collector` (§3) | editable 미설치 | `<root>` 에서 `uv sync` 후 `uv run python -m catalog.download_apks` |
| `INSTALL_FAILED_NO_MATCHING_ABIS` (§5) | APK ABI ≠ 에뮬레이터 ABI | §3 을 호스트 ABI(`--abi $ABI`)로 다시 다운로드. Apple Silicon=arm64-v8a, Intel=x86_64 |
| PlayStore 다운로드 실패 (§3) | gplaydl 토큰/네트워크 | `uv run gplaydl --help` 동작 확인, 재시도. F-Droid 만 받으려면 `--source fdroid` |
| `run-as: unknown package` (§6-b) | client 미설치 | §6 먼저(또는 `--skip-client` 해제). 폴백 `adb root` 로 직접 기록 |
| 수집이 서버에 연결 안 됨 | prefs IP/port ≠ 서버, 또는 접근성/캡처 미허용 | §6-b prefs 와 `monkey-collect run --port` 일치 확인, §6-a 토글 확인, 첫 세션 화면 캡처 허용 |
| 부팅 직후 §5 대량 설치 실패 | PM 미준비 | §1 의 `pm list packages android` 대기 루프 통과 후 재실행 |

## 호스트 아키텍처 주의 (Apple Silicon)

QEMU2 는 aarch64 호스트(Apple M-series)에서 x86_64 system image 를 못 돌린다. **Apple Silicon = `arm64-v8a` 이미지 + `arm64-v8a` APK**, Intel Mac = `x86_64`. §0 이 `uname -m` 으로 `$SYS_IMG`/`$ABI` 를 자동 결정하므로, AVD 생성(§0-a)·F-Droid 다운로드(§3)·디바이스 설치(§5)가 한 ABI 로 정렬된다. `--abi` 로 강제 override 할 수 있으나, 에뮬레이터 이미지와 APK ABI 가 어긋나면 §5 가 `NO_MATCHING_ABIS` 로 실패한다.

> **연관 스킬**: server IP/port 변경은 `update-client-ip <ip> [--port P]`, 기본값(`10.0.2.2:12345`) 복원은 `reset-defaults`. 둘 다 §6-b 와 동일한 on-device prefs 기록 메커니즘을 쓴다.
