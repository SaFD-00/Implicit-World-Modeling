---
name: update-client-ip
description: Monkey-Collector Android client(com.monkey.collector) 의 server IP/port 를 on-device SharedPreferences(collector_settings) 에 직접 기록해 적용 — 런타임에 읽는 값이라 재빌드/재설치 불필요. 원격 토폴로지(에뮬레이터≠서버)면 서버의 LAN IP 를 박는다. <ip> 만 주면 port 는 12345 유지, --serial 미지정 시 MobileGPT-V2-2 AVD 자동 탐색
argument-hint: "<ip-address> [--port P] [--serial S]"
---

# Update Client IP

> **⚠️ 호스트 전제: macOS + 부팅된 `MobileGPT-V2-2` 에뮬레이터.** SDK 경로(`~/Library/Android/sdk`)·`adb` 직접 호출이 macOS 기준이다. **HOST IP 선택**: 에뮬레이터와 Python 서버가 같은 머신이면 `10.0.2.2`(에뮬레이터→호스트 루프백)면 충분하다. 서버가 다른 머신(원격 토폴로지)이면 **서버의 LAN IP** 를 줘야 한다 — 에뮬레이터 안에서 `127.0.0.1`/`localhost` 는 에뮬레이터 자신을, `10.0.2.2` 는 에뮬레이터를 띄운 호스트를 가리킨다.

Monkey-Collector 의 단일 Android client(`com.monkey.collector`)가 접속할 **server IP/port** 를 바꾼다. client 는 이 값을 컴파일타임 상수가 아니라 **on-device SharedPreferences `collector_settings`** (`server_ip`/`server_port`) 에 보관하고 `CollectorService` 가 **런타임에 읽으므로, 값만 바꾸면 재빌드·재설치가 필요 없다.**

> **MobileGPT-V2 의 `update-client-ip` 와의 핵심 차이**: 그쪽은 `MobileGPTGlobal.java` 의 `HOST_IP`/`HOST_PORT` 가 **컴파일타임 `static final` 상수**라 client 2종을 `assembleDebug` 로 **재빌드 후 재설치**해야 했다. Monkey-Collector 는 client 1종 + **런타임 prefs** 라, 이 커맨드는 **adb 로 prefs 파일만 기록**하고 앱을 force-stop 해 새 값을 읽게 한다(빌드·설치 단계 없음). `reset-defaults` 가 이 커맨드의 역방향이다(IP/port 를 기본 `10.0.2.2`/`12345` 로 되돌림).

## Options

`$ARGUMENTS` 를 파싱하여 아래 옵션을 처리:

| 옵션 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `<ip-address>` | ✓ | — | 첫 번째 positional 인자. 새 `server_ip` (예: `192.168.0.112`). IPv4 dotted-quad 검증. `10.0.2.2` 면 에뮬레이터→호스트 루프백 |
| `--port P` | | `12345` | 새 `server_port`. 1–65535 검증. 미지정 시 **현재 prefs 값 유지**(prefs 가 없으면 12345). 서버는 `monkey-collect run --port` 와 동일 port 에서 listen 해야 함 |
| `--serial S` | | (자동) | 대상 adb 시리얼. 미지정 시 `MobileGPT-V2-2` AVD 를 `adb devices` + `emu avd name` 으로 역추출 |

**검증** (실패 시 즉시 중단, 사용법 출력):
- `<ip-address>` 누락 → 중단.
- IPv4 dotted-quad(`A.B.C.D`, 각 옥텟 0–255) 아니면 → 중단.
- `--port` 가 1–65535 밖이면 → 중단.
- 대상 serial 을 못 찾으면(에뮬레이터 미부팅) → 중단하고 `setup-collector`(또는 `emulator -avd MobileGPT-V2-2`) 안내.

## Target

| Client | applicationId | prefs 파일 (on-device) | 키 |
|--------|---------------|------------------------|----|
| Monkey-Collector | `com.monkey.collector` | `/data/data/com.monkey.collector/shared_prefs/collector_settings.xml` | `server_ip`(string) · `server_port`(int) |

> client 가 디바이스에 설치돼 있어야 한다(`setup-collector` 또는 `app/` 에서 `assembleDebug` 후 `adb install`). 미설치면 `run-as` 가 `unknown package` 로 실패한다.

## Process

### 1. 인자 파싱 & serial 결정

```bash
SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}"
PKG="com.monkey.collector"
PREFS_DIR="/data/data/$PKG/shared_prefs"
PREFS="$PREFS_DIR/collector_settings.xml"

IP="$1"   # 첫 positional
# --port / --serial 파싱은 본문 컨벤션대로. PORT 미지정이면 빈 값으로 두고 §2 에서 현재값 채움.

# IPv4 검증
echo "$IP" | grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$' || { echo "ERROR: invalid IPv4: $IP"; exit 1; }
for o in ${IP//./ }; do [ "$o" -le 255 ] 2>/dev/null || { echo "ERROR: octet out of range: $IP"; exit 1; }; done

# serial 자동 탐색 (--serial 미지정 시) — MobileGPT-V2-2 만 대상
if [ -z "$SERIAL" ]; then
  while read -r serial state; do
    case "$serial" in emulator-*) ;; *) continue;; esac
    [ "$state" = "device" ] || continue
    name="$(adb -s "$serial" emu avd name 2>/dev/null | tr -d '\r' | head -1)"
    [ "$name" = "MobileGPT-V2-2" ] && { SERIAL="$serial"; break; }
  done < <(adb devices | sed '1d;/^$/d')
fi
[ -z "$SERIAL" ] && { echo "ERROR: MobileGPT-V2-2 emulator not found. Boot it (setup-collector) first."; exit 1; }
```

### 2. 현재 값 확인

`run-as`(디버그 빌드라 root 불요)로 현재 prefs 를 읽어 표시. `--port` 미지정이면 여기서 현재 `server_port` 를 채운다(없으면 12345):

```bash
CUR_XML="$(adb -s "$SERIAL" shell run-as "$PKG" cat "$PREFS" 2>/dev/null | tr -d '\r')"
CUR_IP="$(printf '%s' "$CUR_XML"   | grep -oE 'name="server_ip">[^<]*'      | sed 's/.*>//')"
CUR_PORT="$(printf '%s' "$CUR_XML" | grep -oE 'name="server_port" value="[0-9]+"' | grep -oE '[0-9]+')"
echo "current: server_ip=${CUR_IP:-<unset>} server_port=${CUR_PORT:-<unset>}"
[ -z "$PORT" ] && PORT="${CUR_PORT:-12345}"
echo "$PORT" | grep -Eq '^[0-9]+$' && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { echo "ERROR: invalid port: $PORT"; exit 1; }
```

### 3. prefs 기록 (재빌드 없음)

SharedPreferences XML 을 구성해 **`run-as` 로 앱 전용 디렉터리에 직접 기록**한다(앱 uid 로 실행되므로 소유권·권한이 자동으로 맞다):

```bash
XML="<?xml version='1.0' encoding='utf-8' standalone='yes' ?>
<map>
    <string name=\"server_ip\">$IP</string>
    <int name=\"server_port\" value=\"$PORT\" />
</map>"

printf '%s\n' "$XML" | adb -s "$SERIAL" shell run-as "$PKG" \
  sh -c "mkdir -p '$PREFS_DIR' && cat > '$PREFS'"
adb -s "$SERIAL" shell run-as "$PKG" chmod 660 "$PREFS" 2>/dev/null
```

> **폴백 (run-as 불가 시)**: 일부 이미지/릴리스 빌드는 `run-as` 가 막힐 수 있다. 에뮬레이터는 보통 rootable 이므로 root 로 직접 기록한다:
> ```bash
> adb -s "$SERIAL" root >/dev/null 2>&1; adb -s "$SERIAL" wait-for-device
> printf '%s\n' "$XML" | adb -s "$SERIAL" shell "mkdir -p '$PREFS_DIR' && cat > '$PREFS'"
> OWN="$(adb -s "$SERIAL" shell stat -c '%U:%G' /data/data/$PKG 2>/dev/null | tr -d '\r')"
> adb -s "$SERIAL" shell "chown $OWN '$PREFS' && chmod 660 '$PREFS'"
> ```

### 4. reload (prefs 캐시 무효화)

SharedPreferences 는 프로세스 메모리에 캐시되므로, 실행 중이던 앱/서비스는 디스크 변경을 즉시 못 읽는다. **force-stop** 으로 프로세스를 내려 다음 기동 때 새 값을 읽게 한다:

```bash
adb -s "$SERIAL" shell am force-stop "$PKG"
```

> AccessibilityService(`CollectorService`)는 토글이 켜져 있으면 시스템이 곧 재기동하며 `getSharedPreferences("collector_settings")` 로 새 IP/port 를 읽는다. 토글이 꺼져 있으면 앱을 한 번 열거나 `setup-collector` 의 접근성 활성화 단계를 먼저 적용한다.

### 5. 결과 요약

```
update-client-ip (com.monkey.collector @ <SERIAL>)
  server_ip   : <CUR_IP|unset> → <IP>
  server_port : <CUR_PORT|unset> → <PORT>
  reloaded    : am force-stop  (다음 기동 시 적용)
```

## Notes

- **빌드/설치 안 함.** prefs 파일만 기록한다(MobileGPT-V2 와 달리 `assembleDebug`/`install` 없음).
- **멱등**: 같은 IP/port 를 다시 기록해도 안전(force-stop 만 반복).
- **port 기본 유지**: `<ip>` 만 주면 현재 port 를 보존한다. 서버를 다른 port 로 띄울 때만 `--port` 를 함께 준다 — 서버(`monkey-collect run --port P`)와 반드시 일치해야 TCP 연결이 성립한다.
- **AccessibilityService 전제**: 이 커맨드는 prefs 만 바꾼다. 수집이 돌려면 client 의 접근성 토글이 켜져 있어야 한다(`setup-collector` §6-a).
- **MediaProjection 동의**는 별개 — 수집 세션 시작 시 1회 UI 동의가 필요하다(adb 로 완전 자동화 불가, `appops … PROJECT_MEDIA allow` 는 best-effort).
- **역방향**: 실험/공유 후 기본값으로 되돌리려면 `reset-defaults` 를 쓴다(`server_ip=10.0.2.2`, `server_port=12345`).
