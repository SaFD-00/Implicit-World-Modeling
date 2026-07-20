---
name: reset-defaults
description: Monkey-Collector Android client(com.monkey.collector) 의 on-device server prefs(collector_settings) 를 canonical 기본값(server_ip=10.0.2.2, server_port=12345)으로 되돌림 — update-client-ip 의 역방향. 런타임 prefs 라 소스/git 파일은 건드리지 않고 재빌드도 없음. 멱등. --serial 미지정 시 MobileGPT-V2-2 AVD 자동 탐색
argument-hint: "[--serial S]"
---

# Reset Defaults

> **⚠️ 호스트 전제: macOS + 부팅된 `MobileGPT-V2-2` 에뮬레이터.** `adb` 직접 호출이 macOS 기준이다.

실험/공유 전에 client 의 server 접속 설정을 **canonical 기본값으로 되돌린다.** Monkey-Collector client(`com.monkey.collector`)는 server IP/port 를 **on-device SharedPreferences `collector_settings`** 에 보관하므로, 이 커맨드는 그 prefs 의 `server_ip` 를 `10.0.2.2`(에뮬레이터→호스트 루프백), `server_port` 를 `12345`(서버 기본 port)로 기록한다.

`update-client-ip` 의 **역방향**에 해당한다 — `update-client-ip` 가 실제 IP/PORT 를 박는다면, 이 커맨드는 그것을 기본값으로 되돌린다.

> **MobileGPT-V2 의 `reset-defaults` 와의 차이**: 그쪽은 `MobileGPTGlobal.java` 의 컴파일타임 `HOST_IP` 를 placeholder 로, config yaml 6개의 sweep 키를 canonical 로 되돌리는 **커밋 전 소스 정리** 였다. Monkey-Collector 는 server IP 가 **런타임 prefs(소스/git 에 안 박힘)** 이고 sweep yaml 도 없으므로, 되돌릴 **커밋 대상 소스 흔적이 없다.** 따라서 이 커맨드는 **on-device prefs 만** 기본값으로 되돌린다(소스/git 파일·`catalog/apps.csv` 는 건드리지 않음).

## Options

`$ARGUMENTS` 를 파싱하여 아래 옵션을 처리:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--serial S` | (자동) | 대상 adb 시리얼. 미지정 시 `MobileGPT-V2-2` AVD 를 `adb devices` + `emu avd name` 으로 역추출 |

**검증**: 대상 serial 을 못 찾으면(에뮬레이터 미부팅) → 중단하고 `setup-emulator`(또는 `emulator -avd MobileGPT-V2-2`) 안내. client 미설치면 `run-as` 가 실패 → "client 미설치" 안내 후 종료.

## Target — canonical 기본값

| 키 | canonical 기본값 | 출처 |
|----|-----------------|------|
| `server_ip` | `10.0.2.2` | `MainActivity.kt` 의 `getString("server_ip", "10.0.2.2")` 기본값 = 에뮬레이터→호스트 루프백 |
| `server_port` | `12345` | `CollectorService.kt`/`MainActivity.kt` 의 `getInt("server_port", 12345)` = 서버(`monkey-collect run`) 기본 port |

prefs 파일: `/data/data/com.monkey.collector/shared_prefs/collector_settings.xml`

## Process

### 1. serial 결정

```bash
PKG="com.monkey.collector"
PREFS_DIR="/data/data/$PKG/shared_prefs"
PREFS="$PREFS_DIR/collector_settings.xml"
DEF_IP="10.0.2.2"; DEF_PORT="12345"

if [ -z "$SERIAL" ]; then
  while read -r serial state; do
    case "$serial" in emulator-*) ;; *) continue;; esac
    [ "$state" = "device" ] || continue
    name="$(adb -s "$serial" emu avd name 2>/dev/null | tr -d '\r' | head -1)"
    [ "$name" = "MobileGPT-V2-2" ] && { SERIAL="$serial"; break; }
  done < <(adb devices | sed '1d;/^$/d')
fi
[ -z "$SERIAL" ] && { echo "ERROR: MobileGPT-V2-2 emulator not found. Boot it (setup-emulator) first."; exit 1; }
```

### 2. 현재 값 확인 (멱등 체크)

```bash
CUR_XML="$(adb -s "$SERIAL" shell run-as "$PKG" cat "$PREFS" 2>/dev/null | tr -d '\r')"
CUR_IP="$(printf '%s' "$CUR_XML"   | grep -oE 'name="server_ip">[^<]*'      | sed 's/.*>//')"
CUR_PORT="$(printf '%s' "$CUR_XML" | grep -oE 'name="server_port" value="[0-9]+"' | grep -oE '[0-9]+')"
echo "current: server_ip=${CUR_IP:-<unset>} server_port=${CUR_PORT:-<unset>}"
if [ "$CUR_IP" = "$DEF_IP" ] && [ "$CUR_PORT" = "$DEF_PORT" ]; then
  echo "변경 없음 (이미 기본값)"; exit 0
fi
```

### 3. 기본값 기록

`update-client-ip` 와 동일한 `run-as` 기록 메커니즘(루트 불요):

```bash
XML="<?xml version='1.0' encoding='utf-8' standalone='yes' ?>
<map>
    <string name=\"server_ip\">$DEF_IP</string>
    <int name=\"server_port\" value=\"$DEF_PORT\" />
</map>"

printf '%s\n' "$XML" | adb -s "$SERIAL" shell run-as "$PKG" \
  sh -c "mkdir -p '$PREFS_DIR' && cat > '$PREFS'"
adb -s "$SERIAL" shell run-as "$PKG" chmod 660 "$PREFS" 2>/dev/null
```

> **폴백 (run-as 불가)**: `adb -s "$SERIAL" root` 후 직접 기록 + `chown <app uid>` (update-client-ip §3 폴백과 동일).

### 4. reload

```bash
adb -s "$SERIAL" shell am force-stop "$PKG"
```

### 5. 결과 요약

```
reset-defaults (com.monkey.collector @ <SERIAL>)
  server_ip   : <CUR_IP> → 10.0.2.2
  server_port : <CUR_PORT> → 12345
  reloaded    : am force-stop
```

이미 기본값이면 "변경 없음 (이미 기본값)" 출력(§2 에서 종료).

## Notes

- **on-device prefs 만 건드린다.** 소스/git 파일(`MobileGPTGlobal.java` 같은 상수)·`catalog/apps.csv`·`.env` 는 변경하지 않는다 — Monkey-Collector 는 server IP 가 런타임 prefs 라 커밋되는 소스 흔적이 없다.
- **빌드/설치 안 함.** 재빌드 불필요(런타임 prefs).
- **멱등**: 이미 `10.0.2.2`/`12345` 면 아무 것도 안 한다.
- **역방향 관계**: 이후 실제 사용 전 `update-client-ip <서버 IP> [--port P]` 로 다시 채운다. 서버를 호스트에서 기본 port(12345)로 띄우면 reset 만으로 바로 동작한다(`10.0.2.2:12345`).
- `catalog/apps.csv` 의 `installed` 컬럼(=`sync-installed` 가 갱신하는 디바이스 상태)을 커밋 전에 되돌리고 싶다면 이 커맨드의 범위 밖이다 — `git checkout -- catalog/apps.csv` 로 처리한다.
