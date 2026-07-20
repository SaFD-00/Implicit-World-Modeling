# 앱별 더미 데이터 시드 — 공용 규약 (§6-e)

setup-collector §6-e 에서 explorable 앱에 소량(앱당 3~5건)의 더미 데이터를 넣어, collector 가 리스트/상세/편집 화면을 탐색할 콘텐츠를 확보한다. 모든 시더는 **venv-free**(plain adb: on-device `sqlite3` / `content insert` / `am` intent / `adb push` + MediaStore broadcast).

이 파일은 모든 시더가 따르는 공용 규약이다. 앱별 명령은:
- [seed-pim.md](seed-pim.md) — 연락처 · 시계 알람 · 캘린더 · SMS
- [seed-notes-tasks.md](seed-notes-tasks.md) — Markor · Simple Notes · org.tasks · Joplin
- [seed-media-misc.md](seed-media-misc.md) — RetroMusic · OpenTracks · Broccoli · Expense

> 레시피는 MobileGPT-V2 `setup-emulator` 에서 **Pixel6/API36 실삽입·UI 검증**된 것을 가져왔고, **Monkey-Collector 의 Pixel6-2/API33 에서 카탈로그 교집합(연락처·Simple Calendar·Markor·org.tasks·RetroMusic·OpenTracks·Joplin) 재검증 완료**. `$SER` = 대상 serial(예 `emulator-5554`).

> **MC 카탈로그 스코프**: Simple Notes·Simple Clock·Simple SMS·Broccoli·Expense 는 현재 `catalog/apps.csv` 에 없어 자연 skip(`has` 가드). 카탈로그에 추가되면 그대로 동작한다.

## 0. 전제

```bash
SER="${SERIAL:-emulator-5554}"
adb -s "$SER" root >/dev/null 2>&1   # google_apis 이미지 → 루트 가능. on-device /system/bin/sqlite3 사용
has(){ adb -s "$SER" shell pm list packages | grep -q "package:$1"; }   # 설치 여부
```
- 시더는 **설치된 앱만** 처리: 각 앱 블록을 `has <pkg> && { ... }` 로 감싼다(선택 설치/`--only` 호환).
- 루트가 안 되는 환경이면 app-sqlite 시딩은 skip + 경고(provider/intent/files 계열은 일부 동작).
- **멱등**: 각 앱은 sentinel(§5) 미존재일 때만 삽입 → 재실행/초기화-후-재실행에 안전.

## 1. on-device sqlite3 — heredoc-stdin 패턴 (권장)

중첩 따옴표 escape 지옥을 피하려고 **SQL 을 `adb shell sqlite3 <db>` 의 stdin 으로 heredoc 전달**한다.

```bash
adb -s "$SER" shell "sqlite3 /data/data/<pkg>/databases/<db>" <<'SQL'
INSERT INTO <table> (...) VALUES (...);
SQL
```
- `<<'SQL'`(따옴표 heredoc) → 호스트 셸이 내용을 안 건드림. 디바이스 sqlite3 가 stdin 으로 읽음.
- 검증: `adb -s "$SER" shell "sqlite3 <db> 'SELECT COUNT(*) FROM <table>;'"`

## 2. prelaunch + WAL — app-sqlite 공통 절차

대부분 app-sqlite 앱은 **DB 가 첫 실행 때 생성**되고 **WAL 모드**라 앱이 켜져 있으면 잠금이 걸린다:

1. DB 없으면 1회 실행해 생성(prelaunch): `adb -s "$SER" shell "[ -f $db ]" || { am start -n <pkg/launch>; sleep 5; }`
2. insert 전 `am force-stop <pkg>` (DB/WAL 비점유).
3. sqlite3 insert(§1 heredoc).
4. org.tasks 처럼 flush 필요하면 insert 후 `am force-stop` 한 번 더.

> 직접 sqlite3 삽입은 Room/provider 검증을 우회 → **NOT NULL 전부 지정**, FK/UNIQUE 준수 필수. FTS 미러(Joplin `notes_fts`, Broccoli `recipes_fts`)는 트리거가 자동 채움(직접 삽입 금지; 단 Joplin `notes_normalized` 는 직접 삽입 → 트리거가 fts 채움).

## 3. provider / intent / files 계열

- **Contacts**: 시스템 ContactsContract provider. `content insert`(root) raw_contact 후 data row → provider 가 display_name 계산. **직접 sqlite3 금지**(seed-pim.md).
- **SMS**: `content insert --uri content://sms`(root) → provider 가 threads/canonical 자동. **body 에 콜론(`:`) 금지**.
- **files (Markor/RetroMusic)**: `adb push`. 미디어는 push 후 MediaStore rescan(§4).

## 4. MediaStore rescan (RetroMusic 등 미디어)

```bash
adb -s "$SER" shell '
for f in /sdcard/Music/demo_song_*.mp3; do
  [ -e "$f" ] && am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d "file://$f" >/dev/null 2>&1
done'
```

## 5. 멱등성 가드 (재실행 시 중복 방지)

각 시더는 삽입 전 **sentinel** 확인 후 이미 있으면 skip:

| 앱 | sentinel |
|---|---|
| Contacts | `content query … raw_contacts --where "display_name='Alice Johnson'"` 비었을 때만 |
| Calendar(Simple) | `SELECT COUNT(*) FROM events WHERE source='mg-seed'` = 0 |
| Markor | `[ -f /sdcard/Documents/markor/meeting-notes.md ]` 아닐 때만 |
| Tasks(org.tasks) | `SELECT COUNT(*) FROM tasks WHERE remoteId LIKE 'mg-seed-%'` = 0 |
| RetroMusic | `[ -f /sdcard/Music/demo_song_1.mp3 ]` 아닐 때만 |
| Joplin | `SELECT COUNT(*) FROM notes WHERE id='00000000000000000000000000000002'` = 0 |
| OpenTracks | `SELECT COUNT(*) FROM tracks WHERE name='Morning Run'` = 0 |
| (Simple Notes/Clock/SMS/Broccoli/Expense — 카탈로그 추가 시) | seed-pim/notes-tasks/media-misc 참조 |

## 6. 권한 gotcha (검증 중 발견)

- **Clock — `SCHEDULE_EXACT_ALARM`**: 활성 알람(`is_enabled=1`) 삽입 시 앱 로드에서 정확알람 스케줄 시도 → API31+ `SecurityException` 토스트. 시드 후 `appops set com.simplemobiletools.clock SCHEDULE_EXACT_ALARM allow`.
- **RetroMusic — `READ_MEDIA_AUDIO`**: 보통 granted. 없으면 `pm grant code.name.monkey.retromusic android.permission.READ_MEDIA_AUDIO`. (곡 안 보이는 주원인은 권한 아니라 **짧은 곡 필터** — seed-media-misc.md.)

## 7. 호스트 미디어 생성 (RetroMusic)

RetroMusic mp3 는 호스트에서 ffmpeg `anullsrc` 무음 mp3 + ID3 로 1회 생성. ffmpeg 미존재 시 그 앱만 skip + 경고. 상세 seed-media-misc.md.

## 8. 검증 (§8 에서 인용)

```bash
adb -s "$SER" shell "content query --uri content://com.android.contacts/data/phones --projection display_name:data1"        # 연락처
adb -s "$SER" shell "sqlite3 /data/data/com.simplemobiletools.calendar.pro/databases/events.db 'SELECT COUNT(*) FROM events;'"  # 일정
adb -s "$SER" shell "ls /sdcard/Documents/markor/ | wc -l"                                                                    # Markor
adb -s "$SER" shell "sqlite3 /data/data/org.tasks/databases/database \"SELECT COUNT(*) FROM tasks WHERE remoteId LIKE 'mg-seed-%';\""  # 할일
adb -s "$SER" shell "content query --uri content://media/external/audio/media --projection title:duration"                    # 음악
# Joplin/OpenTracks 는 각 DB SELECT COUNT(*)
```
