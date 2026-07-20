# 더미 데이터 시드 — PIM (연락처 · 시계 알람 · 캘린더 · SMS)

공용 규약은 [seed-helpers.md](seed-helpers.md). `SER="${SERIAL:-emulator-5554}"`, `adb -s "$SER" root` 전제. 레시피는 MobileGPT-V2 의 `setup-emulator`(Pixel6/API36)에서 가져와 Monkey-Collector(Pixel6-2/API33)에서 재검증.

> **MC 카탈로그**: 연락처(Simple Contacts)·Simple Calendar 는 설치됨(검증 완료). Simple Clock·Simple SMS 는 현재 카탈로그 미포함이라 `has` 가드로 자연 skip — 추가되면 아래 레시피가 그대로 동작한다.

---

## 1. Contacts (연락처)

- **패키지/소비**: 시스템 `com.android.providers.contacts` provider → Simple Contacts(`com.simplemobiletools.contacts.pro`) + Google Contacts 공용.
- **저장방식**: provider-content. **직접 sqlite3 금지** — display_name 계산·aggregation 우회 시 UI 에 이름 안 뜸. 반드시 `content insert`.
- **⚠️ provider-레벨 시드 — 앱 설치 여부와 무관하게 항상 실행됨(실측 확인)**: 아래 가드는 다른 레시피의 `has X && { ... }` 패턴과 달리 **삽입 블록을 감싸지 않는다** — `has A || has B || true` 는 `|| true` 때문에 항상 exit 0 이 되고, 그 결과값이 버려진 채 다음 줄로 그대로 진행된다. 즉 Simple Contacts Pro 가 미설치여도 시스템 provider(및 미리 설치돼 있던 Google Contacts)에 연락처 5건이 삽입된다. 의도적으로 provider-레벨 시드(카탈로그 앱 설치와 무관하게 항상 채움)이므로 두고, 이 사실만 명시한다 — Calendar(아래 §3)처럼 앱 미설치 시 자연 skip 을 원하면 `has com.simplemobiletools.contacts.pro && { ... }` 로 바꿔라.

```bash
has com.simplemobiletools.contacts.pro || has com.google.android.contacts || true   # 항상 true — 게이트 아님, provider-레벨 시드 표시용
CDB=/data/data/com.android.providers.contacts/databases/contacts2.db
if ! adb -s "$SER" shell "content query --uri content://com.android.contacts/raw_contacts --where \"display_name='Alice Johnson'\"" 2>/dev/null | grep -q display_name; then
  addc(){ # $1 given  $2 family  $3 phone
    adb -s "$SER" shell "content insert --uri content://com.android.contacts/raw_contacts --bind aggregation_mode:i:0" >/dev/null
    local RID; RID=$(adb -s "$SER" shell "sqlite3 $CDB 'SELECT MAX(_id) FROM raw_contacts;'" | tr -d '\r')
    adb -s "$SER" shell "content insert --uri content://com.android.contacts/data --bind raw_contact_id:i:$RID --bind mimetype:s:vnd.android.cursor.item/name --bind 'data1:s:$1 $2' --bind 'data2:s:$1' --bind 'data3:s:$2'" >/dev/null
    adb -s "$SER" shell "content insert --uri content://com.android.contacts/data --bind raw_contact_id:i:$RID --bind mimetype:s:vnd.android.cursor.item/phone_v2 --bind 'data1:s:$3' --bind data2:i:2" >/dev/null
  }
  addc Alice   Johnson  "+1 415-555-0100"
  addc Bob     Smith    "+1 415-555-0101"
  addc Carol   Williams "+1 415-555-0102"
  addc David   Brown    "+1 415-555-0103"
  addc Emma    Davis    "+1 415-555-0104"
fi
adb -s "$SER" shell "content query --uri content://com.android.contacts/data/phones --projection display_name:data1 --sort 'display_name ASC'"
```
- **gotcha**: phone `data2`=type(2=mobile). `data1` 공백은 device-side 단일따옴표로 보호. provider 자동 인덱싱 → broadcast 불필요.
- **검증(API33)**: 5명(이름+번호) 정상.

---

## 2. Clock 알람 (Simple Clock)

- **패키지**: `com.simplemobiletools.clock`. (`SET_ALARM` intent 는 Google Deskclock 으로 가므로 탐색대상 Simple Clock 엔 DB 직접 삽입.)
- **저장방식**: `…/databases/alarms.db`, **테이블명 `contacts`(실제는 알람 저장)**.
- **스키마**: `time_in_minutes`(0–1439), `days` 비트마스크(bit0=월…bit6=일; 31=월–금, 96=토일, 127=매일), `sound_uri='content://settings/system/alarm_alert'`.

```bash
has com.simplemobiletools.clock && {
  PKG=com.simplemobiletools.clock; DB=/data/data/$PKG/databases/alarms.db
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell am start -n "$PKG/$PKG.activities.MainActivity" >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM contacts WHERE label='Wake Up';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO contacts (time_in_minutes,days,is_enabled,vibrate,sound_title,sound_uri,label) VALUES
(390,31,1,1,'Default (Cesium)','content://settings/system/alarm_alert','Morning Workout'),
(420,127,1,1,'Default (Cesium)','content://settings/system/alarm_alert','Wake Up'),
(600,96,1,0,'Default (Cesium)','content://settings/system/alarm_alert','Weekend Brunch');
SQL
  fi
  adb -s "$SER" shell appops set $PKG SCHEDULE_EXACT_ALARM allow   # 활성 알람 → 정확알람 권한(SecurityException 방지)
}
```
- **gotcha**: `is_enabled=1` 알람 + 권한 없으면 API31+ `SecurityException` 토스트 → `appops … SCHEDULE_EXACT_ALARM allow` 필수.

---

## 3. Calendar (Simple Calendar Pro)

- **패키지**: `com.simplemobiletools.calendar.pro`. `…/databases/events.db`, 테이블 `events`.
- **스키마**: events 26개 컬럼 **전부 NOT NULL** → 전부 지정. `start_ts/end_ts`=epoch **초**, `time_zone='Asia/Seoul'`, `source='mg-seed'`(멱등 sentinel).

```bash
has com.simplemobiletools.calendar.pro && {
  DB=/data/data/com.simplemobiletools.calendar.pro/databases/events.db
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell am start -n "com.simplemobiletools.calendar.pro/.activities.MainActivity" >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop com.simplemobiletools.calendar.pro
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM events WHERE source='mg-seed';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO events (start_ts,end_ts,title,location,description,reminder_1_minutes,reminder_2_minutes,reminder_3_minutes,reminder_1_type,reminder_2_type,reminder_3_type,repeat_interval,repeat_rule,repeat_limit,repetition_exceptions,attendees,import_id,time_zone,flags,event_type,parent_id,last_updated,source,availability,color,type) VALUES
(1783072800,1783076400,'Team Standup','Office','Daily sync',10,-1,-1,0,0,0,0,0,0,'[]','','mg-seed','Asia/Seoul',0,1,0,0,'mg-seed',0,0,0),
(1783159200,1783166400,'Lunch with Client','Cafe','Partnership discussion',30,-1,-1,0,0,0,0,0,0,'[]','','mg-seed','Asia/Seoul',0,1,0,0,'mg-seed',0,0,0),
(1783288800,1783296000,'Project Review','Meeting Room B','Q3 deliverables',15,-1,-1,0,0,0,0,0,0,'[]','','mg-seed','Asia/Seoul',0,1,0,0,'mg-seed',0,0,0),
(1783422000,1783425600,'Dentist Appointment','Clinic','Annual checkup',60,-1,-1,0,0,0,0,0,0,'[]','','mg-seed','Asia/Seoul',0,1,0,0,'mg-seed',0,0,0);
SQL
  fi
}
```
- **검증(API33)**: events 4행, 월간 그리드에 마커 표시.

---

## 4. SMS / Messages (Simple SMS Messenger)

- **패키지**: `com.simplemobiletools.smsmessenger`(기본 SMS role). 시스템 telephony provider → 자체 캐시.
- **저장방식**: `content insert --uri content://sms`(root) → provider 가 `threads`+`canonical_addresses` 자동.

```bash
has com.simplemobiletools.smsmessenger && {
  MMS=/data/data/com.android.providers.telephony/databases/mmssms.db
  if [ "$(adb -s "$SER" shell "sqlite3 $MMS 'SELECT COUNT(*) FROM threads;'" | tr -d '\r')" = "0" ]; then
    ins(){ adb -s "$SER" shell "content insert --uri content://sms --bind address:s:$1 --bind 'body:s:$2' --bind date:l:$3 --bind read:i:$4 --bind type:i:$5"; }
    # type 1=수신,2=발신. body 에 콜론(:) 금지!
    ins +15551234567 "Hi there, are we still on for lunch?"            1782900000000 1 1
    ins +15551234567 "Yes, noon works! See you then."                 1782900060000 1 2
    ins +15559876543 "Your package has been delivered."               1782900120000 0 1
    ins +15559876543 "Thanks for the update!"                         1782900180000 1 2
    ins +15550001111 "Reminder - dentist appointment tomorrow at 3pm." 1782900240000 0 1
  fi
  adb -s "$SER" shell am start -n "com.simplemobiletools.smsmessenger/.activities.MainActivity" >/dev/null 2>&1; sleep 5
}
```
- **gotcha**: body 콜론(`:`) 금지(bind `col:type:value` 파서 깨짐), 공백 body 는 단일따옴표 보호.
