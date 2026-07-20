# 더미 데이터 시드 — 노트 & 할 일 (Markor · Simple Notes · org.tasks · Joplin)

공용 규약은 [seed-helpers.md](seed-helpers.md). `SER="${SERIAL:-emulator-5554}"`, `adb -s "$SER" root` 전제. 레시피는 MobileGPT-V2 의 `setup-emulator`(Pixel6/API36)에서 가져와 Monkey-Collector(Pixel6-2/API33)에서 재검증.

> **MC 카탈로그**: Markor·org.tasks·Joplin 설치됨(검증 완료). Simple Notes 는 현재 미포함 → `has` 가드 자연 skip.

---

## 1. Markor (마크다운 노트)

- **패키지**: `net.gsantner.markor`. 저장방식 = **files**(DB 없음). 노트북 폴더 `/sdcard/Documents/markor`. MediaStore rescan 불필요(Markor 가 폴더 직접 스캔).

```bash
has net.gsantner.markor && {
  D=/sdcard/Documents/markor
  adb -s "$SER" shell mkdir -p "$D"
  if ! adb -s "$SER" shell "[ -f $D/meeting-notes.md ]"; then
    TMP="$(mktemp -d)"
    printf '# Meeting Notes\n\n- Discussed Q3 roadmap\n- Decided sprint timeline\n- Assigned action items\n' > "$TMP/meeting-notes.md"
    printf '# Shopping List\n\n- Milk\n- Bread\n- Apples\n- Chicken\n- Rice\n'                              > "$TMP/shopping-list.md"
    printf '# Vacation Plans\n\nDates: July 15-30\nDestination: Mountains\nActivities: hiking, camping\n'   > "$TMP/vacation-plans.md"
    printf '# Homemade Pizza\n\n## Ingredients\n- Dough\n- Tomato sauce\n- Mozzarella\n\n## Steps\nBake at 220C for 12-15 min.\n' > "$TMP/recipe-pizza.md"
    for f in "$TMP"/*.md; do adb -s "$SER" push "$f" "$D/" >/dev/null; done
    rm -rf "$TMP"
  fi
}
```
- **검증(API33)**: .md 4개 표시. host `printf > file` 후 push 가 device-side redirect escape 보다 안전.

---

## 2. Simple Notes (Simple Notes Pro)

- **패키지**: `com.simplemobiletools.notes.pro`. `…/databases/notes.db`, 테이블 `notes`(전부 NOT NULL: `type=0`,`path=''`,`protection_type=-1`,`protection_hash=''`).

```bash
has com.simplemobiletools.notes.pro && {
  PKG=com.simplemobiletools.notes.pro; DB=/data/data/$PKG/databases/notes.db
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell am start -n "$PKG/$PKG.activities.MainActivity" >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM notes WHERE title='Shopping List';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO notes (title,value,type,path,protection_type,protection_hash) VALUES
('Shopping List','Milk, Eggs, Bread, Cheese, Apples',0,'',-1,''),
('Meeting Notes','Discussed project timeline and deliverables for Q3.',0,'',-1,''),
('Ideas','App concept: a todo manager with calendar sync.',0,'',-1,''),
('Travel Plans','Book flights and hotel for summer vacation in July.',0,'',-1,'');
SQL
  fi
}
```

---

## 3. Tasks (org.tasks)

- **패키지**: `org.tasks`. **DB 파일명 `database`**(`…/databases/database`), 테이블 `tasks`.
- **스키마 요점**: NOT NULL 정수 다수(미사용 0). `remoteId` **UNIQUE**(`mg-seed-00x` → 멱등 sentinel). 날짜=**ms**. `dueDate=0`=기한없음. `importance`(0~3) 우선순위 색. insert 후 `am force-stop` 로 WAL flush.

```bash
has org.tasks && {
  PKG=org.tasks; DB=/data/data/$PKG/databases/database
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM tasks WHERE remoteId LIKE 'mg-seed-%';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO tasks (title,importance,dueDate,hideUntil,created,modified,completed,deleted,notes,estimatedSeconds,elapsedSeconds,timerStart,notificationFlags,lastNotified,recurrence,repeat_from,calendarUri,remoteId,collapsed,parent,read_only) VALUES
('Grocery Shopping',2,1783152000000,0,1782900000000,1782900000000,0,0,'Milk, eggs, bread, vegetables',0,0,0,0,0,NULL,0,NULL,'mg-seed-001',0,0,0),
('Project Planning Meeting',1,1783238400000,0,1782900000000,1782900000000,0,0,'Prepare agenda and slides',0,0,0,0,0,NULL,0,NULL,'mg-seed-002',0,0,0),
('Read Documentation',3,1783497600000,0,1782900000000,1782900000000,0,0,'Review API docs and tutorials',0,0,0,0,0,NULL,0,NULL,'mg-seed-003',0,0,0),
('Pay Electricity Bill',0,1783584000000,0,1782900000000,1782900000000,0,0,'Due end of month',0,0,0,0,0,NULL,0,NULL,'mg-seed-004',0,0,0);
SQL
  fi
  adb -s "$SER" shell am force-stop $PKG   # WAL flush
}
```
- **검증(API33)**: "My Tasks" 에 4건(제목·메모·기한·우선순위 색). **이 스킬 갱신 시 라이브 검증으로 collector 가 시드된 58개 UI 요소를 정상 파싱함을 확인.**

---

## 4. Joplin

- **패키지**: `net.cozic.joplin`. `…/databases/joplin.sqlite`. 테이블 `folders`,`notes`,`notes_normalized`.
- **규칙**: id=32자 hex(고정 → 멱등), `parent_id`(note)=folder.id, 시간=ms. UI 목록은 `notes` 면 충분. FTS 까지면 `notes_normalized` 에도 삽입 → 트리거가 `notes_fts` 자동(직접 삽입 금지).

```bash
has net.cozic.joplin && {
  PKG=net.cozic.joplin; DB=/data/data/$PKG/databases/joplin.sqlite
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; sleep 6; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM notes WHERE id='00000000000000000000000000000002';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO folders (id,title,created_time,updated_time,user_created_time,user_updated_time) VALUES
('00000000000000000000000000000001','Demo Notebook',1782900000000,1782900000000,1782900000000,1782900000000);
INSERT INTO notes (id,parent_id,title,body,created_time,updated_time,user_created_time,user_updated_time) VALUES
('00000000000000000000000000000002','00000000000000000000000000000001','Welcome to Joplin','This is a demo note. You can edit and organize notes here.',1782900000000,1782900000000,1782900000000,1782900000000),
('00000000000000000000000000000003','00000000000000000000000000000001','Grocery List','- Milk\n- Bread\n- Eggs\n- Coffee',1782900100000,1782900100000,1782900100000,1782900100000),
('00000000000000000000000000000004','00000000000000000000000000000001','Project Ideas','Build a note-taking app with calendar integration.',1782900200000,1782900200000,1782900200000,1782900200000);
INSERT INTO notes_normalized (id,title,body,user_created_time,user_updated_time,parent_id) VALUES
('00000000000000000000000000000002','welcome to joplin','this is a demo note. you can edit and organize notes here.',1782900000000,1782900000000,'00000000000000000000000000000001'),
('00000000000000000000000000000003','grocery list','milk bread eggs coffee',1782900100000,1782900100000,'00000000000000000000000000000001'),
('00000000000000000000000000000004','project ideas','build a note-taking app with calendar integration.',1782900200000,1782900200000,'00000000000000000000000000000001');
SQL
  fi
}
```
- **검증(API33)**: notes 3개(Welcome/Grocery List/Project Ideas).
