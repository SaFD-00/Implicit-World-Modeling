# 더미 데이터 시드 — 음악 & 기타 DB 앱 (RetroMusic · OpenTracks · Broccoli · Expense)

공용 규약은 [seed-helpers.md](seed-helpers.md). `SER="${SERIAL:-emulator-5554}"`, `adb -s "$SER" root` 전제. 레시피는 setup-emulator(Pixel6/API36)에서 가져와 Monkey-Collector(Pixel6-2/API33)에서 재검증.

> **MC 카탈로그**: RetroMusic·OpenTracks 설치됨(검증 완료). Broccoli·Expense 는 현재 미포함 → `has` 가드 자연 skip.

---

## 1. RetroMusic (음악)

- **패키지**: `code.name.monkey.retromusic`(+ Simple Music Player 동일 MediaStore 소비).
- **저장방식**: files — mp3 → `/sdcard/Music` + MediaStore rescan(자체 DB 아님).
- **호스트 생성**: ffmpeg `anullsrc` 무음 mp3 + ID3.

> **gotcha (중요)**: RetroMusic 기본 **"짧은 곡 필터"(~20초)** 로 10초 곡은 Songs 목록에서 **숨겨진다**. → mp3 길이 **≥ ~30초**(권장 45~95s). (READ_MEDIA_AUDIO 는 보통 granted — 권한이 아니라 길이가 원인.)

```bash
has code.name.monkey.retromusic || has com.simplemobiletools.musicplayer || true
if ! adb -s "$SER" shell "[ -f /sdcard/Music/demo_song_1.mp3 ]"; then
  if command -v ffmpeg >/dev/null 2>&1; then
    TMP="$(mktemp -d)"
    gen(){ ffmpeg -y -f lavfi -i anullsrc=r=44100:cl=stereo -t "$4" -q:a 9 -c:a libmp3lame \
      -metadata title="$2" -metadata artist="$3" -metadata album="Demo Album" "$TMP/$1" >/dev/null 2>&1; }
    gen demo_song_1.mp3 "Sunrise"    "The Testers" 75   # ≥30s (짧은곡 필터 회피)
    gen demo_song_2.mp3 "Ocean Drive" "The Testers" 50
    gen demo_song_3.mp3 "Night City"  "Demo Band"   95
    adb -s "$SER" shell mkdir -p /sdcard/Music
    for f in "$TMP"/*.mp3; do adb -s "$SER" push "$f" /sdcard/Music/ >/dev/null; done
    adb -s "$SER" shell 'for f in /sdcard/Music/demo_song_*.mp3; do [ -e "$f" ] && am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d "file://$f" >/dev/null 2>&1; done'
    rm -rf "$TMP"
  else echo "WARN: ffmpeg 없음 → RetroMusic 음악 시드 skip"; fi
fi
has code.name.monkey.retromusic && {
  adb -s "$SER" shell pm grant code.name.monkey.retromusic android.permission.READ_MEDIA_AUDIO 2>/dev/null
  adb -s "$SER" shell am force-stop code.name.monkey.retromusic
}
```
- **검증(API33)**: 75/50/95초 곡 3개 → audio media 3행, RetroMusic Songs 표시.

---

## 2. OpenTracks (운동 기록)

- **패키지**: `de.dennisguse.opentracks`. `…/databases/database.db`, 테이블 `tracks`.
- **스키마 요점(⚠️ 버전 드리프트 실측, v4.27.2/versionCode 6682 기준)**: 구버전 문서에 있던 `category`/`numpoints`/`totaldistance`/`totaltime`/`movingtime`/`avgspeed`/`avgmovingspeed`/`icon` 컬럼은 **더 이상 존재하지 않는다**. 실제 컬럼(`.schema tracks` 로 실측): `uuid,name,description,activity_type,activity_type_localized,time_offset,time_start,time_stop,duration_total,duration_moving,distance,speed_max,altitude_min,altitude_max,altitude_gain,altitude_loss`. `uuid` BLOB **UNIQUE** → `X'HEX'`(32 hex). 시간(`time_*`)=ms epoch, 기간(`duration_*`)=ms. `name` 고유(sentinel).

```bash
has de.dennisguse.opentracks && {
  PKG=de.dennisguse.opentracks; DB=/data/data/$PKG/databases/database.db
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM tracks WHERE name='Morning Run';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO tracks (uuid,name,description,activity_type,activity_type_localized,time_offset,time_start,time_stop,duration_total,duration_moving,distance,speed_max,altitude_min,altitude_max,altitude_gain,altitude_loss) VALUES
(X'A1B2C3D4E5F60011223344556677AA01','Morning Run','5 km run at the park','running','Running',0,1782900000000,1782901500000,1500000,1500000,5000.0,4.5,10.0,25.0,15.0,12.0),
(X'A1B2C3D4E5F60011223344556677AA02','Scenic Bike Tour','Lakeside loop','biking','Road biking',0,1782990000000,1782993600000,3600000,3600000,15000.0,8.0,5.0,40.0,35.0,30.0),
(X'A1B2C3D4E5F60011223344556677AA03','Evening Walk','Neighborhood stroll','walking','Walking',0,1783080000000,1783081200000,1200000,1200000,2000.0,2.2,8.0,12.0,4.0,4.0);
SQL
  fi
}
```
- **검증(v4.27.2/API33)**: tracks 3행(Morning Run/Scenic Bike Tour/Evening Walk), sentinel `name='Morning Run'` 확인.

---

## 3. Broccoli (레시피)

- **패키지**: `com.flauschcode.broccoli`. **DB 파일명 `broccoli`**, 테이블 `recipes`. `favorite` NOT NULL. FTS 미러 `recipes_fts` 트리거 자동(직접 삽입 금지).

```bash
has com.flauschcode.broccoli && {
  PKG=com.flauschcode.broccoli; DB=/data/data/$PKG/databases/broccoli
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM recipes WHERE title='Avocado Toast with Egg';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO recipes (title,imageName,description,servings,preparationTime,source,ingredients,directions,favorite) VALUES
('Avocado Toast with Egg','','A healthy breakfast choice.','1 serving','10 mins','','2 slices bread, 1 avocado, 1 egg, salt, pepper, chili flakes','Toast bread, mash avocado on top, add a fried egg, season and serve.',0),
('Spicy Tuna Wraps','','Quick weekday meal.','2 servings','15 mins','','1 can tuna, mayo, sriracha, 2 tortillas, lettuce, cucumber','Mix tuna with mayo and sriracha, spread on tortillas, add veggies, roll up.',0),
('Greek Salad Pita Pockets','','Fresh and light.','3 servings','20 mins','','Pita, lettuce, cucumber, tomato, feta, olives, Greek dressing','Fill pita pockets with veggies, feta and dressing.',1),
('Vegetarian Chili','','Hearty one-pot dinner.','6 servings','45 mins','','Onion, garlic, peppers, tomatoes, beans, corn, chili seasoning','Saute veggies, add tomatoes and beans, simmer until tender.',0);
SQL
  fi
}
```

---

## 4. Expense (Arduia Expense)

- **패키지**: `com.arduia.expense`. `…/databases/accounting.db`, 테이블 `expense`. `amount`=**센트**(2550=$25.50), `category`=정수 1–11(3=Food,6=Entertainment,7=Transportation), 날짜=ms.

```bash
has com.arduia.expense && {
  PKG=com.arduia.expense; DB=/data/data/$PKG/databases/accounting.db
  adb -s "$SER" shell "[ -f $DB ]" || { adb -s "$SER" shell am start -n "$PKG/.ui.MainActivity" >/dev/null 2>&1; sleep 5; }
  adb -s "$SER" shell am force-stop $PKG
  if [ "$(adb -s "$SER" shell "sqlite3 $DB \"SELECT COUNT(*) FROM expense WHERE name='Groceries';\"" | tr -d '\r')" = "0" ]; then
    adb -s "$SER" shell "sqlite3 $DB" <<'SQL'
INSERT INTO expense (name,amount,category,note,created_date,modified_date) VALUES
('Groceries',2550,3,'Weekly shopping',1782900000000,1782900000000),
('Gas Station',4500,7,'Fuel for car',1782950000000,1782950000000),
('Restaurant Meal',3575,3,'Dinner with friends',1783000000000,1783000000000),
('Movie Tickets',1500,6,'Entertainment',1783050000000,1783050000000);
SQL
  fi
}
```
- **gotcha**: `amount`=센트(합계 $121.25 = 12125센트 확인).
