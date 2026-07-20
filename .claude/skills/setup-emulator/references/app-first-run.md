# 앱별 첫 실행 온보딩 관찰 (§5-a)

카탈로그 앱 12종(system 2 + F-Droid/PlayStore 10)을 §5 설치 직후, §6-e 시드 이전에 1회씩 실행해 온보딩(캐러셀/로그인유도/특수퍼미션)을 통과하고 "탐색 가능 상태"에 도달하는 절차를 실측한 기록. SKILL.md §5-a 에서 위임.

## 공용 규약

- **설치 시점 권한**: §5 의 `adb install -r -g` 가 런타임 퍼미션을 선grant 하므로, 이 문서의 앱들은 대부분 별도 퍼미션 다이얼로그 없이 온보딩만 통과하면 된다.
- **좌표 하드코딩 금지**: 버튼 위치는 해상도/버전에 따라 바뀐다. `adb shell uiautomator dump /sdcard/window_dump.xml && adb pull /sdcard/window_dump.xml` 로 덤프 후 버튼 텍스트 노드의 `bounds="[x1,y1][x2,y2]"` 중심을 계산해 `input tap`. 이 라운드에서 실제 쓴 헬퍼: `_skill_verify_20260714/helper.sh`(`shot`/`dumpui`/`findtext`/`tapcenter`/`launch`/`fstop`) + `dump_and_find.py`(`&amp;` 등 XML 엔티티를 `html.unescape` 로 정규화해 텍스트 매칭).
- **XML 이스케이프 주의**: uiautomator dump 는 `&`를 `&amp;`로 이스케이프한다. 텍스트를 직접 grep 할 때(예: 버튼 라벨에 `&`가 있는 경우) 이스케이프된 형태로 매칭하거나 `dump_and_find.py`처럼 `html.unescape` 로 정규화해야 한다.
- **launch 커맨드**: 프로토콜 기본값 `monkey -p <pkg> -c android.intent.category.LAUNCHER 1` 은 이 환경(emulator-5556)에서 **12개 앱 전부 no-op**(포커스 변화 없음, `pidof` 로 미기동 확인)이었다. 대신 `am start -n <resolved-activity>` 를 쓴다:
  ```bash
  ACT=$(adb -s "$SERIAL" shell cmd package resolve-activity --brief "$PKG" | tail -1)
  adb -s "$SERIAL" shell am start -n "$ACT"
  ```
- **멱등성 확인법**: 온보딩 통과 후 `am force-stop <pkg>` → 재실행 → 온보딩이 재출현하지 않고 바로 메인 화면으로 직행하는지 확인. 이 라운드 12앱 중 11개는 1회성(재출현 없음), YouTube Music 만 예외(아래 참조).
- **지도류 앱의 uiautomator dump 실패**: 연속 애니메이션(위치 마커 펄스 등)이 있는 화면에서 `uiautomator dump`가 `ERROR: could not get idle state.`로 계속 실패할 수 있다(OsmAnd 실측, 다른 11개 앱은 전부 정상). 이 경우 dump 기반 bounds 추출을 포기하고 **스크린샷 + 픽셀 스캔**으로 우회한다: `adb exec-out screencap -p`로 이미지를 받아 `python3 PIL`로 버튼 색상 밴드(예: 링크색 파란 텍스트)를 스캔해 실제 bounds 를 역산한다. 눈대중 좌표 추정은 실패하기 쉽다(OsmAnd 사례에서 5회 이상 무반응 tap 발생) — 픽셀 스캔이 신뢰도가 높다.

---

## 1. code.name.monkey.retromusic (Retro Music)

- **화면 시퀀스**: `MainActivity` 진입 시 "What's New" 변경로그 다이얼로그(WebView, 명시적 닫기 버튼 없음)가 메인 화면 위에 오버레이 → `KEYCODE_BACK` 으로 dismiss → "For you" 메인 탭.
- **통과 커맨드**: `am start -n code.name.monkey.retromusic/.activities.MainActivity` → `input keyevent KEYCODE_BACK`.
- **탐색 가능 앵커**: 하단 탭 `For you`/`Songs`/`Albums`/`Artists`/`Playlists` 노출.
- **판정**: no-login-usable. "Welcome, User Name" 은 로그인 계정이 아니라 고정 placeholder(로그인 기능 자체 없음).
- **특이사항**: 버전 미변경 시 changelog 배너 재노출 안 함(세션/버전당 1회성 추정).

## 2. com.android.settings (Settings)

- **화면 시퀀스**: `Settings` 액티비티 진입 즉시 메인 설정 리스트. 온보딩/ToS/로그인 다이얼로그 전혀 없음.
- **통과 커맨드**: `am start -n com.android.settings/.Settings` — 조작 불필요.
- **탐색 가능 앵커**: `Search settings` 검색바 + `Network & internet`/`Apps`/`Battery` 등 리스트.
- **판정**: no-login-usable. 시스템 앱으로 로그인 개념 자체가 없음.

## 3. com.google.android.apps.maps (Google Maps) — 재검증됨

- **화면 시퀀스**: `MapsActivity` 최초 진입 시 로그인 유도(`Make it your map`) → 우상단 `SKIP` tap → 메인 `Explore` 지도 화면.
- **통과 커맨드**: `am start -n com.google.android.apps.maps/com.google.android.maps.MapsActivity` → `SKIP` bounds 중심 tap.
- **탐색 가능 앵커**: 검색창(`Search here`), `Layers`/`Re-center`/`Directions`, 하단 탭 `Explore`/`Go`/`Saved`/`Contribute`/`Updates`.
- **판정**: **no-login-usable로 재분류**. `SKIP` 이후 검색/경로안내/익스플로어 피드까지 로그인 없이 전부 동작 확인(부가 기능인 즐겨찾기 저장·검색 기록 동기화만 로그인 필요).
- **특이사항**: `references/google-login.md` 의 기존 가정(로그인 필요 6종에 Maps 미포함이지만 과거 `login-wall` 추정이 있었다면)과 달리 이번 재검증으로 로그인 없이 완전 탐색 가능함이 확인됨 — **`google-login.md` 자체는 이번 라운드에서 갱신하지 않는다**(별도 결정 필요, out of scope).

## 4. com.google.android.apps.youtube.music (YouTube Music) — 비멱등 예외

- **화면 시퀀스**: `MusicActivity` 진입 시 로그인 유도(`Open the world of music`) → 하단 텍스트 링크 `DEVICE FILES ONLY` tap → 로컬 라이브러리(`Playlists`/`Albums`/`Songs`/`Artists`).
- **통과 커맨드**: `am start -n com.google.android.apps.youtube.music/.activities.MusicActivity` → `DEVICE FILES ONLY` bounds 중심 tap.
- **탐색 가능 앵커**: 로컬 라이브러리 리스트(각 항목 `>` chevron).
- **판정**: **partial**. 로컬 파일 브라우징은 로그인 없이 가능하나, 스트리밍/온라인 카탈로그 검색·재생 등 핵심 기능은 로그인 필요.
- **특이사항 — 이 라운드 12앱 중 유일한 비멱등 사례**: `am force-stop` 후 재실행하면 로그인 유도 화면이 **다시 나타난다**(다른 11개 앱과 달리 "DEVICE FILES ONLY" 선택이 영속화되지 않음). 재수집 세션에서 "한 번 통과했으니 재출현 안 함"을 가정하면 안 되고, 매 launch 마다 재통과 필요.

## 5. com.google.android.deskclock (Clock)

- **화면 시퀀스**: `DeskClock` 진입 즉시 메인 시계 화면. 비차단성 프라이버시 정책 툴팁 배너만 있음(닫지 않아도 무방).
- **통과 커맨드**: `am start -n com.google.android.deskclock/com.android.deskclock.DeskClock` — 조작 불필요.
- **탐색 가능 앵커**: 하단 탭 `Alarm`/`Clock`/`Timer`/`Stopwatch`/`Bedtime`.
- **판정**: no-login-usable.

## 6. com.google.android.dialer (전화/Dialer)

- **화면 시퀀스**: `GoogleDialtactsActivity` 진입 즉시 메인 `Favorites` 화면. 다이얼로그 없음(런타임 퍼미션은 `-g` 로 선grant).
- **통과 커맨드**: `am start -n com.google.android.dialer/.extensions.GoogleDialtactsActivity` — 조작 불필요.
- **탐색 가능 앵커**: 검색바 `Search contacts & places`, 하단 탭 `Favorites`/`Recents`/`Contacts`/`Voicemail`, 다이얼패드 FAB.
- **판정**: no-login-usable.

## 7. de.dennisguse.opentracks (OpenTracks)

- **화면 시퀀스**: `IntroductionActivity` 2페이지 캐러셀(프라이버시 소개 → OSMDashboard 안내) → FAB(다음) 2회 tap → `TrackListActivity`(메인).
- **통과 커맨드**: `am start -n de.dennisguse.opentracks/.introduction.IntroductionActivity` → FAB bounds 중심 tap ×2.
- **탐색 가능 앵커**: 검색바, `Markers`/`Settings`, 하단 `Record` FAB.
- **판정**: no-login-usable. 위치 권한 다이얼로그는 관찰되지 않음(설치 시 선grant 추정).
- **특이사항**: force-stop 후 재실행 시 `resolve-activity` 는 여전히 `IntroductionActivity` 를 가리키나 내부적으로 `TrackListActivity` 로 즉시 리다이렉트(인트로 재출현 없음).

## 8. net.cozic.joplin (Joplin)

- **화면 시퀀스**: 온보딩 자체 없음 — `MainActivity` 진입이 곧 메인 `All notes` 리스트(앱이 기본 제공하는 튜토리얼 노트 5개가 자동 생성돼 있으나 사용자 입력 아님).
- **통과 커맨드**: `am start -n net.cozic.joplin/.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: 좌측 nav drawer(`All notes`/`Notebooks`/`Trash`/`Tags`/`Configuration`/`Synchronize`).
- **판정**: no-login-usable.

## 9. net.gsantner.markor (Markor)

- **화면 시퀀스**: `IntroActivity` 5페이지 캐러셀(`NEXT` ×4 → `DONE`) → 저장소 권한 다이얼로그(`OK`) → 시스템 "All files access" 특수퍼미션 화면(토글 tap) → `KEYCODE_BACK` → `MainActivity`(파일 브라우저).
- **통과 커맨드**: `am start -n net.gsantner.markor/.activity.MainActivity`(자동으로 IntroActivity 리다이렉트) → `NEXT` ×4 → `DONE` → 권한 `OK` → All files access 토글 tap → `KEYCODE_BACK`.
- **탐색 가능 앵커**: `/storage/emulated/0/Documents` 파일 브라우저, 하단 탭 `Files`/`To-Do`/`QuickNote`/`More`.
- **판정**: no-login-usable. 온보딩 + 저장소 특수퍼미션 둘 다 로그인과 무관.

## 10. net.osmand (OsmAnd) — dump 실패 우회 사례

- **화면 시퀀스**: `MapActivity` 진입 시 "Welcome" 오버레이(위치 검색 → 지역 감지 → 지도 다운로드 권유) → 하단 `SKIP DOWNLOAD` tap → 메인 지도 화면.
- **통과 커맨드/이슈**: `am start -n net.osmand/.plus.activities.MapActivity`. **이 화면에서 uiautomator dump 가 계속 `ERROR: could not get idle state.` 로 실패**(지도 연속 애니메이션 추정) → 픽셀 스캔으로 `SKIP DOWNLOAD` 실제 bounds(`x=[762,1045] y=[2254,2279]`)를 찾아 tap. 눈대중 좌표(y≈1459)로는 5회 이상 무반응이었음 — 공용 규약의 "지도류 dump 실패" 항목 참조.
- **탐색 가능 앵커**: 지구본/검색 아이콘, 나침반, `+`/`-` 줌, 위치 버튼.
- **판정**: no-login-usable. 500MB 지도 다운로드는 `SKIP DOWNLOAD` 로 건너뛰어도 팬/줌 탐색 가능(지도 데이터 시드는 이번 스코프 밖).

## 11. org.tasks (Tasks.org)

- **화면 시퀀스**: 최초 실행 시 계정 선택 화면(`Add account`/`Continue without sync`/`Import Tasks.org backup`) → `Continue without sync` tap → `My Tasks`(메인).
- **통과 커맨드**: `am start -n org.tasks/com.todoroo.astrid.activity.TaskListActivity` → `Continue without sync` 텍스트 노드 상위 clickable 영역 tap.
- **탐색 가능 앵커**: `+` FAB, 정렬/검색/마이크 툴바.
- **판정(온보딩만)**: no-login-usable — "Continue without sync" 로 로그인 없이 CRUD 가능.
- **알려진 collector 이슈(온보딩과 별개)**: 시드 데이터(mg-seed 4건)가 정상 렌더링된 `TaskListActivity` 화면에서도 **`monkey-collect run` 수집 시 Step 0 무한 정체**가 관측됨(interactable-element 판정이 "no interactable"로 declining → back press → 앱 종료로 오인 → 외부앱 스톰 반복). 이는 온보딩/시드 문제가 아니라 collector 코드(screen_matching/UI 파서) 레벨 이슈로 추정 — 미해결, 별도 트래킹 필요. 상세는 SKILL.md Troubleshooting 표 및 [run-and-verify.md](run-and-verify.md).

## 12. org.videolan.vlc (VLC)

- **화면 시퀀스**: `OnboardingActivity` 첫 페이지(`Welcome to VLC!`) → 좌하단 `SKIP` tap → `MainActivity`(메인).
- **통과 커맨드**: `am start -n org.videolan.vlc/.StartActivity` → `SKIP` bounds 중심 tap.
- **탐색 가능 앵커**: 상단 탭 `VIDEOS`/`PLAYLISTS`, 하단 탭 `Video`/`Audio`/`Browse`/`Playlists`/`More`.
- **판정**: no-login-usable. 미디어 없음 안내("No media files found...")와 저장소 접근 배너는 로그인과 무관 — 그대로 두고 진행 가능.
