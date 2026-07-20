# 앱별 첫 실행 온보딩 관찰 (§5-a, batch 2)

U1이 라이브로 확정한 신규 navigable 27종(F-Droid 20 + PlayStore standalone-APK 7)의 첫 실행 온보딩을 Pixel6-3(`emulator-5558`)에서 1회씩 실측한 기록. 기존 12종(retromusic/settings/deskclock/dialer/opentracks/joplin/markor/osmand/tasks/vlc/maps/youtube.music)은 [app-first-run.md](app-first-run.md)에 이미 문서화돼 있어 이 문서의 대상이 아니다.

**공용 규약은 [app-first-run.md](app-first-run.md) 상단과 동일**(좌표 하드코딩 금지·dump→bounds tap, `&amp;` 정규화, `monkey -c LAUNCHER` no-op이므로 `am start -n`, 지도류는 dump 실패 시 스크린샷+픽셀 스캔) — Pixel6-3에서 실측했다는 점만 다르다. 헬퍼는 `_skill_pixel6_3_20260714/helper.sh`(SERIAL=emulator-5558) + `dump_and_find.py`.

## 이번 라운드에서 추가로 확인된 공용 규약(12종 문서에는 없던 것)

- **dump 실패는 "지도류"에 국한되지 않는다**: `org.koreader.launcher.fdroid`(전자책 리더, 자체 렌더링 캔버스)와 `com.xatori.Plugshare`(EV 충전소 지도)도 `uiautomator dump`가 빈 화면 또는 `ERROR: could not get idle state.`를 반환했다. 두 경우 다 스크린샷으로 우회해 앵커 도달을 확인했다. **dump가 비정상적으로 짧거나 실패하면, 직전에 성공한 다른 앱의 stale `/sdcard/window_dump.xml`을 pull해올 수 있다** — 내용이 현재 foreground 앱과 안 맞으면 즉시 재-dump하거나 스크린샷으로 교차검증할 것(`com.xatori.Plugshare`에서 실제로 겪음: `com.wunderground.android.weather`의 XML을 잘못 pull).
- **MANAGE_EXTERNAL_STORAGE 시스템 설정 화면에서 "back"은 요청 앱으로 안 돌아올 수 있다**: 권한을 grant한 뒤 `KEYCODE_BACK`을 누르면 Android가 해당 설정 액티비티를 요청 앱의 태스크가 아니라 별개 태스크로 열어둔 경우, back이 recents상 다른 앱(예: 방금 백그라운드로 간 이전 타깃 앱)으로 떨어질 수 있다(`org.koreader.launcher.fdroid` 권한 부여 후 `org.isoron.uhabits`로 떨어진 사례). back에 의존하지 말고 **`am start -n`으로 대상 앱을 명시적으로 재실행**할 것.
- **랜드스케이프 강제 앱**(`troop.com.freedcam`)은 dump의 bounds 좌표계 자체가 2400x1080(가로)으로 나온다 — tap 좌표를 그 좌표계 그대로 쓰면 된다(별도 회전 변환 불필요, `input tap`이 실제 화면 방향을 따라간다).

---

## 1. app.organicmaps (OrganicMaps)

- **화면 시퀀스**: `DownloadResourcesActivity`(실제로는 `DownloadResourcesLegacyActivity`로 리다이렉트) 진입 시 "Download the world overview map"(69MB) 게이트 → `DOWNLOAD` 외 스킵 옵션 없음(뒤로가기는 앱 자체를 종료) → 다운로드 완료(에뮬레이터 네트워크에서 실측 약 20초) 후 자동으로 `MwmActivity`(메인 지도)로 전환.
- **통과 커맨드**: `am start -n app.organicmaps/.DownloadResourcesActivity` → `DOWNLOAD` bounds 중심 tap → 다운로드 완료 대기(약 20초).
- **탐색 가능 앵커**: 지도 화면의 `Search`/`Bookmarks`/`Menu`/`Help` FAB, `Zoom in`/`Zoom out`/`My Position`.
- **판정**: no-login-usable. 로그인 개념 자체 없음.
- **특이사항**: 다운로드 게이트는 **스킵 불가**(뒤로가기 시 앱 종료, 재실행하면 게이트 재출현) — 반드시 DOWNLOAD를 누르고 완료까지 기다려야 한다. 완료 후에는 멱등(`force-stop`→재실행 시 바로 `MwmActivity`로 직행, 게이트 재출현 없음).

## 2. com.flauschcode.broccoli (Broccoli 레시피 관리자)

- **화면 시퀀스**: 온보딩 없음 — `MainActivity` 진입이 곧 메인 "Recipes" 리스트("No recipes found.").
- **통과 커맨드**: `am start -n com.flauschcode.broccoli/.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: `Categories` 스피너, `Search`, `New Recipe` FAB.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 3. com.fsck.k9 (K-9 Mail / Thunderbird for Android) — login-required로 정정

- **화면 시퀀스**: `FeatureLauncherActivity` 진입 시 "K-9 Mail" 웰컴 화면(`Get started`/`Import settings`) → `Get started` tap → 계정 설정 화면("Sign in with Thundermail" / "Scan QR code" / 이메일 주소 입력 + `Next`).
- **통과 커맨드(막힌 지점까지)**: `am start -n com.fsck.k9/net.thunderbird.app.common.MainActivity` → `Get started` bounds 중심 tap.
- **탐색 가능 앵커**: 없음 — 계정(이메일) 없이는 받은편지함 자체가 존재하지 않는다. `Import settings`도 기존 백업 파일이 필요해 마찬가지로 막힘.
- **판정**: **login-required**(메일 클라이언트 구조상 guest/skip 경로 없음).
- **특이사항**: triage.tsv 원 노트("reaches anchor screen without login")는 웰컴 스플래시를 앵커로 오인한 것이었음 — `Get started` 다음 화면(계정 설정)이 실제 첫 실질 화면이고 여기서 막힌다. triage.tsv에 정정 사유 append 완료.

## 4. com.ichi2.anki (AnkiDroid)

- **화면 시퀀스**: `IntentHandler`가 `IntroductionActivity`로 리다이렉트 → "Study less / Remember more" 웰컴(`Get Started` vs `Sync from AnkiWeb`) → `Get Started` tap → 퍼미션 안내 화면("All files access" 스위치 + `Continue`) → 스위치 tap 시 시스템 "Allow access to manage all files" 설정 화면으로 이동 → 토글 on → `am start`로 앱 재진입(또는 `KEYCODE_BACK`, 단 아래 특이사항 참조) → 자동으로 `DeckPicker`(메인)로 진행.
- **통과 커맨드**: `am start -n com.ichi2.anki/.IntentHandler` → `Get Started` bounds 중심 tap → "All files access" 스위치 bounds 중심 tap → 시스템 설정에서 "Allow access to manage all files" 행 bounds 중심 tap → `am start -n com.ichi2.anki/.IntentHandler` 재실행(또는 `KEYCODE_BACK`이 우연히 앱으로 돌아올 수도 있으나 보장 안 됨 — 공용 규약 참조).
- **탐색 가능 앵커**: `DeckPicker`의 `Add` FAB, "Collection is empty" 안내, `Sync`/`More options` 툴바.
- **판정**: no-login-usable. `Sync from AnkiWeb`은 선택 사항이고 `Get Started`만으로 로그인 없이 완전 사용 가능.
- **특이사항**: 퍼미션 화면의 `Continue` 버튼은 스위치를 먼저 켜지 않으면 무반응(같은 화면 유지) — 스위치를 직접 tap해야 시스템 설정으로 넘어간다. 멱등(재실행 시 바로 `DeckPicker`).

## 5. com.jerboa (Jerboa, Lemmy 클라이언트)

- **화면 시퀀스**: `MainActivity` 진입 시 후원 안내 다이얼로그("Support Jerboa" + 체인지로그, `Done`) → Google 개발자 인증 경고 다이얼로그(`OK`) → **로그인/인스턴스 선택 없이 곧장** 기본 인스턴스의 공개 피드(Local/Active) 표시.
- **통과 커맨드**: `am start -n com.jerboa/.MainActivity` → `Done` bounds 중심 tap → `OK` bounds 중심 tap.
- **탐색 가능 앵커**: 피드의 게시물(upvote/downvote/bookmark/more), 하단 탭 `Home`/`Search`/`Inbox`/`Saved`/`Profile`.
- **판정**: no-login-usable. U1이 미확인으로 남긴 "post-dismiss 화면"은 인스턴스 선택 게이트가 아니라 바로 사용 가능한 연합 피드였음 — login-wall 우려는 기각.
- **특이사항**: 멱등(재실행 시 두 다이얼로그 모두 재출현 없이 바로 피드).

## 6. com.nononsenseapps.feeder (Feeder RSS 리더)

- **화면 시퀀스**: 온보딩 없음 — `MainActivity` 진입이 곧 기본 제공 "Feeder News" 피드로 채워진 메인 리스트.
- **통과 커맨드**: `am start -n com.nononsenseapps.feeder/.ui.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: `All feeds` 드로어, `Search`/`Filter`, `Mark all as read`.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 7. com.simplemobiletools.musicplayer (Simple Music Player)

- **화면 시퀀스**: 온보딩 없음 — `SplashActivity.Orange`가 즉시 `MainActivity`로 리다이렉트, 하단 탭 `Playlists`/`Folders`/`Artists`/`Albums`/`Tracks` 노출.
- **통과 커맨드**: `am start -n com.simplemobiletools.musicplayer/.activities.SplashActivity.Orange` — 조작 불필요.
- **탐색 가능 앵커**: 상단 `Search`/`Sort by`/`Equalizer`, 하단 탭 5종.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 8. de.danoeh.antennapod (AntennaPod)

- **화면 시퀀스**: 온보딩 없음 — `SplashActivity`가 즉시 `MainActivity`(Home)로 진입, "Welcome to AntennaPod! You are not subscribed to any podcasts yet." 안내.
- **통과 커맨드**: `am start -n de.danoeh.antennapod/.activity.SplashActivity` — 조작 불필요.
- **탐색 가능 앵커**: 하단 탭 `Home`/`Queue`/`Inbox`/`Subscriptions`/`More`.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 9. de.markusfisch.android.binaryeye (Binary Eye, 바코드/QR 스캐너)

- **화면 시퀀스**: 온보딩 없음(이 라운드 실측 시점) — `SplashActivity`가 즉시 `CameraActivity`(스캔 화면)로 진입.
- **통과 커맨드**: `am start -n de.markusfisch.android.binaryeye/.activity.SplashActivity` — 조작 불필요.
- **탐색 가능 앵커**: "Scan code" 카메라 뷰, `History`/`Compose barcode`/`Toggle flash`.
- **판정**: no-login-usable.
- **특이사항**: triage.tsv 힌트("Simple/Advanced mode picker")는 이번 실측에서 재현되지 않음 — U1의 최초 트리아지 실행 때 1회성으로 나타났다가 이후 재출현하지 않는 다이얼로그였을 가능성. 최초 설치 후 첫 실행이 아니라면 이 다이얼로그를 기대하지 말 것.

## 10. fr.neamar.kiss (KISS 런처/검색)

- **화면 시퀀스**: 온보딩 없음 — `MainActivity` 진입이 곧 "Start searching for anything" 팁 카드 + 즐겨찾기 아이콘(Phone/Contacts/Chrome) + 하단 검색바가 있는 메인 화면.
- **통과 커맨드**: `am start -n fr.neamar.kiss/.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: 하단 `Search apps, contacts, …` 입력창, `Display app list`/`Menu`.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 11. me.zhanghai.android.files (Files, 파일 관리자)

- **화면 시퀀스**: `FileListActivity` 진입 시 "App needs access to manage all files" 다이얼로그(`Cancel`/`OK`) → `OK` tap 시 시스템 "Allow access to manage all files" 설정 화면 → 토글 on → `KEYCODE_BACK` → 자동으로 내부 저장소 파일 리스트로 복귀.
- **통과 커맨드**: `am start -n me.zhanghai.android.files/.filelist.FileListActivity` → `OK` bounds 중심 tap → 시스템 설정에서 "Allow access to manage all files" 행 bounds 중심 tap → `KEYCODE_BACK`.
- **탐색 가능 앵커**: "Internal shared storage" 파일/폴더 그리드, `Search`/`View and sort`.
- **판정**: no-login-usable.
- **특이사항**: 이 앱은 `back`이 정상적으로 요청 앱(Files)으로 돌아왔음(공용 규약의 "back이 안 돌아올 수 있다" 사례와 달리 이 케이스는 정상 동작) — 앱마다 다를 수 있으니 back 후 `curfocus`로 확인 권장. 멱등(재실행 시 바로 파일 리스트).

## 12. net.programmierecke.radiodroid2 (RadioDroid, 인터넷 라디오)

- **화면 시퀀스**: 온보딩 없음 — `ActivityMain` 진입이 곧 실시간 방송국 리스트(Local/Top Click/Top Vote/Changed Lately 탭)가 채워진 "Stations" 메인 화면.
- **통과 커맨드**: `am start -n net.programmierecke.radiodroid2/.ActivityMain` — 조작 불필요.
- **탐색 가능 앵커**: `Search`/`Sleep timer`, 하단 탭 `Stations`/`Favorites`/`History`/`Alarm`/`Settings`.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 13. net.sourceforge.opencamera (Open Camera)

- **화면 시퀀스**: 온보딩 없음(이 라운드 실측 시점) — `MainActivity` 진입이 곧 카메라 촬영 화면.
- **통과 커맨드**: `am start -n net.sourceforge.opencamera/.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: `Take Photo`/`Switch to video mode`/`Gallery`, 좌상단 `Lock exposure`/`Exposure`/`Popup settings`/`Settings`.
- **판정**: no-login-usable.
- **특이사항**: triage.tsv 힌트("help overlay OK dismiss first")는 이번 실측에서 재현되지 않음 — binaryeye와 마찬가지로 최초 1회성 다이얼로그였을 가능성.

## 14. org.billthefarmer.diary (Diary)

- **화면 시퀀스**: 온보딩 없음 — `Diary` 액티비티 진입이 곧 오늘 날짜 일기 편집 화면(WebView).
- **통과 커맨드**: `am start -n org.billthefarmer.diary/.Diary` — 조작 불필요.
- **탐색 가능 앵커**: 상단 날짜("Jul 14, 2026"), `Cancel`/`Previous`/`More options`.
- **판정**: no-login-usable.
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 15. org.billthefarmer.tuner (Tuner, 악기 튜너)

- **화면 시퀀스**: 온보딩 없음 — `Tuner` 액티비티 진입이 곧 튜닝 UI("Equal Temperament" 드롭다운 포함).
- **통과 커맨드**: `am start -n org.billthefarmer.tuner/.Tuner` — 조작 불필요.
- **탐색 가능 앵커**: 상단 `Equal Temperament` 스피너, 메인 튜닝 게이지 영역.
- **판정**: no-login-usable. 마이크 권한 다이얼로그는 관찰되지 않음(설치 시 `-g` 선grant 추정).
- **특이사항**: 없음(온보딩 없음, 재실행 시 바로 메인 화면 — 멱등).

## 16. org.isoron.uhabits (Loop Habit Tracker)

- **화면 시퀀스**: 온보딩 없음(이 라운드 실측 시점) — `MainActivity` 진입이 곧 "Habits" 리스트("You have no active habits").
- **통과 커맨드**: `am start -n org.isoron.uhabits/.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: `Add habit`/`Filter`/`More options`.
- **판정**: no-login-usable.
- **특이사항**: triage.tsv 힌트("fullscreen-mode Got it dismiss first")는 이번 실측에서 재현되지 않음(최초 1회성 추정).

## 17. org.koreader.launcher.fdroid (KOReader) — dump 실패 사례

- **화면 시퀀스**: `MainActivity` 진입 시 "Please allow the app to manage all files." 다이얼로그(`OK`) → 시스템 "Allow access to manage all files" 설정 화면 → 토글 on → **`back`이 요청 앱이 아니라 직전 타깃 앱(`org.isoron.uhabits`)으로 떨어짐**(공용 규약 참조) → `am start -n`으로 KOReader 명시 재실행 → `ImmersiveModeConfirmation`("Viewing full screen... Got it") → `Got it` tap → 내장 "QUICKSTART GUIDE" 문서가 자동으로 열림.
- **통과 커맨드**: `am start -n org.koreader.launcher.fdroid/org.koreader.launcher.MainActivity` → `OK` bounds 중심 tap → 시스템 설정에서 "Allow access to manage all files" 행 bounds 중심 tap → `am start -n org.koreader.launcher.fdroid/org.koreader.launcher.MainActivity`(재실행, back 대신) → `Got it` bounds 중심 tap.
- **탐색 가능 앵커**: QUICKSTART GUIDE 문서 뷰(리더 자체가 곧 앵커 — 문서를 읽고 있는 상태가 "탐색 시작"). **이 화면은 uiautomator dump가 빈 FrameLayout만 반환**(SDL/자체 렌더링 캔버스 추정) — 스크린샷으로만 확인 가능.
- **판정**: no-login-usable.
- **특이사항**: 퍼미션 부여 후 첫 재실행에서만 `ImmersiveModeConfirmation` 토스트가 뜨고, 이후 재실행(멱등 확인용 force-stop→재실행)에서는 재출현하지 않음 — 문서 뷰로 바로 직행. dump가 항상 실패하므로 이 앱을 대상으로 한 replay 스크립트는 처음부터 스크린샷 기반으로 설계할 것.

## 18. org.schabi.newpipe (NewPipe)

- **화면 시퀀스**: `MainActivity` 진입 시 "Keep Android Open" Google 개발자 인증 경고 다이얼로그(`SOLUTION`/`DETAILS`/`OK`) → `OK` tap → "Check for updates" 확인 다이얼로그(`NO`/`YES`) → 선택 후 곧장 메인 화면(Live/What's New/Subscriptions/Bookmarked Playlists 탭, YouTube 라이브 스트림 목록).
- **통과 커맨드**: `am start -n org.schabi.newpipe/.MainActivity` → `OK` bounds 중심 tap → `NO` bounds 중심 tap(업데이트 확인 비활성화 — `YES`를 선택해도 무방).
- **탐색 가능 앵커**: `Search`, 하단 없음(상단 탭 구조) — Live/What's New/Subscriptions/Bookmarked Playlists.
- **판정**: no-login-usable.
- **특이사항**: 멱등(재실행 시 두 다이얼로그 모두 재출현 없이 바로 메인 화면).

## 19. org.wikipedia (Wikipedia)

- **화면 시퀀스**: `DefaultIcon`이 `InitialOnboardingActivity`로 리다이렉트 → 4단계 캐러셀("All the world's knowledge" → "Data & Privacy" → "Read in more than 300 languages" → "Follow your curiosity") → 마지막 페이지에서 "What are you interested in?"(관심사 선택, 최소 1개 선택 안 하면 `Next` 비활성 텍스트 표시) → `Skip` tap → 홈 피드(Featured article 등).
- **통과 커맨드**: `am start -n org.wikipedia/.DefaultIcon` → `Forward` bounds 중심 tap ×3 → 관심사 화면에서 `Skip` bounds 중심 tap.
- **탐색 가능 앵커**: 홈 피드의 "Today - Jul 14, 2026" / "Featured article", 하단 탭 `Home`/`Saved`/`Search`/`Activity`/`More`.
- **판정**: no-login-usable. 계정 생성은 전체 과정에서 한 번도 강제되지 않음.
- **특이사항**: 멱등(재실행 시 `InitialOnboardingActivity` 없이 바로 `DefaultIcon`이 홈 피드로 직행).

## 20. troop.com.freedcam (FreeDcam) — 랜드스케이프 강제 + 비멱등 튜토리얼

- **화면 시퀀스**: `resolve-activity`가 `ResolverActivity`(모호)를 반환하므로 `freed.cam.ActivityFreeDcamMain`을 직접 지정 실행 → 랜드스케이프로 강제 전환된 카메라 UI 위에 4단계 제스처 튜토리얼 오버레이("Swipe left→right: Settings" → "right→left: close Settings" → "bottom→top: Manuals" → "top→bottom: close Manual", 각 `NEXT`, 마지막은 `CLOSE` + "don't show again" 체크박스) → `CLOSE` tap → 전체 매뉴얼 카메라 UI(ISO/셔터/노출/모드/촬영버튼).
- **통과 커맨드**: `am start -n troop.com.freedcam/freed.cam.ActivityFreeDcamMain` → `NEXT` bounds 중심 tap(랜드스케이프 2400x1080 좌표계, 공용 규약 참조) ×3 → `CLOSE` bounds 중심 tap(같은 위치).
- **탐색 가능 앵커**: 촬영 버튼(우측 큰 원형), 좌측 하단 ISO/Auto/셔터스피드/EV/버스트 카운터, 우측 온도(WB)/플래시/AF모드/AE 토글.
- **판정**: no-login-usable.
- **특이사항**: 이 라운드 비멱등 사례 2건 중 하나(**튜토리얼 오버레이 재출현** 유형; 나머지는 §22 Chess의 게스트 세션 로그인 게이트 재출현) — "don't show again" 체크박스를 tap하지 않으면(4단계 `NEXT`/`CLOSE`만으로 통과) `force-stop`→재실행 시 튜토리얼이 **다시 나타난다**. 재수집 세션에서 "한 번 통과했으니 재출현 안 함"을 가정하면 안 되고, 매 launch 마다 4탭 재통과가 필요하거나, 통과 시 체크박스까지 tap해 영속화해야 한다.

## 21. com.ajnsnewmedia.kitchenstories (Kitchen Stories, 레시피)

- **화면 시퀀스**: `MainActivity` 진입 시 3단계 웰컴 캐러셀("Welcome to Kitchen Stories" × 3장, 각 `Continue`) → 4번째 화면은 "Get started"(회원가입 유도, `Continue with Google`/이메일/`Register with email`/`Log in here`) — 우상단 `X`(close) tap → "How your free trial works" 유료 체험 유도 화면(우상단 `X`) → `X` tap → 홈 피드(Editor's Choice/For You 탭, Today's Recipe)로 직행.
- **통과 커맨드**: `am start -n com.ajnsnewmedia.kitchenstories/.MainActivity` → `Continue` bounds 중심 tap ×3 → 회원가입 화면 우상단 `X` bounds 중심 tap → 체험 안내 화면 우상단 `X` bounds 중심 tap.
- **탐색 가능 앵커**: 홈 피드의 "Today's Recipe", "Our Latest Recipes", 하단 탭 `Home`/`Search`/`My Recipes`/`Shopping List`/`Profile`.
- **판정**: no-login-usable. 회원가입/유료체험 화면 둘 다 `X`로 닫으면 로그인 없이 완전 탐색 가능.
- **특이사항**: 멱등(재실행 시 캐러셀·회원가입·체험안내 전부 재출현 없이 바로 홈 피드).

## 22. com.chess (Chess.com) — 비멱등 로그인 유도

- **화면 시퀀스**: `SplashActivity` → `SignupActivity`("Play. Learn. Have Fun!", `Continue with Google`/`Continue with Email`/`Play as Guest`) → `Play as Guest` tap → `HomeActivity`(Play Online/Solve Puzzles/Daily Puzzle/Play Bots/Play Coach/Learn 그리드).
- **통과 커맨드**: `am start -n com.chess/.splash.SplashActivity` → `Play as Guest` bounds 중심 tap.
- **탐색 가능 앵커**: `HomeActivity`의 카드 그리드, 하단 탭 `Home`/`Puzzles`/`Learn`/`Watch`/`More`.
- **판정**: no-login-usable.
- **특이사항**: 이 라운드 비멱등 사례 2건 중 하나(**게스트 세션 로그인 게이트 재출현** 유형; 나머지는 §20 FreeDcam 튜토리얼 오버레이) — `force-stop` 후 재실행하면 `SignupActivity`가 **다시 나타난다**(YouTube Music·FreeDcam과 같은 패턴 — 게스트 세션이 프로세스 재시작을 못 버팀). 재수집 세션마다 `Play as Guest` 재통과가 필요.

## 23. com.espn.score_center (ESPN) — 6단계 스킵 체인

- **화면 시퀀스**: `EspnLaunchActivity` → (느린 로드, 실측 약 10~12초) → `EspnOnboardingActivity`(`SIGN UP`/`LOG IN`/`SIGN UP LATER`) → `SIGN UP LATER` tap → `EditionSwitchActivity`("Select Edition", 지역 라디오 버튼 목록, 기본값 U.S. - English) → `Next` tap → `FavoriteSportsActivity`("Tap your favorite leagues", `Skip`) → `Skip` tap → `FavoriteTeamsActivity`("Tap your favorite teams", `Skip`) → `Skip` tap → `FavoriteContributorsActivity`("Add Reporters", `Finish`) → `Finish` tap → "You didn't customize the app!" 확인 다이얼로그(`GO BACK`/`OK`) → `OK` tap → `ClubhouseBrowserActivity`(홈 뉴스피드).
- **통과 커맨드**: `am start -n com.espn.score_center/com.espn.sportscenter.ui.EspnLaunchActivity` → (10초 이상 대기) → `SIGN UP LATER` bounds 중심 tap → `Next` bounds 중심 tap → `Skip` bounds 중심 tap → `Skip` bounds 중심 tap → `Finish` bounds 중심 tap → `OK` bounds 중심 tap.
- **탐색 가능 앵커**: 홈 피드 기사 목록, 하단 탭 `Home`/`Scores`/`Watch`/`Verts`/`More`.
- **판정**: no-login-usable. 리그/팀/기자 커스터마이징 전부 스킵 가능.
- **특이사항**: 첫 화면 렌더링까지 지연이 커서(10초 이상) dump/tap 전에 충분히 대기해야 한다(그렇지 않으면 빈 화면 또는 스플래시를 앵커로 오인). 멱등(재실행 시 6단계 전부 생략하고 바로 뉴스피드 — 단, 초기 로드 지연은 재실행 때도 동일하게 발생).

## 24. com.iudesk.android.photo.editor (Photo Editor)

- **화면 시퀀스**: 온보딩 없음(이 라운드 실측 시점) — `MainActivity` 진입이 곧 도구 그리드(Gallery/New/Tools/Gallery Apps/Recent Photos/Rate/Camera/Batch).
- **통과 커맨드**: `am start -n com.iudesk.android.photo.editor/app.activity.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: `Built-In Gallery`, 도구 그리드 8종.
- **판정**: no-login-usable.
- **특이사항**: triage.tsv 힌트("changelog dialog OK first")는 이번 실측에서 재현되지 않음(최초 1회성 추정).

## 25. com.thetrainline (Trainline)

- **화면 시퀀스**: `SplashScreenActivity` → `OnboardingActivity`(쿠키/개인정보 동의, "Your privacy and us", `Accept cookies`/`Choose cookies`) → `Accept cookies` tap → "Welcome aboard" 로그인 유도 화면(Sign in video, `Sign in` 버튼만 노출 — 스킵 링크 없음, `back` 누르면 앱 전체가 종료됨) → **재실행하면 이 로그인 유도 화면 없이 바로 `HomeActivity`로 직행**(즉 이 화면은 매 세션 재출현하는 게이트가 아니라 최초 동의 직후 1회성 넛지).
- **통과 커맨드**: `am start -n com.thetrainline/.activities.home_screen.SplashScreenActivity` → `Accept cookies` bounds 중심 tap → (최초 실행 시 "Welcome aboard" 화면이 뜨면 `back` 또는 재실행으로 넘어감 — 아래 특이사항 참조).
- **탐색 가능 앵커**: `HomeActivity`의 "Search all trains" 검색바, "Popular journeys", 하단 탭 `Search`/`My Tickets`/`Account`.
- **판정**: no-login-usable. `Search`/`My Tickets`는 로그인 없이 사용 가능, `Account` 탭만 `Sign in` 유도.
- **특이사항**: triage.tsv의 "post-dismiss unconfirmed"를 해소 — 로그인 유도 화면에서 `back`을 누르면 앱이 통째로 종료되지만, 그 상태로 **재실행하면 쿠키 동의도 로그인 유도도 건너뛰고 곧장 `HomeActivity`**에 도달한다(즉 쿠키 동의 완료가 진짜 게이트이고, 로그인 유도는 매 세션 다시 볼 필요 없는 1회성 화면). 재수집 스크립트는 `Accept cookies` 한 번만 통과시키면 됨.

## 26. com.wunderground.android.weather (Weather Underground)

- **화면 시퀀스**: `WeatherHomeActivity`(스플래시) → `GdprWUOnBoardingActivity`("Location and Your Weather", `I UNDERSTAND`/`TURN LOCATION OFF`) → `I UNDERSTAND` tap → `OnBoardingScreenActivity`("Where would you like to view the weather?", `CURRENT LOCATION (GPS)`/`SEARCH FOR A LOCATION`) → `CURRENT LOCATION (GPS)` tap → 알림 권한 안내(`ALLOW`/`DON'T ALLOW`) → `ALLOW` tap → `HomeScreenActivity`(현재 위치 "Mountain View, CA" 기준 날씨).
- **통과 커맨드**: `am start -n com.wunderground.android.weather/.ui.splash.WeatherHomeActivity` → `I UNDERSTAND` bounds 중심 tap → `CURRENT LOCATION (GPS)` bounds 중심 tap → `ALLOW` bounds 중심 tap.
- **탐색 가능 앵커**: `HomeScreenActivity`의 `CURRENT CONDITIONS`, `WUNDERMAP`(지도), `DAILY FORECAST`, 상단 위치명/검색/설정.
- **판정**: no-login-usable.
- **특이사항**: 멱등(재실행 시 3단계 전부 생략하고 바로 `HomeScreenActivity`).

## 27. com.xatori.Plugshare (PlugShare, EV 충전소 지도) — dump 실패 사례

- **화면 시퀀스**: `SplashScreenActivity`→`MainActivity`(지도+바텀시트). 최초 1회 `Welcome to PlugShare`+`GET STARTED` 웰컴이 존재하나(U1 트리아지 `com.xatori.Plugshare-01.xml` 확인) U1 트리아지에서 소모돼 U2 실측 시 재현 안 됨 — replay는 `GET STARTED`가 뜨면 tap, 없으면 그대로 진행하도록 방어적으로 설계.
- **통과 커맨드**: `am start -n com.xatori.Plugshare/com.xatori.plugshare.ui.main.MainActivity` — 조작 불필요.
- **탐색 가능 앵커**: 지도 위 충전소 마커, 하단 바텀시트의 `Search`/`All Filters`/`Available`/`2+ Chargers`/`Fast` 필터 칩과 충전소 목록("1804 N Shoreline Blvd" 등), 하단 탭 `Map`/`Trips`/`Bookmarks`/`Me`.
- **판정**: no-login-usable.
- **특이사항**: **지도 애니메이션 때문에 `uiautomator dump`가 실패**(`ERROR: could not get idle state.`) — 스크린샷으로 확인. 이 실패 상태에서 pull한 XML이 직전에 성공했던 다른 앱(`com.wunderground.android.weather`)의 stale 덤프였던 사례 발생(공용 규약 참조) — 이 앱은 replay 스크립트에서 처음부터 스크린샷 기반으로 설계할 것. 멱등(재실행 시 바로 지도+리스트).
