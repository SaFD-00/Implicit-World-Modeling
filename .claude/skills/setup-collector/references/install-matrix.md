# 카탈로그 앱별 설치/로그인/온보딩 커버리지

## Scope Predicate: 설치 자동화 가능(navigable)

설치 자동화 가능한 앱(navigable-candidate) = 다음 모두를 만족:
- APK 캐시 또는 System 내장(apk_cached = yes/built-in)
- `adb install -r -g` 설치 성공 (installs = yes)
- 런타임 권한 grant 이후 비크래시 실행
- 로그인 게이트 없이 탐색 가능 앵커 도달 (로컬 콘텐츠/UI 앞)

## 버킷별 커버리지

| 버킷 | 개수 | 설명 |
|------|------|------|
| navigable-candidate | 36 | 자동화 가능 |
| login-wall | 4 | 계정 필수(로그인 게이트) |
| login-required | 1 | 로그인 필요(비-Google; k9=이메일 IMAP 계정) |
| login-required(Google) | 10 | Google 계정 필수(설치 미시도 또는 실패) |
| crash-or-blank·install-failed | 80 | 설치/실행 실패 |

합계 = 36+4+1+10+80 = 131 (triage 데이터 행과 일치).

## 앱 목록 (버킷별 정렬)

| package_id | app_name | source | apk_cached | installs | launch_bucket | login_required | onboarding_ref | notes |
|-------------|----------|--------|-----------|---------|----------------|-----------------|-----------------|-------|
| app.organicmaps | Organic Maps | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | download 69MB map gate before browse, no login |
| code.name.monkey.retromusic | Retro Music | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| com.ajnsnewmedia.kitchenstories | Kitchen Stories | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | welcome carousel Continue, no login gate seen yet |
| com.android.settings | Settings | System | built-in | no | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented;System;no apk (platform built-in) |
| com.chess | Chess.com | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | prominent login UI but 'Play as Guest' bypass link present |
| com.espn.score_center | ESPN | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | slow load (~10s) before onboarding renders; SIGN UP/LOG IN/SIGN UP LATER guest bypass |
| com.flauschcode.broccoli | Broccoli | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| com.google.android.deskclock | Clock | PlayStore | yes | no | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented;preinstalled(baseline system image); apk-reinstall-failed(MISSING_SPLIT, harmless) |
| com.google.android.dialer | Phone | System | built-in | no | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented;System;no apk (platform built-in) |
| com.ichi2.anki | Anki | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| com.iudesk.android.photo.editor | Photo Editor | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | changelog dialog OK first |
| com.jerboa | Lemmy | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | donation/changelog dialog (Done) first; post-dismiss screen unconfirmed (server/login possible, shallow scope stopped here) |
| com.nononsenseapps.feeder | Feeder | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| com.simplemobiletools.musicplayer | Simple Music Player | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| com.thetrainline | Trainline | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | immersive-mode Got it dismiss only on this screen; post-dismiss unconfirmed |
| com.wunderground.android.weather | Weather Underground | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | location permission dialog (I UNDERSTAND/TURN LOCATION OFF), no login |
| com.xatori.Plugshare | PlugShare | PlayStore | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | GET STARTED welcome, no login gate seen yet |
| de.danoeh.antennapod | AntennaPod | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| de.dennisguse.opentracks | OpenTracks | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| de.markusfisch.android.binaryeye | Binary Eye | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | Simple/Advanced mode picker, no login |
| fr.neamar.kiss | KISS Launcher | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | launcher/search app |
| me.zhanghai.android.files | Material Files | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | manage-all-files permission dialog first |
| net.cozic.joplin | Joplin | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| net.gsantner.markor | Markor | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| net.osmand | OsmAnd | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| net.programmierecke.radiodroid2 | RadioDroid | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| net.sourceforge.opencamera | Open Camera | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | help overlay OK dismiss first |
| org.billthefarmer.diary | Diary | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| org.billthefarmer.tuner | Tuner | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) |  |
| org.isoron.uhabits | Loop Habit Tracker | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | fullscreen-mode Got it dismiss first |
| org.koreader.launcher.fdroid | KOReader | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | manage-all-files permission dialog first |
| org.schabi.newpipe | NewPipe | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | Keep-Android-Open dev-verification warning dialog (OK) first |
| org.tasks | Tasks.org | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| org.videolan.vlc | VLC | F-Droid | yes | yes | navigable-candidate | no | [app-first-run.md](app-first-run.md) | documented |
| org.wikipedia | Wikipedia | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | multi-step intro carousel, no login required |
| troop.com.freedcam | FreeDCam | F-Droid | yes | yes | navigable-candidate | no | [app-first-run-batch2.md](app-first-run-batch2.md) | resolve-activity returns ResolverActivity (ambiguous); launched via freed.cam.ActivityFreeDcamMain directly |
| com.eventbrite.attendee | Eventbrite | PlayStore | yes | yes | login-wall | yes |  | Continue with email/Facebook Social Sign In gate, no guest option seen; launch needs literal $ escaped in am start -n |
| com.nike.ntc | Nike Training Club | PlayStore | yes | yes | login-wall | yes |  | Join Us / Sign In only, no guest option seen |
| com.nike.plusgps | Nike Run Club | PlayStore | yes | yes | login-wall | yes |  | Join Us / Sign In only, no guest option seen |
| org.joinmastodon.android | Mastodon | F-Droid | yes | yes | login-wall | yes |  | server pick + Log in/Create account gate, no guest-browse option seen |
| com.fsck.k9 | K-9 Mail | F-Droid | yes | yes | login-required | yes | [app-first-run-batch2.md](app-first-run-batch2.md) | Get started welcome; reaches anchor screen without login / U2 정정(2026-07-14): "Get started" 다음은 "Sign in with Thundermail"/QR/이메일 입력뿐인 계정설정 화면이고 skip/guest 없음 — 실제로는 login-required(mail client는 계정 없이 받은편지함 자체가 없음). "reaches anchor without login"은 welcome 스플래시를 앵커로 오인한 것이었음. batch2.md 참조 |
| au.com.shiftyjelly.pocketcasts | Pocket Casts | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| bbc.mobile.news.ww | BBC News | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| ch.protonvpn.android | ProtonVPN | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| cn.wps.moffice_eng | WPS Office | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.Slack | Slack | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.accuweather.android | AccuWeather | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.adsk.sketchbook | Sketchbook | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.agoda.mobile.consumer | Agoda | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.airbnb.android | Airbnb | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.alibaba.aliexpresshd | AliExpress | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.amazon.mShop.android.shopping | Amazon Shopping | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.booking | Booking.com | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.brave.browser | Brave | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.citymapper.app.release | Citymapper | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.clue.android | Clue | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.coingecko.coingeckoapp | CoinGecko | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.contextlogic.wish | Wish | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.dd.doordash | DoorDash | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.deepl.mobiletranslator | DeepL | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.discord | Discord | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.droid4you.application.wallet | Wallet | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.dropbox.android | Dropbox | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.duolingo | Duolingo | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.ebay.mobile | eBay | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.google.android.apps.books | Google Play Books | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.google.android.apps.fitness | Google Fit | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.google.android.apps.magazines | Google News | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.google.android.apps.maps | Google Maps | PlayStore | yes | no | install-failed | - | [app-first-run.md](app-first-run.md) | documented;split/MISSING_SPLIT;documented-on-Pixel6-2-previously; apk here fails on fresh Pixel6-3 |
| com.google.android.apps.nbu.files | Files by Google | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.google.android.apps.youtube.music | YouTube Music | PlayStore | yes | no | install-failed | - | [app-first-run.md](app-first-run.md) | documented;split/MISSING_SPLIT;documented-on-Pixel6-2-previously; apk here fails on fresh Pixel6-3 |
| com.google.android.calculator | Calculator | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.grubhub.android | Grubhub | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.hm.goe | H&M | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.hyperionics.avar | Voice Aloud | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.inditex.zara | ZARA | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.ingka.ikea.app | IKEA | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.instacart.client | Instacart | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.kouzoh.mercari | Mercari | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.kroger.mobile | Kroger | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.memrise.android.memrisecompanion | Memrise | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.microsoft.office.outlook | Microsoft Outlook | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.microsoft.office.word | Microsoft Word | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.mobilefootie.wc2010 | FotMob | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.mufumbo.android.recipe.search | Cookpad | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.myfitnesspal.android | MyFitnessPal | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.niksoftware.snapseed | Snapseed | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.paypal.android.p2pmobile | PayPal | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.pinterest | Pinterest | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.reddit.frontpage | Reddit | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.redfin.android | Redfin | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.spotify.music | Spotify | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.spotlightsix.zentimerlite2 | Insight Timer | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.strava | Strava | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.thomsonreuters.reuters | Reuters | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.ticktick.task | TickTick | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.todoist | Todoist | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.tranzmate | Moovit | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.tripit | TripIt | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.ubercab | Uber | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.ubercab.eats | Uber Eats | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.uniqlo.kr.catalogue | UNIQLO | PlayStore | yes | no | install-failed | - |  | split/INVALID_APK |
| com.walmart.android | Walmart | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.whatsapp | WhatsApp | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.xe.currency | XE Currency | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.yahoo.mobile.client.android.finance | Yahoo Finance | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.zillow.android.zillowmap | Zillow | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.zzkko | SHEIN | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| flipboard.app | Flipboard | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| md.obsidian | Obsidian | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| me.lyft.android | Lyft | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| mnn.Android | AP News | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| net.sharewire.parkmobilev2 | ParkMobile | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| net.skyscanner.android.main | Skyscanner | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| org.coursera.android | Coursera | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| org.khanacademy.android | Khan Academy | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| org.mozilla.firefox | Firefox | PlayStore | yes | no | install-failed | - |  | split/INVALID_APK |
| org.plantnet | PlantNet | PlayStore | yes | no | install-failed | - |  | split/INVALID_APK |
| org.telegram.messenger | Telegram | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| org.thoughtcrime.securesms | Signal | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| tv.twitch.android.app | Twitch | PlayStore | yes | no | install-failed | - |  | split/MISSING_SPLIT |
| com.google.android.apps.docs | Google Drive | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.apps.docs.editors.docs | Google Docs | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.apps.docs.editors.sheets | Google Sheets | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.apps.docs.editors.slides | Google Slides | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.apps.photos | Google Photos | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);INVALID_APK |
| com.google.android.calendar | Google Calendar | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.contacts | Contacts | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.gm | Gmail | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.keep | Google Keep | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);MISSING_SPLIT |
| com.google.android.youtube | YouTube | PlayStore | yes | no | login-required(Google) | yes |  | login-required(Google)-group;no launch attempted (Google account gate);INVALID_APK |
