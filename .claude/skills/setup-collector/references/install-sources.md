# 설치 소스별 카탈로그 APK 확보 (§3)

`catalog/download_apks.py` 가 F-Droid·PlayStore·System 세 소스에서 `catalog/apps.csv` 대상 APK 를 받아 `catalog/apks/{pkg}.apk` 에 캐시하고, 실패/시스템 앱은 `catalog/apks/MISSING.md` 에 기록한다. SKILL.md §3 에서 위임.

## 커맨드 형태

```bash
uv run python -m catalog.download_apks --source all --abi arm64-v8a --playstore-arch arm64 [--only PKG,…] [--force]
```

| 플래그 | 의미 |
|---|---|
| `--source all\|fdroid\|playstore` | 소스 제한(기본 `all`) |
| `--abi` | F-Droid 빌드 ABI 필터 — **호스트와 반드시 일치**(Apple Silicon=`arm64-v8a`) |
| `--playstore-arch arm64\|armv7` | `gplaydl` 아키텍처(Apple Silicon=`arm64`) |
| `--only PKG,…` | 콤마 구분 package_id 로 제한 |
| `--force` | 이미 캐시된 APK 도 재다운로드 |
| `--dry-run` / `--csv` | 실제 다운로드 없이 대상만 확인 / 다른 CSV 지정 |

## F-Droid 경로

- `--source fdroid`(또는 `all`)가 F-Droid 인덱스에서 `--abi` 필터에 맞는 빌드를 받는다.
- **인덱스 이탈 사례(2026-07-14 실측)**: `apps.csv` 에 F-Droid 소스로 등록된 앱 중 일부가 인덱스에서 완전히 빠져 있다 — 예 `com.simplemobiletools.calendar.pro`/`com.simplemobiletools.contacts.pro`/`com.simplemobiletools.gallery.pro`(simplemobiletools 3종), 그 외 `io.github.nicehash.metro`/`com.foobnix.pdf.reader`/`org.gnucash.android` 등 총 16종이 `not in F-Droid index: <pkg>` 로 실패(`MISSING.md` F-Droid 섹션 참조).
  - 증상: 다운로드 실패 + **로컬 APK 캐시에도 없음** + **디바이스에도 이미 미설치** — 즉 현재 카탈로그 파이프라인으로 재현 불가능한 상태.
  - 교훈: **로컬 APK 캐시(`catalog/apks/*.apk`)가 유일한 방어선**이다. 한 번이라도 받아둔 APK 는 계속 로컬 디스크에 보존해야 하며, F-Droid 인덱스가 이후 그 앱을 빼더라도 재다운로드 없이 계속 쓸 수 있다.
  - **복원 옵션(미결정 — 다음 세션 판단 사항)**: (a) F-Droid archive repo 에서 구버전 APK 조회, (b) Fossify 포크(simplemobiletools 후속 프로젝트, 패키지 ID 상이)로 대체. 둘 다 이번 라운드에서 결정하지 않는다.
  - **인덱스는 시간에 따라 변한다**: `net.osmand` 는 2026-07-01 스냅샷(MISSING.md)에서 "not in index"였으나 2026-07-14 재실행에서는 정상 다운로드됨 — 실패 기록을 영구 사실로 취급하지 말고 재시도 가치가 있다.

## PlayStore 경로

- `--source playstore`(또는 `all`)가 `gplaydl` 로 받는다. Apple Silicon 호스트는 `--playstore-arch arm64` 를 반드시 붙인다(§0 이 자동 설정).
- **base-APK-only 한계**: `gplaydl` 은 base APK 만 저장한다 — split APK(언어팩/ABI별 분리)가 있는 앱은 일부 리소스가 빠질 수 있다(설치는 되나 완전하지 않을 수 있음). `MISSING.md` 에 경고 기록.
- 실패 유형(실측 6종): 전부 `App not found`(package_id 오탈자/앱 리네임/지역 제한) 또는 `Token expired`(재인증 후 재시도하면 대개 해결).

### PlayStore 라이브 검증 결과 (Pixel6-3, 2026-07-14)

Pixel6-3(emulator-5558)에서 실측한 PlayStore 107개 앱 설치 결과:
- **standalone base-APK로 설치+탐색 성공**: 약 8개(com.chess, com.espn.score_center, com.wunderground.android.weather, com.xatori.Plugshare, com.ajnsnewmedia.kitchenstories, com.thetrainline, com.iudesk.android.photo.editor 등)
- **INSTALL_FAILED_MISSING_SPLIT**: 86개 — App Bundle 형식 앱으로 base APK만으로는 불완전(split APK 파일들이 필요)
- **INSTALL_FAILED_INVALID_APK**: 5개 — gplaydl 다운로드 오류 또는 손상

**결론**: gplaydl의 base-APK-only 방식은 monolithic APK 방식 앱에만 통한다. split-APK 앱(App Bundle)은 split 파일 전체를 다운로드 + `adb install-multiple`로 동시 설치해야 하며, 현재 파이프라인은 이를 미지원한다. 상용 앱 상당수는 설치되더라도 login-wall로 인해 로그인 없이 탐색 불가능.

## System 앱 (다운로드 불필요)

- `com.android.settings`, `com.google.android.dialer` 같은 플랫폼 내장 앱은 APK 자체가 없다 — `download_apks` 가 자동으로 스킵하고 `MISSING.md` 의 `## System` 섹션에 기록한다. §5 설치 루프에서도 APK 파일이 없어 자연히 제외된다.

## 로컬 캐시 규약

- `catalog/apks/*.apk` 는 `.gitignore` 의 전역 `*.apk` 규칙으로 **추적 안 됨** — repo 에는 없고 로컬 디스크에만 존재한다.
- F-Droid 인덱스 이탈처럼 **재다운로드가 불가능해진 APK 는 이 로컬 캐시가 유일한 사본**이다. 디바이스 재생성(`--recreate`)이나 wipe 는 `catalog/apks/` 를 건드리지 않는다(별도 디렉터리) — 캐시를 수동으로 지우지 않는 한 안전하다.
- `catalog/apks/MISSING.md` 는 일반 `.md` 라 gitignore 미적용(커밋 대상) — 다운로드 실패/시스템 앱 목록의 스냅샷 역할.

## §3 실행 결과 예시 (2026-07-14, Pixel6-2 wipe 직후 재실행)

```
Total targets: 152 (fdroid 45 + playstore 107)
Downloaded: 0 / Skipped(existing): 130 / Failed: 22 (fdroid 16 + playstore 6) / System: 2
```

기존 로컬 캐시(130개)가 이미 완전해서 새로 받은 게 0건이었다 — `catalog/apks/` 의 기존 내용은 중단된 실행의 잔재가 아니라 **정상적으로 누적된 캐시**임을 이번 실행으로 확인(대상 수도 150→152 로 소폭 증가 — `apps.csv` 자체가 시간에 따라 늘어남).
