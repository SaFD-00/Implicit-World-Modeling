# Collector client 빌드 (§6)

Monkey-Collector 안드로이드 client(`com.monkey.collector`)를 gradle 로 빌드해 설치하는 상세. SKILL.md §6 에서 위임.

## 프로젝트 레이아웃 (중첩 주의)

```
$ROOT/                              # Monkey-Collector repo (uv/monkey-collect 실행 위치)
└── app/                            # ← gradle 프로젝트 루트 (gradlew, settings.gradle.kts, local.properties)
    ├── gradlew
    ├── build.gradle.kts            # AGP 8.2.0 / Kotlin 1.9.22
    ├── local.properties            # sdk.dir=… (gitignore — 추적 안 됨)
    └── app/                        # ← app 모듈
        ├── build.gradle.kts        # applicationId com.monkey.collector / compileSdk 34 / minSdk 28
        └── build/outputs/apk/debug/app-debug.apk   # ← 산출 APK
```

- gradle 명령은 **`$ROOT/app`** 에서 실행(`cd "$ROOT/app"`).
- APK 절대경로: `$ROOT/app/app/build/outputs/apk/debug/app-debug.apk` (`app/app` 중첩).

## local.properties (SDK 경로)

`$ROOT/app/local.properties` 에 `sdk.dir` 이 있어야 gradle 이 SDK 를 찾는다. **`.gitignore` 의 `**/local.properties`** 로 추적되지 않으므로(머신마다 경로가 다름) 없으면 생성:

```bash
[ -f "$ROOT/app/local.properties" ] || printf 'sdk.dir=%s\n' "${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}" > "$ROOT/app/local.properties"
```
(`ANDROID_SDK_ROOT`/`ANDROID_HOME` 환경변수로도 잡히지만 `local.properties` 가 가장 견고하다.)

## JDK 17 필수 (빌드 실행)

**AGP 8.2 / Gradle 8.x 는 빌드를 실행하는 JVM 으로 JDK 17 을 요구**한다. 컴파일 산출 bytecode 는 `jvmTarget=1.8`(`sourceCompatibility/targetCompatibility = 1.8`)이지만 — 이건 *출력* 타깃일 뿐 — **gradle 자체를 JDK 8/11 로 돌리면 빌드가 실패**한다(`Unsupported class file major version` 등).

```bash
cd "$ROOT/app"
chmod +x gradlew 2>/dev/null
JAVA_HOME="$(/usr/libexec/java_home -v 17)" ./gradlew :app:assembleDebug
```

- macOS 에 JDK 17 이 없으면 Android Studio 내장 JBR 사용 가능: `JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"`.
- 확인: `JAVA_HOME=$(/usr/libexec/java_home -v 17) java -version` → `openjdk version "17.x"`.

## 설치 (멱등)

```bash
# 미설치이거나 --force 일 때만 빌드/설치 (초기화 시 1회)
if [ "$FORCE" = "true" ] || ! adb -s "$SERIAL" shell pm list packages | grep -q "package:com.monkey.collector"; then
  JAVA_HOME="$(/usr/libexec/java_home -v 17)" ./gradlew :app:assembleDebug
  adb -s "$SERIAL" install -r -g "$ROOT/app/app/build/outputs/apk/debug/app-debug.apk"
fi
```
- `install -r` 은 앱 데이터(prefs 포함)와 접근성 토글을 보존한다(전체 uninstall 후 재설치 시 리셋됨).
- **소스 패치 후엔 반드시 재빌드→재설치**: 클라이언트 측 수정(예 ScreenStabilizer crash guard, CollectorService EXCLUDED_PACKAGES)은 APK 를 다시 깔아야 디바이스에 반영된다. 설치 시각 확인: `adb -s "$SERIAL" shell dumpsys package com.monkey.collector | grep lastUpdateTime`.

## 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| `Unsupported class file major version 61/65` / toolchain 실패 | JDK 8/11 로 gradle 실행 | `JAVA_HOME=$(/usr/libexec/java_home -v 17)` 로 재실행 |
| `SDK location not found` | `local.properties` 없음/`sdk.dir` 미설정 | 위 local.properties 생성 |
| `Warning: ... SDK XML versions up to 3 but ... version 4` | cmdline-tools vs 스튜디오 버전차 | **무해 — 무시**. 빌드 정상 진행 |
| 재빌드해도 APK mtime 그대로 | 소스 무변경 → Gradle up-to-date | 정상(이미 패치 반영분). 강제: `./gradlew clean :app:assembleDebug` |
| 디바이스에 옛 동작/크래시 잔존 | 빌드만 하고 **재설치 안 함** | `install -r` 로 재설치 후 §6-c MediaProjection 재동의 |
