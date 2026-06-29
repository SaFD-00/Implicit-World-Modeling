# Changelog

[Keep a Changelog](https://keepachangelog.com/) 스타일. 시점성 진행은 [DEVLOG.md](./DEVLOG.md),
계획은 [ROADMAP.md](./ROADMAP.md) 참조.

## [Unreleased]

### Added
- `docs/` 루트 문서 허브 + `/project-sync` 설정(`.project-sync.json`) 도입  (2026-06-28)
- Monkey-Collector: OpenRouter 공용 LLM 클라이언트(`llm/client.py`, 기본 `qwen/qwen3.7-plus`) + 화면 의미 그룹핑(`llm/screen_grouper.py`, `--screen-grouping` 플래그)  (2026-06-28)
- Monkey-Collector: `setup-collector` 스킬 `references/` deep-dive 8종(client-build, mediaprojection-accessibility, google-login, run-and-verify, seed-helpers, seed-pim, seed-notes-tasks, seed-media-misc) + 런타임 권한 다이얼로그 adb 자동허용(`collection_loop._try_grant_permission_via_adb`, "While using the app" 우선 탭·deny-guard)  (2026-06-29)

### Changed
- Monkey-Collector: 입력 텍스트 생성을 OpenAI Responses API(gpt-5-nano)에서 공용 OpenRouter `LLMClient`(Chat Completions)로 이전  (2026-06-28)
- Monkey-Collector: 탐색 엔진을 `SmartExplorer`(화면 단위 weighted-random)에서 `LLMGuidedExplorer`(coverage-driven unexplored-first + LLM same-function 압축 + transition-graph 최단경로 navigation, 신규 `pipeline/exploration/` 패키지)로 전면 교체; App/Server TCP·저장 포맷 유지, `networkx` 의존성 추가  (2026-06-29)
- Monkey-Collector: `setup-collector` 스킬을 SKILL.md 오케스트레이션 + `references/` 구조로 재구성(AVD Pixel6-2, 빌드 JDK17/AGP8.2, MediaProjection 재동의·Google 로그인·더미데이터 시드·라이브 검증 단계 추가, 전 단계 멱등)  (2026-06-29)

### Fixed
- Monkey-Collector: MediaProjection 토큰 단발성 reuse-guard + `createVirtualDisplay` graceful-degrade(`ScreenStabilizer`); `EXCLUDED_PACKAGES`(`CollectorService`)·`SYSTEM_PACKAGES`(`screen_guard`)에 gms/gsf/vending 추가로 외부앱 스톰 차단; no-ACK 세션 abort(`session_manager`)  (2026-06-29)

> 검증(2026-06-28): 위 OpenRouter LLM 통합을 실제 API Key + AVD(Pixel6-2)로 라이브 검증 — 정적 504 passed, 모델 슬러그(`qwen/qwen3.7-plus`) 실호출·화면 의미 그룹핑·문맥 입력 생성·비용 귀속·graceful fallback 동작 확인 (VERDICT PASS). 상세는 [DEVLOG.md](./DEVLOG.md).

<!-- project-sync: 릴리스/버전 변경 요약을 Added/Changed/Fixed로 한 줄씩 추가. -->
