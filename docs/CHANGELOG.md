# Changelog

[Keep a Changelog](https://keepachangelog.com/) 스타일. 시점성 진행은 [DEVLOG.md](./DEVLOG.md),
계획은 [ROADMAP.md](./ROADMAP.md) 참조.

## [Unreleased]

### Added
- `docs/` 루트 문서 허브 + `/project-sync` 설정(`.project-sync.json`) 도입  (2026-06-28)
- Monkey-Collector: OpenRouter 공용 LLM 클라이언트(`llm/client.py`, 기본 `qwen/qwen3.7-plus`) + 화면 의미 그룹핑(`llm/screen_grouper.py`, `--screen-grouping` 플래그)  (2026-06-28)

### Changed
- Monkey-Collector: 입력 텍스트 생성을 OpenAI Responses API(gpt-5-nano)에서 공용 OpenRouter `LLMClient`(Chat Completions)로 이전  (2026-06-28)

> 검증(2026-06-28): 위 OpenRouter LLM 통합을 실제 API Key + AVD(Pixel6-2)로 라이브 검증 — 정적 504 passed, 모델 슬러그(`qwen/qwen3.7-plus`) 실호출·화면 의미 그룹핑·문맥 입력 생성·비용 귀속·graceful fallback 동작 확인 (VERDICT PASS). 상세는 [DEVLOG.md](./DEVLOG.md).

<!-- project-sync: 릴리스/버전 변경 요약을 Added/Changed/Fixed로 한 줄씩 추가. -->
