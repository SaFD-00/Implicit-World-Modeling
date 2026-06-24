# AGENTS.md (repo root)

이 저장소는 두 개의 독립 하위 프로젝트로 구성된다. 작업 대상에 맞는 트리오를 본다.

- **[`Implicit-World-Modeling/`](./Implicit-World-Modeling)** — 메인 2-stage VLM 파이프라인. 작업 지침 [`AGENTS.md`](./Implicit-World-Modeling/AGENTS.md), 사용자 가이드 [`README.md`](./Implicit-World-Modeling/README.md), 시스템 레퍼런스 [`ARCHITECTURE.md`](./Implicit-World-Modeling/ARCHITECTURE.md).
- **[`Monkey-Collector/`](./Monkey-Collector)** — Android 데이터 수집기 (자체 README/ARCHITECTURE/AGENTS 트리오).

## 큰 변경 후 지식 동기화

**메인 프로젝트(`Implicit-World-Modeling/`)에서 큰 변경을 한 뒤에는 `/iwm-sync` 스킬을 실행**해 문서 트리오 + Claude 메모리(`MEMORY.md` + memory 파일) + Notion(Experiments/Dev Log/Meeting Notes/Reference DB + 중앙 Timeline 허브)을 한 번에 동기화한다. 규약 상세는 [`Implicit-World-Modeling/AGENTS.md`](./Implicit-World-Modeling/AGENTS.md) §"큰 변경 후 지식 동기화", 스킬은 [`.claude/skills/iwm-sync/SKILL.md`](./.claude/skills/iwm-sync/SKILL.md). 이 규약은 메인 프로젝트 전용이다.

- 대용량 정본 `data/`·`outputs/` 는 gitignore 대상(심볼릭 링크 구조). 산출물 자체는 doc/memory/Notion 에 복사하지 말고 경로·요약만 기록.
