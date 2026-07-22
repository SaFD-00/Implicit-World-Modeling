# Docs — Implicit World Modeling

이 `docs/` 디렉터리는 모노레포 **루트 레벨의 진행 기록·로드맵 허브**다. 코드/설계의 정본은 각 하위
프로젝트의 트리오(README·ARCHITECTURE·AGENTS)에 있고, 여기서는 그것을 **가로지르는 진행·계획·변경
이력**을 모은다. `/project-sync`가 이 폴더에 append한다.

## 이 폴더의 문서

| 파일 | 성격 | 갱신 |
|------|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 모노레포 전체 데이터 흐름(수집 → 학습/평가)을 가로지르는 상위 설계 | 구조/계약 변경 시 |
| [ROADMAP.md](./ROADMAP.md) | 실험·마일스톤·할 일 | 진척 시 |
| [DEVLOG.md](./DEVLOG.md) | 시점성 진행 로그 (append-only) | 변경마다 |

## 정본 문서 (각 하위 프로젝트)

- 루트 개요: [`../README.md`](../README.md) · 작업 라우팅: [`../AGENTS.md`](../AGENTS.md)
- **메인 파이프라인** (`Implicit-World-Modeling/`): [README](../Implicit-World-Modeling/README.md) · [ARCHITECTURE](../Implicit-World-Modeling/ARCHITECTURE.md) · [AGENTS](../Implicit-World-Modeling/AGENTS.md)
- **데이터 수집기** (`Monkey-Collector/`): [README](../Monkey-Collector/README.md) · [ARCHITECTURE](../Monkey-Collector/ARCHITECTURE.md) · [AGENTS](../Monkey-Collector/AGENTS.md)

## 외부 기록

- **Notion 워크스페이스** (실험/Task/Dev Log/Meeting/Reference + Timeline 허브): [📱 Implicit World Modeling](https://app.notion.com/p/Implicit-World-Modeling-264dbb15dd0180a98857dfd5abf92738)
- **프로젝트 메모리** (Claude CLI): `~/.claude/projects/-home-seungwoo-baek-projects-Implicit-World-Modeling/memory/` (`.project-sync.json` 의 `memory.encoded` 와 일치 — 체크아웃 경로가 바뀌면 양쪽을 함께 갱신한다)

> 산출물(모델·평가 raw)은 여기 복사하지 않는다. `data/`·`outputs/` 정본 경로와 커밋 SHA로만 인용한다.
