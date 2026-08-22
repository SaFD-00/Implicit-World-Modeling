# Docs — Implicit World Modeling

이 `docs/` 디렉터리는 모노레포 **루트 레벨의 진행 기록·로드맵 허브**다. 코드/설계의 정본은 각 하위
프로젝트의 트리오(README·ARCHITECTURE·AGENTS)에 있고, 여기서는 그것을 **가로지르는 진행·계획·변경
이력**을 모은다. `/project-sync`가 이 폴더에 append한다. 예외로 **외부(벤더)에서 전달받은 설계 문서**도
여기 원문 그대로 보존하며, 그 문서와 이 저장소의 실제 배선이 어긋나는 지점은 각 문서 상단의 **편차 표가
정본**이다.

## 이 폴더의 문서

| 파일 | 성격 | 갱신 |
|------|------|------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 모노레포 전체 데이터 흐름(수집 → 학습/평가)을 가로지르는 상위 설계 | 구조/계약 변경 시 |
| [ROADMAP.md](./ROADMAP.md) | 실험·마일스톤·할 일 | 진척 시 |
| [DEVLOG.md](./DEVLOG.md) | 시점성 진행 로그 (append-only) | 변경마다 |
| [WM_FORMATS.md](./WM_FORMATS.md) | **벤더 전달 설계 문서** — anti-copy 3-포맷 관측성 분할 (AC_EXP08). 상단 편차 표가 이 저장소 기준 정본 | 원문 갱신·배선 변경 시 |
| [DIFF_TARGETS.md](./DIFF_TARGETS.md) | **벤더 전달 설계 문서** — diff-loss 학습 입력 생성. 상단 편차 표가 이 저장소 기준 정본 | 원문 갱신·배선 변경 시 |
| [CHANGES_v2_cerebra.md](./CHANGES_v2_cerebra.md) | `scripts/diff_loss/` 의 Cerebra 스키마 확장(`*_v2c.py`) 변경 이력 | v2c 변경 시 |

## 정본 문서 (각 하위 프로젝트)

- 루트 개요: [`../README.md`](../README.md) · 작업 라우팅: [`../AGENTS.md`](../AGENTS.md)
- **메인 파이프라인** (`Implicit-World-Modeling/`): [README](../Implicit-World-Modeling/README.md) · [ARCHITECTURE](../Implicit-World-Modeling/ARCHITECTURE.md) · [AGENTS](../Implicit-World-Modeling/AGENTS.md)
- **데이터 수집기** (`Monkey-Collector/`): [README](../Monkey-Collector/README.md) · [ARCHITECTURE](../Monkey-Collector/ARCHITECTURE.md) · [AGENTS](../Monkey-Collector/AGENTS.md)

## 외부 기록

- **Notion 워크스페이스** (실험/Task/Dev Log/Meeting/Reference + Timeline 허브): [📱 Implicit World Modeling](https://app.notion.com/p/Implicit-World-Modeling-264dbb15dd0180a98857dfd5abf92738)
- **프로젝트 메모리** (Claude CLI): `~/.claude/projects/-home-seungwoo-baek-projects-Implicit-World-Modeling/memory/` (`.project-sync.json` 의 `memory.encoded` 와 일치 — 체크아웃 경로가 바뀌면 양쪽을 함께 갱신한다)

> 산출물(모델·평가 raw)은 여기 복사하지 않는다. `data/`·`outputs/` 정본 경로와 커밋 SHA로만 인용한다.
