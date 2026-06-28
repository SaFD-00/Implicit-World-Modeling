# Architecture — 모노레포 상위 흐름

> 이 문서는 **두 하위 프로젝트를 가로지르는** 데이터·실험 흐름만 다룬다. 각 하위 시스템의 상세 설계는
> 패키지 `ARCHITECTURE.md`를 정본으로 본다 — [메인 파이프라인](../Implicit-World-Modeling/ARCHITECTURE.md) ·
> [데이터 수집기](../Monkey-Collector/ARCHITECTURE.md).

## 전체 데이터 흐름

```
Monkey-Collector (Android App + Python Server)
   AccessibilityService → 화면 전환 감지 → 서버가 다음 action 선택 → ADB 실행
   → screenshot + UI XML 세션 저장
        │
        ▼   (data/ 정본 — 하위 프로젝트가 심볼릭 링크로 참조)
Implicit-World-Modeling (2-stage VLM fine-tuning)
   Stage 1  World Modeling     : screenshot + UI XML + action → next UI XML
   Stage 2  Action Prediction  : screenshot + UI XML + task   → action JSON
   비교축    base / stage2 / stage1+stage2
   흐름      train → merge → eval   (outputs/ 정본)
```

## 저장소 레이아웃 규약

- top-level `data/` · `outputs/` 가 **정본**, nested 하위 프로젝트의 `data`/`outputs`는 그 정본으로의
  **심볼릭 링크**. 둘 다 git 비추적(gitignore).
- 환경 분리: 메인 파이프라인 = conda env `implicit-world-modeling` + editable LlamaFactory,
  수집기 = uv `.venv` (Python 3.10).

## 더 보기

- 모델 매트릭스·데이터셋(AC_EXP01~04 / MC / MB)·실행 절차: [메인 README](../Implicit-World-Modeling/README.md)
- 수집기 App/Server 구조: [Monkey-Collector README](../Monkey-Collector/README.md)

<!-- project-sync: 구조/계약(contract) 변경 시 이 파일의 해당 섹션만 갱신. 상세는 패키지 트리오에. -->
