# Roadmap

> 마일스톤·실험·할 일을 추적한다. 실험 **결과/지표는 여기 적지 않고** Notion `🧪 Experiments` DB와
> Timeline 허브를 정본으로 본다. 이 문서는 "무엇을 / 어떤 순서로"의 계획 뷰다.

## 마일스톤

- [x] 모노레포 문서 트리오 정비 (루트 + 하위 프로젝트별 README·ARCHITECTURE·AGENTS)
- [x] 2-stage 파이프라인 자동화 (`scripts/stage{1,2}_{train,merge,eval}.sh`)
- [ ] 실험 매트릭스 완주 (모델 × 데이터셋 × {base / stage2 / stage1+stage2})
- [ ] 결과 종합 및 논문화 (AAAI/ICLR 2027 트랙)

## 실험 트랙

> 상세 설정·결과는 Notion Experiments DB 및 프로젝트 메모리 참조.

- [x] EXP01 / EXP02 / EXP03 — 결과 Notion 기록 완료
- [ ] **EXP04** (EXP03 프롬프트 업그레이드) — stage1 LoRA 학습 진행, Stage 2 보류 (`_STAGE1_ONLY`)

## 할 일 (백로그)

- [ ] EXP04 stage1 완료 후 평가 → Stage 2 진행 여부 결정
- [ ] (추후) Obsidian 동기화 — Vault 있는 환경에서 `/project-sync init` 재실행

<!-- project-sync: task/계획 진척 시 - [ ] / - [x] 상태와 항목만 갱신. -->
