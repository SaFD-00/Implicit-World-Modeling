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
- [ ] **EXP04** (EXP03 프롬프트 업그레이드) — **stage1 LoRA 학습 미착수** (2026-07-13 확인: 이 머신에 학습 프로세스·`data/AndroidControl_EXP04/`·`outputs/` 전부 없음). stage1 YAML 은 2026-07-13 생성기로 재구성 (`configs/train/IWM-AC_EXP04/`, `# [reconstructed 2026-07-13]` 헤더), Stage 2 보류 (`_STAGE1_ONLY`)
- [ ] **EXP05** (절대 픽셀 좌표 840×1876 · AndroidWorld 해상도 정렬) — **Qwen2.5-VL 계열 전용** (`qwen2.5-vl-3b`/`qwen2.5-vl-7b`; **Qwen3-VL 계열(`qwen3-vl-4b`/`qwen3-vl-8b`)은 좌표계 mismatch 로 제외**), Stage 1 전용 (Stage 2 보류). EXP03/04(0–1000 정규화·Qwen3-VL 계열 전용)의 대칭 실험군.
  - [x] 데이터 빌드 — 0711 수정본 적용 + diff loss v2 가중 부여. 빌드 정본은 [`scripts/build_exp05_data.py`](../Implicit-World-Modeling/scripts/build_exp05_data.py) (mirror → 토큰 가중 → 원자 교체)
  - [ ] **데이터 쟁점 4건 조병웅님 확인** — 아래 백로그. **본실험 착수 전 선결**
  - [ ] 학습 — 로컬 2×RTX5090 Full FT 불가(OOM, 실측) → **Vessl A100/H100** 필요
  - [ ] 평가 — xy 좌표 채점 (`--coord-mode xy` / `--match-mode pos`, AC_EXP05 일 때만 적용)
- [ ] **`qwen3-vl-4b` — EXP01–EXP04 모델 자격만 복원, 학습 미착수** (2026-07-13). 커밋 `67a52e5` 가 삭제했던 모델을 레지스트리에 되돌려 **4 모델 2 tier**(7-9B: `qwen3-vl-8b`/`qwen2.5-vl-7b` · 3-4B: `qwen3-vl-4b`/`qwen2.5-vl-3b`) 가 됐다. **아직 한 번도 학습·평가하지 않았다** — 위 EXP01–04 의 완료/진행 표시는 전부 `qwen3-vl-8b`·`qwen2.5-vl-*` 기준이다. EXP05 는 Qwen3-VL 계열 배제라 대상 아님

## 할 일 (백로그)

- [ ] EXP04 stage1 완료 후 평가 → Stage 2 진행 여부 결정
- [ ] **EXP05 데이터 쟁점 — 조병웅님 확인 (본실험 전 선결)**
  - [ ] `wait` 액션 전량 퍼지가 의도인가 — 0710 의 `wait` 4,958건이 전부 빠지고 다른 위치에 400건이 새로 들어왔는데 **399건이 빈 current state** → `wait` 이 "빈 화면이면 wait" degenerate shortcut 이 되고 정상 맥락의 `wait` 예측을 학습하지 못한다
  - [ ] train 6.1% 축소(47,556 → 44,670) 수용 여부 — 퍼지분에 EXP01 train 멤버십 2,548 · test 멤버십 598 포함 → 오염 제거가 아니라 커버리지 축소
  - [ ] action / state 키 대칭 붕괴(400 차이) 허용 여부
  - [ ] 좌표 범위이탈 11키 (0710·0711 공통 · 기존 원천 버그) — 액션 라벨 `coordinate` 가 840×1876 을 벗어난다(`[1682, 975]` 반복, 1682 ≈ 840×2). **10행이 EXP05 산출물에 유입**됐고 `(12571,0) = [421,1979]` 는 **OOD 평가셋 3파일 전부를 오염**
- [ ] **EXP05 Vessl 실행 경로 확보** — org/project/cluster + 데이터 업로드 방식. 제출 스펙 자체는 저장소에 생겼다 (`configs/remote/run.template.yaml` + `scripts/remote_launch.sh` — **제공자 중립**, 제출 커맨드는 `.env` 의 `REMOTE_SUBMIT_CMD` 템플릿으로 주입). 단 **UNVALIDATED** — 이 머신에 제출 CLI 가 없어 실제 제출은 검증하지 못했다
- [ ] EXP05 7:3 분할 비율 교수님 최종 확인
- [ ] diff loss v1 (`token_weight_builder.py`) 의 경계 비대칭 — v2 에서는 고쳤으나 v1 은 **AC_EXP02 재현성 보존을 위해 의도적으로 미수정**. EXP02 재실행이 필요해지면 그때 판단
- [ ] `extract_elements` 의 aria-label 누락 / `without_open_app` 필터 무동작 (`## Action` vs `Action:` 마커 mismatch) — 학습·평가 공유 규약이라 조병웅님 확인 필요. **EXP03 에도 있는 기존 이슈**
- [ ] (추후) Obsidian 동기화 — Vault 있는 환경에서 `/project-sync init` 재실행

<!-- project-sync: task/계획 진척 시 - [ ] / - [x] 상태와 항목만 갱신. -->
