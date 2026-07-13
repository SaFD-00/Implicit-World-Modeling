# Implicit-World-Modeling

모바일 GUI **World Modeling** 이 Action Prediction 성능에 미치는 영향을 검증하는 연구 모노레포다. 두 개의 독립 하위 프로젝트로 구성된다 — 한쪽은 데이터를 **수집**하고, 다른 한쪽은 그 데이터로 2-stage VLM 파이프라인을 **학습·평가**한다.

## 하위 프로젝트

| 디렉토리 | 역할 | 문서 트리오 |
|---------|------|------------|
| [`Implicit-World-Modeling/`](./Implicit-World-Modeling) | 메인 **2-stage fine-tuning 파이프라인** (Qwen 계열 VLM × 데이터셋 매트릭스). Stage 1 = World Modeling, Stage 2 = Action Prediction. LlamaFactory 백엔드 + 단일 노트북/`scripts/` 자동화. | [README](./Implicit-World-Modeling/README.md) · [ARCHITECTURE](./Implicit-World-Modeling/ARCHITECTURE.md) · [AGENTS](./Implicit-World-Modeling/AGENTS.md) |
| [`Monkey-Collector/`](./Monkey-Collector) | Android **GUI 데이터 수집기** (App + Python 서버). AccessibilityService 앱이 화면 전환을 감지하고 서버가 다음 action 을 골라 ADB 로 실행하며 screenshot + XML 세션을 저장한다. | [README](./Monkey-Collector/README.md) · [ARCHITECTURE](./Monkey-Collector/ARCHITECTURE.md) · [AGENTS](./Monkey-Collector/AGENTS.md) |

> 두 프로젝트는 환경·툴체인이 다르다. **Implicit-World-Modeling** 은 conda env (`implicit-world-modeling`) + LlamaFactory editable 설치, **Monkey-Collector** 는 uv 가 관리하는 `.venv` (Python 3.10) 를 쓴다. 작업 대상에 맞는 하위 트리오를 본다.

## 현재 상태 (2026-07-14)

- **Monkey-Collector**: 통제 ablation 결과 budget-loop 가드(D1/D2/D3)의 다양성 이득은 **노이즈와 구별되지 않으며**, 예산의 **44~56% 가 signal timeout 대기에서 소진**되는 것이 진짜 병목으로 드러났다. D3 임계값은 데이터 기반으로 150 → 98 재보정. → [iter6 통제 ablation 보고](./Monkey-Collector/.claude/analysis/2026-07-14_04-05-29_iter6-controlled-ablation/README.md) · [DEVLOG](./docs/DEVLOG.md)

## 핵심 아이디어 — 2-stage 파이프라인 (메인 프로젝트)

- **Stage 1 — World Modeling**: `screenshot + UI XML + action → next UI XML`
- **Stage 2 — Action Prediction**: `screenshot + UI XML + task → action JSON`
- **비교 실험**: `base` (zero-shot) vs `stage2` (base→Stage 2 SFT) vs `stage1+stage2` (World Model→Stage 2 SFT)
- **파이프라인 흐름** (Stage 1 / Stage 2 공통): `train → merge → eval`

자세한 모델 매트릭스·데이터셋(AC_EXP01~05 / MC / MB)·실행 방법은 [`Implicit-World-Modeling/README.md`](./Implicit-World-Modeling/README.md) 를 본다.

## 저장소 레이아웃

```
Implicit-World-Modeling/                 # ← 모노레포 루트 (this README)
├── Implicit-World-Modeling/             # 메인 2-stage VLM 파이프라인 (하위 프로젝트)
├── Monkey-Collector/                    # Android GUI 데이터 수집기 (하위 프로젝트)
├── data/                                # 대용량 데이터 정본 (gitignore) — 하위 프로젝트가 심볼릭 링크로 참조
├── outputs/                             # 학습/평가 산출물 정본 (gitignore) — 하위 프로젝트가 심볼릭 링크로 참조
├── AGENTS.md                            # 루트 에이전트 작업 지침 (하위 트리오로 라우팅)
└── .gitignore
```

> **대용량 정본 규약**: top-level `data/` · `outputs/` 가 정본이고, nested 하위 프로젝트의 `data`/`outputs` 는 이 정본으로의 **심볼릭 링크**다. 둘 다 git 추적 대상이 아니다 (gitignore). 산출물 자체는 문서/메모리에 복사하지 말고 경로·요약만 기록한다.

## 빠른 시작

각 하위 프로젝트는 독립적으로 설치·실행한다.

```bash
# 메인 파이프라인 — conda + LlamaFactory editable
cd Implicit-World-Modeling
conda create -n implicit-world-modeling python=3.12 -y && conda activate implicit-world-modeling
pip install -e ".[llamafactory]" && pip install -e ./LlamaFactory
#   → 이후 실행은 implicit-world-modeling.ipynb 또는 scripts/stage{1,2}_{train,merge,eval}.sh

# 데이터 수집기 — uv
cd Monkey-Collector
uv sync --extra dev && source .venv/bin/activate
#   → monkey-collect CLI
```

상세 설치·`.env`·데이터 준비·실행 절차는 각 하위 프로젝트의 `README.md` 를 따른다.

## 문서 규약

- **루트** [`AGENTS.md`](./AGENTS.md) — 작업 대상에 맞는 하위 트리오로 라우팅하는 진입점.
- **각 하위 프로젝트** — `README.md` (사용자 가이드) · `ARCHITECTURE.md` (시스템 레퍼런스) · `AGENTS.md` (에이전트 작업 지침) 트리오.
