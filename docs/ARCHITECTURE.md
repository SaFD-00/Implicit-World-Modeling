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

## 데이터 계약 주의 (수집 산출물 무결성)

- MC `page_graph.json` node와 `pages/*/page.json`의 `activity`/`first_activity` 메타 라벨은 **~8.8% 페이지에서 stale**하다(171/1945 pages, 18/29앱, 대개 `nexuslauncher`로 오표기). 다만 관측 content(raw.xml modal package·element_lines)는 항상 해당 앱 자신이라 **label-only 결함**이다 — IWM Stage-1/2 변환기는 페이지의 앱/화면 정체를 `activity` 문자열이 아니라 **raw.xml content로 판정**해야 한다(에러 없이 틀린 문자열이 들어오는 silent trap). 참조무결성(dangling-edge 0)·`pages`=`nodes` 일치는 건전.
- **MC 코퍼스 레이아웃은 이원화돼 있다**(2026-07-22): `Monkey-Collector/data/raw/<pkg>/`가 **수집 원본**(세션별 `pages/` + `page_graph.json`), `Monkey-Collector/data/processed/`가 **학습 변환 산출물**(`gui-model_stage1.jsonl` + `images/`)이다. IWM 측이 소비하는 것은 후자다. reset은 `data/raw` + `runtime`만 지우고 `data/processed`는 건드리지 않는다.
- **`convert-all`은 완전중복 예제를 항상 1건만 남긴다**: 중복 키는 `(before_encoded_xml, action_json, after_encoded_xml)` 3튜플이며 **끄는 플래그·config 키가 없다**. 세션(앱) 경계를 넘어 전역으로 적용된다. 현재 산출은 **24앱 11,376 examples**(중복 0, 이미지 1:1).
- **⚠️ 학습 박스 이관 시 세 갈래 레이아웃 gap**(이미지만 옮기면 안 된다): ① **분할** — `configs/lf_dataset/dataset_info.json`의 `IWM-MC_stage1_{train,test}`는 두 파일을 기대하는데 `convert-all`은 단일 통합본을 낸다(`scripts/split_data.py --dataset MC`가 관여, MC는 meta 없어 random split·Stage 2 자동 skip). ② **경로·파일명** — 기대는 `data/MonkeyCollection/implicit-world-modeling_stage1_{train,test}.jsonl`, 산출은 `Monkey-Collector/data/processed/gui-model_stage1.jsonl`. ③ **이미지** — JSONL의 `images` 값은 converter가 하드코딩한 `GUI-Model/images/...` prefix인데 실제 파일은 `data/processed/images/`에 있고, 학습측 `media_dir: ../data`(`configs/train/IWM-MC/stage1_*/`)와 조합하면 조회 경로가 `<IWM data root>/GUI-Model/images/`다. **셋 중 하나만 고치면 절반만 맞춘 상태로 학습이 돈다.** 로컬에서는 IWM data root 자체가 없어 무증상이라 이 gap은 이관 시점에 처음 드러난다.
- 근거: [수집 캠페인 무결성 분석 (2026-07-22)](../Monkey-Collector/.claude/analysis/2026-07-22_07-33-00/gap-stage4-integrity.md) · [convert dedup·레이아웃 이원화 (2026-07-22)](../Monkey-Collector/.claude/devlog/2026-07-22_11-31-30_convert-dedup-and-raw-processed-layout.md).

## 더 보기

- 모델 매트릭스·데이터셋(AC_EXP01~05 / MC / MB)·실행 절차: [메인 README](../Implicit-World-Modeling/README.md)
- 수집기 App/Server 구조: [Monkey-Collector README](../Monkey-Collector/README.md)

<!-- project-sync: 구조/계약(contract) 변경 시 이 파일의 해당 섹션만 갱신. 상세는 패키지 트리오에. -->
