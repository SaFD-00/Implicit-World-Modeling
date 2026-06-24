---
name: iwm-sync
description: |
  Implicit-World-Modeling(IWM) 메인 프로젝트에서 큰 변경을 한 뒤 지식 베이스를 한 번에 동기화한다.
  doc 트리오(README/ARCHITECTURE/AGENTS) + Claude 메모리(MEMORY.md 인덱스 + 개별 memory 파일) +
  Notion(Experiments / Dev Log / Meeting Notes / Reference DB + 중앙 Timeline 허브)을 변경 성격에 맞는 곳에 빠짐없이 기록·갱신한다.
  Use when 사용자가 IWM에서 "큰 변경 했으니 문서/메모리/Notion 같이 정리·동기화·기록해줘", "이번 변경 반영해줘",
  "/iwm-sync" 라고 하거나, 실험 결과·디버깅·코드 수정·회의 발표자료/스크립트를 정리해 남길 때.
  Keywords: iwm, sync, 동기화, 문서, docs, memory, 메모리, notion, 노션, 정리, 기록, 실험 결과, 디버깅, 회의, dev log, meeting notes.
---

# iwm-sync — IWM 지식 동기화

큰 변경(big modification) 직후, 흩어진 세 지식 저장소를 한 번에 일관되게 갱신한다. 원칙: **변경을 분류하고 → 해당하는 저장소에만 정확히 기록한다.** 빠뜨리지 않되, 무관한 곳은 건드리지 않는다.

대상 저장소:
1. **Doc 트리오** — `Implicit-World-Modeling/{README.md, ARCHITECTURE.md, AGENTS.md}` (역할 분담은 AGENTS.md §"문서 동기화 원칙" 참조).
2. **Claude 메모리** — `~/.claude/projects/-data-seungwoo-Implicit-World-Modeling/memory/` 의 `MEMORY.md` 인덱스 + 개별 `*.md` memory 파일.
3. **Notion** — "Implicit World Modeling" 페이지: https://app.notion.com/p/Implicit-World-Modeling-264dbb15dd0180a98857dfd5abf92738
   (구성: Experiments DB · Dev Log DB · Meeting Notes DB · Reference DB + 중앙 **Timeline** 허브 DB. Calendar 섹션 = Timeline 캘린더 뷰.)

## "큰 변경"의 정의 (트리거 기준)

아래 중 하나라도 해당하면 이 스킬을 돌린다. 사소한 오타·로그 문구·주석 한 줄은 제외한다.

- **파이프라인/계약 변경(workflow)**: 새 실험군(EXP0N)·데이터셋·모델 추가, split 규칙·cutoff_len·hyperparameter 정본·HF 네이밍·outputs 디렉토리 구조 변경, shell CLI 플래그 추가/제거.
- **실험 실행 결과**: 새 train/eval 런이 끝나 해석할 수치가 나옴 (Stage1/Stage2 metric, ablation, ratio sweep).
- **디버깅 & 코드 수정**: 비자명한 버그를 잡았거나, 환경/구성 함정(GPU·deepspeed·dataset_info 등)을 해결한 코드/설정 변경.
- **회의 산출물**: 발표 자료·발표 스크립트·의사결정이 나온 미팅.
- **새 관련 연구/레퍼런스**: 실험 설계에 영향을 주는 논문/벤치마크/외부 자료.

판단이 애매하면 "이 변경을 모르는 사람이 한 달 뒤 코드를 보면 헷갈릴까?"를 기준으로 한다 — 그렇다면 큰 변경이다.

## 변경 분류 → 기록 위치 매핑

| 변경 카테고리 | Doc 트리오 | 메모리 | Notion DB |
|---|---|---|---|
| 파이프라인/계약 (workflow) | ✅ 3개 중 해당 파일 | △ 함정/근거면 | Timeline(Workflow) |
| 디버깅 분석 & 코드 수정 | △ 운영 제약이면 AGENTS | ✅ project/feedback | **Dev Log** + Timeline |
| 실험 설정 & 결과 | △ 설계 변경 시 | ✅ reference(위치) | **Experiments** + Timeline |
| 회의 발표자료 & 스크립트 | ✗ | △ 결정이 코드에 영향 시 | **Meeting Notes** + Timeline |
| 관련 연구(related work) | ✗ | △ | **Reference** DB |

(✅=거의 항상, △=해당될 때만, ✗=보통 불필요.)

## 실행 체크리스트

변경 내용을 먼저 사용자에게 1~2줄로 요약·확인한 뒤 순서대로 수행한다.

### 0. 분류
변경을 카테고리(workflow / debugging / experiment / meeting / related-work) 중 하나 이상으로 분류하고, 영향받는 doc 트리오 파일·메모리·Notion DB를 미리 결정한다.

### 1. Doc 트리오 동기화
- AGENTS.md §"문서 동기화 원칙"의 역할 분담을 따른다: **README**=사용자 실행 순서·CLI·모델 테이블, **ARCHITECTURE**=실제 디렉토리/산출물/메트릭/노트북 section 매핑, **AGENTS**="어디를 수정?" 규칙 + 운영 제약 + 빠른 검증.
- 한 사실이 여러 문서에 걸치면 함께 갱신. **추가만, 삭제 지양.** Korean 산문 + English 기술 용어 유지.

### 2. 메모리 동기화
- 비자명한 함정/근거/위치 정보면 memory 파일 추가·갱신. 디버깅/구성 함정 → `type: project|feedback`(**Why:**/**How to apply:**), 결과 위치·caveat → `type: reference`.
- 새 파일은 frontmatter 형식을 기존 memory와 정확히 맞추고, 관련 memory에 `[[wikilink]]` 연결. **`MEMORY.md` 인덱스에 한 줄 추가** (`- [<제목>](<file>.md) — <설명>`).
- 코드 동작·file:line 단정 전에 현재 코드로 재검증(메모리는 시점 관측이라 stale 가능).

### 3. Notion 동기화 (중앙 Timeline 허브)
Notion MCP(`notion-fetch`, `notion-query-data-sources`, `notion-create-pages`, `notion-update-page`) 사용.
- 먼저 `notion-fetch`로 위 페이지를 열어 각 DB(Experiments / Dev Log / Meeting Notes / Reference / Timeline) 스키마를 확인한다 — **속성명을 추측하지 말 것.**
- 분류에 따라 해당 DB에 **dated 항목(append)**을 만들고, **Timeline DB에 같은 날짜의 행을 만들어 relation으로 연결**한다(허브 → Calendar 자동 노출). 기존 항목 덮어쓰기 금지(명백한 갱신만 `notion-update-page`).
  - 디버깅/코드 → **Dev Log**(증상/원인/수정/검증 + commit·파일·관련 memory 링크).
  - 실험 결과 → **Experiments**(모델·DS·ratio·cutoff·hparam 정의 + 원시 수치/해석 분리 + 산출 경로). 기존 결과 계보가 있으면 연결.
  - 회의 → **Meeting Notes**(발표자료 링크/요지 + 발표 스크립트 + 결정).
  - 관련 연구 → **Reference** DB.
- 본문 언어 한국어 + 기술 용어 영어. 수치는 원시값/해석 분리.

### 4. 마무리
무엇을 어디에 기록했는지 사용자에게 체크리스트로 보고(doc 변경 파일 / memory + MEMORY.md / Notion 항목 링크). commit 원하면 doc·memory 변경만 staging (Notion은 git 밖).

## 주의
- 이 스킬은 IWM **메인 프로젝트**(`Implicit-World-Modeling/`) 전용. `Monkey-Collector/`는 자체 트리오를 가진 별개 도메인이라 대상이 아니다(필요 시 사용자에게 물어 범위 확장).
- `data/`·`outputs/`는 대용량 gitignore 정본 → 산출물 자체를 doc/memory/Notion에 복사하지 말고 **경로와 요약만** 기록.
- 무관한 저장소·매핑표의 ✗ 칸은 건너뛴다.
