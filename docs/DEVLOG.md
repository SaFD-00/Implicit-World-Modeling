# Dev Log

시점성 진행 로그 (append-only). 최신 엔트리를 위에 추가한다. 과거 엔트리는 수정·삭제하지 않는다.
상세 결과는 Notion Dev Log / Experiments DB, 계획은 [ROADMAP.md](./ROADMAP.md) 참조.

## 2026-07-02 — Monkey-Collector: page matching을 Mobile3M Unique Page(BM25 + conjunctive diff)로 교체 — LLM-free matching

사용자가 `.claude/references/mobilevlm` 을 참고해 Monkey-Collector 의 page 식별을 MobileGPT-V2 식 **LLM element-set matching** 에서 Mobile3M 의 "Unique Page" 메커니즘(**BM25 후보검색 + conjunctive element/pixel diff 검증**)으로 교체하고, 문서 갱신 후 나눠서 커밋·푸시하기를 원했다. matching 이 **LLM-free** 가 되어 화면당 LLM 호출 비용·복잡도가 사라졌고, `ScreenMatch`/`page_key` 출력 계약을 불변으로 유지해 하위 소비처(page_graph·exploration·storage)는 무변경이다.

- 신규 모듈: `pipeline/screen_matching/bm25.py`(`Bm25Index`, Okapi BM25 k1=1.5/b=0.75, numpy 미사용) · `element_lines.py`(`serialize_element_lines` = encoded XML → element-line 문서, `element_diff_count`/`element_jaccard`).
- `ScreenMatcher.match()` 재작성: 구조지문 prefilter → pending 가드 → element-line 직렬화 → BM25 top-K 후보 → conjunctive verify(element diff `<element_diff_max` **AND** pixel gate `luminance_diff<0.3`) → 첫 통과면 `BM25_MERGE`, 없으면 `NEW`. `match_type` = `NEW`/`STRUCTURAL_IDENTICAL`/`BM25_MERGE`/`PENDING_EMPTY`.
- 참고 논문(top-5 + element diff<5) vs 참고 코드 `arm_graph_para_lock.py`(top-1 argmax + Jaccard>0.5)의 불일치를 발견 → **논문 스펙을 기본값**으로 채택하되 5종 config knob 으로 코드 스펙 전환 가능하게 노출.
- LLM element 추출은 옵션 enrichment(`families`, 탐색 same-function grouping)로 분리 — matching 경로는 절대 LLM 을 호출하지 않는다.
- config knob 5종(6-place 관통 + `MC_SCREEN_MATCHING_*` env + CLI flags): `bm25_top_k=5`, `element_criterion=diff|jaccard`, `element_diff_max=5`, `element_jaccard_min=0.5`, `page_pixel_diff_threshold=0.3`.
- `PageKnowledge.element_lines` 필드 추가(page.json additive 직렬화) — `rehydrate` 가 세션 재개 시 BM25 코퍼스를 재구축하고, 필드 없는 legacy page.json 은 첫 observation 의 raw.xml 로 재계산.
- 구 element-set 코드(`set_classifier.py`)·`cluster_merge_tolerance`/`max_expand_iters` 는 deprecated 로 존치하되 미참조.
- 변경(diffstat, 마지막 4커밋): 21 files changed, 1164 insertions(+), 688 deletions(-).
- 커밋(브랜치 `feat/bm25-page-matching`, origin 푸시 완료): `018208e` feat(screen-matching): add Bm25Index + element-line serializer · `3da2b47` feat(screen-matching): BM25 + conjunctive diff page matching · `d6ecddc` feat(config): screen-matching BM25/diff/pixel knobs · `979ef03` docs: describe BM25 unique-page matching.
- 결과/검증: 유닛 **706 all green**(신규 `test_bm25`/`test_element_lines` + `test_screen_matcher`/`test_rehydrate`/`test_config`/`test_storage` 갱신). ruff/mypy clean(변경 파일). 스크립트 E2E PASS — A→NEW, A재방문→STRUCTURAL, B→BM25_MERGE(page0), C→NEW(page1), C→B 기존 노드 재연결 ⇒ page_graph 2노드/2엣지(방향그래프·중복노드 없음)로 Mobile3M unique-page 목표(탐색폭발 방지) 달성. **라이브 AVD(Pixel6-2) 수집 검증은 미실행(다음 세션)**.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` 는 이번 구현에서 이미 갱신됨. 루트 `docs/{README,ARCHITECTURE}.md` 는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-07-02 — Monkey-Collector: 필터된 재방문도 저장 (`persist_filtered`, per-visit observation)

사용자 질문("`data/com.flauschcode.broccoli/pages/` 에 observation 이 왜 페이지당 하나만 생기나")에서 출발했다. 원인은 prefilter-only 모드(`element_extraction=false`)에서 structural/luminance prefilter 로 dedup 된 재방문 화면이 파일을 전혀 안 쓰고, 재사용 못 하는 새 화면은 새 page 가 되어 "새 page = observation 0" 1:1 이 되기 때문이었다. 사용자가 "filtering 이 되더라도 저장되도록" 원해, 필터된 재방문을 그 page 아래 **자체 observation**(방문마다 `0,1,2,…` per-visit 체인)으로 저장하도록 신규 플래그 `screen_matching.persist_filtered`(기본 ON)를 추가했다. 설계는 3개 독립안(per-visit / frames-subdir / config-gated) 생성→심사→적대적 검증 워크플로로 도출했고, "매처 단독 변경(loop/storage/rehydrate 무변경)"이 최소 blast-radius·정합성 최고로 선정됐다. 코드 변경, 작업트리 미커밋.

- `screen_matcher.py`(핵심): `ScreenMatcher.__init__` 에 `persist_filtered` kwarg 추가, allocate+luminance-append+cap 로직을 `_allocate_observation(page, feat, append_luma)` 헬퍼로 추출. 3개 재사용 종료지점을 플래그로 게이트 — structural prefilter hit·luminance prefilter hit 은 fresh `observation_num` 을 할당(`append_luma=False`: 히트 프레임은 이미 near-dup 이라 ring-buffer churn 방지)하고 `_fp_to_key` 를 최신 obs 로 재지정한 뒤 `is_new_observation=True` 반환; `_record_observation` 의 merge dedup 는 `not persist_filtered` 일 때만 재사용 단락. 플래그 off/`cached_page None` 이면 현행 reuse 반환을 byte-identical 유지. page 정체성(page_key)·LLM 0회 불변, no-overwrite(항상 새 번호).
- config 6-place: `config.py`(builtin default True + `ScreenMatchingConfig.persist_filtered` + env map `MC_SCREEN_MATCHING_PERSIST_FILTERED` + `_from_raw` + `merge_with_cli_args`), `config/run.yaml`(키 + 헤더 canonical-defaults), `cli.py`(`--persist-filtered {on,off}` + `create_screen_matcher` 전달 + 시작 로그), `pipeline/screen_matching/__init__.py`(`create_screen_matcher` 파라미터 전달, None-guard 불변).
- 무변경(설계상 확인): `collection_loop.py`(게이트 632 는 `is_new_observation=True` 오면 그대로 발화), `storage.py`(`save_observation` 이 fresh 번호로 새 dir), `rehydrate.py`(`next_observation_num = max(on-disk obs)+1` 로 per-visit 체인 이어감).
- 변경(작업트리, 8 files): src 5(`screen_matcher.py`·`config.py`·`cli.py`·`screen_matching/__init__.py`·`config/run.yaml`) + tests 3(`test_screen_matcher.py`·`test_config.py`·`test_rehydrate.py`) + 패키지문서 3(`README`·`ARCHITECTURE`·`AGENTS`.md) + repo docs 2(`docs/{DEVLOG,CHANGELOG}.md`).
- 신규 테스트: `test_screen_matcher.py` persist 픽스처 2종 + 6 케이스(structural/luminance 재방문 fresh obs·no-LLM·cache backfill, prefilter-only 체인, merge 할당+append, must-fix#1 luminance 미증가, off 회귀), `test_config.py`(기본 True·env·CLI·full-args), `test_rehydrate.py`(multi-obs 저장→resume→`next_observation_num`·체인 지속).
- 결과/검증: `uv run pytest -q`: **684 passed**(0 failed). 기존 reuse 단언은 픽스처 기본값 `persist_filtered=False` 로 그대로 통과.
- 커밋: 미커밋(작업트리). 직후 project-sync + git-push 예정.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` 의 "재사용=파일 미기록" 서술을 `persist_filtered` 기준(기본 ON=저장, off=미기록)으로 정정 + config/CLI 표·저장 트리 반영. 루트 `docs/{README,ARCHITECTURE}.md`는 패키지 정본을 가리키므로 추가 수정 없음. 앞선 저장 재설계(2026-07-01) 엔트리의 "재사용 관측 파일 미기록"은 append-only 라 수정하지 않고 이 엔트리에서 기본값을 뒤집었음을 명시.
- 트레이드오프: loop-heavy 앱에서 근접 중복 observation 다수 생성(prefilter 가 애초에 피하려던 디스크 비용) — `--persist-filtered off`/`MC_SCREEN_MATCHING_PERSIST_FILTERED=false` 로 기존 절약 동작 복원. resume 간 플래그는 일정하게 유지 권장.
- 카테고리: devlog

## 2026-07-01 — Monkey-Collector: LLM을 입력 텍스트 생성 전용으로 (element_extraction 기본 off + prefilter 결합 해제)

사용자가 LLM을 input text 생성 과정에서만 쓰고 나머지(element 추출·screen matching·action 선택)에는 쓰지 않기를 원해 Monkey-Collector의 LLM 사용을 입력 텍스트 생성 전용으로 기본 전환했다. action 선택은 원래 LLM을 안 썼고(coverage + transition-graph + RNG), 실제 LLM 호출 지점은 `text_generator`(유지)와 `element_extractor`(이번에 기본 off) 둘뿐이라 element_extraction만 끄면 목표가 달성된다. 다만 시각적 재방문 dedup을 담당하던 luminance/structural prefilter가 그동안 element_extraction(=ScreenMatcher 존재)에 묶여 있었으므로, 이 결합을 분리해 element_extraction을 꺼도 prefilter-only ScreenMatcher로 재방문 dedup이 계속 동작하게 했다.

- `llm.element_extraction` 기본값 true→false (`config.py` 3곳 + `config/run.yaml`). `input_mode=api` 는 그대로 유지 → 텍스트 입력 LLM 생성·비용추적 정상 동작.
- prefilter 결합 해제: `create_screen_matcher` 가 extractor 없이도 `luminance_prefilter` on 이면 ScreenMatcher를 **prefilter-only 모드**로 생성하도록 변경(`pipeline/screen_matching/__init__.py`), `ScreenMatcher.match()` 에 `extractor is None` early-return 추가 — structural fingerprint + luminance prefilter + observation dedup 은 유지하되 element-set 경로(step-1/expand/classify)를 우회해 `classify(∅, ∅)` 로 인한 빈-page 붕괴를 방지한다.
- `element_extraction` off 이면 same-function 압축(element family) 없이 pure unexplored-first 로 탐색이 degrade 된다(`pipeline/exploration/memory.py` 의 기존 graceful degrade 경로, 사용자 승인).
- 신규 유닛 테스트: `tests/unit/test_screen_matcher.py` 에 prefilter-only 그룹 3종 + `tests/unit/test_rehydrate.py` 에 prefilter-only resume 케이스 추가. `tests/unit/test_config.py` 의 기본값 단언 갱신(element_extraction=false).
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` + repo `docs/{DEVLOG,CHANGELOG}.md` 갱신 — "luminance prefilter 는 element_extraction on 이 필요" 라는 옛 서술을 prefilter-only 모드로 정정.
- 카테고리: devlog

## 2026-07-01 — Monkey-Collector 저장 레이아웃 재설계: `data/`+`runtime/` 이원화 + page/observation 구조 (MobileGPT-V2 memory/runtime 포팅)

Monkey-Collector의 저장 레이아웃을 flat `data/raw/{package}/` 구조에서 `data/`(memory)와 `runtime/`(세션 진행 상태) 이원화 + page/observation 기반 구조로 재설계했다. MobileGPT-V2의 memory/runtime 분리 설계를 포팅했다. 핵심 동기는 세 가지: (1) luminance/구조적 재방문 화면이 매번 새 파일을 쓰던 낭비 제거(재사용 관측은 파일을 전혀 쓰지 않음), (2) luminance prefilter가 "어떤 observation과 동일한지" 판단하는 실질적 역할을 갖게 됨(기존엔 페이지 식별에만 관여, observation 단위 재사용 판단이 없었음), (3) 세션 재개(resume) 시 ScreenMatcher/page_graph 지식이 디스크에서 복원되도록 수정(기존엔 매번 무조건 reset되어 이미 아는 page를 다시 "새 page"로 재발견하던 pre-existing 버그를 함께 고침). 코드 변경, 작업트리 미커밋.

- 새 디렉토리 구조: `data/{package}/pages/{page_key}/page.json`(고정 anchor, 최초 1회만) + `{observation_num:04d}/{screenshot.png,raw.xml,parsed.xml,hierarchy.xml,encoded.xml,pretty.xml,elements.json}` + `page_graph.json`/`.html`. `runtime/{package}/metadata.json`+`events.jsonl`+`cost.csv`+`activity_coverage.csv`.
- `storage.py`: 구 flat 저장 메서드(`save_screenshot`/`save_xml`/`save_elements`) 제거하고 `save_observation`/`save_page_knowledge`/`load_page_knowledge`/`list_pages`/`list_observations`/`next_frame_index` 신설. `regenerate_xml_variants`는 신규/구형 레이아웃 자동 판별.
- `screen_matcher.py`: `ScreenMatch`에 `observation_num`/`is_new_observation` 추가. 전역 `_luminance_lookup`(페이지 식별, 기존과 동일 역할)과 별개로 page-scoped `_page_luminance_lookup` + `_record_observation` 신설해 "이 페이지의 어떤 observation과 같은가"를 판단 — luminance 비활성 시엔 항상 새 observation 할당(픽셀 비교 근거가 없으므로 안전하게 dedupe 안 함).
- 신규 모듈 `pipeline/screen_matching/rehydrate.py`: 세션 재개 시 `data/{package}/pages/`를 순회해 ScreenMatcher의 registry/`_fp_to_key`/`_counter`를 복원. luminance 지문은 각 observation의 저장된 `screenshot.png`에서 재추출(별도 캐시 파일 미도입, 기존 PIL-only 방침 유지).
- `session_manager.py`: `init_or_resume_session`이 `(session_id, resume_step, is_resumed)` 3-tuple 반환. 신규 `rehydrate_session()`이 `page_graph.json`도 `state.page_graph`로 복원(안 하면 `finalize_session`이 이번 세션에서 재방문한 page만으로 그래프를 덮어써 나머지 이력을 잃는 버그가 있었음). `--new-session`은 `data/`+`runtime/` 두 root를 모두 삭제하도록 수정(한쪽만 지우면 resume 로직이 남은 쪽에서 지식을 복원해 "새 세션"이 되지 않는 문제).
- `domain/page_graph.py`: `PageNode`에 `observation_count` 필드(구 `page_graph.json`은 `.get` 기본값 0으로 로드). `build_graph_from_new_layout` 신설 — `events.jsonl`의 `page_key`/`observation_num`을 직접 읽는 정확한 재구성(구조 근사 불필요). 기존 `build_graph_from_session`(activity+Jaccard 구조 근사)은 `pages/` 없는 마이그레이션 이전 세션의 degrade 경로로 유지.
- `export/converter.py`: `page_key`/`observation_num`으로 before/after 화면을 조인(연속 이벤트가 같은 observation을 가리키면 시각적 변화 없음으로 스킵). `pages/` 없는 세션은 `_convert_session_legacy`로 degrade. `--data-dir`가 존재하지 않을 때 발생하던 미처리 `FileNotFoundError` 크래시를 경고+0건 반환으로 수정(사용자가 실제로 겪은 이슈).
- `cli.py`: run/reset/convert/convert-all/page-map/page-map-all/regenerate의 `--output`/`--session`/`--raw-dir` 플래그를 `--data-dir`/`--runtime-dir`(+`--package`)로 교체. `pipeline/reset.py`의 `resolve_targets`가 두 root 모두 대상으로 삭제하도록 변경.
- `config.py`/`run.yaml`: `collection.output_dir` → `data_dir`+`runtime_dir`, `MC_COLLECTION_OUTPUT_DIR` → `MC_COLLECTION_DATA_DIR`/`MC_COLLECTION_RUNTIME_DIR`.
- 마이그레이션 스크립트는 만들지 않음(사용자 결정) — 기존 flat 레이아웃 세션은 그대로 두고 새 도구가 감지해 자동 degrade.
- 변경(작업트리, 18 tracked files + 3 new files — `rehydrate.py`/`test_rehydrate.py`/`test_session_manager.py`): `README.md`/`ARCHITECTURE.md`/`AGENTS.md`/`.claude/skills/setup-collector/`(SKILL.md + `references/run-and-verify.md`)의 `data/raw` 참조를 전부 새 구조 설명으로 갱신. 루트 `.gitignore`에 `**/runtime*/` 패턴 추가(기존 `**/data*/` 패턴은 `runtime/`을 커버하지 않았음). `git diff --stat`(마지막 5커밋 기준): 18 files changed, 522 insertions(+), 23 deletions(-) — 작업트리 미커밋 변경 전체를 포함하는 수치는 아님.
- 결과/검증: `uv run pytest -q`: **666 passed**(0 failed). `uv run ruff check src/ tests/`: 9건, 이 리팩터 이전 baseline과 정확히 동일(전부 `test_structured_parser.py`의 기존 이슈, 무관). `uv run mypy src/`: 37건, baseline 35건 대비 +2 — `storage.py`에 기존부터 있던 "Optional 경로 str|None" 패턴(accepted-debt로 문서화된 패턴)이 새 메서드에서 반복된 것으로 새로운 종류의 이슈는 아님. Collector+ScreenMatcher(FakeExtractor)+DataWriter를 스크립트로 구동한 수동 스모크 테스트: (1) 동일 화면 재방문 시 관측 파일 0개 생성 확인, (2) SUPERSET_MERGE로 다른 렌더 상태 진입 시 새 observation 폴더 생성 확인(page_graph의 `visit_count=3`, `observation_count=2`로 정확히 일치), (3) 세션 재개 후 같은 page가 재발견되지 않고 `observation_count`가 유지됨을 확인. `monkey-collect convert-all`을 실제 백업 데이터(4개 세션)로 재검증해 941개 예제 생성 확인(레거시 degrade 경로 정상 동작).
- 커밋: 미커밋(작업트리). 직후 project-sync + git-push 예정.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` + `.claude/skills/setup-collector/`(SKILL.md, `references/run-and-verify.md`) 갱신. 루트 `docs/{README,ARCHITECTURE}.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-07-01 — Monkey-Collector luminance prefilter 포팅 — 동일 화면 재방문 시 LLM element-extraction 0회 (MobileGPT-V2 Stage-0)

MobileGPT-V2 의 Stage-0 luminance prefilter 를 Monkey-Collector `ScreenMatcher` 에 포팅했다. 시각적으로 동일한 화면을 재방문할 때 LLM element-extraction 호출을 **0회로 단락**해 수집 비용을 줄인다. 기존 page_key 를 재사용하므로 page_graph/explorer 일관성도 보존된다. 코드 변경(기록 전용 아님), 작업트리 미커밋.

- Stage-0c prefilter: `ScreenMatcher.match()` 에 신규 단계 추가 — screenshot 의 luminance 지문(순수 PIL, BT.601 luma via `convert("L")`, width-100 LANCZOS 리사이즈)을 저장 page 지문과 비교해 차이 픽셀 비율 < `screenshot_diff_threshold` 면 그 `page_key` 로 merge(`match_type=LUMINANCE_PREFILTER`, LLM 0회). **pending guard 뒤에 배치**해 빈-page blackhole 보호를 유지하고, 기존 `page_key` 재사용으로 page_graph/explorer 일관성 보존.
- 신규 모듈 `pipeline/screen_matching/luminance.py`(`extract_luminance_features`, `luminance_diff`): numpy 미사용(기존 선언된 Pillow 재사용), `ImageChops.difference` 히스토그램의 strict `|ΔY|>threshold` 카운트로 차이 픽셀 비율 산출.
- `PageKnowledge.luminance_features`: 세션 인메모리 지문(page 당 cap 10, 디스크 미영속).
- 하이퍼파라미터 4종을 config 6-place 노출: `luminance_prefilter`(기본 ON, 사용자 결정)·`luminance_threshold`(10)·`screenshot_diff_threshold`(0.02)·`luminance_low_res_width`(100) → `config.py` + `config/run.yaml` + `MC_SCREEN_MATCHING_*` env + `cli.py`(`--luminance-prefilter` 등). run.yaml 의 stale 헤더 주석(GREEDY→BFS, element_extraction 섹션)도 동시 정정.
- 주입: `collection_loop` 가 `collector._latest_screenshot` 을 `match()` 에 전달. element_extraction off(matcher 없음)면 prefilter 도 비활성.
- 변경(작업트리, 17 files): src 7(NEW `pipeline/screen_matching/luminance.py` + MOD `screen_matcher.py`·`page_knowledge.py`·`screen_matching/__init__.py`·`pipeline/collection_loop.py`·`config.py`·`cli.py`) + config 1(`config/run.yaml`) + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 4(NEW `tests/unit/test_luminance.py` + MOD `tests/unit/test_{screen_matcher,config,cli}.py`) + repo docs 2(`docs/{DEVLOG,CHANGELOG}.md`).
- 검증: **462 unit tests passed**(신규 `test_luminance.py` + `test_{screen_matcher,config,cli}` 확장). 신규 파일 ruff/mypy clean, baseline 대비 신규 에러 0.
- 커밋: 미커밋(작업트리). 직후 git-push 예정.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` 갱신. 루트 `docs/{README,ARCHITECTURE}.md` 는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-30 — Monkey-Collector converter 프레임-이벤트 정렬 버그 수정 (frame_index 도입)

world-modeling 학습데이터 변환에서 **action 라벨이 엉뚱한 프레임에 붙는 정렬 버그**를 발견·수정했다. `input_text` 처럼 페이지가 안 바뀌는 action 도 `_encoded.xml` 의 `value` 변화로 before≠after 학습쌍이 되는데, 실제 변환 결과는 `input_text` 가 중복되고 Discard 다이얼로그 프레임에 input 라벨이 붙는 등 어긋나 있었다. 근본 원인은 events.jsonl 의 `step`(루프 카운터, 비-저장 반복에서도 증가)과 프레임 파일 인덱스(`step_count`, 저장 시에만 증가)가 다른 체계인데 둘을 잇는 조인 키가 디스크에 없던 것. 코드 변경(저장 포맷 변경 포함), 작업트리 미커밋.

- Fix: 수집 시점에 정상 action 이벤트에 `frame_index = writer.step_count - 1`(before 프레임 파일 인덱스)을 기록(`collection_loop._process_xml_signal`). converter(`export/converter.py`)와 offline page-graph 재빌드(`domain/page_graph.py _load_events`)를 `events.get(step_idx)` 추측·`_find_event_by_index` 폴백에서 **`frame_index` 직접 조인**으로 교체. converter 의 after 프레임은 "다음 action 의 before 프레임"으로 잡아 중간 로딩 프레임을 건너뛴다.
- step 의미 재정의: `state.step += 1` 을 정상 action 경로 1곳만 남기고 **13곳 제거**(signal timeout·permission·system·stale·no_change·keyboard·same-page-stuck·empty-UI·예외). 이제 `step` 은 실제 action 에서만 증가 → 사용자 요청대로 signal timeout 등이 step 을 소비하지 않음. 부수: step 이 상한 역할을 하던 경로 보호용 `idle_iterations` 절대 가드(`max_step*4`) 추가 + stale 경로 `clear_signal_queue` 보강. activity_coverage 기록을 `step_count`(파일 인덱스)로 키잉해 page-graph CSV 폴백 정합 유지.
- 하위호환: `frame_index` 없는 구(舊) 세션은 converter 가 변환 스킵, page-graph 재빌드는 토폴로지만 복원하고 엣지 라벨 `unknown` 으로 degrade → 재수집/regenerate 필요(옛 휴리스틱 폴백 제거).
- 검증: 합성 세션 sanity 로 same-page `input_text` 가 (before=빈 필드 → Input → after=채워진 필드, element_index 정확) 학습쌍으로 정렬됨을 실증. 회귀 테스트 6종(step≠frame_index 조인, empty-UI 건너뜀, no_change_retry/transition:false/frame_index 부재 제외, 마지막 action after 부재) + page-graph 엣지 라벨 회귀 추가. 기존 fixture 가 `step==파일인덱스` 를 우연히 1:1 로 맞춰 버그를 못 잡던 것도 `step=frame_index+100` 으로 어긋나게 수정.
- 변경(작업트리, 11 files): src 3(`pipeline/collection_loop.py`, `export/converter.py`, `domain/page_graph.py`) + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 3(`tests/unit/{test_converter,test_page_graph}.py`, `tests/fixtures/session_fixtures.py`) + repo docs 2(`docs/{DEVLOG,CHANGELOG}.md`).
- 검증: 전체 **pytest passed**, ruff 신규 위반 0(변경 파일 clean; 선재 `test_structured_parser.py` F401 은 그대로).
- 커밋: 미커밋(작업트리). 직후 project-sync + git-push 예정.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` 갱신. 루트 `docs/{README,ARCHITECTURE}.md` 는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-30 — Monkey-Collector external 복구 시 open_app 기록 (navigation 격리)

external app 이탈에서 복구 루프가 타깃 앱을 재실행하는 동작(=사실상의 `open_app`)을 events.jsonl 에
액션으로 기록하기 시작했다 — open_app 학습 데이터 확보용. 단 external 이탈은 의도된 화면 전이가 아니므로
이 open_app 은 **navigation 으로 절대 쓰이지 않게 3중 격리**했다. 코드 변경(기록 동작 추가), 작업트리 미커밋.

- 기록: `return_to_app()`/`recover()` 를 `-> bool`(실제 `launch_app` 여부)로 바꾸고, `collection_loop._handle_external_app` 가 launch 시 `_record_open_app`→`DataWriter.log_open_app` 로 `{action_type:"open_app", package, app_name, step, transition:false, trigger:"external_recovery", from_package}` 를 **excursion 당 1회**(`state.open_app_logged` dedup, 다음 in-app 프레임에서 재무장) 기록. `app_name` 은 apps.csv 조인(`cli._resolve_app_names`→`Collector(app_names=...)`), 없으면 빈 문자열.
- navigation 격리(3중): (1) 복구 후 `state.last_action`/`last_ui_tree` 클리어 → live page graph 엣지 차단, (2) `return_to_app`/`recover` 진입 시 `explorer._last_record` 클리어 → routing memory 전이 차단(모든 복구 호출처의 선재 stale-전이 버그도 동시 수정), (3) 이벤트 `transition:false` + `page_graph._load_events` 스킵 가드 → offline 재빌드(`page-map`)·world-modeling converter 배제.
- 스레드 안전: open_app 은 메인 루프, external_app 콜백은 TCP 수신 스레드에서 events.jsonl/metadata.json 에 동시 기록 → `DataWriter` 에 `threading.Lock` 추가해 append·counter 보호. `OpenApp` dataclass 는 record-only(execute_action·select_action 미경유)지만 `ACTION_REGISTRY` 등록으로 라운드트립 가능.
- 한계: 서버측 기록은 서버 주도 재실행만 포착. 클라이언트(`CollectorService.kt`)가 자체 force-launch 하는 경우는 과소기록될 수 있음. open_app 이벤트는 before/after XML 쌍이 없어 기존 world-modeling 변환으로는 소비되지 않음(의도된 분리) — 학습은 별도 변환에서 `step`/`package` 로 직전 in-app 프레임과 페어링해 파생.
- 변경(작업트리, 18 files): src 7(`domain/actions.py`, `storage.py`, `pipeline/exploration/explorer.py`, `pipeline/collection_loop.py`, `domain/page_graph.py`, `pipeline/collector.py`, `cli.py`) + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 6 + repo docs 2(`docs/{DEVLOG,CHANGELOG}.md`).
- 검증: 전체 **pytest passed**, ruff 신규 위반 0(선재 위반은 그대로). 부수: external 카운터 테스트 2건의 stale 기댓값(`recover==6`)을 실제 동작(reinit 분기도 recover 호출 → 7)에 맞게 수정 — HEAD 에서도 이미 실패하던 선재 red.
- 커밋: 미커밋(작업트리). 직후 project-sync + git-push 예정.
- 카테고리: devlog

## 2026-06-30 — Monkey-Collector 빈 page_0 blackhole 버그 수정 + LLM element description/parameters 디스크 보존

element-set screen matching에서 두 결함을 함께 고쳤다: (1) 세션 첫 로딩/스플래시처럼 interactable이 0개인
화면이 빈 page_0로 등록돼 이후 모든 화면을 흡수하던 **blackhole** 버그, (2) LLM이 추출한 element의
`description`/`parameters`가 디스크 저장 시 누락되던 문제. 코드 변경만(기록 전용 아님), 작업트리 미커밋.

- Fix 1 (blackhole): `ScreenMatcher.match()`에 entry guard 추가 — interactable(button/input) 0개 화면은 LLM 호출 없이 `pending`으로 거부해 page로 등록하지 않음(첫 유효 화면이 page_0이 됨). `set_classifier`에 안전망 — 저장 page B=∅이면 SUPERSET_MERGE 불가→DISJOINT라 어쩌다 등록된 빈 page도 sink가 되지 않음. `collection_loop`는 pending 시 page 노드 생성·`save_elements`를 스킵. 부수: `extract_interactable_indexes`를 root-inclusive(`tree.iter()`)로 수정해 단일 루트 interactable 누락 차단.
- Task 2 (description/parameters): `ElementFamily`에 `description`/`parameters` 필드 추가(끝에 — 하위호환), families 생성부에서 ExtractedElement의 5필드 전부 채움, `DataWriter.save_elements`가 `{step}_elements.json`의 각 element에 description/parameters 키를 직렬화. 최종 element 형태: name/description/parameters/element_index/key_element_index.
- 범위 결정: 버그리포트의 "merge 시 scroll-reveal element 누적"(권장)은 page-identity/tie-break 드리프트 위험으로 이번에서 제외(별도 변경으로 분리). 라이브 스모크도 제외 — 단위/오프라인 검증으로 대체.
- 변경(작업트리, 12 files): src 5 + 패키지문서 3(`Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`) + tests 4. 관련 단위테스트 갱신/추가.
- 커밋: 미커밋(작업트리). 직전 커밋 b0ac999, 직후 git-push 예정.
- 결과/검증: 전체 **545 passed**. 기존 3 failures는 직전 커밋 5cc02c4(max_steps 100→1500)/10d9eee(reinit on timeout)發로 본 변경과 무관.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`만 갱신(작업트리). 루트 `docs/{README,ARCHITECTURE}.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-29 — 화면 그룹핑(ScreenGrouper)을 element-set screen matching으로 교체 (MobileGPT-V2 Node-Clustering 포팅)

LLM "화면 그룹핑"(`llm/screen_grouper.py`, annotation 전용이라 탐색에 미반영)을 MobileGPT-V2의 Node-Clustering을
포팅한 **element-set screen matching**(`pipeline/screen_matching/`)으로 교체했다. 화면당 단일 LLM 호출로 각
element의 `element_index`(같은 기능 family 전체)와 `key_element_index`(대표 anchor)를 함께 추출하고,
set-classification으로 산출한 `page_key`가 page_graph 노드와 탐색 abstract page를 같은 키로 결정한다 — 과거
grouping↔page matching 디커플을 의도적으로 커플링했다. (동시 진행한 app-context 입력 텍스트 생성은 아래 엔트리 참조.)

- 신규 패키지 `pipeline/screen_matching/`: `ScreenMatcher` / `ui_attributes`(UIAttributes 지문 + find_matching_node ancestor-walk + text_blind + get_ui_key_attrib + mask_xml_to_indexes) / `set_classifier` / `page_knowledge`(PageKnowledge·KnowledgeRegistry). 신규 `llm/element_extractor.py` + `llm/prompts/element_extractor_prompt.py`로 MobileGPT-V2의 SubtaskExtractor·TriggerUIAgent 2호출을 1호출로 병합.
- `ScreenMatcher.match` 흐름: ① 구조 지문 pre-filter(exact 재방문 short-circuit, LLM 0회) → ② step-1 text-blind ALL-match(저장 anchor) → supported+remaining → ③ expand(remaining 마스킹 후 재추출, dry/`--max-expand-iters` cap) → ④ set-classify(EQSET/SUPERSET_MERGE/SUBSET_MERGE는 containment-always-merge, OVERLAP만 two-sided tolerance band) → ⑤ dispatch(MERGE=stored page_key frozen / NEW=새 page_key + anchor를 현재 화면에서 fingerprint).
- linchpin은 인덱스 공간 일치(extractor encoded XML = `{step}_encoded.xml` = `SemanticElement.index`, 모두 parse→_renumber→_clear_bounds): V2 파서는 포팅하지 않고 매칭 함수만 MC encoded 스키마(tag/aria-label/alt/text/type/value; id/class 없음)로 적응, distinctive 판정은 aria-label/alt. 산출 `page_key`가 `PageGraph.get_or_create_page_by_match`와 `SemanticState.page_key`(Memory/TransitionGraph/Navigator 키)를 모두 결정. `finalize_session`은 matcher 활성 시 라이브 `state.page_graph`(page_key/element_names)를 그대로 저장, 오프라인 page-map만 구조 지문 재구성.
- CLI: `--screen-grouping` → `--element-extraction {on,off}`(기본 on, `--screen-grouping`은 deprecated alias 유지) + `--cluster-merge-tolerance`(0.2) + `--max-expand-iters`(3). 산출물 `{step}_groups.json` → `{step}_elements.json`, cost.csv agent 라벨 `screen_grouper` → `element_extractor`. degrade: `OPENROUTER_API_KEY` 없거나 off면 matcher 미생성 → `page_key=structure_str` fallback + Memory 압축 없음 = 기존 파이프라인 byte-for-byte.
- 변경(작업트리): 신규 `pipeline/screen_matching/{__init__,ui_attributes,screen_matcher,set_classifier,page_knowledge}.py`·`llm/element_extractor.py`·`llm/prompts/element_extractor_prompt.py`·`tests/unit/test_{ui_attributes,set_classifier,screen_matcher,element_extractor}.py`; 수정 `domain/page_graph.py`·`pipeline/{collection_loop,collector,session_manager,exploration/{explorer,memory,navigator,state,transition_graph}}.py`·`storage.py`·`cli.py`·`llm/__init__.py`·`__init__.py`·`Monkey-Collector/{ARCHITECTURE,README,AGENTS}.md` + 다수 기존 테스트; 삭제 `llm/screen_grouper.py`·`tests/unit/test_screen_grouper.py`.
- 커밋: 미커밋(작업트리). last_sync(2026-06-29T17:56:42+09:00) 이후 신규 커밋 없음 — 직전 커밋 beb6a8c, 곧 커밋·푸시 예정.
- 결과/검증: 전체 테스트 green(신규 test_ui_attributes/test_set_classifier/test_screen_matcher/test_element_extractor + 기존 test_memory_unexplored/test_navigator/test_transition_graph_nav/test_storage/test_page_graph/test_semantic_state 갱신), 신규 파일 ruff 0/mypy clean, 전체 mypy 회귀 0(baseline 28 유지)·ruff 16→10. 라이브 org.tasks E2E PASS(Pixel6-2): element-extraction on → page_graph.json `page_key`(`page_0`,`page_1`)+element_names 저장, `{step}_elements.json` families(element_index+key_element_index), match_type ladder 전부 관측(NEW/SUBSET_MERGE/STRUCTURAL_IDENTICAL×20/DISJOINT/EQSET), pre-filter로 24스텝 중 LLM 2회($0.003), 크래시 0; degrade off → 구조경로 완주(page_key 0·elements.json 0·LLM 0).
- 카테고리: devlog

## 2026-06-29 — input-text LLM 생성에 앱 설명 주입

탐색 중 텍스트 입력값을 LLM(`--input-mode api`)으로 생성할 때 프롬프트에 "현재 어떤 앱을 탐색 중인지"가
전혀 없어 앱 도메인에 안 맞는 입력(쇼핑앱 검색창에 일반 단어 등)이 나올 수 있던 문제를 보완했다.
`catalog/apps.csv`의 사람이 읽을 수 있는 메타데이터를 input-text 프롬프트에 주입하도록 배선했다.

- 설계: 공유 `TextGenerator` 인스턴스(cli에서 1회 생성→explorer의 `ActionMapper`+`Collector` 양쪽 공유)에 세션마다 setter로 앱 설명을 박는 방식 채택. `generate()`에 인자를 스레딩하는 대안은 `TextGenerator`/`ActionMapper`/`Explorer` Protocol 시그니처가 줄줄이 바뀌어 기각.
- 변경(작업트리, src 4): `pipeline/app_catalog.py`(`AppJob.description` 프로퍼티: `app_name (category/sub_category) — notes`, app_name 없으면 package_id 폴백); `pipeline/text_generator.py`(`TextGenerator` 베이스에 no-op `set_app_context` 훅 + `LLMTextGenerator` 오버라이드, `generate()`가 `App under test:` 줄을 프롬프트 앞에 조건부 prepend, SYSTEM_PROMPT에 도메인 맞춤 규칙 1줄); `pipeline/collector.py`(`__init__(app_contexts=...)` + `_run_session`이 패키지 확정 후 **무조건** `set_app_context(self._app_contexts.get(pkg, pkg))` — 공유 generator의 이전 앱 설명 누수 방지, `text_generator=None` 가드); `cli.py`(신규 `_resolve_app_contexts(packages)→dict` 헬퍼: AppCatalog 로드 best-effort, 미등록/누락은 dict 제외→Collector가 package_id 폴백; `Collector(app_contexts=...)` 배선). `_resolve_run_packages`는 테스트가 시그니처 고정이라 미변경.
- 테스트(신규만, 기존 0 수정): `test_app_catalog`(description 4종: full/no-notes/no-category/package_id 폴백), `test_text_generator`(app_context 포함·미설정/공백 시 줄 생략·random no-op), `test_run_resume`(`_resolve_app_contexts` 정상/csv없음→{}/미등록 제외), `tests/integration/test_collector`(map→설명 호출·빈map→package_id 폴백 호출).
- 결과/검증: 전체 **545 passed**(타깃 96 포함), 변경 src+신규 단위테스트 ruff clean(`set_app_context` 베이스 no-op은 의도적 선택훅이라 `# noqa: B027`). 라이브 스모크는 미실행(정적 검증까지).
- 커밋: 미커밋(작업트리).
- 문서: `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md` + `docs/CHANGELOG.md` 갱신.
- 카테고리: devlog

## 2026-06-29 — setup-collector 스킬 전면 갱신 + 라이브 재검증 + 런타임 권한 자동허용

Monkey-Collector `setup-collector` 스킬을 MobileGPT-V2 `setup-emulator` 구조(SKILL.md 오케스트레이션 +
`references/` deep-dive)로 재구성하고, 이전 디버깅 세션의 소스 패치들을 커밋 가능한 상태로 정리·문서화한 뒤
emulator-5554/Pixel6-2에서 end-to-end 라이브 재검증했다. 검증 중 사용자 피드백으로 런타임 권한 다이얼로그
자동허용("While using the app")을 추가 보강했다.

- 변경(작업트리): `.claude/skills/setup-collector/SKILL.md`(AVD MobileGPT-V2-2→Pixel6-2 전수 교체, 빌드 JDK8→JDK17/AGP8.2, APK 경로 `app/app/build/outputs`, `local.properties` 노트, 신규 §6-c MediaProjection 재동의·§6-d Google 로그인·§6-e 더미데이터 시드·§9 라이브 검증, 전 단계 멱등); `.claude/skills/setup-collector/references/` 8종 신설(client-build, mediaprojection-accessibility, google-login, run-and-verify, seed-helpers, seed-pim, seed-notes-tasks, seed-media-misc); `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`(Pixel6-2·JDK17·MediaProjection 단발성/graceful-degrade·EXCLUDED gms·screen_guard·no-ACK abort·권한 자동허용 반영); `.gitignore`(`**/*.secrets.local` 추가).
- 소스 패치(5파일): `app/.../ScreenStabilizer.kt`(MediaProjection 토큰 단발성 reuse-guard + `createVirtualDisplay` try/catch graceful-degrade), `CollectorService.kt`(`EXCLUDED_PACKAGES += gms/gsf/vending`), `pipeline/screen_guard.py`(`SYSTEM_PACKAGES` 확장), `pipeline/session_manager.py`(no-ACK 시 abort), `pipeline/collection_loop.py`(빈-UI 가드를 `get_interactable_elements` 기준으로 + 신규 `_try_grant_permission_via_adb`: permissioncontroller `GrantPermissionsActivity`는 a11y 이벤트 부재로 timeout만 발생 → `adb uiautomator dump`로 clickable 버튼만 스캔해 "While using the app" 우선 탭, deny-guard).
- 커밋: 미커밋(작업트리). last_sync(2026-06-29T15:48:25+09:00) 이후 신규 커밋 없음 — 위 변경은 모두 워킹트리(8 modified + `setup-collector/references/` untracked), 직전 커밋 d7ca522.
- 결과/검증(emulator-5554/Pixel6-2, 패치 APK 재설치): Google 로그인 성공(Accounts:1, seungwoo896); 더미데이터 7앱 주입(연락처5·Simple Calendar4·Markor4·org.tasks4·RetroMusic3·OpenTracks3·Joplin3, API33 재검증); smoke run 2/2 PASS — org.tasks **21 pages/29 transitions**, Drive(`com.google.android.apps.docs`) **7 pages/9 transitions**, client 크래시 0, gms 외부앱 스톰 0(Drive top=100% apps.docs), signal timeout는 탐색 소진 시 자연 종료; 권한 자동허용 단위검증(3-button→"While using the app", 2-button→"Allow" 제목 회피, deny-only→무탭) + 실다이얼로그 수동 탭 CAMERA granted=true 확인, 관련 단위테스트 57+건 PASS.
- 문서: 패키지 정본 `Monkey-Collector/{README,ARCHITECTURE,AGENTS}.md`만 갱신(작업트리). 루트 `docs/ARCHITECTURE.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-29 — Monkey-Collector 탐색 엔진을 LLM-Explorer 방식으로 전면 교체

Monkey-Collector의 탐색 엔진을 기존 `SmartExplorer`(화면 단위 weighted-random)에서
`LLMGuidedExplorer`(참조 구현 LLM-Explorer 포팅: coverage-driven unexplored-first 선택 +
LLM same-function 압축 + UI transition graph 최단경로 navigation)로 완전 교체했다.
App(Kotlin)/Server TCP 아키텍처·데이터 수집 파이프라인·저장 포맷(`data/raw/{pkg}/`, `page_graph.json`,
`events.jsonl`, xml variants, `cost.csv`, `activity_coverage.csv`)은 그대로 유지한다. 이전엔
annotation 전용이라 탐색에 반영되지 않던 same-function grouping(ScreenGrouper)을 이제 탐색에 직접 반영하며,
LLM 키가 없으면 순수 unexplored-first로 graceful degrade한다.

- 변경: working tree 14 modified + `pipeline/exploration/` 신규 8파일(`state`/`memory`/`transition_graph`/`navigator`/`action_mapper`/`explorer`/`constants`/`__init__`) + 단위·통합 테스트 6 신규(`test_semantic_state`/`test_action_mapper`/`test_transition_graph_nav`/`test_memory_unexplored`/`test_navigator`/`test_llm_guided_explorer`); `pipeline/explorer.py`(SmartExplorer) + 전용 테스트 3개 삭제. DI 배선 교체(`cli.py`/`collector.py`/`recovery.py`/`pipeline/__init__.py`/`__init__.py`), `collection_loop`는 `set_raw_xml`→`set_screen_context` 2곳만 수정(계약 유지), 세션마다 `explorer.reset()`로 메모리 격리. `networkx` 의존성 명시 추가(numpy/pandas 불필요).
- 설계: `SemanticState`(state_str/structure_str/encoded-index 기반 SemanticElement), `Memory`((structure_str, element_signature, action_type) 단위 커버리지 + same-function 압축), `TransitionGraph`(networkx structure 그래프 + `shortest_nav_steps`), `Navigator`(`_nav_steps` 큐, 매 step signature 재매칭, 무한루프 가드), `action_mapper`(semantic action→domain Action), `explorer.py`(LLMGuidedExplorer + Explorer Protocol). navigation은 Server 순차 실행(Kotlin App 무변경).
- 커밋: 미커밋(작업트리). last_sync(2026-06-28T22:52:55+09:00) 이후 신규 커밋 없음. 현재 작업트리: explorer.py 등 4 삭제(D) + 15 modified + `exploration/`·신규 테스트 6 untracked.
- 결과/검증: 전체 테스트 green, 새 모듈 ruff/mypy 완전 clean(전체 mypy 부채 baseline 29→28로 감소). Pixel6-2 E2E VERDICT PASS — org.tasks 수집 25 steps 정상 종료(completed_at 기록), page_graph **12 nodes / 15 edges**(이전 SmartExplorer 세션 7 pages 대비 증가), action 다양성 tap15/long_press5/swipe4/press_back1, screen_grouper LLM 10회 호출(same-function 압축 동작), xml/screenshots/groups 25씩 저장, activity coverage 2/48(LocationPicker 지도 화면 도달; 지도 정적화면 timeout은 공통 recovery 경로).
- 문서: 패키지 정본 `Monkey-Collector/{ARCHITECTURE,README,AGENTS}.md` 갱신(Action Space/exploration 패키지 서술) — 본 루트 `docs/ARCHITECTURE.md`는 패키지 정본을 가리키므로 추가 수정 없음.
- 카테고리: devlog

## 2026-06-28 — OpenRouter LLM 통합 라이브 검증 (qwen/qwen3.7-plus, API Key + AVD)

이전에 구현·커밋한 Monkey-Collector OpenRouter 공용 LLM 통합(입력 텍스트 생성 + 화면 의미 그룹핑)을
실제 API Key와 AVD(Pixel6-2)로 라이브 검증했다. 정적 검증과 실제 모델·실제 디바이스 화면 기반
동작 검증을 모두 통과(VERDICT PASS). 이 엔트리는 검증 기록 전용이며 코드 변경은 없다.

- 정적: `uv run pytest -q` 504 passed; 변경 파일(`llm/`, `text_generator.py`, 신규 테스트) ruff clean(전체 트리 16건은 기존 baseline `test_structured_parser.py` F401/I001), 신규 LLM 코드 mypy 0건(`save_groups`는 `is None` 가드로 narrow; 전체 29건은 기존 baseline storage `session_dir str|None` 등), AGENTS.md 게이트=pytest 통과. CLI `run --help`에 `--screen-grouping {on,off}` + `--input-mode {api,random}` 노출 확인.
- 모델 슬러그: `qwen/qwen3.7-plus` 실제 OpenRouter chat 호출 성공('OK' 반환)으로 유효 확정. 잘못된 슬러그는 OpenRouter가 400 "not a valid model ID" 반환 확인.
- 화면 의미 그룹핑(핵심 신규): 실제 org.tasks task-edit 화면 → 의미 그룹 3개(Task name input+save / Info banner / Task attribute selectors) → `xml/0000_groups.json` 저장; 별도 task-list 화면은 독립 grouping(`xml/0001_groups.json`).
- 구조 캐시: 동일 화면 재호출 시 LLM 미호출(screen_grouper cost row 불변, 동일 객체 반환) — 비용 절감 동작 확인.
- 입력 생성: 실제 EditText(`org.tasks:id/title`, "Task name")에 문맥 인지 텍스트 'Buy groceries' 생성(LLMTextGenerator 경로).
- 비용 귀속: cost.csv agent 컬럼 정확 분리(screen_grouper×2 + text_generator×1), 단가 계산 정확(예: 533 in·1851 out → $0.0024344 @ qwen/qwen3.7-plus $0.40/$1.20 per 1M).
- Graceful fallback: 잘못된 `OPENROUTER_MODEL` override → 화면 그룹핑 빈 결과 + 입력 텍스트 random fallback, 둘 다 예외 전파 없음.
- 검증 방식: collection_loop/`cli.cmd_run`과 동일하게 DataWriter·CostTracker·공용 LLMClient·ScreenGrouper·LLMTextGenerator를 wiring한 harness로 실구동(scratchpad 격리, 프로젝트 data/raw 무손상).
- 미검증(정직 보고): 풀 `monkey-collect run` TCP 전 구간은 미실행 — 동반 Android 앱 `com.monkey.collector`가 AVD에 미설치라 TCP 핸드셰이크 상대 부재(해당 transport 계층은 이번 LLM 통합 변경과 무관).
- 변경: 코드 변경 없음(기록 전용). last_sync(2026-06-28T22:24:32+09:00) 이후 신규 커밋 없음; 관련 코드 커밋은 이전 동기화에 반영(af40196·13d1a48·a284c72·879f130·a22b030). 작업트리 미커밋: `.project-sync.json`, `docs/CHANGELOG.md`, `docs/DEVLOG.md`(이전 동기화 산출물).
- 카테고리: devlog

## 2026-06-28 — Monkey-Collector LLM 통합 재구성 (OpenRouter 공용 클라이언트 + 화면 의미 그룹핑)

`.claude/references/LLM-Explorer`(droidbot 기반 탐색기)의 `GPT` 클래스 패턴을 참고해 Monkey-Collector의
LLM 통합을 재구성했다. env 기반 OpenRouter OpenAI-호환 Chat Completions 래퍼인 **공용 클라이언트**
(`llm/client.py`, 기본 모델 `qwen/qwen3.7-plus`, 키 없으면 None → graceful fallback)를 신설하고 두 사용처가
이를 공유한다: (1) 입력 텍스트 생성을 OpenAI Responses API(gpt-5-nano)에서 공용 `LLMClient.chat()`으로
이전(OpenRouter가 Responses API 미지원 → Chat Completions 전환), (2) LLM-Explorer `_gen_state_semantic_info`를
이식한 **화면 의미 그룹핑**(`llm/screen_grouper.py`, `--screen-grouping {on,off}` 기본 on, 요소를 기능별로
묶어 `xml/{step}_groups.json` annotation 저장, 동일 구조 캐시로 비용 절감, 실패 시 수집 흐름 무영향). 비용은
cost.csv의 agent 컬럼(text_generator/screen_grouper)으로 구분하고 cost_tracker에 qwen 단가를 추가했다.

- 변경: `llm/{client,screen_grouper,__init__}.py`(신규), `pipeline/text_generator.py`·`collection_loop.py`·`collector.py`, `storage.py`, `cli.py`, `domain/cost_tracker.py`, `.env.example` + Monkey-Collector README/ARCHITECTURE/AGENTS 동기화 (19 files, +835 −175 @ HEAD~5..HEAD)
- 커밋: af40196 (공용 LLM 패키지), 13d1a48 (text 생성 이전), a284c72 (파이프라인 배선), 879f130 (문서 동기화), a22b030 (루트 .claude gitignore) — main push 완료
- 결과/검증: 전체 504 tests passed (신규 test_llm_client/test_screen_grouper + test_text_generator/test_storage 갱신 포함); 도입분 ruff/mypy 이슈 해결(잔여는 baseline 기존 이슈); `run --help`에 `--screen-grouping` 노출 확인. 라이브 스모크 미실행(Pixel6-2 AVD + OPENROUTER_API_KEY 필요).
- 후속: 모델 슬러그 `qwen/qwen3.7-plus`는 하드-고정 안 함(`OPENROUTER_MODEL` override, 카탈로그 실제 슬러그 확인 권장); 그룹핑은 annotation 저장만이며 explorer 탐색 편향 미적용(UITree index ↔ encoded index 공간 불일치) → 후속 분리 가능.
- 카테고리: devlog

## 2026-06-28 — `/project-sync` 초기 설정 (docs·memory·notion)

`/project-sync init`으로 프로젝트 기록 동기화를 설정했다. 모노레포 루트에 `docs/` 허브를 만들고,
Notion 워크스페이스의 5개 카테고리 DB + Timeline 허브 ID를 랜딩 페이지에서 자동 추출해 config에 등록했다.

- 변경: `.project-sync.json`(신규), `docs/{README,ARCHITECTURE,ROADMAP,CHANGELOG,DEVLOG}.md`(신규)
- 활성 플랫폼: `docs`, `memory`, `notion` — Obsidian은 이 Linux 머신에 Vault가 없어 제외
- 결과/검증: Notion 랜딩 페이지 읽기 접근 검증 완료(5 DB + 허브 ID 추출, config.md 캐시와 일치)
- 카테고리: devlog
