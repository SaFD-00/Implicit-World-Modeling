# Dev Log

시점성 진행 로그 (append-only). 최신 엔트리를 위에 추가한다. 과거 엔트리는 수정·삭제하지 않는다.
상세 결과는 Notion Dev Log / Experiments DB, 계획은 [ROADMAP.md](./ROADMAP.md) 참조.

## 2026-07-14 — Monkey-Collector: S-9 진단 성공·수정 실패 (canvas_merge 기본 OFF, S-9 열림) + package_guard(선재 결함 수정) + MC→IWM 브리지 정적 관통

S-9(지도류 연속 캔버스의 page 지문 파편화)를 오프라인 리플레이로 전수 진단했다 — **원인 규명에는 성공했으나 안전한 수정에는 실패했다.** 부수로 발견한 패키지 교차 병합 선재 결함은 닫았고, 별도로 MC→IWM 학습 브리지가 코드 변경 0 으로 정적 관통했다. 상세는 [분석 보고](../Monkey-Collector/.claude/analysis/2026-07-14_18-10-03_s9-diagnosis-and-bridge/README.md) · [revise 기록](../Monkey-Collector/.claude/devlog/2026-07-14_18-10-03_s9-and-bridge.md).

- **⚠️ 직전 엔트리(F2 osmand)의 픽셀 게이트 추정은 이번 진단으로 반증됐다**: 바로 아래 엔트리는 S-9 의 원인을 "`ScreenMatcher` 의 BM25+픽셀 게이트가 **이산적 UI 를 전제**하기 때문"이라고 추정했다. 오프라인 리플레이로 osmand page-mint 를 전수 분류한 결과 **그 추정은 틀렸다** — 파편 45건 중 픽셀 게이트로 막힌 건 armA 2건 / armB 3건뿐이다. **진짜 지배 원인은 element-blocked**(armA **43/45=95.6%**, armB **22/26=84.6%**): 지도의 축척·거리·주소 텍스트와 상태별 aria-label 이 재방문마다 element-line 문서를 재작성해 `element_diff_max=5` 를 크게 넘긴다(element_diff median armA **13** / armB **13.0**[15 아님 — 초기 median_high 오류를 tier-2 가 정정], max 41/45). 과거 엔트리는 append-only 원칙상 수정하지 않는다.
- **❌ S-9 는 고쳐지지 않았다 — canvas_merge 는 기본 OFF 로 출하했다**: 후보 수정 `canvas_merge`(두 canvas 화면 비교 시 element-line 텍스트-blind + 픽셀 게이트 abstain)는 파편화를 절반으로 줄인다(armA map **46→27**, armB **26→14**, 오프라인 리플레이, cross-package 0건). 그러나 **지도가 아닌 화면까지 병합**한다 — osmand 내비게이션 **드로어(메뉴 15개 = 앱의 메인 허브)** 와 **턴바이턴 추적 모드**가 지도 page 로 흡수된다(a11y 가 이 화면들을 MapActivity 로 보고해 activity 가드로도 못 막고, 드로어 메뉴는 aria-label 없는 TextView 라 blind 시 집합 기반 element 기준이 카디널리티를 버린다). 카디널리티(multiset diff) 복원 시도는 드로어/내비는 분리하나 **수정이 수행한 병합 23건 중 20건도 함께 깨뜨려 무효화**됐다(이 수치·드로어 15·내비 7 은 commit `87095ef` 메시지 기재값이며 원천 JSON 으로 독립 재현하지 못함 — "확인됨"이 아니라 "commit 기재값"). → **canvas_merge 기본 OFF, S-9 는 열려 있다.** 임계값 상향으로도 못 고친다(median 이 임계값의 2.6배).
- **🟢 부수 발견(선재 결함, canvas 와 무관) — 패키지 교차 병합을 `package_guard`(기본 ON)로 닫음**: 게이트 밖 전수 감사에서 BM25 병합 경로가 화면의 소속 앱을 검사하지 않아 **런처 홈 화면이 앱 page 로 흡수**됨을 발견했다 — 라이브 코퍼스에 **이미 오염**(osmand armB **+2**, broccoli **+1**; armA/musicplayer/calendar 0). `package_guard`(기본 ON, fail-open)로 닫아 R2(교차 0/5)·R3(가드 단독 효과 전량 귀속) 통과. **⚠️ 소급 적용 안 됨 — 기존 아카이브는 오염된 채이고 낫게 하려면 재수집이 필요하다**(사용자 확인 게이트).
- **🟡 MC→IWM 학습 브리지 — 정적 관통까지, "학습 실행" 아님**: 이전 "학습 코드가 MC 코퍼스를 참조하지 않는다"는 보고는 **grep false negative**(레지스트리가 `MonkeyCollection`/`IWM-MC` 라는 다른 이름 사용)였고, 배선은 이전부터 존재했다. 첫 관통: musicplayer+calendar → **164 examples(train 155 / test 9)**, 스키마 검증 전 행 통과, `gen_configs --check` 통과, osmand 는 S-9 오염으로 **의도 제외**, **코드 변경 0**. **정직한 상한**: `stage1_train.sh` 는 이 맥에서 **DRY_RUN 조차 불가**(conda env 부재, bash 3.2, `LlamaFactory/` 디렉터리 자체 없음 — 원격 GPU 박스 전용)라 **"학습을 돌렸다"는 주장은 없다.** 입력은 실험 잔여물이지 프로덕션 코퍼스가 아니며, test 9개는 `--stage1-ratio` 기본값 0.95 를 164행에 조정 없이 적용한 결과로 통계적 무의미하다.
- **정직성 — cross-app 일반화 없음**: S-9 진단·수정 시도·package_guard 실측은 전부 osmand 범위다(통제 3앱은 교차 0건 확인용). 사전등록 5/5 예측 적중(prereg mtime 17:28 < 리플레이 산출물 17:46~).
- **tier-2(codex terra)가 결함 2건 적발** — 둘 다 결과를 바꿨다: (1) median 계산 오류 **15→13.0** 정정(보고 수치를 바꿈), (2) canvas 병합이 **드로어·내비 화면까지 흡수**함을 병합된 page 의 실제 화면을 열어 발견(출하 여부 자체를 바꿈 — 자동 게이트만으로는 안 잡혔을 것).
- 변경: `Monkey-Collector/src/monkey_collector/pipeline/screen_matching/offline_replay.py`(신규 리플레이 하네스) · `pipeline/screen_matching/`(canvas_merge·package_guard config wiring) · `config/run.yaml` · tests(canvas_merge 16 / package_guard 8). 문서 정합: `Monkey-Collector/ARCHITECTURE.md` §7(config 표·MC_* 표에 canvas_merge·canvas_min_area_frac·package_guard 3행 추가) · `Monkey-Collector/README.md`(config 키 목록에 3키 추가) · `Monkey-Collector/AGENTS.md`("알려진 한계" S-9 항목 + package_guard 항목).
- 커밋: `87095ef`(fix(matching): guard cross-app page merges; diagnose S-9 without shipping an unsafe fix) — 14 files, +1446 −10.
- 결과/검증: tier-1(직접 재계산) medians·R0–R4 게이트·prereg mtime·`wc -l` 164/155/9 확인 · `pytest 859 passed`(기준선 835 + canvas 16 + guard 8, **브리프 인용값 — 이 세션 재실행 안 함**).
- 후속: **P1** S-9 재설계 — 카디널리티 복원 경로는 이미 반증됨(다시 시도 금지, 필드 단위 텍스트 제외 등 미검증 대안 존재) · **P2** canvas 탐지기의 calendar scrim 오탐(S9.9, forward-looking) · **P3** 오염 아카이브(osmand armB, broccoli) 재수집 · **P4** `split_data.py --dataset MC` 의 `--stage1-ratio` 기본값이 소규모 코퍼스에 부적합.
- **카테고리**: devlog

## 2026-07-14 — Monkey-Collector: F2(server pull)를 osmand 에서 재현 — 예산 주장만 cross-app 승격, 코퍼스는 판정 유보

직전 F2 마일스톤이 musicplayer 1쌍뿐이라 single-case 였던 것을 **두 번째 앱(osmand)** 에서 이진 질문 하나로 검증하고, 측정 도구 결함(S-3)을 고쳐 실전 검증했으며, `AGENTS.md` 에 평가 방법·불변식 9항을 계약화했다. 상세는 [분석 보고](../Monkey-Collector/.claude/analysis/2026-07-14_15-30-41_s1-f2-osmand/README.md) · [revise 기록](../Monkey-Collector/.claude/devlog/2026-07-14_15-30-41_s1-osmand-and-docs.md) · [사전등록](../Monkey-Collector/.claude/handoff/s1-prereg.md).

- **✅ 승격된 것은 이진 주장 하나뿐**: **"F2 는 signal timeout 을 제거하고 그만큼의 예산을 회수한다"** (musicplayer 37→0, osmand **24→0**; 예산 소진 osmand **288s(31.8%)→0s**). 스코프는 **`cross-app` 이되 4앱 중 2앱이고, IME-heavy 앱(broccoli)은 미검증**이다 — 이 병기 문구는 뺄 수 없다(musicplayer 계측에서 키보드 에피소드가 timeout 의 22/37건·예산 31.8% 였는데 poke 가 그 범주까지 닫는지는 모른다).
- **❌ 효과 크기는 승격되지 않는다**: steps 313→429. **`+N%` 로 반올림 금지** — osmand 노이즈 바닥이 미실측이고 arm 당 **n=1** 이다.
- **❌ 코퍼스는 어느 방향으로도 판정하지 않는다 (판정 유보)**: raw pages 71→67 이지만 **이 지표 자체가 오염돼 있다**(아래). 사전등록된 종결점이 아니므로 이득도 손해도 결론짓지 않는다.
- **🔴 신규 P1 (S-9) — 지도류 연속 캔버스의 page 지문 파편화**: osmand 의 지도 화면이 팬/줌마다 다른 page 로 등록된다 — armA 의 **71 pages 중 46개(65%)가 MapActivity** 인데 **지도 방문 수는 두 arm 이 사실상 같다**(266 vs 265). `ScreenMatcher` 의 BM25+픽셀 게이트가 **이산적 UI 를 전제**하기 때문이다. **raw page count 는 지도류 앱에서 코퍼스 지표로 쓸 수 없고**, IWM Stage-1 이 diff-weighted loss 라 "같은 화면인데 다른 page" 가 **학습 신호를 직접 오염**시킨다. 이걸 고치기 전에는 S-6(arm 당 n≥3)를 돌려도 **같은 오염된 자로 재게 된다**. `Monkey-Collector/AGENTS.md` "알려진 한계" 에 문서화.
- **⚠️ 정직성 — 이번 분석에서 나는 두 번 틀렸고 tier-2 가 둘 다 잡았다**: (1) **사전등록 위반** — 나는 prereg 에 "pages 로 결론을 세우지 마라"고 직접 써놓고, pages 가 71→67 로 불리하게 나오자 "코퍼스 이득이 재현되지 않았다"는 **부정 방향의 결론**을 세웠다(과대주장을 피하려다 반대편으로 과대주장). (2) 그 전에 pages 감소를 설명할 온건한 가설(앱 포화)을 세웠다가 **내 데이터로 죽였다** — page 발견 곡선상 두 arm 모두 런의 마지막까지 새 page 를 찾고 있었다(armA 313/313, armB 425/429). 오염을 걷어낸 post-hoc 분해는 armB 에 유리하게 나오지만 **post-hoc 이므로 그쪽으로도 승격하지 않았다** — 유리해서 기준을 바꾸는 건 불리해서 바꾸는 것과 같은 오류다.
- **도구 수정(S-3)이 실전 검증됨**: `instrument_p1.sh` 에 산출물 아카이브 단계를 신설했고, 실제로 armB 의 `--new-session` 이 `data/net.osmand/` 를 지웠으나 아카이브 덕에 armA 의 `page_graph.json` 이 살아남았다 — **이 수정이 없었으면 이번 짝 비교 자체가 불가능**했다. 함께 `measure.sh` 의 오염된 steps 카운터(`grep -cE 'Step [0-9]+:'`)를 metadata SSoT 읽기로 교체(오염 카운터로 세면 352/517, 진짜는 313/429). 직전 `solutions.md` 의 S-3 전제("measure.sh 는 이미 아카이브를 한다")가 **틀렸음**을 발견해 정정했다 — 그 로직은 존재하지 않았고, iter6 가 온전했던 진짜 이유는 baseline arm 이 git worktree 에서 돌아 CWD-상대 `data_dir` 이 갈렸기 때문이다.
- **arm 설계**: F2 에 문서화된 off 스위치(`poke_delay_sec <= 0` → `collection_loop.py:198`)를 발견해 **worktree 없이 같은 코드 + env 비트 하나**(`MC_COLLECTION_POKE_DELAY_SEC=0`)로 baseline arm 을 구성했다(둘 다 `0a92715`, osmand 900s). manipulation check 통과(armA poke **0** / armB **42**). 측정 전 **stale-APK 가짜 음성 경로 배제**(디바이스 `base.apk` MD5 == 로컬 F2 빌드) — `measure.sh` 는 클라이언트를 재설치하지 않으므로 이 확인이 없었으면 "osmand 에선 F2 가 안 통한다"는 가짜 음성이 날 수 있었다. 세션 길이 906.3s vs 901.5s(armA 가 오히려 김) → timeout 감소는 세션 길이 아티팩트가 아니다.
- 변경: `Monkey-Collector/AGENTS.md`(평가 방법·불변식 9항 신설) · `Monkey-Collector/ARCHITECTURE.md` · `Monkey-Collector/README.md` · `.claude/handoff/instrument_p1.sh`·`measure.sh`·`s1-prereg.md`(gitignored 로컬).
- 커밋: `0ecb4cc`(docs: add an evaluation guide and a named invariant contract to AGENTS) — 3 files, +376 −81.
- 결과/검증: tier-1(오케스트레이터 직접) 전 수치 원천 재계산 + `Session complete` ↔ `metadata.total_steps` 교차검증(313=313, 429=429) + page 발견 곡선 + 지도 activity 분해 + 해시 게이트 코드 확인(`CollectorService.kt:342` — poke 프레임은 XML 이 다를 때만 송신, 기여 상한 4.7%) + APK MD5 대조 + **pytest 835 passed**. C1~C12 전부 CONFIRMED. ⚠️ **tier-2 는 codex 2회 실패**(gpt-5.6-terra 미가용, gpt-5.5 10분 한도 초과)**로 advisor/fable 폴백** — **이상적인 교차 플랫폼 검증이 아니었다.** 그럼에도 tier-2 의 최대 산출은 위 서사 오류 2건 적발이다.
- 후속: **S-9(P1)** page 지문 파편화 수정이 **S-6(P1, arm 당 n≥3)의 선행조건** · **S-10(P2)** `reset_app.sh` 가 이전 실험 타깃을 제거하지 않아 armA 가 musicplayer 로 7회 이탈(교차 실험 오염 벡터) · **S-11(P3)** `ARCHIVE_FAIL` 이 `cp -R` 자체 실패는 못 잡음.
- **카테고리**: devlog

## 2026-07-14 — Implicit-World-Modeling: GPU 정책 3축 분기 — 80GB 에서 (3-4B | lora) 는 DeepSpeed offload 를 끈다 (6커밋)

EXP05 stage1 world modeling full FT(`qwen2.5-vl-3b`, A100×2)를 돌리다 **offload 자체가 병목**임을 실측하고, `gpu_policy.py` 의 "deepspeed 는 GPU 무관 **항상 offload**" 불변식을 `(gpu_type, size_class, mode)` **3축 분기**(`_is_no_offload_combo`)로 교체했다. 커밋 `291b26a`·`202fec3`·`b9ccf6e`·`17113f2`·`1f3a827`·`da8ccd2` (6건), 7 files +263 −86. 상세는 [revise 기록](../.claude/devlog/2026-07-14_02-15-00_gpu-policy-no-offload-3axis.md), 정본 매트릭스는 [§2 GPU 정책 · 함정 7](../Implicit-World-Modeling/ARCHITECTURE.md#2-모델-설정).

- **증상 — GPU 가 놀고 있었다**: offload 를 켠 채로는 **165 s/step**(2094 step → 약 4 일)인데 GPU 메모리는 80GB 중 **23~26 GB** 만 쓰고 두 GPU 전력이 **135 W 대 378 W** 로 벌어졌다. 병목이 계산이 아니라 **CPU↔GPU 전송**이라는 신호 — 한쪽 GPU 가 파라미터·optimizer state 를 기다리며 놀고 있었다.
- **무엇이 과했나**: "항상 offload" 의 근거는 **7B full FT 는 offload 를 빼면 확정 OOM**(모델 상태만 GPU 당 ~77 GiB)이라는 실측이다. 이 명제는 **7-9B 에 대해 여전히 참**이지만, **3-4B·lora 에까지 일반화한 것이 과했다.** 두 갈래를 가르는 것은 정확히 **optimizer state 의 크기**다.
- **연 것 / 열지 않은 것**: 80GB(A100/H100) × **3-4B**(full·lora 무관)와 80GB × **lora**(size_class 무관)는 offload 를 끄고 half-batch 반감도 **면제**(`pdbs=2`/`ga=16`). **`7-9B × full` 과 RTX5090 전부는 offload 유지.** `GLOBAL_BATCH_SIZE=64` 불변식은 전 조합에서 유지된다. **lora 가 풀린다고 full 까지 풀리지 않는다** — 이 경계는 `test_a100_7b_lora_is_no_offload_but_full_is_not` 이 대조 고정한다. 같은 술어가 offload 와 half-batch 면제를 **함께** 판정하는 이유는 두 결정이 같은 메모리 실측에서 나오기 때문이다.
- **결과 (실측)**: EXP05 3B full FT 가 **165 → 138 s/step**(약 16 % 단축), GPU 메모리 **23~26 → 64~73 GB**, 전력 불균형 해소. ETA 약 4 일 → **약 3.3 일**.
- **한계 (정직한 천장)**: 이득이 **16 % 에 그쳤다.** 두 GPU 모두 util 100 % 로 실제 연산 중이므로 **남은 병목은 offload 가 아니라 시퀀스 길이 자체**(`cutoff_len 24576` + visual token 2,048개)와 `gradient_checkpointing` 재계산이다. `cutoff_len` 을 내리면 하드 제약 7, `image_max_pixels` 를 내리면 하드 제약 4 에 걸려 **정책으로는 더 줄일 수 없다.** 또한 `pdbs=2` 의 대가로 메모리 여유가 **~7 GB** 뿐이라 더 긴 시퀀스가 배치에 걸리면 **OOM 가능성이 남아 있다** (미해결, 감시 중).
- 변경: `Implicit-World-Modeling/scripts/gpu_policy.py` · `implicit_world_modeling/gen_configs.py` · `scripts/_common.sh` · `tests/test_gpu_policy.py` · `tests/test_gen_configs.py`. 문서 정합: `Implicit-World-Modeling/ARCHITECTURE.md`(§2 GPU 정책 매트릭스 · 함정 7 · 함정 19) · `Implicit-World-Modeling/AGENTS.md`(하드 제약 5).
- 검증: `pytest tests/ -q` **전체 통과**(실패 0, skip 9) · `gen_configs --check` OK — **커밋 YAML 은 불변**(RTX5090 baseline 이라 정책이 바뀌어도 재생성하지 않는다, 하드 제약 6) · `gpu_policy` 4조합 전부 global batch 64 · **실행 검증**: EXP05 3B full FT 재시작 → step 15/2094 진행(loss 0.1735, OOM 없음, 138 s/step).
- 부수 발견(코드 변경 아님): 이 머신의 `/usr/local/cuda` 는 **13.0** 인데 torch 는 **cu12.8** 이라 `_common.sh` 의 CUDA 가드가 학습 진입을 막는다. 시스템 심링크를 건드리지 않고 conda env 의 cu12.8 toolkit 을 가리키는 shim 을 만들어 `CUDA_HOME` 으로 넘기면 통과한다.
- **카테고리**: devlog

## 2026-07-14 — Monkey-Collector: iter6 통제 ablation — budget-loop fix 판정 + D3 임계값 재보정

리셋 프로토콜 paired-arm ablation(pre-fix `6ff8e95` vs fix `fe12f46`, musicplayer+calendar, 900s × 4 run + 동일 코드 반복측정 1 run)으로 budget-loop fix(D1/D2/D3)가 실제로 무엇을 했는지 판정했다. 결론은 **fix의 다양성 이득이 노이즈와 구별되지 않는다**는 것이고, 그 과정에서 진짜 병목이 따로 드러났다. 상세는 [분석 보고](../Monkey-Collector/.claude/analysis/2026-07-14_04-05-29_iter6-controlled-ablation/README.md) · [revise 기록](../Monkey-Collector/.claude/devlog/2026-07-14_04-05-29_iter6-d3-recalibration.md).

- **판정 — 다양성 이득은 노이즈에 묻힌다**: musicplayer(clean isolator, 900s, 리셋 프로토콜)에서 측정된 fix 효과는 **pages +1 / edges +0 / steps −32**다. 그런데 **동일 코드 두 run의 노이즈 바닥이 pages −3 / edges −10 / steps +17** — 효과가 노이즈보다 작다. iter5의 "다양성 +78%" 서사는 **반박**됐다. 이는 효과 부재의 증명이 아니라 **arm당 n=1로는 분해 불가**하다는 확인이며, 판정하려면 arm당 n≥3이 필요하다.
- **steps −32는 fix 탓이 아니다**: 지배 항은 확률적 reinit 사이클(armA=3 / armB=6 / 동일 코드 u3b=4)이다. "가드가 예산을 구했다"는 주장은 `weak-evidence`에 머문다.
- **D2는 발화한다**(musicplayer 2회, calendar 1회) — 메커니즘은 작동하나 **이득은 미입증**이다.
- **calendar arm 쌍은 CONFOUNDED**(armA seed=25 vs armB seed=83)라 판정에서 제외했다. 수집기가 `com.android.providers.calendar`(별도 priv-app DB)에 이벤트를 만들고 `pm clear`가 거기 못 닿아 오염이 다음 arm으로 샌다. **이 쌍의 델타를 fix 효과로 읽으면 안 된다.**
- **🔴 최대 발견 — 예산의 44~56%가 signal timeout 대기에서 소진**된다(5/5 run, 양 arm, 양 앱; 평균 대기 13.0~13.4s로 `signal_timeout_sec=12`와 일치). 가드는 예산의 **소수 지분**을 다투고 있었다. 다음 우선순위 **P1**.
- **D3 임계값 150 → 98 재보정(이번의 유일한 확정 코드 변경)**: 전 아카이브 23개 `events.jsonl`의 max productive gap = 49 → 결정 규칙 `T = max(49×2, 40) = 98 ≤ 120`(규칙 충돌 없음). 오케스트레이터 tier-1과 codex(gpt-5.6-terra) tier-2의 독립 재계산이 일치했다. 조기 포기 위험 없음 — 98을 넘는 tail 실측 두 케이스(iter1 calendar 374, iter4 calendar 160) 모두 그 구간 신규 page 0건이라 98은 진짜 포화 구간만 회수한다. D3 **발화 로직은 무변경**(상수만).
- **측정 프로토콜 함정 2건을 `Monkey-Collector/AGENTS.md`에 문서화**: (1) provider-backed 앱(calendar)은 seed가 `pm clear`에 생존해 수집기가 만든 오염도 다음 arm으로 새는데 `reset_app.sh`는 run 내부 `before==after`만 검사해 못 잡는다. (2) 동일 코드 반복측정의 노이즈가 측정하려는 효과보다 커서 arm당 n=1로는 판정할 수 없다.
- 변경: `Monkey-Collector/src/monkey_collector/config.py` · `pipeline/collector.py` · `config/run.yaml` · `tests/unit/test_config.py`(D3 상수 동기화 지점 7곳) + `Monkey-Collector/AGENTS.md` · `Monkey-Collector/.claude/handoff/HANDOFF.md`. 문서 정합: `Monkey-Collector/ARCHITECTURE.md` §7 config 표(D3 행 신설, `data_dir`/`runtime_dir` CWD-상대 명시).
- 커밋: `abddb00`(D3 임계값 150→98) · `a95049a`(측정 프로토콜 + 함정 2건) · `b503ad3`(iter6 판정 기록) — origin/main push 완료.
- 결과/검증: `pytest` **824 passed**(기준선 불변) · `git diff -- src/monkey_collector/pipeline/collection_loop.py` 빈 출력(발화 로직 무변경) · 잔여 `150` grep 0건 · 회귀 체크 U3b probe 908s 예산 소진 정상 종료(`no_progress_stop` 0건, 이 run max gap 45 < 98).
- 후속: **P1** signal timeout 원인 계측(client logcat + server 신호 타임스탬프 — 예산의 절반이라 가드 튜닝보다 우선) · **P2-a** 효과 측정은 arm당 n≥3 · **P2-b** `reset_app.sh`가 provider-backed 앱을 canonical seed로 복원하고 검사를 `after==canonical`로 변경 · **P3-b** broccoli harness-evict(사용자 확인 게이트). 2h regime의 gap 분포는 미관측이라 그 구간의 조기 포기 위험은 아직 배제되지 않았다.

## 2026-07-13 — Implicit-World-Modeling: 데드 코드 감사 — 도달 불가 심볼 제거 + 노트북 stale 채점기 복제 정리 (4커밋)

`grep` 으로 **호출자 0 을 확정한** 심볼만 제거했다. 삭제 616줄 중 468줄이 노트북의 "정본과 글자 단위 동치" 라 **자칭**하던 채점기 복제본 2개인데, 실제로는 `pos`/`bounds`·xy 채점 경로가 빠진 **구버전**이었다. 브랜치 `chore/dead-code-audit`, 커밋 `db074af`·`a316c9d`·`f2b1a42`·`b7c5be6` (4건), 13 files +51 −616. **동작 변경 0.**

- **셸·Python 도달 불가 심볼 제거 (`db074af`)**: `scripts/_common.sh` 6건(`is_ac_exp01_ratio()`, `EXP01_RATIO_FILE` 맵, write-only `EXP01_RATIO=`, 레거시 `NPROC_PER_NODE_OVERRIDE`, `DS_PREFIX[AC_EXP01_ratio*]` 3키, `HF_SLUG[AC_EXP01]`) + 중복 함수 병합(`ds_eval_suffix`→`ds_model_suffix`, 본문 md5 동일). 이어 `lf_registry`(`QWEN2_VL_CONFIG`/`MODEL_ORDER`/`DS_ORDER`) · `gen_configs`(`STAGE2_VARIANTS` + **항상 빈 문자열이던** optim/seed/save_steps 보간 + `full_lr` fallback) · `_action_eval`(`_FIELD_MATCH_TYPES`) · `setup_llamafactory` dead store 를 제거했다.
- **삭제 근거는 전부 도달 불가 증명**: `parse_args` 는 bare `AC_EXP01` 을 ratio 로 expand 하고 `parse_eval_args` 는 raw 키만 허용한다 → `DS_PREFIX` 의 ratio 키와 `HF_SLUG` 의 bare 키는 **영원히 인덱싱되지 않는다.** "언젠가 쓸지도" 가 아니라 코드 경로상 닿을 수 없음을 확인한 것만 뺐다.
- **노트북 stale 복제 정리 (`a316c9d`, 76→73셀)**: 셀 39 는 셀 40 이 호출하므로 삭제 대신 **정본 `scripts/_hungarian_eval.py` 재수출 shim 23줄**로 치환했다. 이 과정에서 **반환 arity 불일치 버그**(정본은 dict 반환, 구 복제본은 2-tuple → 셀 40 의 `metrics, _ =` 언패킹)도 함께 고쳤다 — 복제본이 정본과 동치가 아니었다는 증거다.
- **문서 수치 drift 교정 (`f2b1a42`)**: 학습 YAML **162 → 160**(EXP05 Qwen3-VL YAML 2개가 `fd4fd77` 에서 삭제됨), EXP05 stage1 YAML **6 → 4**(자격 매트릭스상 Qwen2.5-VL 2모델 × full/lora). 함께 **보존 근거 주석**을 달아 다음 감사에서 다시 데드로 오인하지 않게 했다. `configs/lf_dataset/AndroidControl_EXP04` 심링크 추적 누락도 함께 수정(`b7c5be6`).
- **보존 결정 (데드가 아님)**: `diff_loss/` **v1 4파일은 데드가 아니다** — EXP02 bit-exact 재현의 **유일 경로**다(v1 40/40 vs v2 17/40). `DEEPSPEED_NO_OFFLOAD` 는 프로덕션 호출자 0건이지만 `tests/test_gpu_policy.py:205` 가 값을 고정하는 **테스트 전용 opt-out** 이고, `qwen3_5*` 분기·`QWEN3_5_VL_CONFIG`·`remote_launch.sh` 는 **사용자가 향후 계획으로 보존 결정**했다.
- **검증**: `pytest tests/` → **548 passed, 9 skipped**(베이스라인 동일). `gen_configs --check` → **160 YAML 일치**(완전 동등 비교 + orphan 검출 → byte-identical 증명). `bash -n scripts/*.sh` → exit 0. `nbformat.validate` → **73셀, called-but-undefined 0건**, 셀 39 shim 실행 시 `f.__module__ == '_hungarian_eval'`. tier-2 독립 검증(advisor/fable) **전 주장 CONFIRMED** (codex 는 bwrap 샌드박스 오류로 폴백).
- **카테고리**: devlog

## 2026-07-13 — Implicit-World-Modeling: LlamaFactory 부트스트랩/설정 재구성 — LF 를 git 에서 재구성 가능하게 (12커밋)

"git clone 후 노트북으로 LlamaFactory 를 clone 하고 그 내부 파일·폴더를 수정해야 하는데 매우 복잡하다"는 문제 제기에서 출발했다. 조사해 보니 복잡함은 증상이고 실체는 **복구 불가능성**이었다 — LF 는 pin 없는·gitignore 된·직접 변조되는 서드파티 체크아웃이라, 그 디렉토리를 지우면 anchor 치환으로 패치한 소스 6파일·학습 YAML 74개·in-place 변조된 `dataset_info.json`·런타임 심링크가 **영구 소실**됐다. 게다가 clone 에 커밋 pin 이 없어 upstream HEAD 가 움직이면 anchor 치환이 깨질 수 있었다. 브랜치 `refactor/lf-bootstrap`, 커밋 `17f49a3`..`3917446` (12건), 193 files +11566 −1411.

- **LF 를 git 에서 재구성 가능하게 (`17f49a3`, `973beb1`)**: LF 워킹트리에만 살던 유일본을 전부 repo 로 흡수했다. 이제 **pin `99464b3d` + `patches/llamafactory/{0001-diff-loss,0002-double-ce-fix}.patch`** 를 pristine clone 에 적용한 결과가 **살아있는 LF 와 byte-exact 일치**한다(src 6파일 `diff` 무출력). 부트스트랩은 `bash scripts/setup_llamafactory.sh --install --verify` **한 커맨드**(멱등). 패치 정본이 anchor 문자열 치환 파이썬 스크립트에서 **리뷰 가능한 `.patch` 커밋**으로 바뀌었고, `scripts/diff_loss/apply_llamafactory_patch.py` 는 **은퇴·삭제**했다.
- **설정 소유권 이전 (`f0fbce9`, `bb53bee`)**: 학습 YAML 정본 = `configs/train/` **162개**(생성기 `python -m implicit_world_modeling.gen_configs`), dataset_dir 정본 = `configs/lf_dataset/`(`dataset_info.json` + 상대 심링크 7). 런타임에 LF 내부 상태를 **읽지도 쓰지도 않는다** — `_common.sh` 가 LF 의 `dataset_info.json` 을 in-place 변조하고 LF 안에 심링크를 만들던 경로를 끊었다.
- **GPU 정책 SSoT (`9130b4c`)**: `.env`/노트북 Cell 5/Cell 10/`_common.sh` 4곳에 분산돼 있던 GPU 정책을 `scripts/gpu_policy.py` 단일 정본으로 통합. **커밋 YAML 은 GPU-불변 baseline** 이고 GPU 트리오(pdbs/grad_accum/deepspeed)는 `llamafactory-cli train cfg.yaml key=value` **런타임 override** 로 주입한다(LF `hparams/parser.py` OmegaConf merge) → **하드웨어를 바꿔도 YAML 재생성이 필요 없다.** RTX5090 {1,2} · A100·H100 {1,2,4,8} 전 조합에서 `GLOBAL_BATCH=64` 유지를 실측 확인.
- **deepspeed always-offload 고정 (`9130b4c`)**: as-trained **74/74 YAML 이 전부 `ds_z3_offload_config.json`** 이었고, `ds_z3_config.json`(no-offload)은 **한 번도 실행된 적 없는 죽은 기본값**이었다(생성·학습이 전부 RTX5090 경로였기 때문). 따라서 "A100 이니 offload 를 빼도 된다"는 조건부 분기는 미실행 경로로의 조용한 divergence 였고, 실제로 A100 에서 offload 를 빼면 EXP05 7B full FT 는 모델상태만 GPU당 ~77 GiB 로 **확정 OOM** 이다. GPU 무관 **항상 offload** 로 고정해 이 트랩을 제거했다.
- **diff-loss 이중 CE 수정 (`dbab68d`)**: `use_diff_token_weighted_loss` 경로가 `labels` 를 pop 하지 않아 HF 내부 CE 가 **이중 실행되고 그 결과가 버려지던** 버그 + logits 전량 fp32 upcast 를 고쳤다(chunked CE 도입). **activation peak 20.87 → 10.72 GiB (48.6% 절감)**, 결과는 **bit-exact** — chunk 128/333/4096 × `num_items` 유무 전 조합에서 `|Δloss| = 0.000e+00`, `max|Δgrad| = 0.000e+00`(tolerance 없이 `torch.equal`).
- **EXP03/04 YAML 은 애초에 디스크에 없었다 (`f0fbce9`)**: 전역 `find` **0건**. 노트북의 `_YAML_GEN_DS` allowlist 와 "hand-fix 라 복구 불가, 재생성 금지" 주석은 **존재하지 않는 파일을 지키고 있었다.** 레지스트리 값으로 재구성하고 `# [reconstructed 2026-07-13]` 헤더를 달아 명시했다.
- **노트북 thin wrapper 화 (`2ec9cf3`)**: 정본 로직을 전부 코드로 이관하고 노트북(Cell 3/5/7/10/12/14)은 호출자만 남겼다.
- **원격 제출 스펙 (`e8b1a53`)**: `configs/remote/run.template.yaml` + `scripts/remote_launch.sh`. **제공자 중립** — 코드에 벤더명이 0건이고 제출 커맨드는 `.env` 의 `REMOTE_SUBMIT_CMD` 템플릿으로 주입한다. **UNVALIDATED**(아래 한계 참조).
- **검증 중 잡은 결함 5건 (전부 수정됨)**: (1) `--verify` 가 **스택 패치를 오판**(`b30ba36`) — 각 패치를 독립적으로 `git apply --reverse --check` 했는데 0002 가 0001 이 추가한 라인을 고치므로 0001 단독 역적용은 원리적으로 불가능해 항상 FAIL 이었다(같은 스크립트의 상태 게이트는 "2/2 적용됨"이라 자기모순). 기지 상태 byte-exact 대조로 교체. (2) **`dataset_dir` 미전달**(`bb53bee`) — 커밋 YAML 의 `dataset: IWM-AC_*` 키는 `configs/lf_dataset` 에만 있어 안 넘기면 **fresh clone 에서 학습이 시작조차 못 한다.** (3) **`.env` 가 프로세스 환경을 덮어씀**(`bb53bee`) — `GPU_TYPE=A100 NPROC_PER_NODE=4 bash stage1_train.sh` 가 `.env` 값에 먹혀 무시됐다(= GPU 매트릭스를 쓸 수 없었다는 뜻). (4) **`save_strategy=no` 가 boolean 파싱**(`23daeb3`) — OmegaConf 는 YAML 1.1 규칙이라 따옴표 없는 `no` 를 `False` 로 읽는다. (5) **EXP03 stage2 dataset 미등록**(`3917446`) — 데이터(128MB)와 YAML 12개가 다 있는데 정본에 엔트리가 없어 LF 안쪽에서 죽던 상태. 등록 + `require_dataset_registered` 가드 추가(이제 학습 전에 잡힌다).
- **검증**: `setup_llamafactory.sh --verify` → **exit 0**(pin+0001+0002 == 살아있는 LF, byte-exact). `gen_configs --check` → **exit 0**(162 YAML; as-trained 74개는 `17f49a3` 대비 **수정 라인 0** = byte-exact 재현). `pytest -p no:warnings -q` → **547 passed, 9 skipped, 실패 0**. GPU 매트릭스 **10/10** 조합에서 global batch 64 + always-offload(허용 안 되는 RTX5090×4 는 학습 진입 전 중단). **1-step 실학습 스모크**(EXP02 3B LoRA, RTX5090 1장) → exit 0, `loss=0.3309`, 22초, OOM 없음 — `dataset_dir` override 가 `configs/lf_dataset` 로 해석되고 `token_weights` 가 배치에 비균등(`[1.0, 2.0]`)으로 실림 → **diff-loss 경로 end-to-end 확인**.
- **정직한 한계**: (1) **원격 제출 스펙은 UNVALIDATED** — 이 머신에 제출 CLI 가 없어 스키마·제출 경로를 실행 검증하지 못했다(검증된 것은 YAML 파싱·placeholder 전수 나열·더미값 렌더뿐). (2) **A100/H100 의 CPUAdam JIT 경로는 실기 검증 불가** — always-offload 라 그쪽도 JIT 를 타는데 로컬에 해당 하드웨어가 없다. `_common.sh` 의 CUDA_HOME 가드 fail-fast 가 유일한 방어. (3) **EXP03/04 재구성본은 as-trained 와 다를 수 있다**(원본 소실). (4) **EXP04 는 아직 돌릴 수 없다** — 데이터 자체가 없다. YAML 은 있지만 `require_dataset_registered` 가 학습 전에 명확히 중단시킨다. `docs/ROADMAP.md` 의 EXP04 "학습 진행" 항목을 "미착수" 로 교정했다.
- **카테고리**: devlog

## 2026-07-13 — Implicit-World-Modeling: qwen3-vl-4b 레지스트리 복원 (EXP01–04 자격) + 노트북·문서 서술 코드 정합화

`qwen3-vl-8b` 가 가능한 실험군(EXP01–EXP04)에 `qwen3-vl-4b` 를 추가해 달라는 요청을 처리했다. 조사해 보니 4b 는 신규 모델이 아니라 커밋 `67a52e5`(2026-05-31, "qwen3-vl-4b 삭제·qwen2.5-vl-7b 등록, 7-9B 단일 tier 통합")가 **의도적으로 삭제**한 모델이어서, 신규 설계가 아니라 **삭제 전 설정의 복원**으로 처리했다. 이어 "노트북도 현재 코드 기준으로" 요청이 들어와 노트북 서술의 stale 사실을 전수 정합화했다.

- **레지스트리 복원 (4 모델 2 tier)**: `qwen3-vl-4b`(= `Qwen/Qwen3-VL-4B-Instruct`, template `qwen3_vl_nothink`, size `3-4B`)를 `scripts/_common.sh` 3구조(`MODEL_ID`/`MODEL_TEMPLATE`/`ALL_MODELS`) + 노트북 Cell 5 3구조(`MODEL_FAMILY_CONFIG`/`_MODEL_CONFIG`/`MODEL_ORDER`)에 재등록. `_MODEL_CONFIG` 엔트리는 삭제 전 원문과 **byte-identical**. 이제 **7-9B: `qwen3-vl-8b`·`qwen2.5-vl-7b` / 3-4B: `qwen3-vl-4b`·`qwen2.5-vl-3b`** 체제다.
- **hparam delta 없음**: `_SIZE_CONFIG_AC` 의 tier dict 를 빈 dict 로 유지했다 (`hparam_overrides={}`) — EXP01/EXP02 실측 어댑터와 동일조건 보존 정책 유지. lr/rank 를 새로 도입하지 않았다.
- **모델 자격 vs 학습 이력 분리**: 문서의 "전용성" 문구만 **Qwen3-VL 계열(4b/8b)** 로 확장하고, 실행 이력·산출물 서술은 사실 그대로 뒀다. **`qwen3-vl-4b` 는 아직 한 번도 학습된 적이 없다** — 자격만 복원된 상태이며 노트북 walkthrough 의 `--model` 은 전부 `qwen3-vl-8b` 그대로다.
- **EXP05 는 배제**: 절대 픽셀 좌표(840×1876)라 Qwen3-VL 계열 전체가 좌표·factor **이중 mismatch**다. 기존 8b 배제와 대칭으로 **코드 가드는 신설하지 않고 문서 규약으로만** 배제했다 — 즉 `--model all --dataset AC_EXP05` 는 CLI 상 통과하며(이제 8b + 4b 2개가 통과), `IWM-AC_EXP05/stage1_full/` 에 `qwen3-vl-8b_world-model.yaml` 이 실재하는 것이 그 증거다.
- **노트북 정합화 (12 셀)**: "8 모델"/"12개 모델" → 등록 4 모델, "96 YAML"(8×6×2) → **64 YAML**(4 모델 × `_DATASET_CONFIG` 8 키 × 2 mode, allowlist 해제 시), tier "3 단(2B/3-4B/7-9B)" → **2 단**(7-9B/3-4B, 전부 빈 dict = 미적용), 학습 DS 열거에 **EXP04·EXP05 추가**(Cell 1 표에 2 행 신설), YAML 경로 `custom/Implicit-World-Modeling-{DS}` → 실제 `custom/IWM-{DS}`.
- **Cell 5 자기모순 주석 정정**: "AC_EXP01 은 Stage 2 미정의 — `_STAGE1_ONLY` guard 로 skip" 이 **코드에 반증**됐다(`stage2` 정의 실재 + `_STAGE1_ONLY` = {MC, EXP04, EXP05} 에 EXP01 없음 + `IWM-AC_EXP01_ratio73/stage2_*` 실존 + README 가 sweep 문서화). 함께 `_SIZE_CONFIG_AC` 적용 범위(EXP01/02 → **EXP01–EXP05**, MC 미적용), cutoff_len 24576 목록에 **EXP05 추가**, "등록 모델은 모두 7-9B tier" 를 정정. **주석만 변경 — 실행 코드 불변.**
- **검증**: `bash -n` exit 0 · 레지스트리 ast 정합성(`_common.sh` ↔ 노트북 일치) · 노트북 JSON 유효 + Cell 5 `ast.parse` · stale 패턴 게이트 0 hit · `!bash` 20 줄 HEAD 와 byte 동일 · **pytest 96 passed / 9 skipped**(변경 전후 동일). tier-2 는 codex 가 샌드박스 초기화 실패(`bwrap` loopback)로 2회 미가용이라 advisor(mode=verify) 폴백 판정 — 전건 CONFIRMED / PASS.
- **잔여 이슈(범위 밖)**: `scripts/tmux_exp04_stage1.sh` 가 문서에서 참조되나 레포에 부재(학습 머신 untracked 추정) · ARCHITECTURE 섹션맵의 셀 범위 숫자가 실제 노트북(76 셀)과 불일치(재부여는 교차 참조 파손 위험이라 보류).
- **카테고리**: devlog

## 2026-07-13 — Implicit-World-Modeling: EXP05 stage1 0711 수정본 적용 + diff-loss v2 버그 수정 (GAP1 호출자 신설)

조병웅님의 stage1 수정본(Drive `0711_버젼`)이 모두 도착해(action 파일이 2026-07-12 23:24 마지막 업로드) 다운로드·전수 검증 후 EXP05 데이터를 재생성했다. 재생성에 쓰는 v2 diff-loss 체인의 CONFIRMED 버그를 **먼저** 고치고 재생성은 1회만 수행했다. `data/AndroidControl/_backup_0710/` 에 0710 원천 + 구 EXP05 산출물을 보존한다.

- **0711 검증(전수 + 독립 에이전트 6인 재검증, 5 CONFIRMED / 1 PARTIAL)**: 변경의 정체는 **`wait` 액션 클래스 교체 하나**다. 나머지 7개 액션은 건수까지 완전 동일(click 49,282 / swipe 10,606 / type 5,787 / open 5,208 / navigate_back 2,879 / long_press 160 / navigate_home 28)이고, 기존 `wait` 4,958건을 **전량 퍼지**한 뒤 **전혀 다른 (episode,step) 위치에 `wait` 400건을 action 쪽에만 재생성**했다 (`78,908 − 4,958 + 400 = 74,350` 산술 일치). 공통 73,950건의 변화는 **프롬프트·JSON 포맷 정리뿐**이며 GT 의미는 불변 — state 의 예측 대상 XML 은 전건 동일, action GT 는 compact→spaced 재포맷, action human 은 `[Screenshot]`→`[Current/Next UI Screenshot]`, state system prompt 는 재작성(1,269→1,575자).
- **⚠️ 확인 필요 3건(조병웅님)**: (1) `wait` 전량 퍼지 — 퍼지분 4,958 중 **EXP01 train 멤버십 2,548 · test 멤버십 598** 포함 → 테스트셋 오염 제거가 아니라 **커버리지 축소**(train −6.1%). (2) 신규 `wait` 400건 중 **399건이 빈 current state**(`<node bounds="[0,0][0,0]" point="[0,0]"/>`; 0710 엔 0건) → `wait` 이 "빈 화면이면 wait" **degenerate shortcut** 이 됐고 **74건이 train 에 실제 유입**. (3) **action/state 키 대칭 붕괴** — 0710 은 두 pool 키집합 동일(78,908)이었으나 0711 은 action(74,350) ⊋ state(73,950), 차이가 정확히 그 400건.
- **재생성 실측**: source pool action 74,350 / state 73,950 → mirror **train 44,670**(입력 50,000 / drop 5,330) + test 6종 = **총 60,717행**(0710: 47,556 / 64,787). 가중 train = **state 31,221 + action 13,449**, weight 분포 state `{0.25, 1.0}` / action `{1.0}`, state 출력 토큰의 **53.4%** 가 0.25배 감쇠. **fallback 0건**(fail-closed 기본으로 돌렸는데 diff/weight 실패가 하나도 없었다).
- **GAP1 해소 — `scripts/build_exp05_data.py` 신설**: v2 diff-loss 체인에 **커밋된 호출자가 없어** EXP05 train 이 out-of-band 산출물이었고 fresh clone 이 재현 불가했다(분석 `2026-07-12_153250` 의 CRITICAL). 기록이 없던 0710 생성 명령을 **재현 게이트**로 복원 — committed 0710 train 에서 `token_weights`/`_diff_counts` 를 벗겨 입력으로 되돌린 뒤 재구성 명령으로 재실행해 **400/400 완전 일치**(model `Qwen/Qwen2.5-VL-3B-Instruct`, template auto→qwen, w 1.0/1.0/0.25, metric v2). 이 명령을 스크립트로 고정하고 mirror→가중치→원자 교체를 한 번에 수행하게 했다. tokenizer/revision(`66285546…`)·가중 상수·집계를 `<train>.meta.json` sidecar 에 기록한다.
- **diff-loss v2 버그 수정(v1 은 불가침 — EXP02 재현성)**: (S2-05) `token_weight_builder_v2` 가 **토큰 시작점만** 검사해 element 왼쪽 경계를 걸친 토큰을 놓치고 오른쪽 넘침엔 주던 **비대칭 경계** → **interval overlap**(`tok_cs < char_end and tok_ce > char_start`, zero-length offset 제외)으로 교정하고 중첩 span 은 **max 가중치** 채택(순서 비의존). 실측 영향: state 토큰의 0.65%가 바뀌었고 **전부 0.25→1.0 상향**(하향 0건) — 버그 서명과 정확히 일치. (S2-03) fail-open → `--on-error {fail,uniform,skip}` 도입, 기본 **fail-closed**, fallback 을 성공으로 집계하지 않음. (S2-09) 최종 경로 직접 스트리밍 → **sibling temp + `os.replace` 원자 교체** + `--input == --output` 거부. (S2-08) `--revision` 으로 tokenizer commit 고정 + sidecar 기록.
- **좌표 범위이탈(신규 발견, 0711 무관 — 기존 원천 버그)**: 0710·0711 **양쪽 동일하게** `coordinate`(액션 라벨) 필드에서 840×1876 을 벗어나는 키가 **11개**다(`bounds`/`point` 는 깨끗 — 그래서 기존 문서의 "x_max 840" 수치가 나온 것). 값이 `[1682, 975]` 로 반복되고 **1682 ≈ 840×2** 라 스케일링 버그로 보인다. **현재 EXP05 산출물에 10행이 실려 있다**(train 7 + `test_ood_{action,state,state_without_open_app}` 각 1) — `(12571,0) = [421,1979]` 가 **OOD 평가셋 3파일 전부를 오염**시킨다. 조병웅님 확인 필요.
- **검증**: 신규 `tests/test_diff_loss_v2.py` 10케이스(왼쪽 경계 걸침·비겹침 baseline·zero-length offset·UNCHANGED 유지·중첩 max·input==output 거부·실패 시 부분 산출물 미잔류·action uniform·fallback 미집계·skip 모드) → 전체 **96 passed / 9 skipped**, ruff clean. 최종 train 계약 독립 재확인: 44,670행 = state 31,221 + action 13,449, action weight 정확히 `{1.0}`, state `{0.25, 1.0}`.
- **문서**: `README`/`ARCHITECTURE`/`AGENTS` 의 EXP05 행수·원천(0710→0711)·빌드 정본 경로 갱신, `docs/EXP05_DIFF_LOSS_PLAN.md` §5 데이터 상태 전면 교체 + §6 미결에 aria-label(약 88% 영향)·`without_open_app` 무동작·좌표 이탈 추가.
- **카테고리**: devlog

## 2026-07-12 — Implicit-World-Modeling: gpt-5.6-sol 전면 코드 리뷰 + 형식 통일 정리 (ruff/pytest 정본화 · mirror 3형제 통합 · 문서 stale 교정)

`Implicit-World-Modeling/` 서브프로젝트(~9.2k LOC) 전면 코드 리뷰 후 동작 보존 위주로 정리했다. 리뷰는 `/workflow:adaptive-router` 로 gpt-5.6-sol(effort high) 8-슬라이스 read-only 팬아웃 → 각 correctness/reproducibility 발견을 별도 모델이 refute-by-default 로 교차검증(86개 발견 → 72개 통과, critical 1/high 18/medium 24/low 29). 정리 범위는 사용자 확정(포매팅·린트·문서 + mirror 통합만; diff_loss v1/v2 병합·shell dedup·os.path→pathlib 은 제외).

- **린트/포매팅 정본화(S-1/S-2)**: `pyproject.toml` 에 `[tool.ruff]`(line-length 88, target py311, `LlamaFactory`·`*.ipynb` 제외) + `[tool.ruff.lint]`(select `E4,E7,E9,F,I,UP,B` — **E501 은 formatter 와 충돌해 비활성**) + `[tool.ruff.format]` + `[tool.pytest.ini_options]` + `dev` extras(`ruff`, `pytest`) 추가. 저장소에 lint/test 설정이 **아예 없었고** pytest 가 venv 에 미설치라 856줄 테스트가 실행조차 안 되던 상태를 정본화. `ruff format`(18 파일 reflow) + 안전 autofix 20건(I001 12·F401 5·UP015 3) 적용. `--unsafe-fixes` 미사용. 잔여 lint 10건(B905·B007·E731·F841·E402)은 동작 접촉 가능이라 이연.
- **mirror 3형제 통합(S-3)**: `scripts/mirror_exp03.py`+`mirror_exp04.py`+`mirror_exp05.py` → 단일 `scripts/mirror_experiment.py`(`--experiment {exp03,exp04,exp05}` required, `@dataclass(frozen=True) VariantConfig`, `kind: Literal["stage1","stage2"]` 기반 라우팅, STAGE1_JOBS 7 공통 + STAGE2_JOBS 3 은 exp03 만). 신규 `tests/test_mirror_experiment.py`(config 동등성 + 합성 로직 + 소형 합성 트리 e2e byte-identity 3층). **원본은 동등성 증명 후 삭제**.
- **mirror 통합 검증(실데이터 byte-identity)**: 오케스트레이터가 직접 재실행 — (1) 원본 복원 상태 full suite **95 passed**(신규 13 포함, Layer 3 byte-identity). (2) **실데이터**: 통합 `--experiment exp05` vs 복원 원본 `mirror_exp05` 출력 7파일 **바이트 동일**, 그리고 committed `data/AndroidControl_EXP05/` 는 mirror 출력 + diff-loss `token_weights`/`_diff_counts` 계층이라 그 필드 제거 시 train 47,556행 전부 일치(mismatch 0). (3) **실데이터 stage2 분기**: 통합 `--experiment exp03` 출력 10파일(stage2 3파일 포함)이 committed `data/AndroidControl_EXP03/` 와 **바이트 동일**(EXP03 은 diff-loss 없어 직접 대조). 삭제된 스크립트의 실제 코드 호출자 없음 확인(`git grep`, 노트북·`_common.sh` 참조는 전부 주석/문서/힌트 문자열).
- **문서·주석 stale 교정(S-4)**: 존재하지 않는 `setup.py` 참조 4곳(AGENTS.md:112/117/151, ARCHITECTURE.md:577) → `pyproject.toml` 로 교정. transformers pin `>=4.56.0,<4.57`(stale) → `>=4.57.1,<4.58` 로 5곳 통일, ARCHITECTURE.md 자기모순(19행 vs 577행) 해소. 트리오(README/ARCHITECTURE/AGENTS)의 `mirror_exp0X.py` 참조 ~30곳 → `mirror_experiment.py --experiment exp0X`. `_common.sh` 주석 4곳·`stage2_eval.sh` 헤더(AC_EXP03 지원 누락) 교정. append-only `DEVLOG`/frozen `EXP05_DIFF_LOSS_PLAN`/노트북은 별건으로 미변경.
- **동기화 경로 교정**: `.project-sync.json` 의 `repo_root`·`memory.encoded` 가 다른 호스트(`/data/seungwoo/...`) 기준이라 memory 동기화가 무력화되던 것을 현재 호스트(`/home/seungwoo.baek/projects/...`)로 교정 — 실제 memory 디렉터리로 해석됨을 확인.
- **검증(tier-1 직접)**: 최종 full suite **86 passed, 9 skipped**(skip=삭제된 원본 의존 Layer 2/3, 설계상), `ruff format --check` clean(21 파일), `ruff check` 잔여=이연 10건만, 전 tracked `.py` `py_compile` 통과, `bash -n` 통과. patch-anchor(`apply_llamafactory_patch.py`) 47개 문자열 상수 값이 format 전후 **바이트 동일** 확인(monkeypatch 무손상). tier-2 는 codex(openai) quota 한도로 advisor(fable) 폴백.
- **범위 밖(보고용) 버그**: 리뷰가 찾은 correctness/reproducibility 37건 + cross-file GAP 3건은 로컬 `.claude/analysis/2026-07-12_153250/report.md` 에 기록(이번에 미수정). 핵심: **v2 diff-loss 체인 orphan**(커밋된 호출자 없어 EXP05 diff-loss 데이터 fresh-clone 재현 불가, CRITICAL), 노트북 eval-clone 셀이 xy scoring 과 drift, `_common.sh` 모델↔좌표 규약 미검증 등.
- **미완**: Sol completeness critic 은 codex quota 로 미실행(opus critic 이 완결성 패스 대행) → 한도 회복 후 재시도 예정. 노트북 mirror 참조 7곳·GAP1/2 는 노트북 정합성 별건.
- **워크플로우**: `/workflow:adaptive-router`(advisor plan → gpt-5.6-sol 리뷰 팬아웃 + claude/opus worker 구현 → tier-2 advisor 검증). 분석 산출물은 `.claude/analysis/` (gitignored, 로컬 전용).
- 카테고리: devlog

## 2026-07-12 — Monkey-Collector: iteration 3 검증 + 관련연구 문서화 + iteration 4(R1 value-guided 탐색) — coverage 정체는 reachability ceiling으로 확정

전날 iteration 3(W-A/W1/W2)을 AVD Pixel6-2로 재수집 검증하고, 사용자 요청으로 관련 연구를 조사·문서화한 뒤 최우선 권고(R1)를 iteration 4로 구현했다. 핵심 결론: **R1은 올바르게 구현·검증됐으나 4앱 activity coverage 정체는 exploration-order 문제가 아니라 reachability ceiling(계정·데이터·딥네비·권한 게이트 전제)이라 R1으로 count가 오르지 않는다.** adaptive-router 라우팅: advisor `claude/fable`(지정 `gpt-5.6-sol` hard-unavailable), worker `claude/opus`, tier-2 `codex/gpt-5.6-terra`(플레이키 — 후반 `gpt-5.5` 폴백), 재수집은 worker-held가 두 번 실패해 orchestrator-background로 대체.

- **iteration 3 재수집 검증(`.claude/analysis/2026-07-11_iter3-revalidation/`)**: (W-A `1020d0d`) coverage 분모를 `ActivityCatalog` 조회 계층에서 framework/3p non-navigable(androidx.car.app/billingclient/gms/play.core) 제외 → osmand 16→**11**, calendar 39→**37**(라이브 확인), 방문∩denylist=∅. (W1 `382c377`) `AdbClient.is_keyboard_shown()`(`dumpsys input_method` mInputShown) + ESC dismiss 후 결정론 마커 — musicplayer ESC 유효율 **10/10**(iter2 추론 61% 대체). (W2 `0358be1`) `_fallback` root 랜덤탭에서 SET_TEXT 요소 강등 → keyboard 프레임 **53→39→17.4%**, nexuslauncher drift 2.4→2.3→**1.84/min**, coverage 무퇴행. broccoli stuck-rate: like-for-like로 iter3≈iter1 → **not reproduced**(naive 전체세션 비교는 non-stationarity artifact). cross-app 회귀 없음. pre-existing ruff 4건 정리(`c3eb3cb`).
- **관련연구 문서화(`docs/research/gui-exploration-world-model.md`, `97ce5dd`)**: 4스레드 26편(탐색전략·GUI/디지털 world model·GUI 에이전트·데이터품질) + gap 분석 + R1-R6 권고. LLM-Explorer 원전 특정([arXiv:2505.10593](https://arxiv.org/abs/2505.10593), MobiCom 2025). 최대 gap = `_pick_unexplored`가 `rng.choice`(knowledge-guided 미채택). 2026 preprint는 도구소싱 caveat 명시.
- **iteration 4 — R1(`6ff8e95`)**: `_pick_unexplored`의 uniform `rng.choice`를 결정론 사전식 랭킹 `(novelty, type_prior, uniqueness)`로 교체(동점만 seed 고정 rng). novelty=cross-page 미탐색(인덱스폴백 `@<index>` 예외, `@`+숫자로 정밀 판별해 `@home` 오탐 방지), type_prior TOUCH>SCROLL>SELECT>SET_TEXT, uniqueness=same-function group 미소속. `Memory` 읽기전용 접근자 2개. 후보 집합·select_action 단계·`_fallback`·Navigator 불변. `pytest` **813 passed**, ruff clean, codex tier-2 6 CONFIRMED + `@`-오탐 1건 수정·guard 테스트.
- **R1 재수집 실증(정직한 negative)**: musicplayer(full, W-A 필터 fresh) coverage **5/15 = iter3와 동일**(방문 activity 집합은 달라짐, pages 16 vs 14로 다양성만 소폭↑). calendar(full 223 steps) **2/37 = iter3 2/39와 동일 count**. headroom 최대 앱에서도 count 무증가 → **정체는 reachability ceiling**(calendar 37 중 35는 계정 실상태·이벤트/날짜 네비·딥링크·권한 게이트 전제). R1은 "도달 가능한데 rng이 놓치던" activity를 잡는 것인데 그런 여지가 거의 없었다. R1은 유효한 개선(전이 다양성)이나 coverage count의 레버가 아님이 실증됨.
- **운영 교훈(memory `collector-reconnect-trigger`)**: 재수집 연결은 (0) 서버 시작 전 accessibility 선-disable+force-stop으로 잔존 클라이언트 제거 → (1) 서버 bind 대기 → (2) force-stop+accessibility 재활성화 → (3) 그 뒤 무접촉. 연속 수집 2번째 앱부터 터지는 즉시-disconnect(0 steps)는 잔존 연결 race. iter 후 앱은 `--force` 없으면 "Nothing to collect"로 skip돼 stale 데이터를 새 결과로 오인. harness가 background 수집을 semi-random하게 evict(broccoli·calendar 조기종료) — full run은 hands-off로 대기해야 안정적.
- 커밋(브랜치 `main`, **푸시 안 함**): `1020d0d`·`382c377`·`0358be1`·`c3eb3cb`(iter3) · `97ce5dd`(research) · `6ff8e95`(iter4/R1). 논리 단위 분리.
- 남은 과제: coverage를 더 올리려면 exploration-policy 튜닝이 아니라 **reachability 공략**(로그인 플로우·풍부한 seed 데이터·게이트 화면 targeted 네비)이 필요 — 다른 클래스의 작업. R2(코퍼스 near-dup/다양성 감사)·R3(inverse-dynamics K-step, IWM subproject)·R4(coverage-vs-time velocity)는 미착수.
- 카테고리: devlog

## 2026-07-11 — Monkey-Collector: 수집 데이터 진단 → signal-timeout/launcher-drift 수정 → coverage/keyboard 재수정 (2 iteration, AVD Pixel6-2 재수집 검증)

사용자가 AVD Pixel6-2 로 수집 파이프라인을 업데이트하기를 원해 `analyze → revise → 재수집 검증` 을 데이터가 "잘 쌓였다" 고 판단될 때까지 반복했다(adaptive-router: 지정 advisor `openai/gpt-5.6-sol` 이 이 환경에서 codex CLI 의 gpt-5.6-* 미지원으로 전면 불가 → 가용성 폴백으로 advisor·검증 모두 `claude/fable`, worker `claude/opus`, 재수집 검증은 `claude/sonnet`). 기존 스냅샷 2개(`data_20260703_015219`/`data_20260702_110426`)를 직접 카운트로 진단한 뒤 두 라운드에 걸쳐 코드를 고치고 AVD 재수집으로 실효를 검증했다.

- **진단(`.claude/analysis/2026-07-11_data-collection-diag/`)**: signal timeout **435건**(각 25s, musicplayer 4.32 min/step ≈ 정체), nexuslauncher launcher-drift **140건**(broccoli), persist_filtered near-dup bloat(broccoli top-2 page=631+511 obs=76%), **타깃 4앱 중 3앱이 catalog/activities.json 누락 → coverage GT 무의미(1.0 클램프)**, calendar 실제 1/39. 저장/조인 스키마(events.jsonl page_key/observation_num, transition:false)는 건강 확인 — 문제는 상류 수집.
- **iteration 1(커밋 `45895b5` fix + `7102500` data)**: (P1-2) `collection.signal_timeout_sec`(기본 25→**12s**, config 6-place) + `MAX_SIGNAL_TIMEOUTS=3` → stuck 에피소드 최악 대기 125s→**36s**, nudge 는 1·2회차만; 데드상수 `FIRST_STEPS_NO_BACK` 제거. (P1-3) keyboard/back 이 launcher 로 이탈한 page 를 `back_exit_page_ids` 로 세션 학습 + `return_to_app` launcher-aware(무의미 back 생략). (P1-1) catalog broccoli·musicplayer apps.csv 등록 + device-pull APK 로 4앱 activities 재추출(calendar.pro merge-preserve). tcp_server·클라이언트 .kt 무접촉(APK 재빌드 불요).
- **iteration 1 재수집 검증(`.claude/analysis/2026-07-11_recollection-validation/`, 4앱×30min)**: throughput **2.3~14x** 개선(musicplayer 4.32→0.31 min/step) — P1-2 검증. 그러나 (a) **coverage 측정 결함** 발견: adb `get_current_activity` regex 가 trailing `}` 를 삼키고 activity-alias 가 catalog 부재 → calendar 가 실제 ≥6 activity 방문하고도 1/39 고착, (b) **keyboard-dismiss-back→launcher loop** 가 P1-3 의 D4(press_back 한정 마킹) 사각지대로 musicplayer 스텝 53%·broccoli 22% 소모.
- **iteration 2(커밋 `b85ecdc` fix + `7a46ab7` data)**: (Bug A) adb `_parse_current_activity` char-class regex 로 `}` 제거, `catalog/extract_activities.py` 가 `<activity-alias>` element 단위(zip 페어링 비결정성 회피) alias→target 맵 추출, `activity_coverage.py` 가 alias 방문을 target 으로 해석(분모 미확장·클램프 유지). (Bug B) launcher 이탈 keyboard-back 이 page 를 학습하고, 학습된 page 의 keyboard 는 `KEYCODE_ESCAPE`(최대 2회)→back 폴백으로 dismiss.
- **iteration 2 재검증(`.claude/analysis/2026-07-11_iter2-revalidation/`, calendar+musicplayer)**: (Bug A) **완전 수정** — calendar coverage 1→2+ 상승(`}` 제거, AllInOne alias 산입), musicplayer 3→**5/15**, end-to-end 시뮬로 alias→target 산입·split-APK 미산입 확인. (Bug B) **부분 개선** — ESC 경로 발동 25회(iter1 0회), keyboard 프레임 53%→**39%**, 단 이 AVD IME 에서 ESC 유효율 ~61%(11/18 first-attempt), 나머지는 2회 소진 후 back 폴백 → launcher drift 잔존.
- 검증: `pytest` iteration1 후 **764 passed** → iteration2 후 **784 passed**(신규 `test_adb_parse_activity`·`test_signal_timeout_escalation`·`test_launcher_drift`+`TestAliasResolution`/`TestKeyboardBackExit`, mutation-check). tier-2 advisor(fable) 각 라운드 PASS(주장 전부 CONFIRMED). AVD 재수집으로 empirical 확인. **venv 인터프리터 경로가 stale(`Project`→`Projects` 이동)이라 pytest 실행 불가였던 것도 `uv sync` 로 복구**.
- 커밋(브랜치 `main`, **푸시 안 함**): `45895b5`·`7102500`(iter1) · `b85ecdc`·`7a46ab7`(iter2). 논리 단위 분리(코드+테스트 / catalog 데이터).
- 남은 이슈(iteration 3 후속 과제): musicplayer 검색창 keyboard 의 IME-특이 ESC 무효 프레임(~39%) → IME manager 강제 hide 또는 검색창 반복 input 억제. broccoli stuck-rate 이상치 재현 미측정. osmand coverage(1/16) alias 실효성 미검증.
## 2026-07-11 — Implicit-World-Modeling: EXP05 (AndroidControl_EXP05) 신규 실험군 도입 — AndroidWorld 해상도 정렬 · 절대 픽셀 좌표 · Qwen2.5-VL 전용

어제(2026-07-10) Slack DM(조병웅↔백승우) 논의를 근거로 EXP05 실험군을 파이프라인에 도입했다. EXP05 = **AndroidWorld 해상도 정렬** 실험군으로, base 이미지 1080×2400 을 image budget **1,605,632**(factor 28) 로 smart_resize 한 **절대 픽셀 좌표(840×1876)** 를 쓴다 — Qwen2.5-VL native 와 일치하므로 **Qwen2.5-VL 전용**(EXP03/EXP04 의 0–1000 정규화·Qwen3-VL 전용과 정확한 대칭). AC_EXP01 ratio73 멤버십을 mirror 파생하는 방식(EXP03/EXP04 계보)이다.

- **신규 스크립트**: `Implicit-World-Modeling/scripts/mirror_exp05.py` — `mirror_exp04.py` 정밀 클론(소스 경로 `*_xy_pixel-aligned.jsonl`, 출력 `data/AndroidControl_EXP05/`, 소스 부재 시 traceback 없이 exit 1, docstring 픽셀 정렬 서술). 함수 로직·JOBS 7튜플(stage1 전용) 무변경.
- **파이프라인 배선(`scripts/_common.sh`)**: `AC_EXP05` 전 지점 등록(DS_PREFIX=IWM-AC_EXP05, HF_SLUG=ac-exp05-, DS_DATADIR=AndroidControl_EXP05, parse_args/parse_eval_args, build_infer_cmd cutoff 24576) + 신규 모델 `qwen2.5-vl-3b`(`Qwen/Qwen2.5-VL-3B-Instruct`, template qwen2_vl, 3-4B tier). `stage1_eval.sh` dual-task, `filter_long_samples.py`·`eval_viewer.py` 인식(stage2 맵 제외).
- **노트북 Cell 5/10/14**: qwen2.5-vl-3b + EXP05 config(image 1,605,632/3,136, cutoff 24576, stage1_only, dual-task), Cell 10 `_YAML_GEN_DS` allowlist(EXP03/04 hand-fix YAML 보호), Cell 14 파일부재 dataset skip 완화.
- **데이터**: Google Drive '0710_버젼' 폴더에서 `gdown`으로 raw 2파일(`stage1_0710_{action,state}_pred.jsonl`, 923M+877M) 다운로드 → canonical 이름(`implicit-world-modeling_stage1_{action,state}_xy_pixel-aligned.jsonl`)으로 rename → `mirror_exp05.py` 실행. **출력 stage1 7파일 총 64,787행(train 47,556, drop 2,444)**. 좌표 실측 x_max 840 / y_max 1876(픽셀 정렬 확증), 출력 이미지 경로 AndroidControl/ 100%(myset 0 — mirror 가 EXP01 경로 재사용). YAML 6종 + dataset_info EXP05 5키 생성.
- **검증**: tier-1 결정론적 게이트 직접 재실행(py_compile, 소스 부재 가드 exit1, `bash -n`, `--dataset AC_EXP05` parse→require_yaml exit1, Cell 5 config assert, git scope). tier-2 판정 **GO-with-caveats**(기능 결함 0, caveat=스테일 주석) — codex(openai)가 이 샌드박스 bwrap 오류로 미가용이라 advisor(fable)로 폴백. W3(문서)는 transport stall 실패 후 오케스트레이터 직접 마무리.
- **문서**: `Implicit-World-Modeling/{AGENTS,ARCHITECTURE,README}.md` 에 EXP05(절대 픽셀·Qwen2.5-VL 전용) + qwen2.5-vl-3b 반영(기존 EXP03/04 정규화 규약 변경 보존). "3B/8B 모두" 요청의 8B는 `qwen2.5-vl-7b`로 해석(Qwen2.5-VL 에 8B 부재, qwen3-vl-8b 는 factor·좌표 이중 mismatch).
- **커밋/푸시**: `32b1e57` feat(exp05) code + `d8509e5` docs(exp05) → origin/main push 완료. data/·LlamaFactory/ 산출물은 gitignored(로컬 전용).
- **미완**: 실제 Stage 1 학습(qwen2.5-vl-3b/7b, GPU 멀티시간 — Vessl 또는 로컬 `stage1_train.sh --dataset AC_EXP05`). eval/merge 후속.
- **워크플로우**: `/workflow:adaptive-router`(advisor plan → worker fan-out → 2-tier verify). `/workflow:revise`는 analyze 산출물 입력이 없어 미적용(이 작업은 Slack 기반).
- 카테고리: devlog

## 2026-07-11 — Implicit-World-Modeling: EXP05 diff loss 개편 — v2 매칭(bounds 기반) + 신규 가중 체계(diff 1.0 / non-diff 0.25) · eval xy 액션 스페이스 확장

같은 날 Slack DM + Google Meet 회의(조병웅↔백승우)에서 diff loss 의 **가중치 체계**와 **매칭 로직**을 동시에 바꾸기로 확정했다(조병웅이 `new_diff_loss.zip` = v2 코드 공유). 위 EXP05 도입 엔트리에 이어지는 후속 작업이다.

- **신규 가중 체계 (Qwen Agent World 방식)**: 기존 state transition 의 diff 토큰 **2.0** / non-diff **1.0** → 신규 diff **1.0** / non-diff **0.25**. intermediate action 예측 샘플은 가중치 없이 **uniform 1.0**. 배수 자체는 줄었지만 non-diff 도 같이 낮아져 diff 가 non-diff 대비 **실질 4배** 강해진다.
- **diff loss v2 (v1 과 병존)**: `scripts/diff_loss/` 에 v2 4파일 신규 추가(`hungarian_metric_v2.py`, `hungarian_diff_v2.py`, `token_weight_builder_v2.py`, `preprocess_dataset_v2.py`). v1 4파일은 **무변경**(AC_EXP02 재현성 보존). v1 대비: 위치 cost 가 DOM index → **bounds 중심점 거리**(`W_POS=0.4`, `BOUNDS_NORM=2050.0`=840×1876 대각선, `BOUNDS_TAU=50.0`), `_collect_texts()` 의 자손 텍스트 흡수 제거, `MATCH_THRESHOLD` 1.5→1.7, element 키 index→bounds 폴백, metric key `hungarian_idx`→`hungarian_pos`.
- **v2 가 EXP05 에 필수인 이유**: EXP05 HTML 에는 `index` 속성이 **아예 없다**(실측: index 0개, bounds 48개). v1 builder 는 `index="..."` 를 regex 로 필수 요구하므로, v1 을 쓰면 모든 토큰이 baseline 으로 방치돼 **diff loss 가 조용히 무력화**된다.
- **가중치 적용 함정 2건(해결)**: (1) `token_weight_builder` 의 baseline 이 `[1.0]*n_asst` 이고 `if weight == 1.0: continue` 로 기본값을 스킵하는 구조라, 신규 체계에서는 diff weight 가 바로 그 1.0 이어서 **diff 토큰이 baseline(0.25)에 방치**된다 → baseline 을 `wmap["UNCHANGED"]` 에서 유도하고 스킵 조건을 `if weight == base` 로 변경. (2) action 샘플에 uniform 1.0 분기가 없으면 "diff 없음 → 전부 0.25" 로 오처리된다 → `images` 개수로 판별(1개=state_pred, 2개=action_pred)해 명시적 분기 추가.
- **EXP05 데이터 재생성(실측)**: train **47,556행** = state 33,285(weight 값 `{0.25, 1.0}`) + action 14,271(`{1.0}`). state 샘플 출력 토큰의 **54.1% 가 0.25배로 감쇠**. v1(EXP02) 대조에서 전체 diff 비율은 56.9% vs 55.8% 로 유사하나 세부 분해는 **v2 가 매칭을 개선**(ADDED 45.8%→26.6%, MODIFIED 11.1%→29.2%) — v1 은 매칭 실패로 ADDED 가 부풀었고, v2 는 bounds 좌표 덕에 "같은 자리 텍스트 변경"을 MODIFIED 로 제대로 잡는다.
- **EXP05 배선**: stage1 YAML 6개 + 노트북 Cell 5(SSoT)에 `use_diff_token_weighted_loss: true` 활성화. `token_weights` 는 train jsonl 인라인이라 `dataset_info.json` 컬럼 등록은 **불필요**(`converter.py:226` 이 raw jsonl 에서 직접 읽음, EXP02 선례).
- **eval 채점을 xy 액션 스페이스로 확장**: 액션 스페이스가 xy 로 통일돼 GT 스키마가 `<action>{"action":"click","coordinate":[x,y]}</action>` 로 바뀌었다(키가 `action` — 구 `action_type` 과 다름; swipe 는 `coordinate1`/`coordinate2`). **opt-in 플래그**로 구현해 EXP01~04 채점은 불변 — `_action_eval.py --coord-mode {index,xy}`(기본 index), `_hungarian_eval.py --match-mode {index,pos}`(기본 index), `stage1_eval.sh` 가 **AC_EXP05 일 때만** 전달. codex 검증에서 기본 모드가 신/구 코드 **byte-identical** 확인. xy 모드 규칙: click/long-press 는 pred 좌표가 GT 좌표가 속한 element 의 bbox 안이면 정답(포함 element 없으면 오답 + `no_bbox_n` 별도 집계), scroll/swipe 는 xy1→xy2 벡터의 주 방향(`|dx|>=|dy|` → left/right, else up/down) 일치 시 정답, input_text/type 은 좌표 무관. `tests/test_action_eval_xy.py` 30케이스 추가 → 전체 **82 tests OK**. pred 좌표계 sanity 경고 추가(정규화 좌표 의심 시 stderr 경고, 채점 결과는 불변).
- **검증 — diff loss 가 실제로 작동함을 확증**: 동일 데이터·동일 설정에서 플래그만 바꿔 대조 — step 1 loss 0.6633(ON) vs 1.0391(OFF), step 2 0.2395 vs 1.2479, train_loss **0.4514 vs 1.1435**. loss 가 명확히 달라 `token_weights` 가 collator → trainer → loss 계산까지 실제로 반영됨을 확인.
- **로컬 학습 불가 판정(실측)**: 로컬 2×RTX5090 에서 EXP05 3B Full FT 는 **CUDA OOM**(step 3 에서 8.92GiB 할당 실패) + **157~168 s/it → 총 97~104시간(약 4일)**. 원인은 `cutoff_len 24576` + `max_pixels 1,605,632` 의 비전 토큰으로 시퀀스가 극단적으로 길어진 것 + RTX5090 에 강제되는 ZeRO-3 CPU offload. **본 학습은 Vessl A100/H100 에서 수행**한다(저장소에 Vessl 파이프라인 스크립트 없음 — 운영 지식).
- **데이터 분할 비율**: 회의에서 **7:3 확정**. EXP05 는 이미 AC_EXP01 ratio73 멤버십을 미러하므로 충족 — 추가 작업 없음.
- **미결**: (1) 조병웅의 수정본 stage1 데이터 미업로드(Drive `modifiedTime` 2026-07-10T09:12:02Z 그대로) → 사용자 지시로 **기존 0710 데이터로 진행**. 수정본 도착 시 재다운로드 → mirror → 전처리 재실행 필요. (2) v2 `extract_elements` 의 aria-label 누락 — 포함 조건이 `description` 단독이라 `<div aria-label="Home">` 같은 요소가 제외된다(EXP05 test 300문서 중 117개, element 366개 실측). 학습/평가가 공유하는 규약이라 조병웅 확인 필요. (3) `without_open_app` 필터 무동작 — 필터가 `## Action` 마커를 찾는데 새 프롬프트 포맷은 `Action:` 을 쓴다. **EXP03 에도 존재하는 기존 이슈**(state 와 without_open_app 행수 동일).
- 참고 계획 문서: `docs/EXP05_DIFF_LOSS_PLAN.md`
- 카테고리: devlog

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
