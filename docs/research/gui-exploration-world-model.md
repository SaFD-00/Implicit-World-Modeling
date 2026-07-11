# 관련 연구 — GUI 탐색 & 암묵적 World Modeling

> GUI 상호작용 전이 데이터로 VLM 에 **암묵적 world model** 을 학습시켜 GUI 에이전트의 action prediction 을 개선하려는 이 프로젝트의 근거·비교·적용 권고를 실제 문헌으로 정리한다.
>
> 작성일: 2026-07-11 · 조사 도구: WebSearch/WebFetch (URL 실검증) · 코드 수정 없음(문서 산출만)

---

## 프로젝트 문제 정의

이 모노레포는 두 단계를 연결한다. **(1) Monkey-Collector** 는 Android AccessibilityService + Python 서버로 실제 앱을 자동 탐색해 `(screenshot + UI XML + action → next screenshot + UI XML)` 전이를 대량 수집한다. 탐색 엔진은 coverage-driven unexplored-first + same-function 압축 + 최단경로 navigation 을 수행하는 **LLM-Explorer 포팅**이다. **(2) Implicit-World-Modeling** 은 이 전이 데이터로 2-stage VLM fine-tuning 을 한다 — Stage 1 (World Modeling): `screenshot+XML+action → next XML` 로 환경 dynamics 를 암묵 학습, Stage 2 (Action Prediction): `screenshot+XML+task → action JSON`. 핵심 가설은 **"다음 상태를 예측하도록 사전학습한 world model 이 downstream action prediction 을 돕는다"** 이며, base / stage2-only / (stage1+stage2) 를 비교해 이 이점을 정량화한다.

따라서 이 프로젝트의 성패는 두 축에 달려 있다: **① 수집 데이터의 전이 밀도·다양성·커버리지**(Collector 품질)와 **② 전이 예측 사전학습이 action prediction 으로 전이(transfer)되는지**(WM 가설). 아래 4개 스레드는 이 두 축의 SOTA 근거를 제공한다.

---

## 관련 연구

### 스레드 1 — 자동 GUI/모바일 앱 탐색 전략 (데이터 수집)

- **[LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps](https://arxiv.org/abs/2505.10593)** · MobiCom 2025 · Zhao et al.
  - 핵심: LLM 을 **매 스텝 action 생성**에 쓰지 않고 **앱 지식 유지(knowledge maintenance)** 에만 쓰고, 그 지식으로 action 선택은 LLM-less 하게 수행한다. 원시 탐색 데이터를 state/action merge 로 **Abstract Interaction Graph** 로 추상화하고, 이 지식으로 미탐색 우선 + 컨텍스트-aware test plan 을 세운다. SOTA LLM 기반 대비 **~148x 저비용**에 더 높은 activity coverage.
  - 관련성: **이 프로젝트 탐색 엔진의 직접 원전.** `pipeline/exploration/` 이 포팅한 대상. 다만 포팅본은 coverage 추적 + same-function 압축 + shortest-path navigation 은 가져왔으나, **미탐색 후보 중 선택은 `rng.choice`**(explorer.py `_pick_unexplored`)로 원전의 knowledge-guided 우선순위(가치 기반 랭킹)를 채택하지 않았다 → 아래 gap/권고 참조.

- **[Stoat: Guided, Stochastic Model-Based GUI Testing of Android Apps](https://tingsu.github.io/files/fse17-stoat.pdf)** · FSE 2017 · Su et al.
  - 핵심: 앱 행동을 **stochastic FSM** 으로 모델링(노드=상태, 엣지=입력 이벤트). UI 이벤트 실행 빈도를 기록해 전이 확률 초기값으로 쓰고, 모델 기반으로 high-coverage + diverse 시퀀스를 반복 생성(전이 확률 mutate → 테스트 생성 → 시스템 이벤트 주입 → replay).
  - 관련성: 이 프로젝트의 `transition_graph.py`(page_key 그래프)와 같은 계열. **커버리지-vs-시간을 최적화 목표로 명시**하는 점이 참고 — 현재 DFS/BFS/GREEDY 전략을 비교할 계량 지표의 근거.

- **[DroidBot: A Lightweight UI-Guided Test Input Generator for Android](https://www.researchgate.net/publication/318125264_DroidBot_A_Lightweight_UI-Guided_Test_Input_Generator_for_Android)** · ICSE 2017 (demo) · Li et al.
  - 핵심: 런타임에 동적으로 만든 **UI Transition Graph(UTG)** — 노드=UI 상태, 엣지=상태 전이를 유발한 action — 를 기반으로 UI-guided 입력을 생성. 경량·root-free.
  - 관련성: `page_graph.py`/`transition_graph.py` 의 UTG 개념과 동형. 이 프로젝트가 만드는 것이 바로 "world model 학습용 UTG 코퍼스"임을 보여주는 기준선.

- **[Humanoid: A Deep Learning-based Approach to Automated Black-box Android App Testing](https://arxiv.org/pdf/1901.02633)** · ASE 2019 · Li et al.
  - 핵심: 인간의 앱 상호작용 로그로 **어떤 UI 요소가 인간이 상호작용할 법한지**를 DL 로 학습해, 그 우선순위로 탐색을 유도. 순수 random/model-based 대비 더 의미있는 상태에 도달.
  - 관련성: `_pick_unexplored` 를 uniform-random 에서 **학습된 상호작용 우선순위**로 대체하는 방향의 선례. same-function 압축과 결합 가능.

- **[APE: Practical GUI Testing of Android Applications via Model Abstraction and Refinement](https://helloqirun.github.io/papers/icse19_tianxiao.pdf)** · ICSE 2019 · Gu et al.
  - 핵심: 정적 모델 대신 런타임 정보로 **모델 추상화 수준을 동적으로 refine**(CEGAR 스타일) — 상태 추상화의 크기와 정밀도를 균형. 15개 대형 앱에서 SOTA 대비 커버리지·크래시 우위.
  - 관련성: 이 프로젝트의 `ScreenMatcher`(BM25 unique-page)가 하는 "상태 추상화 정밀도" 문제와 정확히 대응. near-dup 병합 임계값(element diff / pixel gate)이 곧 추상화 granularity 이며, APE 는 이를 고정이 아니라 적응적으로 조정하는 근거.

- **[Fastbot2: Reusable Automated Model-based GUI Testing for Android Enhanced by Reinforcement Learning](https://tingsu.github.io/files/ASE22-industry-Fastbot.pdf)** · ASE 2022 · Cai et al.
  - 핵심: activity coverage 를 높일 가능성이 큰 UI 이벤트를 **RL value 로 선택**하고, 과거 탐색 지식을 재사용(continuous testing). 산업 규모 검증.
  - 관련성: 미탐색 후보를 **가치 기반**으로 고르는 직접 근거(현재 `rng.choice` 대체안). 세션 간 지식 재사용은 이 프로젝트의 resume/rehydrate 와 상보적.

- **[LLMDroid: Enhancing Automated Mobile App GUI Testing Coverage with Large Language Model Guidance](https://dl.acm.org/doi/10.1145/3715763)** · FSE 2025 · (Proc. ACM SE)
  - 핵심: LLM 가이드로 전통 탐색 도구의 coverage 한계를 완화. LLM 을 stuck 탈출·고수준 계획에 선택적으로 투입.
  - 관련성: LLM-Explorer 와 같은 "LLM 을 아껴 쓰는" 최신 흐름. 이 프로젝트가 LLM 을 element 추출/텍스트 생성에만 쓰고 식별·선택은 LLM-free 로 유지하는 설계와 정합.

- **[MobileGPT: Augmenting LLM with Human-like App Memory for Mobile Task Automation](https://arxiv.org/abs/2312.03003)** · MobiCom 2024 · Lee et al.
  - 핵심: explore–select–derive–recall 의 인간형 앱 메모리로 서브태스크를 모듈화·재사용. 185 태스크/18 앱에서 82.7% 자동화, 적응은 98.75%.
  - 관련성: 이 프로젝트의 `ElementExtractor`(MobileGPT-V2 Node-Clustering·subtask/trigger 프롬프트 포팅)의 원전 계열. element family / same-function group 압축의 근거.

- **Mobile3M (3M-page 모바일 GUI 전이 그래프 코퍼스)** · *URL 확인 필요* (search 로 개요만 확인: 49개 중국 앱, 3M UI 페이지, click/scroll/input, screenshot+XML, random-walk directed graph, Parquet)
  - 핵심: 실제 앱을 random walk 로 탐색해 **전이-action 그래프**로 통합한 대규모 코퍼스. unique-page 식별에 BM25 element-line 매칭 사용.
  - 관련성: 이 프로젝트 `pipeline/screen_matching/`(BM25 unique-page matching)이 명시적으로 포팅한 "Mobile3M 메커니즘". 정확한 arXiv/venue 는 미확정이므로 인용 시 재검증 요망(fabrication 회피).

### 스레드 2 — GUI/디지털 world models & 암묵적 world modeling

- **[GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning](https://arxiv.org/html/2505.12493)** · arXiv 2025 · Gao et al.
  - 핵심: **K-step GUI Transition** — `S_t` 와 `S_{t+k}` 두 스크린샷을 주고 그 전이를 유발한 **첫 action 을 예측**하는 self-supervised inverse-dynamics 과제. 인간 주석 불필요(기존 궤적에서 action 추출). 이로 GUI dynamics 를 학습하면 downstream action prediction 이 향상: **AndroidControl-High 최대 +11.2%**(Qwen2.5-VL-7B 70.4% EM), ScreenSpot-v2 +2.5%. k-value 당 단 2K 샘플.
  - 관련성: **이 프로젝트 가설의 가장 직접적·정량적 근거.** "전이를 예측하도록 사전학습 → action 예측 향상"을 4 VLM·4 벤치마크에서 입증. Stage 1 을 forward dynamics(다음 XML 예측) 로 하는 이 프로젝트에, **inverse dynamics(K-step) 를 보조 목표로 추가**하면 근거 있는 강화가 된다(권고 참조).

- **[How Mobile World Model Guides GUI Agents?](https://arxiv.org/abs/2605.10347)** · arXiv 2026 (최근 preprint, tool-sourced) · Xu et al.
  - 핵심: 모바일 world model 이 GUI 에이전트를 어떻게 돕는지 체계 분석 — (a) 예측 modality(renderable code = ID 충실도↑, text feedback = OOD robust), (b) test-time 통합은 **posterior critic 보다 prior perceptual module 로 쓸 때 유리**, (c) imagined trajectory 로부터 전이 가능한 상호작용 경험 획득. (정량치는 요약기 confabulation 위험 있어 재검증 전까지 인용 자제.)
  - 관련성: **이 프로젝트 논지("world model 이 action 을 돕는다")의 정면 최신 근거.** 특히 "world model 은 사후 비평가보다 사전 지각 모듈로 통합하라"는 결론은 Stage1→Stage2 파이프라인 설계 판단에 시사.

- **[VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](https://arxiv.org/html/2510.16907v1)** · arXiv 2025 · (VAGEN team)
  - 핵심: VLM 에이전트가 내부에 **구조화된 world model belief** — 현재를 해석하는 state model + 미래를 예측하는 transition model — 을 세우도록 RL 로 강화.
  - 관련성: 암묵적 world modeling 을 명시적 reasoning 으로 끌어올리는 방향. 이 프로젝트가 Stage1 을 "암묵" 으로 두는 선택의 대안(명시적 transition reasoning) 을 보여줌.

- **[Introducing Dreamer: Scalable RL Using World Models](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/)** · Google Research · Hafner et al. ([Dream to Control, 2020](https://www.researchgate.net/publication/337730042_Dream_to_Control_Learning_Behaviors_by_Latent_Imagination))
  - 핵심: 과거 경험에서 **latent dynamics(다음 latent 상태 예측)** 를 self-supervised 로 학습하고, 그 world model 안에서 상상(imagination)으로 정책을 학습. transition predictor + encoder + decoder 구조.
  - 관련성: "관측→다음관측 예측으로 dynamics 를 학습한다"는 이 프로젝트 Stage 1 의 이론적 뿌리(도메인만 로봇→GUI). GUI 는 이미지+XML 관측이라 latent 대신 XML 을 예측 타깃으로 삼은 변형으로 해석 가능.

- **[MobileWorldBench: Towards Semantic World Modeling for Mobile Agents](https://arxiv.org/html/2512.14014)** · arXiv 2025/26 (tool-sourced)
  - 핵심: 현재 화면 + action 으로 **미래 상태를 예측**하는 VLM 능력을 Next-State-Generation / Next-State-QA 로 벤치마킹.
  - 관련성: 이 프로젝트 Stage 1 (next XML 예측)의 평가 프로토콜 참고. 자체 Hungarian-F1 next-XML 평가와 대응.

- **[Code2World: A GUI World Model via Renderable Code Generation](https://arxiv.org/abs/2602.09856)** · arXiv 2026 (tool-sourced)
  - 핵심: 구조화된 **HTML 을 native 표현**으로 써서 action-조건부 전이를 renderable code 로 예측 — 시각 충실도 + 구조적 controllability 동시 달성.
  - 관련성: 이 프로젝트가 next-state 를 **UI XML**(구조 표현)로 예측하는 선택을 뒷받침. 스크린샷 픽셀 생성보다 구조 표현 예측이 controllable 하다는 논지 공유.

- **[MobileDreamer: Generative Sketch World Model for GUI Agent](https://arxiv.org/html/2601.04035v1)** · arXiv 2026 (tool-sourced)
  - 핵심: action-조건부 예측 상태를 **구조화된 element layout(sketch)** 으로 생성, order-invariant 학습으로 robust next-state.
  - 관련성: next-state 를 요소 레이아웃으로 예측 + **순서 불변 학습** — 이 프로젝트의 XML 예측/Hungarian 매칭 평가와 발상이 겹침(집합 기반 next-state).

### 스레드 3 — GUI 에이전트 & action prediction (Stage 2)

- **[CogAgent: A Visual Language Model for GUI Agents](https://arxiv.org/pdf/2312.08914)** · CVPR 2024 · Hong et al.
  - 핵심: CogVLM 기반 고해상도 cross-module 로 GUI 스크린샷을 처리, PC/web/Android 에서 구조 입력 없이 요소 위치·action 예측.
  - 관련성: screenshot+UI 기반 action prediction 의 대표 baseline. Stage 2 성능 위치를 가늠할 기준.

- **[SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/html/2401.10935v2)** · ACL 2024 · Cheng et al.
  - 핵심: 스크린샷만으로 action 위치 예측, **GUI grounding 사전학습**이 downstream agent 성능의 핵심임을 제시.
  - 관련성: "GUI 사전학습이 downstream 을 돕는다"는 이 프로젝트 가설의 인접 근거(단 grounding vs dynamics 라는 사전학습 목표 차이).

- **[UI-TARS: Pioneering Automated GUI Interaction with Native Agents](https://arxiv.org/html/2501.12326v1)** · arXiv 2025 · Qin et al.
  - 핵심: 스크린샷만 입력하는 native GUI 에이전트, **UI-specific pretraining + 통합 vision-language-action** 로 일반화 강화.
  - 관련성: 대규모 UI pretraining 이 action 예측 일반화를 끌어올린다는 최신 SOTA 근거. 이 프로젝트의 dynamics-pretraining 을 그 계보에 위치.

- **[Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs](https://arxiv.org/abs/2404.05719)** · ECCV 2024 · You et al. (Apple)
  - 핵심: 모바일 UI 특화 MLLM(any-resolution 로 작은 아이콘/텍스트 확대), referring/grounding/reasoning.
  - 관련성: 모바일 UI 이해의 대표 backbone. 스크린샷+요소 이해가 action 예측의 전제임을 보여줌.

- **[AppAgent: Multimodal Agents as Smartphone Users](https://arxiv.org/abs/2312.13771)** · arXiv 2023 · Zhang et al.
  - 핵심: 단순 action space(tap/swipe)로 앱을 조작, **자율 탐색 또는 시연 관찰**로 새 앱 사용법 학습.
  - 관련성: "자율 탐색으로 앱 지식을 얻어 action 에 쓴다" — 이 프로젝트의 Collector→Model 파이프라인과 같은 발상(다만 오프라인 사전학습 대신 온라인 문서화).

- **[Mobile-Agent: Autonomous Multi-modal Mobile Device Agent with Visual Perception](https://arxiv.org/abs/2401.16158)** · arXiv 2024 · Wang et al.
  - 핵심: XML 없이 스크린샷 시각 인식만으로 요소 위치·계획·실행.
  - 관련성: screenshot-only vs screenshot+XML 트레이드오프의 대척점. 이 프로젝트가 **XML 을 입력·예측 타깃으로 유지**하는 선택의 비교군.

- **[AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents](https://arxiv.org/abs/2405.14573)** · arXiv 2024 · Rawles et al. (Google DeepMind)
  - 핵심: 20개 실앱·116개 파라미터화 태스크, adb 로 시스템 상태를 검사하는 ground-truth reward.
  - 관련성: Stage 2 action prediction 을 **실제 태스크 성공률**로 평가할 표준 환경(현재 오프라인 step-accuracy 를 넘어선 OOD 검증 경로). GUI-Shift 도 AndroidControl 로 이 계열 평가.

### 스레드 4 — 탐색 수집 데이터의 품질/커버리지 메트릭

- **[Near-Duplicate Detection in Web App Model Inference](https://tsigalko18.github.io/assets/pdf/2020-Yandrapally-ICSE.pdf)** · ICSE 2020 · Yandrapally et al.
  - 핵심: 모델 추론 시 화면의 **near-duplicate 병합**이 상태 그래프 품질을 좌우함을 정량 분석(구조·시각·하이브리드 기법 비교, 임계값이 상태 폭발/붕괴를 결정).
  - 관련성: 이 프로젝트 `ScreenMatcher`(BM25 element-line + pixel gate)와 `persist_filtered`(방문마다 `0,1,2,…` per-visit observation 체인)의 **near-dup 축적 위험**을 직접 겨냥. WM 학습 코퍼스가 near-dup 로 부풀면 Stage 1 이 "변화 없음" 편향을 학습.

- **[MobileViews: A Million-scale and Diverse Mobile GUI Dataset](https://arxiv.org/abs/2409.14337)** · arXiv 2024 · Gao et al.
  - 핵심: 대규모·**다양성**을 명시 목표로 한 모바일 GUI 데이터셋 수집(스크린샷+뷰 계층).
  - 관련성: WM 학습에 좋은 데이터의 조건 = 규모 + 다양성. 4개 앱 대상인 이 프로젝트의 커버리지/다양성 상한을 가늠하는 참조점.

- **[FineVision: Open Data Is All You Need](https://arxiv.org/pdf/2510.17269)** · arXiv 2025
  - 핵심: 데이터 다양성 계량 기법 — **SSCD copy-detection 임베딩**으로 near-dup 억제/공간 균등 점유, **Effective Rank**(분산이 퍼진 차원 수 = 개념 폭), **β-Recall**(실분포 커버리지).
  - 관련성: 이 프로젝트가 **전이 코퍼스 다양성을 수치화**할 실용 메트릭 세트. 세션 간·앱 간 중복도와 전이 다양성 감사(audit)에 이식 가능.

- **[Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning](https://arxiv.org/pdf/2108.03353)** · UIST 2021 · Wang et al.
  - 핵심: UI 화면의 의미 요약 — 화면 semantic 표현/중복 판별의 초기 기법.
  - 관련성: page_key 식별을 **의미 수준**에서 보강할 대안(현재 BM25 element-line 은 구조 기반). semantic dedup 으로 near-dup 정밀도 향상 여지.

- **Activity/screen coverage-over-time (스레드 1 문헌 재사용).** Stoat/APE/Fastbot2/LLM-Explorer 는 모두 **"고정 시간 내 도달한 unique activity 수"** 를 1차 지표로 삼는다.
  - 관련성: 이 프로젝트는 `domain/activity_coverage.py` 로 최종 coverage 는 재지만, **coverage-vs-time(velocity)·transition-density(엣지/노드) 시계열**은 없다. DFS/BFS/GREEDY 세 전략을 근거로 고르려면 이 시계열 지표가 필요.

---

## 이 프로젝트 vs SOTA (gap 분석)

| 영역 | SOTA 기법 | 이 프로젝트 현재 구현 | 상태 |
| --- | --- | --- | --- |
| 상태 추상화 | BM25 unique-page + pixel gate (Mobile3M), 적응적 refine (APE) | BM25 element-line + luminance gate (Mobile3M 포팅) | **채택** — 단 임계값 고정(APE식 적응 refine 미채택) |
| element 압축 | Node-Clustering / same-function (MobileGPT-V2) | `ElementExtractor` family + `Memory` same-function 압축 | **채택** |
| navigation | 전이 그래프 최단경로 (DroidBot UTG, Stoat FSM) | `TransitionGraph` shortest-path + DFS/BFS/GREEDY | **채택** |
| 미탐색 후보 **선택** | knowledge-guided 랭킹 (LLM-Explorer), RL value (Fastbot2), 학습된 상호작용 우선순위 (Humanoid) | `rng.choice`(long_touch 후순위만) — `_pick_unexplored` | **미채택** ← 최대 gap |
| 탐색 목표 계량 | activity-coverage-vs-time (Stoat/APE/Fastbot2/LLM-Explorer) | 최종 coverage CSV 만, 시계열/velocity 없음 | **부분** |
| 데이터 near-dup 감사 | near-dup 병합 품질 (Yandrapally), 다양성 계량 (FineVision SSCD/Effective Rank) | matcher 단의 dedup 은 있으나 **코퍼스 수준 다양성/near-dup 감사 없음** | **미채택** |
| WM 사전학습 목표 | forward dynamics (Dreamer), **inverse dynamics K-step (GUI-Shift)**, code/sketch next-state (Code2World/MobileDreamer) | Stage 1 forward-only (next XML 예측) | **부분** — inverse-dynamics 보조 목표 미채택 |
| action 평가 | 실태스크 성공률 (AndroidWorld), grounding (ScreenSpot) | 오프라인 step-accuracy / Hungarian-F1 | **부분** — OOD 실행 평가 미채택 |

---

## 적용 권고

value/effort 로 우선순위. **iter3 중복** = 진행 중인 다른 워커(explorer `_fallback` 입력창 재선택 억제 / `catalog_activities.py` coverage 분모 정화 / `adb.py`·`collection_loop.py` keyboard 측정)와 겹치는지.

| # | 기법 | 대상 파일/컴포넌트 | 기대효과 | 난이도 | iter3 중복 |
| --- | --- | --- | --- | --- | --- |
| R1 | **미탐색 후보 value/knowledge-guided 선택**(LLM-Explorer·Fastbot2·Humanoid): `rng.choice` → 요소 유형·전이 신규성·family 크기 기반 랭킹 | `exploration/explorer.py` `_pick_unexplored`, `memory.py` | 스텝당 coverage 효율↑ → **세션당 전이 밀도·다양성↑**(WM 학습 데이터 질 직결) | M | **비중복(신규)** |
| R2 | **코퍼스 수준 near-dup/다양성 감사**(Yandrapally·FineVision): `persist_filtered` per-visit 체인의 near-dup 비율·Effective Rank·SSCD 유사도를 오프라인 리포트 | `pipeline/screen_matching/*`(읽기), `export/converter.py`, 신규 감사 스크립트 | near-dup 코퍼스 bloat 조기 발견 → Stage 1 "무변화 편향" 방지 | M | **비중복(신규)** |
| R3 | **inverse-dynamics 보조 목표(K-step GUI Transition)**(GUI-Shift): Stage 1 에 forward(next XML) + `(S_t,S_{t+k})→first action` 예측을 함께 | `Implicit-World-Modeling/` 학습 파이프라인 | **published +11.2% action-acc 근거**로 Stage1→Stage2 transfer 강화 | M~L | **비중복(별 subproject)** |
| R4 | **coverage-vs-time·transition-density 시계열 계량**(Stoat/APE/Fastbot2): unique activity/page velocity + 엣지/노드 밀도 시계열 로깅 | 신규 eval(읽기), `domain/activity_coverage.py` 인접(분모 로직은 안 건드림) | DFS/BFS/GREEDY 를 **근거 기반**으로 선택·튜닝 | S~M | **부분 인접**(분모=iter3, velocity 시계열은 직교) |
| R5 | **적응적 매칭 granularity refine**(APE CEGAR): element-diff/pixel 임계값을 세션 통계로 조정(상태 폭발/붕괴 감지 시) | `screen_matching/screen_matcher.py` 임계값 | 앱별 상태 추상화 정밀도↑ → 전이 그래프 품질↑ | L | 비중복(신규) |
| R6 | **AndroidWorld 스타일 OOD 실행 평가**: 오프라인 step-acc 를 넘어 실태스크 성공률로 Stage2 검증 | `Implicit-World-Modeling/` 평가 | ID/OOD 일반화 근거 강화 | L | 비중복(별 subproject) |

**우선순위 요약**: R1(고 value·중 effort, Collector 측 최대 gap) → R2(데이터 품질 안전망) → R3(가설의 published 강화) 순. R4 는 저비용 계측이라 빠른 실측 가치.

---

## 참고문헌

### 탐색 전략
- LLM-Explorer, MobiCom 2025 — https://arxiv.org/abs/2505.10593
- Stoat, FSE 2017 — https://tingsu.github.io/files/fse17-stoat.pdf
- DroidBot, ICSE 2017 — https://www.researchgate.net/publication/318125264_DroidBot_A_Lightweight_UI-Guided_Test_Input_Generator_for_Android
- Humanoid, ASE 2019 — https://arxiv.org/pdf/1901.02633
- APE, ICSE 2019 — https://helloqirun.github.io/papers/icse19_tianxiao.pdf
- Fastbot2, ASE 2022 — https://tingsu.github.io/files/ASE22-industry-Fastbot.pdf
- LLMDroid, FSE 2025 — https://dl.acm.org/doi/10.1145/3715763
- MobileGPT, MobiCom 2024 — https://arxiv.org/abs/2312.03003
- Mobile3M (코퍼스, ScreenMatcher 포팅 원전) — *정확한 URL 확인 필요*

### World models
- GUI-Shift, 2025 — https://arxiv.org/html/2505.12493
- How Mobile World Model Guides GUI Agents?, 2026 preprint — https://arxiv.org/abs/2605.10347
- VAGEN, 2025 — https://arxiv.org/html/2510.16907v1
- Dreamer (Dream to Control), Google Research — https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/
- MobileWorldBench, 2025/26 preprint — https://arxiv.org/html/2512.14014
- Code2World, 2026 preprint — https://arxiv.org/abs/2602.09856
- MobileDreamer, 2026 preprint — https://arxiv.org/html/2601.04035v1

### GUI 에이전트 & action prediction
- CogAgent, CVPR 2024 — https://arxiv.org/pdf/2312.08914
- SeeClick, ACL 2024 — https://arxiv.org/html/2401.10935v2
- UI-TARS, 2025 — https://arxiv.org/html/2501.12326v1
- Ferret-UI, ECCV 2024 — https://arxiv.org/abs/2404.05719
- AppAgent, 2023 — https://arxiv.org/abs/2312.13771
- Mobile-Agent, 2024 — https://arxiv.org/abs/2401.16158
- AndroidWorld, 2024 — https://arxiv.org/abs/2405.14573

### 데이터 품질/커버리지
- Near-Duplicate Detection in Web App Model Inference, ICSE 2020 — https://tsigalko18.github.io/assets/pdf/2020-Yandrapally-ICSE.pdf
- MobileViews, 2024 — https://arxiv.org/abs/2409.14337
- FineVision, 2025 — https://arxiv.org/pdf/2510.17269
- Screen2Words, UIST 2021 — https://arxiv.org/pdf/2108.03353

> **인용 주의**: 2026 preprint(2605.10347·2602.09856·2601.04035·2512.14014)와 GUI-Shift 는 도구(WebFetch/WebSearch) 출력으로만 확인했다 — 실존하나 요약기가 세부 수치를 confabulate 할 수 있어 **정량치 재인용 시 원문 재확인** 필요. Mobile3M 은 개요만 확인되어 정확한 서지정보 미확정(fabrication 회피 위해 URL 확인 필요로 표기).
