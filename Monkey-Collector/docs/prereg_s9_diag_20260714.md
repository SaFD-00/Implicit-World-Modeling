# S-9 진단 사전등록 — 지도 지문 파편화 원인 (2026-07-14)

측정 실행 전 작성. 아래 분류 정의·판정 규칙은 데이터를 본 뒤 변경하지 않는다.

## 목적

armA/armB의 osmand 아카이브를 오프라인 리플레이하여, `MapActivity` 페이지가
새로 mint되는 각 이벤트의 원인을 `retrieval-miss` / `element-blocked` /
`pixel-blocked` 세 범주 중 하나로 분류한다.

## 하네스 충실도 게이트 (측정보다 먼저 통과해야 함)

현재 코드(`0a92715` 이후 `screen_matching/` 무변경 확인됨)로 리플레이했을 때:

- armA: 전체 페이지 **71**, MapActivity 페이지 **46**
- armB: 전체 페이지 **67**, MapActivity 페이지 **26**

이 정확히 재현되어야 한다. 불일치 시 하네스가 유죄 — gate 통계로 진행하지
않고 불일치 내용을 그대로 보고하고 중단한다. 숫자를 맞추기 위해 하네스를
조작하지 않는다.

"MapActivity 페이지"의 판정 기준은 `page_graph.json`의 각 노드에 기록된
`activity` 필드(= 그 페이지가 mint된 순간의 activity, first-seen 기준)이다.
이는 사후에 관측된 아카이브 사실이며 사전등록 대상이 아니다 — 단지 게이트
비교 대상의 정의를 명확히 하기 위해 여기 기록한다.

## 리플레이 방법

`events.jsonl`을 `step` 오름차순으로 정렬 후, `page_key`가 있는 이벤트만
(외부 앱 인터럽트 이벤트 제외) 순서대로 처리한다. 각 이벤트의
`(page_key, observation_num)`으로 해당 관측 디렉토리의
`raw.xml`/`encoded.xml`/`screenshot.png`를 로드하고, 이벤트의
`activity_name`과 함께 **새로 생성한** `ScreenMatcher` 인스턴스(설정값은
`config/run.yaml` 정본 그대로: `luminance_prefilter=true`,
`luminance_threshold=10`, `screenshot_diff_threshold=0.02`,
`luminance_low_res_width=100`, `persist_filtered=true`, `bm25_top_k=5`,
`element_criterion=diff`, `element_diff_max=5`, `element_jaccard_min=0.5`,
`page_pixel_diff_threshold=0.3`, `extractor=None`)에 순서대로 투입한다.

세션당 하나의 fresh matcher (armA/armB 별도 인스턴스, rehydrate 없음 — 원본
라이브 세션도 처음부터 빈 상태로 시작했으므로).

## 분류 정의

각 **MapActivity mint 이벤트**(`is_new_page=True` 이고 해당 이벤트의
`activity_name`에 `"MapActivity"`가 포함된 경우, **세션의 첫 페이지는 제외**
— 레지스트리가 비어 있어 BM25 조회 자체가 없으므로 분류 대상이 아님)마다,
mint 직전 matcher 상태에서 BM25 top-5를 재현하여 다음 세 범주 중 하나로
분류한다:

- **retrieval-miss**: BM25 top-5 후보 중 (이 리플레이에서 지금까지 mint된)
  기존 MapActivity 페이지가 **단 하나도 없음**.
- **element-blocked**: top-5 중 MapActivity 후보가 있으나, **그 전부**가
  element criterion(`|A△B| < element_diff_max=5`)을 실패. 관측된 최소
  `element_diff_count` 값을 기록한다.
- **pixel-blocked**: element criterion을 통과한 MapActivity 후보가
  **있으나**, **그 전부**가 pixel gate(`min luminance-diff fraction <
  page_pixel_diff_threshold=0.3`)를 실패. 관측된 최소 luminance-diff
  fraction 값을 기록한다.

세 범주는 상호배타적이고 전수를 포괄한다(정의상 위 셋 중 정확히 하나에
해당). BM25 쿼리 자체가 비어 있던 경우(빈 element-line 문서)는
`retrieval-miss`로 취급하고 그 사실을 별도로 표시한다.

## 판정 규칙

- **두 arm 모두**에서 한 범주가 strict majority(해당 arm의 분류 대상 mint
  이벤트 수의 >50%)를 차지하면 → 그 범주를 원인으로 판정.
  - verdict: `confirmed`
  - scope: `single-case` (osmand 한 앱, 두 run — cross-app 일반화 아님)
- 어느 한 arm이라도 majority가 없거나, 두 arm의 majority 범주가 서로
  다르면 → verdict: `needs-user-input`, 두 arm의 분포 전체를 보고한다.

## 제약 재확인

- `screen_matcher.py` 등 `screen_matching/` 산출 코드는 수정하지 않는다
  (읽기 전용 도입 — 신규 리플레이 모듈만 추가).
- 아카이브 데이터는 읽기 전용. `data/`/`runtime/` 라이브 디렉토리 미접촉.
- 결과가 가설(픽셀 게이트가 원인일 것이라는 사전 추측)과 다르면 가설을
  버린다. 확증이 목표가 아니다.

---

# U2-fix 검증 사전등록 — canvas-gated text-blind element criterion (2026-07-14)

리플레이 결과를 보기 **전에** 작성. 아래 게이트·판정 규칙은 데이터를 본 뒤
변경하지 않는다. 게이트가 실패하면 실패로 보고한다 — 숫자를 맞추기 위해
문턱·탐지기·게이팅 규칙을 사후 조정하지 않는다.

## 진단 요약 (위 S-9 측정의 결과 — 이 수정의 전제)

MapActivity page mint 이벤트의 원인은 **element-blocked**로 판정됐다
(armA 43/45 = 95.6%, armB 22/26 = 84.6%; verdict `confirmed`, scope
`single-case`). 뷰포트 상태 텍스트(축척 `100 ft`/`200 ft`, 거리 `0.25`/`5666 mi`,
주소, 상태별 aria-label)가 element-line identity를 오염시킨다.
element-blocked 건의 `min_element_diff_count`는 armA median 13(max 41),
armB median 15(max 45) — 문턱(`element_diff_max=5`) 상향으로는 고칠 수 없다.

## 수정안 (측정 전 확정)

**canvas-gated text-blind element criterion.**

1. **canvas 탐지기** `is_canvas_screen(raw_xml, min_area_frac=0.7)`: raw XML의
   노드 중 (a) 자식 없는 leaf, (b) `clickable="true"` 또는
   `long-clickable="true"`, (c) bounds 면적 ≥ `min_area_frac` × 화면 면적
   (= 노드 bounds 면적의 최댓값) 인 노드가 하나라도 있으면 canvas 화면.
   파싱 실패/빈 bounds → False. 클래스명(SurfaceView 등) 기반 탐지는 쓰지
   않는다 — osmand 지도 표면은 a11y 트리에서 leaf `android.view.View`다.
2. **blind 투영**: 직렬화 시점에 노드 **텍스트만** 빈 문자열로 만든 두 번째
   element-line 문서(`element_lines_blind`). `aria-label`을 포함한 식별
   속성은 전부 유지한다(구조 전용 투영은 카디널리티 붕괴로 사전 기각됨).
3. **both-sides 게이팅**: 현재 화면과 후보 페이지가 **둘 다** canvas일 때만
   (a) element criterion을 blind 집합으로 계산하고 (b) 픽셀 게이트를
   abstain한다. 그 외 모든 쌍은 기존 경로 그대로.
4. **문턱 불변**: `element_diff_max=5` / `element_jaccard_min=0.5` /
   `page_pixel_diff_threshold=0.3` / `screenshot_diff_threshold=0.02` 를 그대로
   재사용한다. 새 문턱 노브를 추가하지 않는다.
5. BM25 검색(corpus/top_k)·구조 prefilter·observation identity는 불변.

노브: `screen_matching.canvas_merge`(기본 true),
`screen_matching.canvas_min_area_frac`(기본 0.7). 리플레이 CLI:
`--canvas-merge {on,off}`.

## 검증 게이트 (순서대로, Gate 1은 hard-stop)

측정은 **오프라인 리플레이 전용**이다 — 실기기 재수집으로 검증하지 않는다.
입력: osmand는 `data_s1_f2_osmand_20260714/{armA_poke_off,armB_poke_on}/`,
통제 3앱은 `data/`+`runtime/`의 아카이브
(`com.simplemobiletools.musicplayer` 9 pages / `com.google.android.calendar`
28 / `com.flauschcode.broccoli` 4).

- **Gate 1 (OFF 충실도, hard-stop)**: `--canvas-merge off` 에서
  armA 전체 71 / map 46, armB 전체 67 / map 26 을 정확히 재현하고
  per-event 불일치가 0이어야 한다. 실패 시 Gate 2 이하를 보지 않고
  중단·보고한다 (하네스가 유죄).
- **Gate 2 (효과)**: `--canvas-merge on` 에서 **두 arm 모두** map page가
  **30% 이상 감소**해야 한다 — armA ≤ 32, armB ≤ 18 이 합격 하한.
  실측치를 그대로 보고한다(체리피킹 금지).
- **Gate 3 (osmand 비지도 회귀)**: ON에서 (라이브 mint였는데) 병합된
  **비-MapActivity 페이지의 전수 목록**을 보고한다. 허용되는 것은 그 페이지
  자신의 obs-0 raw.xml이 **canvas-detected인 경우뿐**이다(both-sides 규칙상
  구성적으로 보장돼야 한다). canvas-detected가 아닌 페이지가 하나라도 병합되면
  **FAIL**. 목록은 은닉 없이 전수 보고한다.
- **Gate 4 (통제 앱 불변)**: musicplayer / calendar / broccoli 를 ON·OFF로
  각각 리플레이하여 **ON 결정 ≡ OFF 결정 (event-for-event)** 이어야 한다.
  OFF가 라이브 페이지 수(9/28/4)와도 일치하면 병기한다. OFF부터 라이브와
  어긋나면(이 앱들에서 하네스가 미검증인 탓일 수 있다) 그 사실을 그대로
  보고하고 ON≡OFF 기준으로만 판정한다.

## 효과 주장의 상한

Gate 2가 통과해도 주장할 수 있는 것은 **"osmand 두 run의 오프라인 리플레이에서
지도 페이지 파편화가 줄었다"** 까지다. 수집 품질 개선·cross-app 일반화는
이 측정으로 주장하지 않는다 (scope: `single-case`).

---

# U2-repair 사전등록 — same-package 병합 가드 (2026-07-14)

리플레이 결과를 보기 **전에** 작성. 게이트가 실패하면 문턱·탐지기·게이트 정의를
사후 조정하지 않는다 — 실패를 그대로 보고한다.

## 결함 (U2-fix 검증 후 게이트 밖 감사에서 확정)

U2-fix의 Gate 1/2/4는 재실행으로 PASS 확인됐다. 그러나 **Gate 3의 기준이
불완전했다** — "병합된 비지도 페이지가 canvas-detected인가"만 물었고 **"같은
앱인가"를 묻지 않았다.** 전수 감사 결과:

```
armB ON: replay page 0  ← {nexuslauncher/NexusLauncherActivity,
                           net.osmand/DrawerLayout, net.osmand/MapActivity}
         replay page 26 ← {nexuslauncher/NexusLauncherActivity, net.osmand/Dialog}
```

**안드로이드 런처 홈 화면이 osmand 지도와 같은 page로 병합됐다.** page_graph의
엣지가 거짓이 되고, 다운스트림 학습은 "앱이 바뀌었는데 화면이 같다"를 배운다.

**이 오염 채널은 선재한다 — canvas fix가 도입한 것이 아니라 확장했다.** 수정 전
라이브 코퍼스에서도 관측된다: armB 라이브 page 0 = {launcher, MapActivity},
page 34 = {launcher, Dialog}; **broccoli 라이브** page 0(MainActivity)이 step
4·8의 launcher 이벤트를 흡수(비-canvas 경로, 텍스트가 살아 있는 상태로).
armA / musicplayer / calendar는 교차 0건. 기전: BM25 병합 경로는 activity/package를
**애초에 검사하지 않는다**(구조 prefilter만 `(activity, fp)` 키를 쓴다) — 텍스트가
사실상 유일한 방어선이었고, blind 투영이 그것을 걷어냈다.

## 수정안 (측정 전 확정)

**BM25 병합 경로 전체에 same-package 가드.**

1. **가드 층위 = package**(activity 아님). baseline이 같은 패키지의 window 라벨
   교차 병합(DrawerLayout↔MapActivity)을 설계상 허용하며,
   `domain/page_graph.py::_canonical_activity`의 "window 라벨은 identity가
   아니다" 선례와 일치한다. activity 가드는 baseline이 스스로 하는 병합을
   재파편화한다.
2. **가드 범위 = BM25 병합 전체**(canvas 쌍 한정 아님). 선재 결함이 armB와
   broccoli 라이브 **양쪽에서** 측정됐으므로, canvas 경로만 막는 것은 측정된
   오염 채널을 알면서 열어두는 것이다.
3. **knob `screen_matching.package_guard`, 기본 ON.** 사용자 선택지가 아니라
   리플레이 충실도 앵커(OFF = 수정 전 코드 재현)다.
4. **fail-open**: 어느 한쪽 package가 빈 문자열이면 가드 **abstain**(병합 허용).
   파싱 실패로 파편화를 만들지 않는다.
5. package = activity 문자열(`package/window.Class`)의 `/` 앞부분. 페이지 쪽은
   mint 시점 activity(`PageKnowledge.first_activity`, additive 저장 + legacy
   resume 폴백).
6. 문턱·canvas 탐지기·BM25 retrieval·구조 prefilter·observation identity 불변.
   새 문턱 노브 없음.

## 사전 예측 (결과를 보기 전에 기록)

- (a) armA `(canvas=on, guard=on)` = 현재 ON 결과와 동일(51 pages / 27 map) —
  armA의 ON 흡수 집합에 교차가 0건이라 가드가 막을 병합이 없다.
- (b) armA `(off, on)` = 라이브와 동일(71 / 46, 불일치 0).
- (c) armB `(off, on)`의 라이브 대비 차이는 **전부 launcher 병합 차단에 귀속
  가능**해야 한다.
- (d) broccoli `(off, on)` = steps 4·8 차단 + 하류 캐스케이드만 — 예상 5 pages.
- (e) musicplayer / calendar는 **모든 조합에서 불변**(단일 패키지 스트림).

## 게이트 (전부 오프라인 리플레이. 실기기 재수집 금지.)

- **R0 (충실도, hard-stop)**: `canvas=off, guard=off` → armA 71/46, armB 67/26,
  통제 3앱 9/28/4, per-event 불일치 전부 0. 가드 코드는 knob OFF에서 완전히
  불활성이어야 한다. 실패 시 **중단·보고**.
- **R1 (효과 유지)**: `canvas=on, guard=on` → armA map ≤ **32**, armB map ≤ **18**
  (U2-fix의 하한 그대로). 가드가 효과를 죽이면 안 된다.
- **R2 (교차 금지, hard fail)**: `canvas=on, guard=on` → **어떤 replay page도 2개
  이상의 서로 다른 package를 흡수하지 않는다** — osmand 2 arm + 통제 3앱 전부.
  흡수 activity 집합 전수 표를 첨부한다.
- **R3 (선재 결함 격리)**: `canvas=off, guard=on` → 가드 단독 효과. armA /
  musicplayer / calendar는 라이브와 동일해야 하고, armB / broccoli의 차이는
  **전부 교차 병합 차단에 귀속 가능**해야 한다(각 차이를 차단된 병합 쌍으로
  지목한다). **귀속 불가능한 차이 = FAIL.**
- **R4 (canvas 불활성, 통제)**: 통제 3앱에서 `(on, on)` ≡ `(off, on)`
  event-for-event.
- **pytest**: 851(U2-fix 기준선) + 신규 가드 테스트 전부 green.

## 효과 주장의 상한

**"osmand 두 run의 오프라인 리플레이에서 파편화가 줄었고, 측정된 교차-패키지
오염이 제거됐다"** 까지. 수집 품질 개선·cross-app 일반화는 주장하지 않는다.
U2-fix가 노출한 **같은-패키지 내부의 오병합**(서로 다른 MapActivity-hosted 설정
화면끼리)은 이 수정의 대상이 **아니며** 여전히 미해결이다.
