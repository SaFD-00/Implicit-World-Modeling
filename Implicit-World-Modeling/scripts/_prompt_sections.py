"""GT/eval 프롬프트를 섹션 dict 로 분해하는 공용 파서.

왜 별도 모듈인가
----------------
프롬프트는 두 계열이고, **한 계열만 보는 파서는 다른 계열에서 조용히 빈 문자열을
돌려준다.** 그 실패 모드가 2026-07-30 의 woa 사고였다 — `--exclude-action` 이 두
계열의 마커·키를 모두 놓쳐 전 실험에서 단 한 행도 거르지 못했는데, 산출물은 정상적인
모양으로 남아서 아무도 알아채지 못했다.

그래서 파서는 **한 벌만 존재해야 한다.** `_compare_site`(정성 비교 사이트)와
`_state_diff_eval`(copy-bias 진단 채점기)이 각자 복제본을 들고 있으면 계열이 하나 더
생겼을 때 한쪽만 고쳐지고 다른 쪽이 조용히 틀린다. 이 모듈이 그 한 벌이다.

의도적으로 의존성이 없다 (표준 라이브러리 `re` 뿐). 정본 채점 경로가 이 파서를
쓰므로, bs4·scipy·sacrebleu 같은 선택적 의존성이 여기 들어오면 그 의존성이 빠진
환경에서 채점 자체가 죽는다.
"""

from __future__ import annotations

import re

# GT 프롬프트는 두 계열이다 (2026-08-01 전수 실측):
#   A) EXP01~04 : '## Current State' / '## Next State' / '## Action'
#   B) EXP05~07 : 'Current UI State:' / 'Next UI State:' / 'Action:' /
#                 'Task Instruction:' / 'Action History:'
SECTION_MARKERS: list[tuple[str, str]] = [
    ("current_state", r"^##\s*Current State\s*$"),
    ("current_state", r"^Current UI State:\s*$"),
    ("next_state", r"^##\s*Next State\s*$"),
    ("next_state", r"^Next UI State:\s*$"),
    ("action", r"^##\s*Action\s*$"),
    ("action", r"^Action:\s*$"),
    ("instruction", r"^Task Instruction:\s*$"),
    ("history", r"^Action History:\s*$"),
]
_MARKER_RE = re.compile(
    "|".join(f"(?P<g{i}>{pat})" for i, (_, pat) in enumerate(SECTION_MARKERS)),
    re.MULTILINE,
)
_SCREENSHOT_LINE_RE = re.compile(r"^\[[^\]\n]*Screenshot\]\s*$", re.MULTILINE)
_ROLE_USER_RE = re.compile(r"^user\s*$", re.MULTILINE)
_ROLE_ASSISTANT_RE = re.compile(r"^assistant\s*$", re.MULTILINE)

# 필수 섹션 — 없으면 파싱 실패로 집계한다.
REQUIRED_SECTIONS = {
    "state": ("current_state", "action"),
    "action": ("current_state", "next_state"),
    "stage2": ("current_state", "instruction"),
}


def parse_prompt(prompt: str) -> dict[str, str]:
    """prompt 문자열의 user 파트를 섹션 dict 로 분해한다. 실패 시 키가 비어 있다.

    입력은 두 가지다. 둘 다 받는다:
      - 렌더된 chat prompt (`generated_predictions*.jsonl` 의 `prompt`) — role 줄 있음
      - GT 원문 (`test.jsonl` 의 `messages[1]["value"]`) — role 줄 없음
    """
    body = prompt
    m_user = _ROLE_USER_RE.search(body)
    if m_user is not None:
        body = body[m_user.end() :]
    for m_asst in reversed(list(_ROLE_ASSISTANT_RE.finditer(body))):
        body = body[: m_asst.start()]
        break

    hits = []
    for m in _MARKER_RE.finditer(body):
        idx = int(m.lastgroup[1:])
        hits.append((m.start(), m.end(), SECTION_MARKERS[idx][0]))

    out: dict[str, str] = {}
    for i, (_, end, name) in enumerate(hits):
        stop = hits[i + 1][0] if i + 1 < len(hits) else len(body)
        chunk = _SCREENSHOT_LINE_RE.sub("", body[end:stop]).strip()
        if chunk and name not in out:
            out[name] = chunk
    return out
