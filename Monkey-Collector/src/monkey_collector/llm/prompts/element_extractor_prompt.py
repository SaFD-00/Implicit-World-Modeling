"""Single-call element-extraction prompt.

Merges MobileGPT-V2 Node-Clustering's TWO prompts into one. The reference
extracts in two LLM calls — ``SubtaskExtractorAgent`` proposes the same-function
family (every member index, its ``trigger_ui_index``) and a separate
``TriggerUIAgent`` then picks the representative anchor(s). Here a single call
emits BOTH per element:

* ``element_index``     — the FULL same-function family (every member index),
                          the V2 ``trigger_ui_index`` candidate set.
* ``key_element_index`` — the 1-3 representative anchor(s), the V2
                          ``TriggerUIAgent`` selection.

The system prompt fuses V2's extraction rules (same-function grouping, canonical
verb prefixes, KNOWN-exclusion / always-on-extract semantics) with V2's
representative-selection criteria (stability, lowest-index preference). Indices
are the encoded ``index`` attribute, which aligns 1:1 with
``SemanticElement.index`` (same parse → renumber pipeline).
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a smartphone assistant that analyzes mobile app screens.

Given the HTML of one app screen (delimited by <screen></screen>), list the high-level functions ("elements") a user can perform on it. Each interactable element is tagged <button> or <input> and carries an integer `index` attribute.

A 'KNOWN ELEMENTS' list MAY be provided:
- Non-empty: those goals are already covered elsewhere. Return ONLY elements NOT in that list. Return {"elements": []} if nothing new remains — do not invent elements.
- Empty: list ALL elements visible on the screen.

For each element output:
1. name — a GENERAL action name, not specific to this screen ('call_contact', not 'call_Bob'). Prefer canonical verb prefixes: open_ / close_ / view_ / edit_ / save_ / cancel_ / delete_ / send_ / select_ / toggle_ / navigate_ / scroll_ / search_.
2. description — what the action does.
3. parameters — info needed to execute it, phrased as questions ({"contact_name": "Which contact?"}).
4. element_index — EVERY `index` of the interactable element(s) in this element's SAME-FUNCTION FAMILY. When many controls invoke the same handler / open the same screen-template differing only by content (list rows, grid tiles, an icon-launcher grid), this is ONE element listing ALL their indices. Otherwise it is the one (or few) indices that start the action.
5. key_element_index — the 1-3 REPRESENTATIVE anchor index(es) chosen FROM element_index: the most stable, structurally-first entry point(s). Prefer the LOWEST index; prefer elements with a clear label (aria-label/text). The matcher statically covers the rest of the family from these anchors, so do NOT enumerate every family member here — pick the representative(s) only. key_element_index MUST be a subset of element_index.

***Rules***
- Merge related actions into one higher-level goal ('input_name' + 'input_email' -> 'fill_in_info'). NEVER merge two triggers that open DIFFERENT screens — those are separate elements.
- SAME-FUNCTION GROUP: rows/tiles that open the SAME screen-template (same handler, only content differs) are ONE element with all indices in element_index and the content as a parameter. Reserve per-row elements for rows whose destinations are structurally DIFFERENT (e.g. Settings -> Display vs Sound).
- When a control re-orders / filters / switches the view-mode of the SAME screen, emit ONE element with the choice as a parameter (select_sort_order {order: by_title | by_date}), never one per option.
- Capture navigation triggers (hamburger/drawer, overflow menus, FABs, settings/profile icons, tabs, list rows that open another page) and secondary/meta items (License, FAQ, Privacy, Version) — each opens its own screen.
- Do NOT extract completion STEPS handled by the explorer loop: dialog/bottom-sheet confirm/cancel/dismiss buttons, internal widget adjustments (color/date/time picker, slider), or individual fields of one form (already merged).
- The HTML may be a partial/masked view (UIs already covered may be stripped). Extract only what is present; do not speculate about hidden parts.

Response Format (JSON, no markdown, no prose):
{"elements": [
  {
    "name": "<action_name>",
    "description": "<what this action does>",
    "parameters": {"<param_name>": "<question>"},
    "element_index": [<index>, ...],
    "key_element_index": [<index>, ...]
  }
]}"""


def get_user_prompt(encoded_xml: str, known_names: list[str] | None) -> str:
    known_str = "\n".join(f"- {n}" for n in known_names) if known_names else "(none)"
    return f"""KNOWN ELEMENTS already covered elsewhere (do NOT re-extract or paraphrase):
{known_str}

HTML code of the mobile app screen:
<screen>{encoded_xml}</screen>

Extract all elements from the screen, EXCLUDING any whose goal matches a known element.
Return {{"elements": []}} if there is no new element. Do NOT invent elements to fill the response.
Response:"""


def get_messages(encoded_xml: str, known_names: list[str] | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": get_user_prompt(encoded_xml, known_names)},
    ]
