"""Tunable constants for the LLM-guided exploration engine.

Ported from the reference ``LLM-Explorer`` (``input_policy3.py`` module-level
constants). Names and values are preserved so the exploration behaviour matches
the original policy; see each constant for its role in the engine.
"""

from __future__ import annotations

# Probability of taking a purely random action instead of the unexplored-first
# / navigation strategy. The reference keeps this at 0.0 — exploration is driven
# entirely by the unexplored-action frontier, not chance.
RANDOM_EXPLORE_PROB: float = 0.0

# Two states are treated as the same abstract page when their interactable
# element sets differ by at most this many element signatures.
MAX_NUM_DIFF_ELEMENTS_IN_SIMILAR_STATES: int = 2

# A same-function element group is only applied for coverage compression when
# the screen has at least this many interactable elements (mirrors the
# reference's guard against grouping trivially small screens).
MIN_SIZE_SAME_FUNCTION_ELEMENT_GROUP: int = 5

# After this many transitions are recorded for one (element, action) pair with a
# consistent effect, further repetitions of structurally-similar actions are
# skipped as redundant.
SKIP_SIMILAR_ACTION_THRESHOLD: int = 4

# Maximum number of consecutive steps spent on the same screen frame before the
# engine forces a back press to escape a stuck page.
MAX_EXPLORE_CURRENT_STATE_TIME: int = 10

# Maximum number of steps a single navigation plan may run before it is
# abandoned (guards against a navigation queue that never reaches its target).
MAX_NAVIGATE_NUM_AT_ONE_TIME: int = 10

# Maximum number of steps without discovering a new activity before the engine
# considers the app exhausted.
MAX_EXPLORED_ACTIVITIES_NOT_INCREASE_TIME: int = 100
