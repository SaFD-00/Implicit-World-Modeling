"""Tunable constants for the LLM-guided exploration engine.

Ported from the reference ``LLM-Explorer`` (``input_policy3.py`` module-level
constants). Names and values are preserved so the exploration behaviour matches
the original policy; see each constant for its role in the engine.

The *traversal strategy* (DFS | BFS | GREEDY) is configured separately, in
``config/run.yaml`` (``exploration.strategy``), the ``MC_EXPLORATION_STRATEGY``
env var, or the ``--strategy`` CLI flag — see ``monkey_collector.config``. The
constants here are engine internals shared by all three strategies.
"""

from __future__ import annotations

# Maximum number of steps a single navigation plan may run before it is
# abandoned (guards against a navigation queue that never reaches its target).
MAX_NAVIGATE_NUM_AT_ONE_TIME: int = 10
