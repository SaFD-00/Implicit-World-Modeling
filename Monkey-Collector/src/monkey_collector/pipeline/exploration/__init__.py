"""LLM-guided exploration engine.

Replaced a legacy weighted-random explorer with an engine ported from the
reference ``LLM-Explorer``: coverage-driven unexplored-first selection,
LLM-derived same-function element compression, and shortest-path navigation to
unexplored screens over a UI transition graph.

Public surface:
    - ``Explorer``: the Protocol the collection loop depends on.
    - ``LLMGuidedExplorer``: the engine implementation.
"""

from __future__ import annotations

from monkey_collector.pipeline.exploration.explorer import Explorer, LLMGuidedExplorer

__all__ = ["Explorer", "LLMGuidedExplorer"]
