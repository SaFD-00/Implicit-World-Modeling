"""Tests for time/steps session-budget enforcement in run_collection_loop.

Covers:
  - `_has_budget` pure-function unit tests (time mode, steps mode).
  - Loop deadline termination end-to-end (time mode) via an injected clock.
"""

from unittest.mock import MagicMock

from monkey_collector.pipeline.collection_loop import (
    CollectionState,
    _has_budget,
    run_collection_loop,
)


class TestHasBudgetPure:
    def test_time_mode_before_deadline(self):
        state = CollectionState(budget_mode="time")
        assert _has_budget(state, clock=lambda: 5.0, deadline=10.0) is True

    def test_time_mode_after_deadline(self):
        state = CollectionState(budget_mode="time")
        assert _has_budget(state, clock=lambda: 15.0, deadline=10.0) is False

    def test_steps_mode_before_max(self):
        state = CollectionState(step=2, max_step=5, budget_mode="steps")
        assert _has_budget(state, clock=lambda: 0.0, deadline=None) is True

    def test_steps_mode_at_max(self):
        state = CollectionState(step=5, max_step=5, budget_mode="steps")
        assert _has_budget(state, clock=lambda: 0.0, deadline=None) is False


class TestLoopDeadlineTermination:
    def test_time_budget_expires_loop_returns(self):
        """A time-budgeted session ends via the deadline, not the idle backstop.

        `now` is a 3-value iterator: [0 (deadline calc), 0 (1st while-check,
        still < 10), 100 (2nd while-check, past 10)] — the deadline (10s)
        expires on the second while-evaluation, after one benign no_change
        iteration.
        """
        clock_values = iter([0, 0, 100])

        def fake_now():
            return next(clock_values)

        server = MagicMock()
        server.get_latest_signal.return_value = ("no_change", None, None)

        collector = MagicMock()
        collector.server = server
        collector.xml_timeout = 1
        collector.action_delay = 0

        state = CollectionState(
            step=0, max_step=1000, budget_mode="time", max_duration_sec=10,
        )

        run_collection_loop(collector, state, "com.test.app", now=fake_now)

        # Deadline termination, not the idle backstop: get_latest_signal was
        # called far fewer times than max_idle (max(1000*4, 20) = 4000).
        assert server.get_latest_signal.call_count < 4000
