"""Regression test: a stale finish from the previous session must not end a
freshly-launched session at step 0.

run_queue reuses one server/client connection across apps, so a trailing
"finish" from the prior session's teardown can linger in the signal queue.
run_collection_loop must drain the queue before its first read.
"""

from unittest.mock import MagicMock

from monkey_collector.pipeline.collection_loop import CollectionState, run_collection_loop


class TestStaleSignalClearedAtStart:
    def test_queue_cleared_before_first_read(self):
        calls = []
        server = MagicMock()
        server.clear_signal_queue.side_effect = lambda: calls.append("clear")

        def get_sig(timeout):
            calls.append("get")
            return ("finish", None, None)

        server.get_latest_signal.side_effect = get_sig

        collector = MagicMock()
        collector.server = server
        collector.xml_timeout = 1

        state = CollectionState(step=0, max_step=5)
        run_collection_loop(collector, state, "com.test.app")

        # The clear must run before any signal is consumed, so a stale finish
        # sitting in the real queue is discarded instead of ending step 0.
        assert calls and calls[0] == "clear"
        assert "get" in calls
