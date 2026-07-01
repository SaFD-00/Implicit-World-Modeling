"""Tests for monkey_collector.cli — CLI argument parsing."""

from unittest.mock import patch

import pytest


class TestRunArgsParsing:
    def test_defaults(self):
        """YAML-covered params default to None (sentinel for 'use config').

        Effective values come from monkey_collector.config (see test_config.py);
        argparse only records whether the user supplied a flag.
        """
        from monkey_collector.cli import main

        with patch(
            "sys.argv", ["monkey-collect", "run", "--apps", "all"]
        ), patch("monkey_collector.cli.cmd_run") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.apps == ["all"]
            assert args.steps is None
            assert args.seed is None
            assert args.port is None
            assert args.input_mode is None
            assert args.strategy is None
            assert args.config is None
            assert args.new_session is False
            assert args.force is False
            assert args.luminance_prefilter is None
            assert args.luminance_threshold is None
            assert args.screenshot_diff_threshold is None
            assert args.luminance_low_res_width is None

    def test_apps_required(self):
        from monkey_collector.cli import main

        with patch("sys.argv", ["monkey-collect", "run"]), \
                pytest.raises(SystemExit):
            main()

    def test_all_flags(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "run",
            "--apps", "com.test.app", "com.other.app",
            "--steps", "50",
            "--seed", "99",
            "--port", "54321",
            "--new-session",
            "--force",
            "--input-mode", "random",
        ]), patch("monkey_collector.cli.cmd_run") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.apps == ["com.test.app", "com.other.app"]
            assert args.steps == 50
            assert args.seed == 99
            assert args.port == 54321
            assert args.new_session is True
            assert args.force is True
            assert args.input_mode == "random"

    def test_strategy_flag(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "run", "--apps", "all", "--strategy", "DFS",
        ]), patch("monkey_collector.cli.cmd_run") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.strategy == "DFS"

    def test_luminance_flags(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "run", "--apps", "all",
            "--luminance-prefilter", "off",
            "--luminance-threshold", "25",
            "--screenshot-diff-threshold", "0.05",
            "--luminance-low-res-width", "64",
        ]), patch("monkey_collector.cli.cmd_run") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.luminance_prefilter == "off"
            assert args.luminance_threshold == 25
            assert args.screenshot_diff_threshold == 0.05
            assert args.luminance_low_res_width == 64

    def test_config_flag(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "run", "--apps", "all", "--config", "/tmp/x.yaml",
        ]), patch("monkey_collector.cli.cmd_run") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.config == "/tmp/x.yaml"

class TestConvertArgsParsing:
    def test_required_args(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "convert",
            "--package", "com.test.app",
            "--output", "/data/output.jsonl",
            "--images-dir", "/data/images",
        ]), patch("monkey_collector.cli.cmd_convert") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "data"
            assert args.runtime_dir == "runtime"
            assert args.package == "com.test.app"
            assert args.output == "/data/output.jsonl"
            assert args.images_dir == "/data/images"

    def test_with_label(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "convert",
            "--package", "com.test.app",
            "--output", "/out.jsonl",
            "--images-dir", "/img",
            "--label", "5",
        ]), patch("monkey_collector.cli.cmd_convert") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.label == 5

    def test_custom_roots(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "convert",
            "--data-dir", "/other/data",
            "--runtime-dir", "/other/runtime",
            "--package", "com.test.app",
            "--output", "/out.jsonl",
            "--images-dir", "/img",
        ]), patch("monkey_collector.cli.cmd_convert") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "/other/data"
            assert args.runtime_dir == "/other/runtime"


class TestConvertAllArgsParsing:
    def test_required_args(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "convert-all",
            "--data-dir", "/data",
            "--runtime-dir", "/runtime",
            "--output", "/data/output.jsonl",
            "--images-dir", "/data/images",
        ]), patch("monkey_collector.cli.cmd_convert_all") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "/data"
            assert args.runtime_dir == "/runtime"
            assert args.output == "/data/output.jsonl"
            assert args.images_dir == "/data/images"

    def test_defaults(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "convert-all",
            "--output", "/data/output.jsonl",
            "--images-dir", "/data/images",
        ]), patch("monkey_collector.cli.cmd_convert_all") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "data"
            assert args.runtime_dir == "runtime"


class TestNoCommand:
    def test_no_command_exits(self):
        """No command -> SystemExit(1)."""
        from monkey_collector.cli import main

        with patch("sys.argv", ["monkey-collect"]), pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestRunArgsDefaults:
    def test_default_output(self):
        """data-dir/runtime-dir/delay default to None sentinel; config supplies the value."""
        from monkey_collector.cli import main

        with patch(
            "sys.argv", ["monkey-collect", "run", "--apps", "all"]
        ), patch("monkey_collector.cli.cmd_run") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir is None
            assert args.runtime_dir is None
            assert args.delay is None


class TestSyncInstalledArgsParsing:
    def test_defaults(self):
        from monkey_collector.cli import main

        with patch(
            "sys.argv", ["monkey-collect", "sync-installed"]
        ), patch("monkey_collector.cli.cmd_sync_installed") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.apps_csv == "catalog/apps.csv"

    def test_custom_apps_csv(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "sync-installed",
            "--apps-csv", "/tmp/apps.csv",
        ]), patch("monkey_collector.cli.cmd_sync_installed") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.apps_csv == "/tmp/apps.csv"

class TestResetArgsParsing:
    def test_all_flag(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "reset", "--all", "--yes",
        ]), patch("monkey_collector.cli.cmd_reset") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.all is True
            assert args.yes is True
            assert args.dry_run is False
            assert args.data_dir == "data"
            assert args.runtime_dir == "runtime"
            assert args.apps is None

    def test_packages(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "reset",
            "--apps", "com.a,com.b",
            "--data-dir", "/tmp/data",
            "--runtime-dir", "/tmp/runtime",
            "--dry-run",
        ]), patch("monkey_collector.cli.cmd_reset") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.apps == "com.a,com.b"
            assert args.data_dir == "/tmp/data"
            assert args.runtime_dir == "/tmp/runtime"
            assert args.dry_run is True


class TestPageMapArgsParsing:
    def test_required_args(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "page-map",
            "--package", "com.test.app",
        ]), patch("monkey_collector.cli.cmd_page_map") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "data"
            assert args.runtime_dir == "runtime"
            assert args.package == "com.test.app"
            assert args.threshold == 0.85
            assert args.output is None
            assert args.no_open is False

    def test_threshold_and_no_open(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "page-map",
            "--package", "com.test.app",
            "--threshold", "0.5",
            "--output", "/tmp/graph.html",
            "--no-open",
        ]), patch("monkey_collector.cli.cmd_page_map") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.threshold == 0.5
            assert args.output == "/tmp/graph.html"
            assert args.no_open is True


class TestPageMapAllArgsParsing:
    def test_defaults(self):
        from monkey_collector.cli import main

        with patch("sys.argv", ["monkey-collect", "page-map-all"]), \
                patch("monkey_collector.cli.cmd_page_map_all") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "data"
            assert args.runtime_dir == "runtime"
            assert args.threshold == 0.85
            assert args.no_open is False


class TestRegenerateArgsParsing:
    def test_defaults(self):
        from monkey_collector.cli import main

        with patch("sys.argv", ["monkey-collect", "regenerate"]), \
                patch("monkey_collector.cli.cmd_regenerate") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "data"

    def test_custom_data_dir(self):
        from monkey_collector.cli import main

        with patch("sys.argv", [
            "monkey-collect", "regenerate",
            "--data-dir", "/data/other",
        ]), patch("monkey_collector.cli.cmd_regenerate") as mock_cmd:
            main()
            args = mock_cmd.call_args[0][0]
            assert args.data_dir == "/data/other"
