"""Tests for monkey_collector.pipeline.screen_guard — screen classification."""

from monkey_collector.pipeline.screen_guard import (
    LAUNCHER_PACKAGES,
    SYSTEM_PACKAGES,
    find_dialog_button,
    is_keyboard,
    is_launcher,
    is_permission_dialog,
    is_system_screen,
)
from tests.conftest import make_tree


class TestIsKeyboard:
    def test_soft_input_window(self):
        assert is_keyboard("com.app/android.inputmethodservice.SoftInputWindow")
        assert is_keyboard("SoftInputWindow")

    def test_normal_activity(self):
        assert not is_keyboard("com.test.app/.MainActivity")

    def test_empty(self):
        assert not is_keyboard("")


class TestIsPermissionDialog:
    def test_permissioncontroller(self):
        assert is_permission_dialog("com.google.android.permissioncontroller")
        assert is_permission_dialog("com.android.permissioncontroller")

    def test_package_installer(self):
        assert is_permission_dialog("com.android.packageinstaller")

    def test_target_app_is_not_permission(self):
        assert not is_permission_dialog("com.test.app")

    def test_empty(self):
        assert not is_permission_dialog("")


class TestIsSystemScreen:
    def test_systemui(self):
        assert is_system_screen("com.android.systemui")

    def test_android(self):
        assert is_system_screen("android")

    def test_empty_is_system(self):
        # empty package is treated as a system/unknown screen
        assert is_system_screen("")

    def test_target_app_is_not_system(self):
        assert not is_system_screen("com.test.app")


class TestIsLauncher:
    def test_nexuslauncher(self):
        assert is_launcher("com.google.android.apps.nexuslauncher")

    def test_launcher3(self):
        assert is_launcher("com.android.launcher3")

    def test_target_app_is_not_launcher(self):
        assert not is_launcher("com.test.app")

    def test_gms_is_not_launcher(self):
        # A gms/store surface is a system screen but NOT the launcher, so a
        # Back drift there must not be learned as a back-exit page.
        assert not is_launcher("com.google.android.gms")

    def test_empty_is_not_launcher(self):
        assert not is_launcher("")

    def test_launcher_packages_subset_of_system(self):
        # Invariant: every launcher package is also a system package, so a
        # launcher drift still counts as leaving the app.
        assert LAUNCHER_PACKAGES <= SYSTEM_PACKAGES


class TestFindDialogButton:
    def test_prefers_allow_over_deny(self):
        tree = make_tree([
            {"clickable": True, "text": "Deny", "bounds": (0, 0, 100, 100)},
            {"clickable": True, "text": "Allow", "bounds": (200, 0, 300, 100)},
        ])
        button = find_dialog_button(tree)
        assert button is not None
        assert button.text == "Allow"

    def test_while_using_wins_first(self):
        tree = make_tree([
            {"clickable": True, "text": "Allow", "bounds": (0, 0, 100, 100)},
            {"clickable": True, "text": "While using the app", "bounds": (200, 0, 400, 100)},
        ])
        button = find_dialog_button(tree)
        assert button is not None
        assert "While using" in button.text

    def test_matches_content_desc(self):
        tree = make_tree([
            {"clickable": True, "content_desc": "허용", "bounds": (0, 0, 100, 100)},
        ])
        button = find_dialog_button(tree)
        assert button is not None
        assert button.content_desc == "허용"

    def test_no_match_returns_none(self):
        tree = make_tree([
            {"clickable": True, "text": "Some other label", "bounds": (0, 0, 100, 100)},
        ])
        assert find_dialog_button(tree) is None

    def test_case_insensitive(self):
        tree = make_tree([
            {"clickable": True, "text": "ALLOW", "bounds": (0, 0, 100, 100)},
        ])
        button = find_dialog_button(tree)
        assert button is not None
