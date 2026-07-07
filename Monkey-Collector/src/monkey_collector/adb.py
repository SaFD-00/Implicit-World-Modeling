"""ADB command wrapper using subprocess."""

import os
import re
import shutil
import subprocess
import time

from loguru import logger

# Monkey-Collector is locked to this AVD. `AdbClient` resolves its emulator
# serial at construction time and prefixes every adb invocation with
# `-s <serial>`, so other emulators or real devices attached at the same
# time are ignored.
REQUIRED_AVD_NAME = "Pixel6-2"

# Characters that need escaping for adb shell input text
_SPECIAL_CHARS = re.compile(r'([\\\"\'`\s&|;<>()$!~{}*?#])')


def _find_adb() -> str:
    """Locate the adb binary, checking PATH then common SDK locations."""
    found = shutil.which("adb")
    if found:
        return found
    android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if android_home:
        candidate = os.path.join(android_home, "platform-tools", "adb")
        if os.path.isfile(candidate):
            return candidate
    # macOS default location
    default = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
    if os.path.isfile(default):
        return default
    return "adb"  # fallback, will raise FileNotFoundError if missing


def _escape_text_for_adb(text: str) -> str:
    """Escape text for safe use with ``adb shell input text``."""
    text = text.replace(" ", "%s")
    text = _SPECIAL_CHARS.sub(r'\\\1', text)
    return text


def _list_emulator_serials(adb_path: str) -> list[str]:
    """Return serials of online emulators (e.g. ``['emulator-5554']``)."""
    result = subprocess.run(
        [adb_path, "devices"], capture_output=True, text=True, timeout=10
    )
    serials: list[str] = []
    for line in result.stdout.splitlines()[1:]:  # skip "List of devices attached"
        parts = line.strip().split()
        if len(parts) >= 2 and parts[1] == "device" and parts[0].startswith("emulator-"):
            serials.append(parts[0])
    return serials


def _resolve_avd_serial(adb_path: str, avd_name: str) -> str:
    """Return the emulator serial whose AVD name matches *avd_name*.

    Queries each online emulator with ``adb -s <serial> emu avd name`` and
    returns the first match. Raises ``RuntimeError`` with a user-friendly
    message if no match is found.
    """
    serials = _list_emulator_serials(adb_path)
    for serial in serials:
        result = subprocess.run(
            [adb_path, "-s", serial, "emu", "avd", "name"],
            capture_output=True, text=True, timeout=10,
        )
        # `emu avd name` prints the AVD name on the first line, then "OK".
        first_line = (result.stdout or "").strip().splitlines()[0:1]
        name = first_line[0].strip() if first_line else ""
        if name == avd_name:
            return serial
    raise RuntimeError(
        f"AVD '{avd_name}' is not running. "
        f"Start it with: emulator -avd {avd_name}\n"
        f"Currently online emulators: {serials or 'none'}"
    )


class AdbClient:
    """Wrapper for ADB shell commands, locked to a single AVD.

    On construction, resolves the emulator serial of ``REQUIRED_AVD_NAME``
    (``Pixel6-2``) and prefixes every adb invocation with
    ``-s <serial>``. If the AVD is not currently running, construction
    fails with a ``RuntimeError``.
    """

    def __init__(self):
        self._adb = _find_adb()
        self._serial = _resolve_avd_serial(self._adb, REQUIRED_AVD_NAME)
        logger.info(
            f"AdbClient bound to AVD '{REQUIRED_AVD_NAME}' (serial={self._serial})"
        )

    def _cmd_prefix(self) -> list[str]:
        return [self._adb, "-s", self._serial]

    def shell(self, command: str, timeout: int | None = None) -> str:
        """Run an ADB shell command and return stdout."""
        cmd = self._cmd_prefix() + ["shell", command]
        logger.debug(f"ADB: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0 and result.stderr:
            logger.warning(f"ADB stderr: {result.stderr.strip()}")
        return result.stdout.strip()

    def launch_app(self, package: str) -> str:
        """Launch an app's main launcher activity via am start."""
        resolve_output = self.shell(
            f"cmd package resolve-activity --brief "
            f"-a android.intent.action.MAIN "
            f"-c android.intent.category.LAUNCHER {package}"
        )
        for line in reversed(resolve_output.strip().split("\n")):
            line = line.strip()
            if "/" in line:
                # Escape '$' (Java inner-class separator) so the
                # device shell doesn't treat it as a variable.
                line = line.replace("$", "\\$")
                return self.shell(f"am start -n {line}")
        # Fallback: let Android resolve the intent (no random events)
        return self.shell(
            f"am start -a android.intent.action.MAIN "
            f"-c android.intent.category.LAUNCHER {package}"
        )

    def force_stop(self, package: str) -> str:
        """Force stop an app."""
        return self.shell(f"am force-stop {package}")

    def get_device_resolution(self) -> tuple[int, int]:
        """Get device screen resolution."""
        output = self.shell("wm size")
        size_str = output.split(":")[-1].strip()
        w, h = size_str.split("x")
        return int(w), int(h)

    def press_back(self) -> str:
        """Press the back button."""
        return self.shell("input keyevent KEYCODE_BACK")

    def press_home(self) -> str:
        """Press the home button."""
        return self.shell("input keyevent KEYCODE_HOME")

    def tap(self, x: int, y: int) -> str:
        """Tap at the given (x, y) coordinates."""
        return self.shell(f"input tap {x} {y}")

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300
    ) -> str:
        """Perform a swipe gesture."""
        return self.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, text: str) -> str:
        """Type text into the currently focused input field."""
        if not text:
            return ""
        escaped = _escape_text_for_adb(text)
        return self.shell(f"input text {escaped}")

    def clear_text_field(self) -> str:
        """Select all text in the focused field and delete it."""
        self.shell("input keyevent KEYCODE_MOVE_END")
        self.shell("input keycombination 113 29")  # Ctrl+A
        return self.shell("input keyevent KEYCODE_DEL")

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> str:
        """Long-press at (x, y) via a zero-movement swipe."""
        return self.swipe(x, y, x, y, duration_ms)

    def install(self, apk_path: str) -> str:
        """Install an APK."""
        cmd = self._cmd_prefix() + ["install", "-r", apk_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip()

    def get_current_package(self) -> str:
        """Return the package name of the current foreground app."""
        # Field name varies by Android version: "mResumedActivity" (older) vs.
        # "topResumedActivity"/"ResumedActivity" (API 33+, e.g. Pixel6-2's
        # google_apis image). Grepping the "ResumedActivity" substring matches
        # all of them.
        output = self.shell(
            "dumpsys activity activities | grep ResumedActivity"
        )
        match = re.search(r'(\S+/\S+)', output)
        if match:
            activity = match.group(1).strip()
            if "/" in activity:
                return activity.split("/", 1)[0]
            return activity
        return ""

    def get_current_activity(self) -> str:
        """Return the full activity component name of the foreground activity.

        Returns a string like ``com.test.app/.MainActivity`` or empty string
        on failure.
        """
        output = self.shell(
            "dumpsys activity activities | grep ResumedActivity"
        )
        match = re.search(r'(\S+/\S+)', output)
        if match:
            return match.group(1).strip()
        return ""

    def get_declared_activities(self, package: str) -> list[str]:
        """Return all declared Activity component names for *package*.

        Parses the ``Packages:`` section from ``dumpsys package`` to get
        **all** manifest-declared activities.  Falls back to the Activity
        Resolver Table (intent-filter-only) when the Packages section
        yields no results.  Returns sorted list of component strings.
        """
        try:
            output = self.shell(f"dumpsys package {package}", timeout=15)
        except Exception as e:
            logger.warning(f"Failed to get declared activities: {e}")
            return []

        activities = self._parse_package_activities(output, package)
        if not activities:
            activities = self._parse_resolver_activities(output, package)

        return sorted(activities)

    # ------------------------------------------------------------------
    # Private helpers for dumpsys parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_package_activities(output: str, package: str) -> set[str]:
        """Parse activities from the ``Packages:`` section of dumpsys.

        This section lists **all** manifest-declared activities regardless
        of whether they have intent filters.
        """
        activities: set[str] = set()
        activity_pattern = re.compile(rf'({re.escape(package)}/\S+)')

        in_packages = False
        in_target_pkg = False
        in_activities = False
        activities_indent: int | None = None

        for line in output.splitlines():
            stripped = line.strip()

            if not in_packages:
                if stripped == "Packages:":
                    in_packages = True
                continue

            if not in_target_pkg:
                if f"Package [{package}]" in stripped:
                    in_target_pkg = True
                continue

            indent = len(line) - len(line.lstrip())

            if not in_activities:
                if stripped == "activities:":
                    in_activities = True
                    activities_indent = indent
                    continue
                # Another top-level package started — stop searching.
                if stripped.startswith("Package ["):
                    break
                continue

            # Inside activities: subsection.
            # A sibling or parent section at same/lower indent ends it.
            if stripped and indent <= activities_indent:
                break

            match = activity_pattern.search(stripped)
            if match:
                activities.add(match.group(1))

        return activities

    @staticmethod
    def _parse_resolver_activities(output: str, package: str) -> set[str]:
        """Parse activities from the Activity Resolver Table (fallback).

        Only returns activities that have intent filters registered.
        """
        activities: set[str] = set()
        in_activity_section = False
        activity_pattern = re.compile(rf'({re.escape(package)}/\S+)')

        for line in output.splitlines():
            stripped = line.strip()
            if "Activity Resolver Table:" in stripped:
                in_activity_section = True
                continue
            if in_activity_section and stripped.startswith(
                ("Receiver Resolver", "Service Resolver",
                 "Provider Resolver", "Permissions:")
            ):
                break
            if in_activity_section:
                match = activity_pattern.search(stripped)
                if match:
                    activities.add(match.group(1))

        return activities

    def wait_for_idle(self, timeout: float = 2.0) -> None:
        """Wait for the UI to settle after an action."""
        time.sleep(min(timeout, 1.0))
