"""Screen classification guards: keyboard, permission dialog, system screen.

Pure helpers that decide *what kind* of screen the collector is looking at,
using only the XML metadata (top_package, activity_name) and the parsed
``UITree``.  Used by the collection loop to dismiss keyboards, auto-handle
permission dialogs, and detect drift into system screens.
"""

from __future__ import annotations

from monkey_collector.xml.ui_tree import UIElement, UITree

# System / launcher packages that are never the target app.  A screen owned by
# one of these means we have drifted out of the app under collection.
SYSTEM_PACKAGES: frozenset[str] = frozenset({
    "",
    "android",
    "com.android.systemui",
    "com.google.android.permissioncontroller",
    "com.android.permissioncontroller",
    "com.android.shell",
    "com.android.packageinstaller",
    "com.google.android.packageinstaller",
})

# Packages that present a permission / install grant dialog we can act on.
_PERMISSION_PACKAGES: frozenset[str] = frozenset({
    "com.google.android.permissioncontroller",
    "com.android.permissioncontroller",
    "com.android.packageinstaller",
    "com.google.android.packageinstaller",
})

# Button labels for permission dialogs, ordered by preference.  We prefer
# *granting* so exploration can proceed into the gated screen; failing that we
# dismiss (deny / back).
PERMISSION_BUTTON_KEYWORDS: tuple[str, ...] = (
    "while using",
    "앱 사용 중에만",
    "사용 중에만",
    "allow",
    "허용",
    "ok",
    "확인",
    "yes",
    "deny",
    "거부",
    "don't allow",
    "cancel",
    "취소",
)


def is_keyboard(activity_name: str) -> bool:
    """Return True if the screen is the soft input (keyboard) window."""
    return "SoftInputWindow" in (activity_name or "")


def is_permission_dialog(top_package: str) -> bool:
    """Return True if the foreground package is a permission/install dialog."""
    return (top_package or "") in _PERMISSION_PACKAGES


def is_system_screen(top_package: str) -> bool:
    """Return True if the foreground package is a system/launcher screen."""
    return (top_package or "") in SYSTEM_PACKAGES


def find_dialog_button(
    ui_tree: UITree,
    keywords: tuple[str, ...] = PERMISSION_BUTTON_KEYWORDS,
) -> UIElement | None:
    """Find the best clickable button matching *keywords*, by preference order.

    Matches ``text`` or ``content_desc`` case-insensitively.  Keywords are
    tried in order, so the first keyword that matches any clickable element
    wins (grant before deny).  Returns None if nothing matches.
    """
    clickable = [e for e in ui_tree.get_clickable_elements()]
    for keyword in keywords:
        kw = keyword.lower()
        for elem in clickable:
            label = f"{elem.text} {elem.content_desc}".lower()
            if kw in label:
                return elem
    return None
