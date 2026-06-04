"""Tests for page_graph matching robustness: activity normalization,
transient-overlay stripping, and missing-activity cross matching."""

from monkey_collector.domain.page_graph import (
    PageGraph,
    _canonical_activity,
    compute_xml_fingerprint,
)

_BASE_XML = """<hierarchy>
  <node class="android.widget.FrameLayout" bounds="[0,0][1080,1920]">
    <node class="android.widget.Button" resource-id="com.app:id/ok" text="OK" clickable="true" bounds="[0,0][100,100]"/>
    <node class="android.widget.TextView" resource-id="com.app:id/title" text="Title" bounds="[0,200][500,300]"/>
  </node>
</hierarchy>"""

# Same page with a transient snackbar overlay on top.
_WITH_SNACKBAR = """<hierarchy>
  <node class="android.widget.FrameLayout" bounds="[0,0][1080,1920]">
    <node class="android.widget.Button" resource-id="com.app:id/ok" text="OK" clickable="true" bounds="[0,0][100,100]"/>
    <node class="android.widget.TextView" resource-id="com.app:id/title" text="Title" bounds="[0,200][500,300]"/>
    <node class="android.widget.TextView" resource-id="com.app:id/snackbar_text" text="Saved!" bounds="[0,1800][1080,1900]"/>
  </node>
</hierarchy>"""


class TestCanonicalActivity:
    def test_empty(self):
        assert _canonical_activity("") == ""
        assert _canonical_activity("   ") == ""

    def test_keyboard_collapses(self):
        assert _canonical_activity("com.x/...SoftInputWindow") == ""

    def test_real_activity_kept(self):
        assert _canonical_activity("com.app/.MainActivity") == "com.app/.MainActivity"


class TestTransientStripping:
    def test_snackbar_does_not_change_fingerprint(self):
        assert compute_xml_fingerprint(_BASE_XML) == compute_xml_fingerprint(_WITH_SNACKBAR)

    def test_snackbar_matched_to_same_page(self):
        g = PageGraph()
        p0 = g.get_or_create_page("com.app/.MainActivity", _BASE_XML, 0)
        p1 = g.get_or_create_page("com.app/.MainActivity", _WITH_SNACKBAR, 1)
        assert p0 == p1
        assert len(g.nodes) == 1


class TestCrossActivityMatching:
    def test_missing_then_known_merges(self):
        g = PageGraph()
        p0 = g.get_or_create_page("", _BASE_XML, 0)
        p1 = g.get_or_create_page("com.app/.MainActivity", _BASE_XML, 1)
        assert p0 == p1
        assert len(g.nodes) == 1
        # stored label is upgraded once the real activity is observed
        assert g.nodes[0].activity == "com.app/.MainActivity"

    def test_known_then_missing_merges(self):
        g = PageGraph()
        p0 = g.get_or_create_page("com.app/.MainActivity", _BASE_XML, 0)
        p1 = g.get_or_create_page("", _BASE_XML, 1)
        assert p0 == p1
        assert len(g.nodes) == 1

    def test_keyboard_label_merges_with_base(self):
        g = PageGraph()
        p0 = g.get_or_create_page("com.app/.MainActivity", _BASE_XML, 0)
        p1 = g.get_or_create_page("com.x/...SoftInputWindow", _BASE_XML, 1)
        assert p0 == p1

    def test_distinct_known_activities_not_merged(self):
        g = PageGraph()
        p0 = g.get_or_create_page("com.app/.ActivityA", _BASE_XML, 0)
        p1 = g.get_or_create_page("com.app/.ActivityB", _BASE_XML, 1)
        assert p0 != p1
        assert len(g.nodes) == 2
