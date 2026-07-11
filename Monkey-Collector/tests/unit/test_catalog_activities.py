"""Tests for monkey_collector.catalog_activities — static ground-truth loader."""
import json

import pytest

from monkey_collector.catalog_activities import ActivityCatalog


@pytest.fixture(autouse=True)
def _reset_catalog():
    ActivityCatalog.reset()
    yield
    ActivityCatalog.reset()


class TestLoad:
    def test_loads_activities(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({
            "org.tasks": {
                "app_name": "Tasks",
                "activities": ["org.tasks/X", "org.tasks/Y"],
            },
        }))
        cat = ActivityCatalog.instance(path)
        assert cat.is_loaded()
        assert cat.get_declared("org.tasks") == ["org.tasks/X", "org.tasks/Y"]

    def test_unknown_package_returns_none(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({"org.tasks": {"activities": []}}))
        cat = ActivityCatalog.instance(path)
        assert cat.get_declared("com.unknown") is None

    def test_missing_file_not_loaded(self, tmp_path):
        cat = ActivityCatalog.instance(tmp_path / "nope.json")
        assert not cat.is_loaded()
        assert cat.get_declared("any.package") is None

    def test_corrupt_json_not_loaded(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text("{ not valid json")
        cat = ActivityCatalog.instance(path)
        assert not cat.is_loaded()
        assert cat.get_declared("any.package") is None

    def test_top_level_not_object(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps(["not", "an", "object"]))
        cat = ActivityCatalog.instance(path)
        assert not cat.is_loaded()

    def test_returns_copy_not_internal_list(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({"p": {"activities": ["a", "b"]}}))
        cat = ActivityCatalog.instance(path)
        first = cat.get_declared("p")
        first.append("c")
        assert cat.get_declared("p") == ["a", "b"]

    def test_singleton_caches_first_path(self, tmp_path):
        path1 = tmp_path / "first.json"
        path1.write_text(json.dumps({"p": {"activities": ["x"]}}))
        cat1 = ActivityCatalog.instance(path1)
        cat2 = ActivityCatalog.instance(tmp_path / "other.json")
        assert cat1 is cat2
        assert cat2.get_declared("p") == ["x"]

    def test_entry_without_activities_key(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({"p": {"app_name": "P"}}))
        cat = ActivityCatalog.instance(path)
        assert cat.is_loaded()
        assert cat.get_declared("p") == []


class TestAliases:
    def test_loads_aliases(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({
            "p": {
                "activities": ["p/p.Real"],
                "aliases": {"p/p.Alias": "p/p.Real"},
            },
        }))
        cat = ActivityCatalog.instance(path)
        assert cat.get_aliases("p") == {"p/p.Alias": "p/p.Real"}

    def test_missing_aliases_key_returns_empty_dict(self, tmp_path):
        # Old-format catalog file (no "aliases" key) → registered package
        # yields an empty map, not None.
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({"p": {"activities": ["p/p.Real"]}}))
        cat = ActivityCatalog.instance(path)
        assert cat.get_aliases("p") == {}

    def test_unknown_package_aliases_returns_none(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({"p": {"activities": [], "aliases": {}}}))
        cat = ActivityCatalog.instance(path)
        assert cat.get_aliases("com.unknown") is None

    def test_aliases_not_loaded_returns_none(self, tmp_path):
        cat = ActivityCatalog.instance(tmp_path / "absent.json")
        assert not cat.is_loaded()
        assert cat.get_aliases("p") is None

    def test_aliases_returns_copy(self, tmp_path):
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({
            "p": {"activities": ["p/p.Real"], "aliases": {"p/a": "p/p.Real"}},
        }))
        cat = ActivityCatalog.instance(path)
        first = cat.get_aliases("p")
        first["p/b"] = "p/p.Real"
        assert cat.get_aliases("p") == {"p/a": "p/p.Real"}


class TestNonNavigableFilter:
    """Framework/3p SDK activities are excluded from the coverage denominator."""

    def _catalog(self, tmp_path, activities, aliases=None):
        path = tmp_path / "activities.json"
        entry = {"activities": activities}
        if aliases is not None:
            entry["aliases"] = aliases
        path.write_text(json.dumps({"p": entry}))
        return ActivityCatalog.instance(path)

    def test_drops_car_app_permission_activity(self, tmp_path):
        cat = self._catalog(tmp_path, [
            "p/p.MainActivity",
            "p/androidx.car.app.CarAppPermissionActivity",
        ])
        assert cat.get_declared("p") == ["p/p.MainActivity"]

    def test_drops_billingclient_proxy_activity(self, tmp_path):
        cat = self._catalog(tmp_path, [
            "p/p.MainActivity",
            "p/com.android.billingclient.api.ProxyBillingActivity",
        ])
        assert cat.get_declared("p") == ["p/p.MainActivity"]

    def test_drops_gms_google_api_activity(self, tmp_path):
        cat = self._catalog(tmp_path, [
            "p/p.MainActivity",
            "p/com.google.android.gms.common.api.GoogleApiActivity",
        ])
        assert cat.get_declared("p") == ["p/p.MainActivity"]

    def test_drops_play_core_activities(self, tmp_path):
        cat = self._catalog(tmp_path, [
            "p/p.MainActivity",
            "p/com.google.android.play.core.common.PlayCoreDialogWrapperActivity",
            "p/com.google.android.play.core.missingsplits.PlayCoreMissingSplitsActivity",
        ])
        assert cat.get_declared("p") == ["p/p.MainActivity"]

    def test_keeps_app_namespace_permission_activity(self, tmp_path):
        """Own-namespace screens are kept even if named like a permission gate."""
        cat = self._catalog(tmp_path, [
            "p/p.notificationpermission.RequestNotificationPermissionsActivity",
        ])
        assert cat.get_declared("p") == [
            "p/p.notificationpermission.RequestNotificationPermissionsActivity",
        ]

    def test_alias_dropped_when_target_non_navigable(self, tmp_path):
        cat = self._catalog(
            tmp_path,
            ["p/p.MainActivity", "p/com.google.android.gms.common.api.GoogleApiActivity"],
            aliases={
                "p/p.Alias": "p/com.google.android.gms.common.api.GoogleApiActivity",
                "p/p.RealAlias": "p/p.MainActivity",
            },
        )
        assert cat.get_aliases("p") == {"p/p.RealAlias": "p/p.MainActivity"}

    def test_osmand_denominator_16_to_11(self, tmp_path):
        """Real catalog data: 5 framework activities dropped."""
        activities = [
            "net.osmand/androidx.car.app.CarAppPermissionActivity",
            "net.osmand/com.android.billingclient.api.ProxyBillingActivity",
            "net.osmand/com.google.android.gms.common.api.GoogleApiActivity",
            "net.osmand/com.google.android.play.core.common.PlayCoreDialogWrapperActivity",
            "net.osmand/com.google.android.play.core.missingsplits.PlayCoreMissingSplitsActivity",
            "net.osmand/net.osmand.plus.activities.MapActivity",
            "net.osmand/net.osmand.plus.download.DownloadActivity",
            "net.osmand/net.osmand.plus.help.HelpActivity",
            "net.osmand/net.osmand.plus.myplaces.MyPlacesActivity",
            "net.osmand/net.osmand.plus.activities.ContributionVersionActivity",
            "net.osmand/net.osmand.plus.activities.PrintDialogActivity",
            "net.osmand/net.osmand.plus.activities.RestartActivity",
            "net.osmand/net.osmand.plus.activities.search.GeoIntentActivity",
            "net.osmand/net.osmand.plus.plugins.development.LogcatActivity",
            "net.osmand/net.osmand.plus.plugins.development.TestVoiceActivity",
            "net.osmand/net.osmand.plus.wikivoyage.explore.WikivoyageExploreActivity",
        ]
        path = tmp_path / "activities.json"
        path.write_text(json.dumps({"net.osmand": {"activities": activities}}))
        cat = ActivityCatalog.instance(path)
        assert len(cat.get_declared("net.osmand")) == 11
