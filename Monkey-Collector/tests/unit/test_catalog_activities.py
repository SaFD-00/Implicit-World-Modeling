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
