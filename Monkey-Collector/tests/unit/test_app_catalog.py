"""Tests for monkey_collector.pipeline.app_catalog — AppCatalog CSV parser + filter."""

from __future__ import annotations

from pathlib import Path

import pytest

from monkey_collector.pipeline.app_catalog import AppCatalog, AppJob

SAMPLE_CSV = (
    "category,sub_category,app_name,package_id,source,priority,notes\n"
    "Shopping,General,Amazon,in.amazon.mShop.android.shopping,PlayStore,High,top e-commerce\n"
    "Shopping,General,eBay,com.ebay.mobile,PlayStore,Medium,auction\n"
    "Social,Chat,WhatsApp,com.whatsapp,PlayStore,High,messaging\n"
    "Social,Chat,Telegram,org.telegram.messenger,PlayStore,Low,messaging\n"
    "Utility,System,Calculator,com.example.calc,System,Medium,builtin\n"
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "apps.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


@pytest.fixture
def catalog(sample_csv: Path) -> AppCatalog:
    return AppCatalog.load(sample_csv)


def test_load_from_sample_csv(catalog: AppCatalog) -> None:
    apps = catalog.filter()
    assert len(apps) == 5
    assert all(isinstance(a, AppJob) for a in apps)
    first = apps[0]
    assert first.category == "Shopping"
    assert first.app_name == "Amazon"
    assert first.package_id == "in.amazon.mShop.android.shopping"
    assert first.source == "PlayStore"
    assert first.priority == "High"


def test_load_handles_bom(tmp_path: Path) -> None:
    path = tmp_path / "apps_bom.csv"
    path.write_text("\ufeff" + SAMPLE_CSV, encoding="utf-8")
    cat = AppCatalog.load(path)
    assert len(cat.filter()) == 5


def test_load_normalizes_header_whitespace_and_case(tmp_path: Path) -> None:
    content = (
        " Category , Sub_Category , App_Name , Package_Id , Source , Priority , Notes \n"
        "Shopping,General,Amazon,com.amazon,PlayStore,High,note\n"
    )
    path = tmp_path / "apps_spaced.csv"
    path.write_text(content, encoding="utf-8")
    cat = AppCatalog.load(path)
    apps = cat.filter()
    assert len(apps) == 1
    assert apps[0].app_name == "Amazon"


def test_filter_by_categories(catalog: AppCatalog) -> None:
    apps = catalog.filter(categories=["Shopping"])
    assert len(apps) == 2
    assert {a.app_name for a in apps} == {"Amazon", "eBay"}


def test_filter_by_priorities(catalog: AppCatalog) -> None:
    apps = catalog.filter(priorities=["High"])
    assert len(apps) == 2
    assert {a.app_name for a in apps} == {"Amazon", "WhatsApp"}


def test_filter_combined(catalog: AppCatalog) -> None:
    apps = catalog.filter(categories=["Shopping"], priorities=["High"])
    assert len(apps) == 1
    assert apps[0].app_name == "Amazon"


def test_filter_none_returns_all(catalog: AppCatalog) -> None:
    assert len(catalog.filter(categories=None, priorities=None)) == 5
    assert len(catalog.filter()) == 5


def test_case_insensitive(catalog: AppCatalog) -> None:
    lower = catalog.filter(categories=["shopping"])
    upper = catalog.filter(categories=["Shopping"])
    assert {a.package_id for a in lower} == {a.package_id for a in upper}
    assert len(lower) == 2

    hi_lower = catalog.filter(priorities=["high"])
    assert len(hi_lower) == 2


def test_filter_trims_whitespace(catalog: AppCatalog) -> None:
    apps = catalog.filter(categories=["  Shopping  "], priorities=[" High "])
    assert len(apps) == 1
    assert apps[0].app_name == "Amazon"


def test_filter_unknown_category_returns_empty(catalog: AppCatalog) -> None:
    assert catalog.filter(categories=["NonExistent"]) == []


def test_categories_and_priorities_methods(catalog: AppCatalog) -> None:
    assert catalog.categories() == ["Shopping", "Social", "Utility"]
    assert catalog.priorities() == ["High", "Low", "Medium"]


def test_app_job_is_frozen() -> None:
    job = AppJob(
        category="A",
        sub_category="B",
        app_name="C",
        package_id="d.e.f",
        source="PlayStore",
        priority="High",
    )
    with pytest.raises(Exception):  # noqa: B017
        job.category = "Z"  # type: ignore[misc]


def test_installed_column_missing_defaults_to_false(catalog: AppCatalog) -> None:
    for app in catalog.filter():
        assert app.installed is False


def test_installed_column_parsed(tmp_path: Path) -> None:
    content = (
        "category,sub_category,app_name,package_id,source,priority,notes,installed\n"
        "Shopping,General,Amazon,com.amazon,PlayStore,High,top,true\n"
        "Shopping,General,eBay,com.ebay,PlayStore,Medium,auction,False\n"
        "Social,Chat,Signal,org.sig,PlayStore,High,msg,1\n"
        "Social,Chat,Telegram,org.tg,PlayStore,Low,msg,\n"
        "Utility,System,Calc,com.calc,System,Medium,builtin,YES\n"
    )
    path = tmp_path / "apps_installed.csv"
    path.write_text(content, encoding="utf-8")
    cat = AppCatalog.load(path)
    apps = {a.package_id: a.installed for a in cat.filter()}
    assert apps == {
        "com.amazon": True,
        "com.ebay": False,
        "org.sig": True,
        "org.tg": False,
        "com.calc": True,
    }


def test_filter_by_installed(tmp_path: Path) -> None:
    content = (
        "category,sub_category,app_name,package_id,source,priority,notes,installed\n"
        "A,B,One,p.one,PlayStore,High,,true\n"
        "A,B,Two,p.two,PlayStore,High,,false\n"
        "A,B,Three,p.three,PlayStore,High,,true\n"
    )
    path = tmp_path / "apps.csv"
    path.write_text(content, encoding="utf-8")
    cat = AppCatalog.load(path)

    installed = cat.installed_apps()
    assert {a.package_id for a in installed} == {"p.one", "p.three"}

    uninstalled = cat.filter(installed=False)
    assert {a.package_id for a in uninstalled} == {"p.two"}

    assert len(cat.filter(installed=None)) == 3


def test_find_by_package(catalog: AppCatalog) -> None:
    app = catalog.find_by_package("com.whatsapp")
    assert app is not None
    assert app.app_name == "WhatsApp"
    assert catalog.find_by_package("com.nope.missing") is None


def _job(**overrides) -> AppJob:
    base = dict(
        category="Shopping",
        sub_category="General",
        app_name="Amazon Shopping",
        package_id="com.amazon.mShop",
        source="PlayStore",
        priority="High",
        notes="Top e-commerce; complex UI with search/filters/cart",
    )
    base.update(overrides)
    return AppJob(**base)


def test_description_full() -> None:
    assert _job().description == (
        "Amazon Shopping (Shopping/General) — "
        "Top e-commerce; complex UI with search/filters/cart"
    )


def test_description_without_notes() -> None:
    assert _job(notes="").description == "Amazon Shopping (Shopping/General)"


def test_description_without_category() -> None:
    assert _job(category="", sub_category="", notes="").description == "Amazon Shopping"


def test_description_falls_back_to_package_id() -> None:
    job = _job(app_name="", category="", sub_category="", notes="")
    assert job.description == "com.amazon.mShop"
