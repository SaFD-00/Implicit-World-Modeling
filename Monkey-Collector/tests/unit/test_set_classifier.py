"""Tests for the cluster-assignment set classifier."""

from monkey_collector.pipeline.screen_matching.set_classifier import classify


def test_eqset():
    c = classify({"a", "b"}, {"a", "b"}, 0.2)
    assert c.match_type == "EQSET" and c.is_merge


def test_superset_always_merges():
    c = classify({"a", "b", "c"}, {"a", "b"}, 0.2)
    assert c.match_type == "SUPERSET_MERGE" and c.is_merge


def test_subset_always_merges():
    c = classify({"a"}, {"a", "b"}, 0.2)
    assert c.match_type == "SUBSET_MERGE" and c.is_merge


def test_empty_a_is_disjoint_not_subset():
    # Recognizing nothing on the screen must not be read as a subset merge.
    c = classify(set(), {"a", "b"}, 0.2)
    assert c.match_type == "DISJOINT" and not c.is_merge


def test_disjoint():
    c = classify({"x", "y"}, {"a", "b"}, 0.2)
    assert c.match_type == "DISJOINT" and not c.is_merge


def test_overlap_in_band_merges():
    # |A∩B|=3, N=|B|=4, M=|A|=4, tol=0.5 → bands [2,6] both → merge.
    a = {"a", "b", "c", "x"}
    b = {"a", "b", "c", "y"}
    c = classify(a, b, 0.5)
    assert c.match_type == "OVERLAP_MERGE" and c.is_merge


def test_overlap_out_of_band_new():
    # A rich screen sharing one incidental element with a small page must NOT
    # collapse: |A∩B|=1, N=1? No — make B large, A large, tiny overlap.
    a = {"a", "b", "c", "d", "shared"}
    b = {"shared", "e", "f", "g", "h"}
    c = classify(a, b, 0.2)
    # |A∩B|=1; N=5 band=[4,6]; 1 not in band → OVERLAP_NEW
    assert c.match_type == "OVERLAP_NEW" and not c.is_merge
