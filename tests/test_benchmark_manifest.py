from scripts.benchmark import ALLOWED_RIGHTS


def test_rights_allowlist_is_explicit():
    assert ALLOWED_RIGHTS == {"owned", "consented", "licensed"}
    assert "public" not in ALLOWED_RIGHTS
