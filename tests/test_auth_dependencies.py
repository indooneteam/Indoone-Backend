from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.dependencies import current_user_id, enforce_user_match, require_owned_file


def test_current_user_id_requires_principal() -> None:
    with pytest.raises(HTTPException) as exc:
        current_user_id(SimpleNamespace(state=SimpleNamespace(principal_id="")))
    assert exc.value.status_code == 401


def test_current_user_id_rejects_whitespace_principal() -> None:
    with pytest.raises(HTTPException) as exc:
        current_user_id(SimpleNamespace(state=SimpleNamespace(principal_id="   ")))
    assert exc.value.status_code == 401


def test_user_match_rejects_cross_user_access() -> None:
    request = SimpleNamespace(state=SimpleNamespace(principal_id="user-one"))
    with pytest.raises(HTTPException) as exc:
        enforce_user_match(request, "user-two")
    assert exc.value.status_code == 403


def test_file_owner_guard_accepts_matching_principal() -> None:
    request = SimpleNamespace(state=SimpleNamespace(principal_id="user-one"))
    assert require_owned_file(request, "user-one") == "user-one"


def test_file_owner_guard_rejects_missing_owner() -> None:
    request = SimpleNamespace(state=SimpleNamespace(principal_id="user-one"))
    with pytest.raises(HTTPException) as exc:
        require_owned_file(request, "")
    assert exc.value.status_code == 403
