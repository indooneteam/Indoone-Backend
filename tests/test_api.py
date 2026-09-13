from app.api.request_context import (
    clear_principal_id,
    get_principal_id,
    require_principal_id,
    set_principal_id,
)


def test_principal_context_can_be_cleared() -> None:
    set_principal_id("user-one")
    assert require_principal_id() == "user-one"
    clear_principal_id()
    assert get_principal_id() == ""
