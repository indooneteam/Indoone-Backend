import pytest


def test_fcm_sender_requires_token():
    from app.notifications.fcm import send_data_notification

    with pytest.raises(ValueError):
        send_data_notification(token=" ", title="x", body="y")
