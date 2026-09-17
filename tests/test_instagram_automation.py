import pytest

from app.capabilities.instagram_automation import plan_automation


def test_plan_comment_rule() -> None:
    payload = {
        "event_id": "e1",
        "entries": [
            {"changes": [{"field": "comments", "value": {"id": "c1", "text": "Hello Indoone"}}]}
        ],
    }
    rules = [
        {"trigger": "comment", "action": "reply_comment", "keyword": "indoone", "response": "Thanks!"}
    ]
    result = plan_automation(payload, rules)
    assert result["dry_run"] is True
    assert result["matched_rule_indexes"] == [0]
    assert result["actions"][0]["action"] == "reply_comment"
    assert result["actions"][0]["comment_id"] == "c1"


def test_plan_message_rule() -> None:
    payload = {
        "event_id": "e2",
        "entries": [
            {"messaging": [{"sender": {"id": "u1"}, "message": {"text": "price please"}}]}
        ],
    }
    rules = [
        {"trigger": "message", "action": "send_message", "keyword": "price", "response": "Please check our latest pricing."}
    ]
    result = plan_automation(payload, rules)
    assert result["matched_rule_indexes"] == [0]
    assert result["actions"][0]["recipient_id"] == "u1"


def test_invalid_rule_is_rejected() -> None:
    with pytest.raises(ValueError):
        plan_automation({"entries": []}, [{"trigger": "unknown", "action": "send_message", "response": "x"}])
