from __future__ import annotations

_WHATSAPP_WEBHOOK_PATH = "/api/integrations/whatsapp/webhook"


def is_whatsapp_webhook_path(path: str) -> bool:
    return path == _WHATSAPP_WEBHOOK_PATH
