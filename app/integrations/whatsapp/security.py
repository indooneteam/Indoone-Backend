from __future__ import annotations

_WHATSAPP_WEBHOOK_PATH = "/api/integrations/whatsapp/webhook"
_TELEGRAM_WEBHOOK_PATH = "/api/telegram/webhook/incoming"
_INSTAGRAM_WEBHOOK_PATH = "/api/integrations/instagram/webhook"
_INSTAGRAM_CALLBACK_PATH = "/api/integrations/instagram/callback"


def is_whatsapp_webhook_path(path: str) -> bool:
    return path == _WHATSAPP_WEBHOOK_PATH


def is_public_integration_path(path: str) -> bool:
    return path in {
        _WHATSAPP_WEBHOOK_PATH,
        _TELEGRAM_WEBHOOK_PATH,
        _INSTAGRAM_WEBHOOK_PATH,
        _INSTAGRAM_CALLBACK_PATH,
    }
