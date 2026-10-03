from __future__ import annotations

import hashlib
import logging
from typing import Any

from firebase_admin import firestore


logger = logging.getLogger("indoone.notifications")
_TOKEN_COLLECTION = "notification_tokens"


def _db() -> firestore.Client:
    from app.notifications.fcm import _firebase_app

    return firestore.client(app=_firebase_app())


def _token_document_id(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register_token(*, user_id: str, token: str) -> None:
    normalized = token.strip()
    if not normalized:
        raise ValueError("FCM registration token is required")

    document = _db().collection(_TOKEN_COLLECTION).document(_token_document_id(normalized))
    document.set(
        {
            "userId": user_id,
            "token": normalized,
            "platform": "android",
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )


def user_tokens(*, user_id: str) -> list[tuple[str, Any]]:
    documents = (
        _db()
        .collection(_TOKEN_COLLECTION)
        .where("userId", "==", user_id)
        .stream()
    )
    return [(document.id, document.get("token")) for document in documents if document.get("token")]


def delete_token(document_id: str) -> None:
    _db().collection(_TOKEN_COLLECTION).document(document_id).delete()
