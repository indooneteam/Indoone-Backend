from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import firebase_admin
from firebase_admin import credentials, messaging


_DEFAULT_PROJECT_ID = "indoone"


def _project_id() -> str:
    return os.getenv("FIREBASE_PROJECT_ID", _DEFAULT_PROJECT_ID).strip() or _DEFAULT_PROJECT_ID


def _firebase_app() -> firebase_admin.App:
    try:
        return firebase_admin.get_app()
    except ValueError:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if credentials_path:
            path = Path(credentials_path).expanduser()
            if not path.is_file():
                raise RuntimeError(
                    f"GOOGLE_APPLICATION_CREDENTIALS does not point to a file: {path}"
                )
            credential = credentials.Certificate(str(path))
            return firebase_admin.initialize_app(
                credential,
                {"projectId": _project_id()},
            )

        return firebase_admin.initialize_app(options={"projectId": _project_id()})


def send_data_notification(
    *,
    token: str,
    title: str,
    body: str,
    category: str = "updates",
    route: str | None = None,
) -> str:
    normalized_token = token.strip()
    if not normalized_token:
        raise ValueError("FCM registration token is required")

    data: dict[str, str] = {
        "category": category.strip().lower() or "updates",
        "title": title.strip(),
        "body": body.strip(),
    }
    if route and route.strip():
        data["route"] = route.strip()

    message = messaging.Message(
        token=normalized_token,
        data=data,
        android=messaging.AndroidConfig(
            priority="high",
            ttl=3600,
            restricted_package_name="com.indoone.authenticator",
        ),
    )

    return messaging.send(message, app=_firebase_app())
