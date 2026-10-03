from __future__ import annotations

import argparse

from app.notifications.fcm import send_data_notification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send one production-style FCM data notification to an Indoone device."
    )
    parser.add_argument("--token", required=True, help="FCM registration token for the target device")
    parser.add_argument("--title", default="New message", help="Notification title")
    parser.add_argument(
        "--body",
        default="You have a new message in Indoone.",
        help="Notification body",
    )
    parser.add_argument("--category", default="ai", help="Indoone notification category id")
    parser.add_argument("--route", default="chat", help="In-app route to open on tap")
    args = parser.parse_args()

    message_id = send_data_notification(
        token=args.token,
        title=args.title,
        body=args.body,
        category=args.category,
        route=args.route,
    )
    print("FCM_SENT=1")
    print(f"MESSAGE_ID={message_id}")


if __name__ == "__main__":
    main()
