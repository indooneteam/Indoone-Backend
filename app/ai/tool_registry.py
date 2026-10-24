"""Single source of truth for every tool the AI can call.

This module is the ONE place a tool is defined. To add a tool, add one
``ToolSpec`` entry below: its name, description, model-facing parameter
schema, governance (risk / approval / connector / capability / limits), and
where its executor lives. Nothing else in the codebase should define a tool.

Everything downstream reads from here:

* ``TOOL_SPECS``                 -- the catalogue (used by the dispatcher,
                                    approval policy and connector checks).
* ``function_declarations()``    -- the model-facing function-calling schemas.
* ``GITHUB_TOOL_DECLARATIONS``   -- the read-only GitHub declarations, derived
                                    from the specs (no hand-kept duplicate).
* ``EMAIL_TOOL_DECLARATIONS``    -- the email plugin declarations, kept here so
                                    every tool definition lives in one file.

NOTE (known overlap to resolve in a later step): the legacy ``gmail_*`` specs
below are the deterministic agent path, while the email plugin exposes its own
``*_email*`` / ``*_draft*`` tools in ``EMAIL_TOOL_DECLARATIONS``. Both families
are listed here so the duplication is visible in one place instead of hidden
across modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Tool spec
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ToolSpec:
    """One callable tool.

    ``parameters`` is the model-facing JSON schema of the arguments.
    Backend-injected fields (for example ``user_id``) are never listed there --
    the executor adds them from the authenticated request.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    # How the executor receives input: "json" (object) or "text" (raw string).
    payload_style: str = "json"
    # Governance.
    risk: str = "read"
    requires_approval: bool = False
    timeout_seconds: float = 20.0
    max_output_chars: int = 20_000
    categories: tuple[str, ...] = ()
    connector_id: str | None = None
    capability: str | None = None
    # Where the executor lives, for maintainers.
    module: str = "app.ai.tools"


def _object(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "OBJECT", "properties": properties}
    if required:
        schema["required"] = list(required)
    return schema


_STRING = {"type": "STRING"}
_INTEGER = {"type": "INTEGER"}
# Contact records are client-supplied objects. Gemini requires `items` on every
# ARRAY parameter, so the element schema is declared once and reused.
_CONTACT_ITEM = {
    "type": "OBJECT",
    "properties": {
        "name": {"type": "STRING", "description": "Contact name."},
        "phone": {"type": "STRING", "description": "Contact phone number."},
        "email": {"type": "STRING", "description": "Contact email address."},
        "contact_id": {"type": "STRING", "description": "Optional client contact id."},
    },
}


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

TOOL_SPECS: tuple[ToolSpec, ...] = (
    # ---- local reasoning / text ------------------------------------------ #
    ToolSpec(
        "calculator",
        "Evaluate safe numeric arithmetic.",
        parameters=_object(
            {"expression": {"type": "STRING", "description": "Arithmetic expression, e.g. 12*(3+4)."}},
            ("expression",),
        ),
        payload_style="text",
        categories=("reasoning",),
    ),
    ToolSpec(
        "text_stats",
        "Count text characters, words, and lines.",
        parameters=_object({"text": {"type": "STRING", "description": "Text to measure."}}, ("text",)),
        payload_style="text",
        categories=("text",),
    ),
    ToolSpec(
        "json_summary",
        "Summarize the top-level JSON structure.",
        parameters=_object({"json": {"type": "STRING", "description": "JSON document to summarize."}}, ("json",)),
        payload_style="text",
        categories=("data",),
    ),
    # ---- coding ----------------------------------------------------------- #
    ToolSpec(
        "code_analysis",
        "Perform safe static code analysis.",
        parameters=_object({"code": {"type": "STRING", "description": "Source code to analyze."}}, ("code",)),
        payload_style="text",
        categories=("coding",),
    ),
    ToolSpec(
        "code_fix_suggestions",
        "Produce static repair guidance for code.",
        parameters=_object({"code": {"type": "STRING", "description": "Source code to repair."}}, ("code",)),
        payload_style="text",
        categories=("coding",),
    ),
    ToolSpec(
        "code_transform",
        "Apply deterministic code transformations.",
        parameters=_object({"code": {"type": "STRING", "description": "Source code to transform."}}, ("code",)),
        payload_style="text",
        categories=("coding",),
    ),
    ToolSpec(
        "sandbox_execution",
        "Execute guarded code in the sandbox.",
        parameters=_object({"code": {"type": "STRING", "description": "Code to run in the sandbox."}}, ("code",)),
        payload_style="text",
        risk="execution",
        requires_approval=True,
        categories=("coding", "execution"),
    ),
    # ---- contacts / phone ------------------------------------------------- #
    ToolSpec(
        "contacts_search",
        "Search a client-authorized contact snapshot.",
        parameters=_object(
            {
                "contacts": {"type": "ARRAY", "description": "Contact records supplied by the client.", "items": _CONTACT_ITEM},
                "query": {"type": "STRING", "description": "Name or phrase to search for."},
                "limit": {"type": "INTEGER", "description": "Maximum matches to return."},
            },
            ("query",),
        ),
        categories=("contacts",),
    ),
    ToolSpec(
        "contact_resolve",
        "Resolve a client-authorized contact.",
        parameters=_object(
            {
                "contacts": {"type": "ARRAY", "description": "Contact records supplied by the client.", "items": _CONTACT_ITEM},
                "query": {"type": "STRING", "description": "Name or phrase to resolve."},
            },
            ("query",),
        ),
        categories=("contacts",),
    ),
    ToolSpec(
        "phone_call",
        "Prepare a phone call action.",
        parameters=_object(
            {
                "phone_number": {"type": "STRING", "description": "Number to call."},
                "contact_name": {"type": "STRING", "description": "Display name for the number."},
            },
            ("phone_number",),
        ),
        risk="external",
        requires_approval=True,
        categories=("phone", "external"),
    ),
    ToolSpec(
        "phone_call_contact",
        "Prepare a phone call action for a contact.",
        parameters=_object(
            {
                "contacts": {"type": "ARRAY", "description": "Contact records supplied by the client.", "items": _CONTACT_ITEM},
                "query": {"type": "STRING", "description": "Contact name to call."},
            },
            ("query",),
        ),
        risk="external",
        requires_approval=True,
        categories=("phone", "external"),
    ),
    # ---- Gmail (legacy deterministic agent path) -------------------------- #
    ToolSpec(
        "gmail_search",
        "Search the authenticated user's Gmail mailbox.",
        parameters=_object(
            {
                "query": {"type": "STRING", "description": "Gmail search query."},
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
                "max_results": {"type": "INTEGER", "description": "Maximum messages to return."},
            },
            ("query",),
        ),
        categories=("gmail", "external"),
        connector_id="gmail",
        capability="messages.search",
    ),
    ToolSpec(
        "gmail_read",
        "Read one message from the authenticated user's Gmail mailbox.",
        parameters=_object({"message_id": {"type": "STRING", "description": "Gmail message id."}}, ("message_id",)),
        categories=("gmail", "external"),
        connector_id="gmail",
        capability="messages.read",
    ),
    ToolSpec(
        "gmail_compose",
        "Create an email draft with the local Indoone AI without sending it.",
        parameters=_object({"request": {"type": "STRING", "description": "What the draft should say."}}, ("request",)),
        connector_id="gmail",
        capability="messages.read",
        requires_approval=True,
        categories=("gmail", "email", "external"),
    ),
    ToolSpec(
        "gmail_email_safety",
        "Read and analyze authenticated Gmail messages with the local Indoone Email Safety behavior.",
        parameters=_object(
            {
                "query": {"type": "STRING", "description": "Search phrase identifying the messages."},
                "message_id": {"type": "STRING", "description": "Exact message id, if known."},
                "user_request": {"type": "STRING", "description": "The user's original request."},
                "limit": {"type": "INTEGER", "description": "Number of messages to analyze."},
            },
        ),
        categories=("gmail", "email_safety", "external"),
        connector_id="gmail",
        capability="messages.read",
    ),
    ToolSpec(
        "gmail_reply",
        "Reply to an authenticated Gmail message.",
        parameters=_object(
            {
                "message_id": {"type": "STRING", "description": "Message id to reply to."},
                "body": {"type": "STRING", "description": "Reply body."},
            },
            ("message_id", "body"),
        ),
        risk="external",
        requires_approval=True,
        categories=("gmail", "email", "external"),
        connector_id="gmail",
        capability="messages.send",
    ),
    ToolSpec(
        "gmail_send",
        "Send an email from the authenticated user's Gmail account.",
        parameters=_object(
            {
                "to": {"type": "STRING", "description": "Recipient address."},
                "subject": {"type": "STRING", "description": "Subject line."},
                "body": {"type": "STRING", "description": "Email body."},
            },
            ("to", "subject", "body"),
        ),
        risk="external",
        requires_approval=True,
        categories=("gmail", "external"),
        connector_id="gmail",
        capability="messages.send",
    ),
    # ---- GitHub ----------------------------------------------------------- #
    ToolSpec(
        "github_list_repositories",
        "List repositories accessible to the authenticated GitHub user, ordered by recent update.",
        parameters=_object(
            {
                "page": {"type": "INTEGER", "description": "Page number, starting at 1."},
                "per_page": {"type": "INTEGER", "description": "Number of repositories, 1 through 30."},
            },
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repos.read",
    ),
    # ---- GitHub: the rest of the flow ------------------------------------- #
    # Reading a branch, a pull request's files, the checks and one commit. These
    # are read-only, so they need no approval and belong in the list the model may
    # call on its own.
    ToolSpec(
        "github_list_branches",
        "List a repository's branches, so work can be started from the right one.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "per_page": {"type": "INTEGER", "description": "Number of branches, 1 through 100."},
            },
            required=("repository",),
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repos.read",
    ),
    ToolSpec(
        "github_list_pull_request_files",
        "The diff of a pull request: which files changed, how much, and the patch for each.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "number": {"type": "INTEGER", "description": "Pull request number."},
                "per_page": {"type": "INTEGER", "description": "Number of files, 1 through 100."},
            },
            ("repository", "number"),
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repos.read",
    ),
    ToolSpec(
        "github_list_check_runs",
        "The CI status for a branch or commit, so a change is not merged before it is green. Reports an overall conclusion: success, failure, running or none.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "ref": {"type": "STRING", "description": "Branch name or commit sha. Omit for the default branch."},
            },
            required=("repository",),
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repos.read",
    ),
    ToolSpec(
        "github_get_commit",
        "One commit: its message, author, and the files it touched with their patches.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "sha": {"type": "STRING", "description": "The commit id, 7 to 40 hex characters."},
            },
            ("repository", "sha"),
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repos.read",
    ),
    # ---- GitHub: the writes the model may propose ------------------------- #
    # Each of these changes a repository, so requires_approval is set. The model
    # proposes; the user confirms. They are never reached without it.
    ToolSpec(
        "github_create_branch",
        "Create a branch from a ref, so a change can be made without touching the default branch. Requires the user's explicit approval.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "branch": {"type": "STRING", "description": "The new branch name."},
                "from_ref": {"type": "STRING", "description": "Branch or commit to start from. Defaults to the default branch."},
            },
            ("repository", "branch"),
        ),
        risk="write",
        requires_approval=True,
        categories=("github", "external", "write"),
        connector_id="github",
        capability="repos.write",
    ),
    ToolSpec(
        "github_delete_branch",
        "Delete a branch, the clean-up after a pull request is merged. Requires the user's explicit approval.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "branch": {"type": "STRING", "description": "The branch to delete."},
            },
            ("repository", "branch"),
        ),
        risk="write",
        requires_approval=True,
        categories=("github", "external", "write"),
        connector_id="github",
        capability="repos.write",
    ),
    ToolSpec(
        "github_create_pull_request",
        "Open a pull request from one branch into another. Requires the user's explicit approval.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "title": {"type": "STRING", "description": "The pull request title."},
                "head": {"type": "STRING", "description": "The branch with the change."},
                "base": {"type": "STRING", "description": "The branch to merge into. Defaults to main."},
                "body": {"type": "STRING", "description": "The description."},
            },
            ("repository", "title", "head"),
        ),
        risk="write",
        requires_approval=True,
        categories=("github", "external", "write"),
        connector_id="github",
        capability="repos.write",
    ),
    ToolSpec(
        "github_merge_pull_request",
        "Merge a pull request, squash by default. Only when its checks are green. Requires the user's explicit approval.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Repository as owner/name."},
                "number": {"type": "INTEGER", "description": "The pull request number."},
                "method": {"type": "STRING", "enum": ["squash", "merge", "rebase"], "description": "Merge method. Defaults to squash."},
            },
            ("repository", "number"),
        ),
        risk="write",
        requires_approval=True,
        categories=("github", "external", "write"),
        connector_id="github",
        capability="repos.write",
    ),
    ToolSpec(
        "github_list_assigned_issues",
        "List open issues assigned to the authenticated GitHub user.",
        parameters=_object(
            {
                "page": {"type": "INTEGER", "description": "Page number, starting at 1."},
                "per_page": {"type": "INTEGER", "description": "Number of issues, 1 through 30."},
            },
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="issues.read",
    ),
    ToolSpec(
        "github_list_pull_requests",
        "List open pull requests for a repository the authenticated GitHub user can access.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name, when known."},
                "repository_hint": {"type": "STRING", "description": "Repository name the user mentioned, when known."},
                "page": {"type": "INTEGER", "description": "Page number, starting at 1."},
                "per_page": {"type": "INTEGER", "description": "Number of pull requests, 1 through 30."},
            },
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="pull_requests.read",
    ),
    ToolSpec(
        "github_list_repository_contents",
        "List folders and files under a path in a GitHub repository.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name, if known."},
                "repository_hint": {"type": "STRING", "description": "Repository name mentioned by the user, if known."},
                "path": {"type": "STRING", "description": "Folder path; empty means repository root."},
                "ref": {"type": "STRING", "description": "Optional branch or commit."},
            },
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repo.contents.read",
    ),
    ToolSpec(
        "github_read_repository_file",
        "Read a bounded text file from an accessible GitHub repository; known credential formats are redacted.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name, if known."},
                "repository_hint": {"type": "STRING", "description": "Repository name mentioned by the user, if known."},
                "path": {"type": "STRING", "description": "File path inside the repository."},
                "ref": {"type": "STRING", "description": "Optional branch or commit."},
            },
            ("path",),
        ),
        max_output_chars=32000,
        timeout_seconds=30.0,
        categories=("github", "external"),
        connector_id="github",
        capability="repo.files.read",
    ),
    ToolSpec(
        "github_search_repository_code",
        "Search code in a selected accessible GitHub repository.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name, if known."},
                "repository_hint": {"type": "STRING", "description": "Repository name mentioned by the user, if known."},
                "query": {"type": "STRING", "description": "Function, symbol, or text to search for."},
                "per_page": {"type": "INTEGER", "description": "Number of results, 1 through 30."},
            },
            ("query",),
        ),
        max_output_chars=24000,
        categories=("github", "external"),
        connector_id="github",
        capability="repo.code.search",
    ),
    ToolSpec(
        "github_list_repository_commits",
        "List recent commits, optionally filtered by repository path.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name, if known."},
                "repository_hint": {"type": "STRING", "description": "Repository name mentioned by the user, if known."},
                "path": {"type": "STRING", "description": "Optional file path."},
                "per_page": {"type": "INTEGER", "description": "Number of commits, 1 through 30."},
            },
        ),
        categories=("github", "external"),
        connector_id="github",
        capability="repo.history.read",
    ),
    ToolSpec(
        "github_compare_repository_commits",
        "Compare two branches or commits and list changed files.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name, if known."},
                "repository_hint": {"type": "STRING", "description": "Repository name mentioned by the user, if known."},
                "base": {"type": "STRING", "description": "Base branch or commit."},
                "head": {"type": "STRING", "description": "Branch or commit to compare with base."},
            },
            ("base", "head"),
        ),
        max_output_chars=24000,
        categories=("github", "external"),
        connector_id="github",
        capability="repo.compare.read",
    ),
    ToolSpec(
        "github_update_repository_file",
        "Commit a user-confirmed text-file update to an accessible GitHub repository.",
        parameters=_object(
            {
                "repository": {"type": "STRING", "description": "Exact repository owner/name."},
                "path": {"type": "STRING", "description": "File path inside the repository."},
                "content": {"type": "STRING", "description": "New file content."},
                "message": {"type": "STRING", "description": "Commit message."},
                "sha": {"type": "STRING", "description": "Blob SHA of the file being replaced, for updates."},
                "create": {"type": "BOOLEAN", "description": "True to create a new file."},
                "branch": {"type": "STRING", "description": "Target branch; defaults to the default branch."},
            },
            ("repository", "path", "content", "message"),
        ),
        risk="external",
        requires_approval=True,
        max_output_chars=4000,
        categories=("github", "external", "write"),
        connector_id="github",
        capability="repo.files.write",
    ),
    # ---- Google Calendar -------------------------------------------------- #
    ToolSpec(
        "google_calendar_list_calendars",
        "List the authenticated user's Google calendars.",
        parameters=_object(
            {
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
                "max_results": {"type": "INTEGER", "description": "Maximum calendars to return."},
            },
        ),
        categories=("google_calendar", "external"),
        connector_id="google_calendar",
        capability="events.read",
    ),
    ToolSpec(
        "google_calendar_list_events",
        "List events from a Google calendar, optionally filtered by time range or query.",
        parameters=_object(
            {
                "calendar_id": {"type": "STRING", "description": "Calendar id; defaults to primary."},
                "time_min": {"type": "STRING", "description": "RFC3339 lower time bound."},
                "time_max": {"type": "STRING", "description": "RFC3339 upper time bound."},
                "query": {"type": "STRING", "description": "Free-text search over events."},
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
                "max_results": {"type": "INTEGER", "description": "Maximum events to return."},
            },
        ),
        categories=("google_calendar", "external"),
        connector_id="google_calendar",
        capability="events.read",
    ),
    ToolSpec(
        "google_calendar_get_event",
        "Read one Google calendar event by id.",
        parameters=_object(
            {
                "calendar_id": {"type": "STRING", "description": "Calendar id; defaults to primary."},
                "event_id": {"type": "STRING", "description": "Event id."},
            },
            ("event_id",),
        ),
        categories=("google_calendar", "external"),
        connector_id="google_calendar",
        capability="events.read",
    ),
    ToolSpec(
        "google_calendar_create_event",
        "Create a Google calendar event. Requires the user's explicit approval.",
        parameters=_object(
            {
                "calendar_id": {"type": "STRING", "description": "Calendar id; defaults to primary."},
                "event": {
                    "type": "OBJECT",
                    "description": "Google Calendar event resource.",
                    "properties": {
                        "summary": {"type": "STRING", "description": "Event title."},
                        "start": {"type": "STRING", "description": "RFC3339 start date-time."},
                        "end": {"type": "STRING", "description": "RFC3339 end date-time."},
                        "location": {"type": "STRING"},
                        "description": {"type": "STRING"},
                    },
                },
            },
            ("event",),
        ),
        risk="external",
        requires_approval=True,
        categories=("google_calendar", "external", "write"),
        connector_id="google_calendar",
        capability="events.create",
    ),
    ToolSpec(
        "google_calendar_update_event",
        "Update an existing Google calendar event. Requires the user's explicit approval.",
        parameters=_object(
            {
                "calendar_id": {"type": "STRING", "description": "Calendar id; defaults to primary."},
                "event_id": {"type": "STRING", "description": "Event id to update."},
                "event": {
                    "type": "OBJECT",
                    "description": "Fields to change on the event.",
                    "properties": {
                        "summary": {"type": "STRING"},
                        "start": {"type": "STRING", "description": "RFC3339 start date-time."},
                        "end": {"type": "STRING", "description": "RFC3339 end date-time."},
                        "location": {"type": "STRING"},
                        "description": {"type": "STRING"},
                    },
                },
            },
            ("event_id", "event"),
        ),
        risk="external",
        requires_approval=True,
        categories=("google_calendar", "external", "write"),
        connector_id="google_calendar",
        capability="events.update",
    ),
    ToolSpec(
        "google_calendar_delete_event",
        "Delete a Google calendar event. Requires the user's explicit approval.",
        parameters=_object(
            {
                "calendar_id": {"type": "STRING", "description": "Calendar id; defaults to primary."},
                "event_id": {"type": "STRING", "description": "Event id to delete."},
            },
            ("event_id",),
        ),
        risk="external",
        requires_approval=True,
        categories=("google_calendar", "external", "write"),
        connector_id="google_calendar",
        capability="events.delete",
    ),
    # ---- Google Drive ----------------------------------------------------- #
    ToolSpec(
        "google_drive_list_files",
        "List or search the authenticated user's Google Drive files.",
        parameters=_object(
            {
                "query": {"type": "STRING", "description": "File name text to search for."},
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
                "page_size": {"type": "INTEGER", "description": "Number of files, 1 through 1000."},
            },
        ),
        categories=("google_drive", "external"),
        connector_id="google_drive",
        capability="files.search",
    ),
    ToolSpec(
        "google_drive_get_file",
        "Read metadata, and optionally the text, of a Google Drive file.",
        parameters=_object(
            {
                "file_id": {"type": "STRING", "description": "Drive file id."},
                "download": {"type": "BOOLEAN", "description": "Download the file content when true."},
                "export_mime_type": {"type": "STRING", "description": "Export type for Google Docs files, e.g. text/plain."},
            },
            ("file_id",),
        ),
        categories=("google_drive", "external"),
        connector_id="google_drive",
        capability="files.read",
    ),
    ToolSpec(
        "google_drive_upload_file",
        "Upload a text file to the authenticated user's Google Drive. Requires the user's explicit approval.",
        parameters=_object(
            {
                "filename": {"type": "STRING", "description": "File name to create."},
                "mime_type": {"type": "STRING", "description": "MIME type, e.g. text/plain."},
                "content": {"type": "STRING", "description": "File text content."},
                "parent_id": {"type": "STRING", "description": "Optional parent folder id."},
            },
            ("filename", "mime_type", "content"),
        ),
        risk="external",
        requires_approval=True,
        categories=("google_drive", "external", "write"),
        connector_id="google_drive",
        capability="files.upload",
    ),
    # ---- Google Photos ---------------------------------------------------- #
    ToolSpec(
        "google_photos_list_media",
        "List the authenticated user's Google Photos media items.",
        parameters=_object(
            {
                "page_size": {"type": "INTEGER", "description": "Number of items, 1 through 100."},
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
            },
        ),
        categories=("google_photos", "external"),
        connector_id="google_photos",
        capability="media.search",
    ),
    ToolSpec(
        "google_photos_search_media",
        "Search Google Photos media, optionally filtered by album, type, or order.",
        parameters=_object(
            {
                "page_size": {"type": "INTEGER", "description": "Number of items, 1 through 100."},
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
                "album_id": {"type": "STRING", "description": "Restrict to one album id."},
                "media_type": {"type": "STRING", "enum": ["ALL_MEDIA", "PHOTO", "VIDEO"]},
                "order": {"type": "STRING", "enum": ["MEDIA_ITEM_DATE_DESC", "MEDIA_ITEM_DATE_ASC"]},
            },
        ),
        categories=("google_photos", "external"),
        connector_id="google_photos",
        capability="media.search",
    ),
    ToolSpec(
        "google_photos_get_media_item",
        "Read one Google Photos media item by id.",
        parameters=_object(
            {"media_item_id": {"type": "STRING", "description": "Media item id."}},
            ("media_item_id",),
        ),
        categories=("google_photos", "external"),
        connector_id="google_photos",
        capability="media.read",
    ),
    ToolSpec(
        "google_photos_list_albums",
        "List the authenticated user's Google Photos albums.",
        parameters=_object(
            {
                "page_size": {"type": "INTEGER", "description": "Number of albums, 1 through 50."},
                "page_token": {"type": "STRING", "description": "Pagination token from a prior result."},
            },
        ),
        categories=("google_photos", "external"),
        connector_id="google_photos",
        capability="media.search",
    ),
    # ---- YouTube ---------------------------------------------------------- #
    ToolSpec(
        "youtube_search",
        "Search public YouTube videos, channels, or playlists.",
        parameters=_object(
            {
                "query": {"type": "STRING", "description": "Search query."},
                "resource_type": {"type": "STRING", "enum": ["video", "channel", "playlist"]},
                "max_results": {"type": "INTEGER", "description": "Number of results, 1 through 50."},
                "page_token": {"type": "STRING"},
            },
            ("query",),
        ),
        categories=("youtube", "external"),
        connector_id="youtube",
        capability="search.read",
    ),
    ToolSpec(
        "youtube_get_videos",
        "Read details for specific YouTube videos.",
        parameters=_object(
            {"video_ids": {"type": "ARRAY", "description": "YouTube video ids.", "items": {"type": "STRING"}}},
            ("video_ids",),
        ),
        categories=("youtube", "external"),
        connector_id="youtube",
        capability="videos.read",
    ),
    ToolSpec(
        "youtube_get_my_channel",
        "Read the authenticated user's YouTube channel.",
        parameters={},
        categories=("youtube", "external"),
        connector_id="youtube",
        capability="videos.read",
    ),
    ToolSpec(
        "youtube_list_my_videos",
        "List videos from the authenticated channel's uploads.",
        parameters=_object(
            {
                "max_results": {"type": "INTEGER", "description": "Number of videos, 1 through 50."},
                "page_token": {"type": "STRING"},
            },
        ),
        categories=("youtube", "external"),
        connector_id="youtube",
        capability="videos.read",
    ),
    # ---- Instagram -------------------------------------------------------- #
    ToolSpec(
        "instagram_get_profile",
        "Read the authenticated Instagram profile.",
        parameters={},
        categories=("instagram", "external"),
        connector_id="instagram",
        capability="profile.read",
    ),
    ToolSpec(
        "instagram_list_media",
        "List the authenticated Instagram account's media.",
        parameters=_object(
            {
                "limit": {"type": "INTEGER", "description": "Number of items, 1 through 100."},
                "after": {"type": "STRING", "description": "Pagination cursor from a prior result."},
            },
        ),
        categories=("instagram", "external"),
        connector_id="instagram",
        capability="media.read",
    ),
    ToolSpec(
        "instagram_get_media_details",
        "Read details for one Instagram media item.",
        parameters=_object({"media_id": {"type": "STRING", "description": "Instagram media id."}}, ("media_id",)),
        categories=("instagram", "external"),
        connector_id="instagram",
        capability="media.read",
    ),
    # ---- Facebook --------------------------------------------------------- #
    ToolSpec(
        "facebook_list_pages",
        "List the Facebook Pages the authenticated user manages.",
        parameters={},
        categories=("facebook", "external"),
        connector_id="facebook",
        capability="pages.read",
    ),
    ToolSpec(
        "facebook_list_page_posts",
        "List posts on a Facebook Page.",
        parameters=_object(
            {
                "page_id": {"type": "STRING", "description": "Facebook Page id."},
                "limit": {"type": "INTEGER", "description": "Number of posts, 1 through 100."},
                "after": {"type": "STRING", "description": "Pagination cursor from a prior result."},
            },
            ("page_id",),
        ),
        categories=("facebook", "external"),
        connector_id="facebook",
        capability="pages.read",
    ),
    # ---- Canva ------------------------------------------------------------ #
    ToolSpec(
        "canva_list_designs",
        "List the authenticated user's Canva designs.",
        parameters=_object(
            {
                "query": {"type": "STRING", "description": "Optional name search."},
                "continuation": {"type": "STRING", "description": "Pagination token from a prior result."},
            },
        ),
        categories=("canva", "external"),
        connector_id="canva",
        capability="designs.read",
    ),
    ToolSpec(
        "canva_get_design",
        "Read one Canva design.",
        parameters=_object({"design_id": {"type": "STRING"}}, ("design_id",)),
        categories=("canva", "external"),
        connector_id="canva",
        capability="designs.read",
    ),
    ToolSpec(
        "canva_get_export_formats",
        "List the export formats available for a Canva design.",
        parameters=_object({"design_id": {"type": "STRING"}}, ("design_id",)),
        categories=("canva", "external"),
        connector_id="canva",
        capability="designs.read",
    ),
    ToolSpec(
        "canva_create_export_job",
        "Start an export job for a Canva design. Requires the user's explicit approval.",
        parameters=_object(
            {
                "design_id": {"type": "STRING"},
                "format": {
                    "type": "OBJECT",
                    "description": "Export format payload, e.g. {\"type\": \"PDF\"}.",
                    "properties": {"type": {"type": "STRING"}},
                },
            },
            ("design_id",),
        ),
        risk="external",
        requires_approval=True,
        categories=("canva", "external", "write"),
        connector_id="canva",
        capability="exports.create",
    ),
    # ---- Local helpers ---------------------------------------------------- #
    ToolSpec(
        "data_analysis",
        "Summarize CSV, JSON, or TSV data: row count, numeric statistics, duplicates.",
        parameters=_object(
            {
                "content": {"type": "STRING", "description": "The CSV/JSON/TSV text."},
                "filename": {"type": "STRING", "description": "File name; its extension selects the parser (e.g. data.csv)."},
            },
            ("content",),
        ),
        categories=("data",),
    ),
    ToolSpec(
        "task_schedule_build",
        "Validate and normalize a schedule specification before it is stored.",
        parameters=_object(
            {
                "schedule": {"type": "STRING", "description": "Schedule expression or description."},
                "timezone": {"type": "STRING", "description": "IANA timezone; defaults to UTC."},
                "retry_limit": {"type": "INTEGER", "description": "Retries, 0 through 20."},
                "misfire_policy": {"type": "STRING", "enum": ["skip", "run_once"]},
            },
            ("schedule",),
        ),
        categories=("productivity",),
    ),
)


# --------------------------------------------------------------------------- #
# Lookups and policy helpers
# --------------------------------------------------------------------------- #

def get_tool_spec(name: str) -> ToolSpec | None:
    normalized = name.strip().lower()
    return next((item for item in TOOL_SPECS if item.name == normalized), None)


def tool_names() -> frozenset[str]:
    return frozenset(item.name for item in TOOL_SPECS)


def auto_approved_tool_names() -> frozenset[str]:
    return frozenset(item.name for item in TOOL_SPECS if not item.requires_approval)


def requires_approval(name: str) -> bool:
    spec = get_tool_spec(name)
    return True if spec is None else spec.requires_approval


def describe_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "description": item.description,
            "parameters": dict(item.parameters),
            "payload_style": item.payload_style,
            "risk": item.risk,
            "requires_approval": item.requires_approval,
            "timeout_seconds": item.timeout_seconds,
            "max_output_chars": item.max_output_chars,
            "categories": list(item.categories),
            "connector_id": item.connector_id,
            "capability": item.capability,
            "module": item.module,
        }
        for item in TOOL_SPECS
    ]


# --------------------------------------------------------------------------- #
# Model-facing function declarations
# --------------------------------------------------------------------------- #

def tool_declaration(spec: ToolSpec) -> dict[str, Any]:
    """One model-facing declaration for a tool (parameters omitted when empty)."""
    declaration: dict[str, Any] = {"name": spec.name, "description": spec.description}
    if spec.parameters:
        declaration["parameters"] = dict(spec.parameters)
    return declaration


def function_declarations(
    names: tuple[str, ...] | None = None,
    *,
    connector_id: str | None = None,
    read_only: bool = False,
) -> list[dict[str, Any]]:
    """Return Gemini ``functionDeclarations`` built from the catalogue.

    ``names``        restrict to these tool names (in catalogue order).
    ``connector_id`` restrict to tools for one connector.
    ``read_only``    exclude tools that require approval.
    """
    wanted = set(names) if names is not None else None
    declarations: list[dict[str, Any]] = []
    for spec in TOOL_SPECS:
        if wanted is not None and spec.name not in wanted:
            continue
        if connector_id is not None and spec.connector_id != connector_id:
            continue
        if read_only and spec.requires_approval:
            continue
        declarations.append(tool_declaration(spec))
    return declarations


# The read-only GitHub tools the model may call. Derived from the catalogue so
# the schema can never drift from the spec.
GITHUB_TOOL_DECLARATIONS: list[dict[str, Any]] = function_declarations(
    connector_id="github", read_only=True
)


# The GitHub writes the model may propose. requires_approval is set on each spec,
# so function_declarations(read_only=True) above cannot pick them up by accident:
# they live in their own list and are only ever reached through the approval path.
GITHUB_WRITE_TOOL_DECLARATIONS: list[dict[str, Any]] = function_declarations(
    names=(
        "github_create_branch",
        "github_delete_branch",
        "github_create_pull_request",
        "github_merge_pull_request",
    ),
    connector_id="github",
    read_only=False,
)


# --------------------------------------------------------------------------- #
# Email plugin declarations
# --------------------------------------------------------------------------- #
# Kept here so every tool definition lives in one file. The email plugin
# imports this list rather than defining its own copy.

EMAIL_TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "list_recent_emails",
        "description": "Get the newest messages in the selected mailbox Inbox. Use for requests such as check my mail, show latest/recent emails, or what arrived recently. This is read-only.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"], "description": "Mailbox to use when clear from the user's request; omit if not specified."},
                "limit": {"type": "INTEGER", "description": "Number of messages to return, between 1 and 200. Defaults to 200 unless the user asks for fewer."},
            },
        },
    },
    {
        "name": "count_inbox_emails",
        "description": "Count the messages in the selected mailbox Inbox independently of the displayed list limit. Use whenever the user asks how many emails/messages are in the Inbox. Return the exact count only when exact is true; if exact is false, report the count as a lower bound and say the mailbox is larger than the checked range. This is read-only.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"], "description": "Mailbox to count when clear from the user's request; omit if not specified."},
            },
        },
    },
    {
        "name": "search_emails",
        "description": "Search a connected mailbox for emails matching a subject, sender, topic, or keyword. Use the user's specific search words. This is read-only.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "A concise search phrase, subject, sender, or topic."},
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "limit": {"type": "INTEGER", "description": "Number of matching messages, between 1 and 200. Defaults to 200 unless the user asks for fewer."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_email",
        "description": "Read the full body of a particular email. Prefer message_id from prior tool results; otherwise search by a specific subject/sender phrase. If multiple matches exist, ask the user to disambiguate instead of choosing arbitrarily.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Specific subject or sender phrase to identify the email."},
                "message_id": {"type": "STRING", "description": "Exact message id from a prior email tool result, if known."},
                "account_id": {"type": "STRING", "description": "Account id returned by an earlier result; required with folder_id for reading a Zoho message by id."},
                "folder_id": {"type": "STRING", "description": "Folder id returned by an earlier result; required with account_id for reading a Zoho message by id."},
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
            },
        },
    },
    {
        "name": "prepare_email_reply",
        "description": "Prepare a reply to one specific Zoho Mail message and wait for the user's explicit CONFIRM. Search by a specific subject, sender, or topic. Never send without confirmation.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["zoho_mail"]},
                "query": {"type": "STRING", "description": "Specific subject, sender, or phrase identifying exactly one email to reply to."},
                "body": {"type": "STRING", "description": "The reply body to show the user before confirmation."},
            },
            "required": ["query", "body"],
        },
    },
    {
        "name": "list_email_drafts",
        "description": "List saved drafts in the selected Gmail or Zoho Mail account. This is read-only.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "limit": {"type": "INTEGER", "description": "Number of drafts to return, between 1 and 20."},
            },
        },
    },
    {
        "name": "create_email_draft",
        "description": "Save a new email draft. This does not send an email. Use the user's requested recipient, subject, and message, and ask for missing details in the final answer when needed.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "to": {"type": "STRING", "description": "Recipient email address, if provided."},
                "subject": {"type": "STRING", "description": "Email subject."},
                "body": {"type": "STRING", "description": "Draft email body."},
            },
            "required": ["body"],
        },
    },
    {
        "name": "update_email_draft",
        "description": "Update an existing draft matching a specific subject or phrase. Use only when the user asks to edit/change a draft. If the match is ambiguous, do not modify anything and ask which draft.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "query": {"type": "STRING", "description": "Specific subject or phrase to find the existing draft."},
                "instructions": {"type": "STRING", "description": "Requested changes to the existing draft."},
            },
            "required": ["query", "instructions"],
        },
    },
    {
        "name": "prepare_email_to_send",
        "description": "Prepare an email and ask for explicit confirmation. This tool must never send directly. Use only when the user explicitly asks to send an email; show recipient, subject, and full body and wait for CONFIRM.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "to": {"type": "STRING", "description": "One recipient email address."},
                "subject": {"type": "STRING", "description": "Email subject."},
                "body": {"type": "STRING", "description": "Email body."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "prepare_email_deletion",
        "description": "Find matching emails and prepare a deletion-to-Trash confirmation preview. Never delete directly. Use only when the user explicitly asks to delete/remove email; ask for a filter if their request is broad.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "query": {"type": "STRING", "description": "Specific subject, sender, topic, or other limiting search phrase. Must not be empty or a broad all-mail request."},
                "limit": {"type": "INTEGER", "description": "Maximum number of matching messages to include in the preview, between 1 and 20."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "prepare_email_draft_deletion",
        "description": "Find one saved draft by a specific subject or phrase and prepare deletion. Never delete directly; the user must confirm.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "provider": {"type": "STRING", "enum": ["gmail", "zoho_mail"]},
                "query": {"type": "STRING", "description": "Specific draft subject or phrase. Broad all-drafts requests are not allowed."},
            },
            "required": ["query"],
        },
    },
]
