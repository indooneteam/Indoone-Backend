"""Backward-compatible import path for GitHub write operations.

The implementations are owned by app.plugins.github.writes.
"""

from app.plugins.github.writes import github_comment_issue_or_pr, github_create_issue

__all__ = ["github_comment_issue_or_pr", "github_create_issue"]
