"""The page a share link opens.

Rendered on the server, not in the browser: the transcript is already in the database, and
a fetch-then-render page would need JavaScript to show text that was known before the
response was written. This way the page works with scripting off, has no bundle to
download, and shows the whole thread in the first paint.

Read-only by construction: there is no input, no form and no write path anywhere in this
module.
"""

from __future__ import annotations

import html
import re

_TITLE_LIMIT = 90

# Fenced code first, because nothing inside a fence should be touched by the inline rules.
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"(?<![\"(])(https?://[^\s<>\"]+)")


def _inline(text: str) -> str:
    """Inline markdown, applied to text that is ALREADY escaped.

    Order matters: a link's label may contain bold, but a URL must never be re-formatted
    into something clickable twice.
    """
    escaped = html.escape(text, quote=False)

    escaped = _LINK.sub(
        lambda match: (
            f'<a href="{match.group(2)}" rel="noopener noreferrer nofollow" target="_blank">'
            f"{match.group(1)}</a>"
        ),
        escaped,
    )
    escaped = _INLINE_CODE.sub(r"<code>\1</code>", escaped)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _BARE_URL.sub(
        lambda match: (
            f'<a href="{match.group(1)}" rel="noopener noreferrer nofollow" target="_blank">'
            f"{match.group(1)}</a>"
        ),
        escaped,
    )

    return escaped.replace("\n", "<br/>")


def render_body(text: str) -> str:
    """A message body as HTML: code blocks preserved, everything else escaped."""
    parts: list[str] = []
    cursor = 0

    for match in _FENCE.finditer(text or ""):
        before = (text or "")[cursor : match.start()]

        if before.strip():
            parts.append(f'<div class="text">{_inline(before.strip())}</div>')

        language = match.group(1).strip()
        code = html.escape(match.group(2).rstrip("\n"), quote=False)
        label = f'<div class="code-label">{html.escape(language)}</div>' if language else ""
        parts.append(f'<div class="code">{label}<pre><code>{code}</code></pre></div>')
        cursor = match.end()

    tail = (text or "")[cursor:]

    if tail.strip():
        parts.append(f'<div class="text">{_inline(tail.strip())}</div>')

    return "".join(parts) or '<div class="text"></div>'


_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 16px 48px;
  background: #FAF9FC; color: #242129;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Kannada", sans-serif;
  line-height: 1.55;
}
.banner {
  position: sticky; top: 0; z-index: 2;
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  margin: 0 -16px 18px; padding: 10px 16px;
  background: rgba(250, 249, 252, .92); backdrop-filter: blur(6px);
  border-bottom: 1px solid #E5E0ED;
  font-size: 13px; color: #6B6478;
}
.banner a {
  flex: none; padding: 7px 14px; border-radius: 999px;
  background: #703BE2; color: #fff; text-decoration: none; font-weight: 600; font-size: 13px;
}
main { max-width: 720px; margin: 0 auto; }
h1 { font-size: 19px; line-height: 1.3; margin: 6px 0 18px; word-break: break-word; }
.row { display: flex; margin-bottom: 10px; }
.row.user { justify-content: flex-end; }
.bubble {
  max-width: 88%; padding: 10px 13px; border-radius: 14px;
  background: #fff; border: 1px solid #E5E0ED;
}
.row.user .bubble { background: #F4F2FF; border-color: #DED6F6; }
.who { font-size: 11px; color: #6B6478; margin-bottom: 4px; }
.text { font-size: 14px; word-wrap: break-word; overflow-wrap: anywhere; }
.text a { color: #5E2DD2; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
.text code { background: #F1EEF8; padding: 1px 5px; border-radius: 5px; }
.code { margin: 8px 0; border: 1px solid #E5E0ED; border-radius: 10px; overflow: hidden; background: #fff; }
.code-label { font-size: 11px; color: #6B6478; padding: 5px 10px; background: #F7F5FB; border-bottom: 1px solid #E5E0ED; }
.code pre { margin: 0; padding: 10px 12px; overflow-x: auto; }
.empty { color: #6B6478; font-size: 14px; }
.footer { max-width: 720px; margin: 26px auto 0; font-size: 12px; color: #8B8497; text-align: center; }
"""


def _page(title: str, inner: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="robots" content="noindex, nofollow"/>
<title>{html.escape(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="banner">
  <span>Shared via Indoone AI</span>
  <a href="https://indoone.ai" rel="noopener">Try Indoone</a>
</div>
<main>
{inner}
</main>
<p class="footer">This is a read-only copy of a conversation.</p>
</body>
</html>"""


def render_share_page(title: str, messages: list[tuple[str, str]]) -> str:
    """The transcript page. Nothing here accepts input."""
    heading = (title or "").strip() or "Shared conversation"
    heading = heading[:_TITLE_LIMIT]

    if not messages:
        inner = (
            f"<h1>{html.escape(heading)}</h1>"
            '<p class="empty">This conversation has no messages yet.</p>'
        )

        return _page(f"{heading} — Indoone AI", inner)

    rows: list[str] = []

    for role, content in messages:
        is_user = role == "user"
        who = "You" if is_user else "Indoone AI"
        rows.append(
            f'<div class="row {"user" if is_user else "ai"}">'
            f'<div class="bubble"><div class="who">{who}</div>{render_body(content)}</div>'
            f"</div>"
        )

    inner = f"<h1>{html.escape(heading)}</h1>" + "".join(rows)

    return _page(f"{heading} — Indoone AI", inner)


def render_not_found() -> str:
    """The page for a link that does not exist, and for one that has been withdrawn.

    Deliberately the same answer for both: telling a stranger which of the two it is would
    say whether a share id ever existed.
    """
    inner = (
        "<h1>Chat not found or link expired</h1>"
        '<p class="empty">This link does not point to a shared conversation. '
        "It may have been withdrawn by the person who created it.</p>"
    )

    return _page("Chat not found — Indoone AI", inner)
