"""Agent 14 - the web supervisor: the internet, through one narrow door.

A model that answers from memory about "the latest version" invents one, and a model asked
to link a source invents a URL that looks right. Both are worse than saying nothing, because
a plausible wrong answer is not questioned.

So the web is reached here, and only here:

* **A URL is checked before it is fetched.** ``http://127.0.0.1:8000`` is our own admin port;
  ``http://169.254.169.254/latest/meta-data/`` is the instance's credentials; a name that
  resolves to a private range is the same attack wearing a domain. The check resolves the
  host and refuses anything that is not a public address, on the standard ports, over
  http(s). This is the one place in the product where an attacker-supplied string becomes a
  request from inside our network.
* **A page is text, not a document.** Scripts, styles, navigation, cookie banners and
  trackers are removed before anything is read, and what is left is bounded: a page past
  6,000 characters is cut at a sentence with a notice, because the point of a page is the
  paragraph that answers the question.
* **A result says where it came from.** Title, domain, and a clean URL with the tracking
  parameters stripped. A citation that cannot be opened is not a citation.

It executes the search and the fetch through the provider that already exists
(``app/web_research``); this is the perimeter around it.
"""

from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import socket
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

MAX_RESULTS = 8
MAX_CONTENT_CHARS = 6_000
MAX_QUERY_CHARS = 400
ALLOWED_PORTS = {None, 80, 443}
MAX_REDIRECTS = 3

# Advertising, tracking and affiliate hosts. A result from one of these is not a source.
_BLOCKED_HOSTS = (
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "adservice.google.com", "ads.yahoo.com", "adnxs.com", "taboola.com",
    "outbrain.com", "criteo.com", "quantserve.com", "scorecardresearch.com",
    "clickbank.net", "amzn.to", "bit.ly", "tinyurl.com",
)

# Tracking parameters, stripped from every URL this agent hands back.
_TRACKING_PARAMS = ("fbclid", "gclid", "dclid", "msclkid", "mc_eid", "igshid", "ref_src", "yclid")

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

_MESSAGES = {
    "BLOCKED_URL": {
        "kn": "ಆ ಲಿಂಕ್ ಸುರಕ್ಷಿತವಲ್ಲ, ಹಾಗಾಗಿ ತೆರೆಯಲಾಗಿಲ್ಲ.",
        "hi": "वह लिंक सुरक्षित नहीं है, इसलिए नहीं खोला गया।",
        "en": "That link is not safe to open, so I did not fetch it.",
    },
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
    "NOT_FOUND": {
        "kn": "ಆ ಪುಟ ಸಿಗಲಿಲ್ಲ.",
        "hi": "वह पेज नहीं मिला।",
        "en": "That page was not found.",
    },
    "UNAVAILABLE": {
        "kn": "ಆ ತಾಣ ಈಗ ಲಭ್ಯವಿಲ್ಲ.",
        "hi": "वह साइट अभी उपलब्ध नहीं है।",
        "en": "That site is not responding right now.",
    },
    "TIMEOUT": {
        "kn": "ಆ ತಾಣ ತುಂಬಾ ನಿಧಾನವಾಗಿದೆ.",
        "hi": "वह साइट बहुत धीमी है।",
        "en": "That site is taking too long to answer.",
    },
}

_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "web_search",
        "description": "Search the live web. Use when the answer depends on something recent.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "freshness": {"type": "string", "enum": ["day", "week", "month", "year"]},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_fetch_page",
        "description": "Read a page as clean text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extract_main_text": {"type": "boolean", "default": True},
            },
            "required": ["url"],
        },
    },
)

_REQUIRED = {"web_search": ("query",), "web_fetch_page": ("url",)}


@dataclass(slots=True)
class WebToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class SearchResultItem:
    title: str
    url: str
    snippet: str
    domain: str


@dataclass(slots=True)
class WebSupervisorResult:
    success: bool
    data: Any = None
    sanitized_output: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_WEB_SUPERVISOR_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_tool_declarations() -> list[dict[str, Any]]:
    return [json.loads(json.dumps(declaration)) for declaration in _TOOL_DECLARATIONS]


# ---------------------------------------------------------------- the perimeter


def _resolver(host: str) -> list[str]:
    """Every address a name points at. Injected in tests; DNS in production."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return []

    return [str(info[4][0]) for info in infos]


def _public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False

    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_safe_url(url: str, *, resolver: Callable[[str], list[str]] | None = None) -> tuple[bool, str | None]:
    """Whether this URL may be fetched from inside our network.

    The hostname is not enough: a domain can resolve to 127.0.0.1, and that is the attack.
    So the name is resolved and every address it answers with must be public.
    """
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in {"http", "https"}:
        return False, "only http and https are allowed"

    host = (parsed.hostname or "").strip()

    if not host:
        return False, "no host"

    try:
        port = parsed.port
    except ValueError:
        return False, "invalid port"

    if port not in ALLOWED_PORTS:
        return False, f"port {port} is not allowed"

    if host.casefold() in {"localhost", "localhost.localdomain"} or host.casefold().endswith(".localhost"):
        return False, "localhost is not allowed"

    # A literal address is checked directly, without a lookup.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        return (True, None) if _public(str(literal)) else (False, "private address")

    addresses = (resolver or _resolver)(host)

    if not addresses:
        return False, "the host does not resolve"

    if not all(_public(address) for address in addresses):
        return False, "the host resolves to a private address"

    return True, None


def _strip_tracking(url: str) -> str:
    """Remove the parameters that only exist to follow the person who clicked."""
    parsed = urlparse(url)

    if not parsed.query:
        return url

    kept = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not name.casefold().startswith("utm_") and name.casefold() not in _TRACKING_PARAMS
    ]

    return urlunparse(parsed._replace(query=urlencode(kept)))


def _blocked_host(host: str) -> bool:
    lowered = (host or "").casefold()
    return any(lowered == blocked or lowered.endswith("." + blocked) for blocked in _BLOCKED_HOSTS)


# ---------------------------------------------------------------- the purifier


class _Reader(HTMLParser):
    """Keeps the prose, drops the machinery."""

    _DROP = {"script", "style", "nav", "footer", "header", "noscript", "iframe", "svg", "form"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str = ""
        self._dropping = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._DROP:
            self._dropping += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._DROP and self._dropping:
            self._dropping -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._dropping:
            return

        if self._in_title:
            self.title += data
            return

        self.parts.append(data)


def purify_html(raw_html: str) -> str:
    """Readable text from a page, bounded."""
    reader = _Reader()

    try:
        reader.feed(raw_html or "")
        reader.close()
    except Exception:  # malformed markup must not lose the page
        pass

    text = "".join(reader.parts)
    text = html.unescape(text)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANK_LINES.sub("\n\n", text).strip()

    if len(text) > MAX_CONTENT_CHARS:
        window = text[:MAX_CONTENT_CHARS]
        cut = window.rfind("\n")

        if cut < MAX_CONTENT_CHARS // 2:
            # No paragraph break near the end: end on a sentence, not mid-word.
            sentences = list(_SENTENCE_END.finditer(window))
            cut = sentences[-1].end() if sentences else window.rfind(" ")

        text = text[: cut if cut > 0 else MAX_CONTENT_CHARS].rstrip()
        text += "\n\n[Web content truncated for budget]"

    return text


# ---------------------------------------------------------------- errors


def _error(code: str, language: str) -> WebSupervisorResult:
    table = _MESSAGES.get(code, _MESSAGES["INVALID_ARGUMENT"])
    key = (language or "en").strip().casefold()[:2]
    message = table.get(key, table["en"])

    return WebSupervisorResult(
        success=False,
        error_code=code,
        error_message=message,
        sanitized_output=message,
    )


def _status_of(exc: Exception) -> int:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value

    text = str(exc).casefold()

    if "timeout" in text or "timed out" in text:
        return -1
    for status in (404, 403, 401, 429, 500, 502, 503, 504):
        if str(status) in text:
            return status

    return 0


def _error_for_exception(exc: Exception, language: str) -> WebSupervisorResult:
    status = _status_of(exc)

    if status == -1:
        return _error("TIMEOUT", language)
    if status == 404:
        return _error("NOT_FOUND", language)
    return _error("UNAVAILABLE", language)


# ---------------------------------------------------------------- validation


def validate_arguments(call: WebToolCall) -> tuple[dict[str, Any] | None, str | None]:
    if call.tool_name not in _REQUIRED:
        return None, "unknown tool"

    arguments = dict(call.arguments or {})

    for name in _REQUIRED[call.tool_name]:
        value = arguments.get(name)
        if value is None or not str(value).strip():
            return None, f"missing argument {name!r}"

    if call.tool_name == "web_search":
        query = " ".join(str(arguments["query"]).split())[:MAX_QUERY_CHARS]
        arguments["query"] = query

        try:
            arguments["limit"] = max(1, min(MAX_RESULTS, int(arguments.get("limit", 5))))
        except (TypeError, ValueError):
            arguments["limit"] = 5

        freshness = str(arguments.get("freshness") or "").casefold()
        arguments["freshness"] = freshness if freshness in {"day", "week", "month", "year"} else None

    return arguments, None


# ---------------------------------------------------------------- execution


async def validate_and_execute(
    call: WebToolCall,
    *,
    searcher: Callable[..., Any] | None = None,
    fetcher: Callable[..., Any] | None = None,
    resolver: Callable[[str], list[str]] | None = None,
    language: str = "en",
) -> WebSupervisorResult:
    """Search or fetch, through the perimeter, and never raise."""
    started = time.perf_counter()

    arguments, problem = validate_arguments(call)

    if problem is not None:
        result = _error("INVALID_ARGUMENT", language)
        result.sanitized_output = f"{result.error_message} ({problem})"
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    assert arguments is not None

    if call.tool_name == "web_search":
        search = searcher or _default_searcher()

        try:
            raw = await search(arguments["query"], arguments["limit"])
        except Exception as exc:
            result = _error_for_exception(exc, language)
            result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
            return result

        items: list[SearchResultItem] = []

        for entry in raw or []:
            url = _strip_tracking(str(getattr(entry, "url", "") or entry.get("url", "") if isinstance(entry, dict) else getattr(entry, "url", "")))
            title = str(getattr(entry, "title", "") if not isinstance(entry, dict) else entry.get("title", ""))
            snippet = str(getattr(entry, "snippet", "") if not isinstance(entry, dict) else entry.get("snippet", ""))
            domain = (urlparse(url).hostname or "").removeprefix("www.")

            if not url or not title or _blocked_host(domain):
                continue

            items.append(
                SearchResultItem(
                    title=title.strip()[:200],
                    url=url,
                    snippet=" ".join(snippet.split())[:400],
                    domain=domain,
                )
            )

        if not items:
            result = _error("NOT_FOUND", language)
            result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
            return result

        lines = [f"- {item.title} ({item.domain}): {item.snippet}" for item in items]
        chips = [{"label": f"Source: {items[0].domain}", "action": f"open_url:{items[0].url}"}]
        chips.append({"label": "Search More", "action": "expand_search"})

        return WebSupervisorResult(
            success=True,
            data=[
                {"title": item.title, "url": item.url, "snippet": item.snippet, "domain": item.domain}
                for item in items
            ],
            sanitized_output="\n".join(lines),
            action_chips=chips,
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    url = _strip_tracking(str(arguments["url"]).strip())
    allowed, reason = is_safe_url(url, resolver=resolver)

    if not allowed:
        result = _error("BLOCKED_URL", language)
        result.sanitized_output = f"{result.error_message} ({reason})"
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    fetch = fetcher or _default_fetcher()

    try:
        raw = await fetch(url)
    except Exception as exc:
        result = _error_for_exception(exc, language)
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    body = raw if isinstance(raw, str) else str(raw or "")
    text = purify_html(body) if arguments.get("extract_main_text", True) else body[:MAX_CONTENT_CHARS]
    domain = (urlparse(url).hostname or "").removeprefix("www.")

    return WebSupervisorResult(
        success=True,
        data={"url": url, "domain": domain, "characters": len(text)},
        sanitized_output=text,
        action_chips=[
            {"label": f"Source: {domain}", "action": f"open_url:{url}"},
            {"label": "Search More", "action": "expand_search"},
        ],
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def _default_searcher() -> Callable[..., Any]:
    """The research provider that already exists, so there is one search implementation."""

    async def run(query: str, limit: int) -> list[Any]:
        from app.web_research.research import DuckDuckGoResearchProvider

        return await DuckDuckGoResearchProvider().search(query, limit=limit)

    return run


def _default_fetcher() -> Callable[..., Any]:
    async def run(url: str) -> str:
        import asyncio
        import urllib.request

        def blocking() -> str:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "IndooneBot/1.0 (+https://indoone.in)"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read(2_000_000).decode(charset, errors="replace")

        return await asyncio.to_thread(blocking)

    return run
