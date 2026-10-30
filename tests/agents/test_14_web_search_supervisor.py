"""Agent 14 - the web supervisor.

The SSRF tests are the important ones: this is the only place in the product where a string
an attacker chose becomes a request from inside our own network.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agents import load

web = load("14_web_search_supervisor")

PUBLIC = lambda host: ["93.184.216.34"]  # noqa: E731
PRIVATE = lambda host: ["10.0.0.7"]  # noqa: E731


def run(call, **kwargs):
    return asyncio.run(web.validate_and_execute(call, **kwargs))


# ---------------------------------------------------------------- 1. declarations


def test_the_declarations_are_function_calling_shaped() -> None:
    declarations = web.get_tool_declarations()

    assert [item["name"] for item in declarations] == ["web_search", "web_fetch_page"]
    for item in declarations:
        assert item["parameters"]["type"] == "object"


def test_a_caller_cannot_change_the_declarations() -> None:
    web.get_tool_declarations()[0]["name"] = "web_scrape_everything"

    assert web.get_tool_declarations()[0]["name"] == "web_search"


# ---------------------------------------------------------------- 2. the perimeter


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/health",
        "http://[::1]/x",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://172.16.0.9/x",
        "http://0.0.0.0/x",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/x",
        "https://example.com:8080/x",
        "http://example.com:22/x",
    ],
)
def test_a_dangerous_url_is_refused(url: str) -> None:
    allowed, reason = web.is_safe_url(url, resolver=PUBLIC)

    assert allowed is False
    assert reason


def test_a_public_url_passes() -> None:
    allowed, reason = web.is_safe_url("https://docs.python.org/3/", resolver=PUBLIC)

    assert allowed is True
    assert reason is None


def test_a_name_that_resolves_to_a_private_address_is_the_attack() -> None:
    """The hostname looks public; the answer is our own network."""
    allowed, reason = web.is_safe_url("https://looks-fine.example.com/x", resolver=PRIVATE)

    assert allowed is False
    assert "private" in reason


def test_a_name_that_does_not_resolve_is_refused() -> None:
    allowed, reason = web.is_safe_url("https://nowhere.invalid/x", resolver=lambda host: [])

    assert allowed is False
    assert "does not resolve" in reason


def test_the_fetch_refuses_before_any_request_is_made() -> None:
    called: list[str] = []

    async def fetcher(url):
        called.append(url)
        return "<html><body>secret</body></html>"

    result = run(web.WebToolCall("web_fetch_page", {"url": "http://169.254.169.254/latest/meta-data/"}), fetcher=fetcher)

    assert result.success is False
    assert result.error_code == "BLOCKED_URL"
    assert called == [], "a blocked URL must never be requested"


# ---------------------------------------------------------------- 3. the purifier


def test_the_purifier_keeps_the_prose_and_drops_the_machinery() -> None:
    page = """
    <html><head><title>Python 3.14</title>
    <style>body{color:red}</style></head>
    <body>
      <script>alert('x')</script>
      <nav><a href="/">Home</a></nav>
      <header>Site header</header>
      <p>The release adds template strings.</p>
      <footer>Copyright</footer>
      <noscript>Enable JS</noscript>
    </body></html>
    """

    text = web.purify_html(page)

    assert "The release adds template strings." in text
    for dropped in ("alert(", "color:red", "Home", "Site header", "Copyright", "Enable JS"):
        assert dropped not in text


def test_a_huge_page_is_cut_with_a_notice() -> None:
    page = "<p>" + ("This is a sentence about the release. " * 2000) + "</p>"

    text = web.purify_html(page)

    assert len(text) < web.MAX_CONTENT_CHARS + 100
    assert "[Web content truncated for budget]" in text
    assert text.endswith("[Web content truncated for budget]")


def test_a_small_page_is_left_whole() -> None:
    assert web.purify_html("<p>Short page.</p>") == "Short page."


def test_malformed_markup_does_not_lose_the_page() -> None:
    assert "still readable" in web.purify_html("<p>still readable<div><span>")


# ---------------------------------------------------------------- 4. tracking and spam


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a?utm_source=x&utm_medium=y&id=7",
        "https://example.com/a?id=7&fbclid=abc",
        "https://example.com/a?gclid=abc&msclkid=z",
    ],
)
def test_tracking_parameters_are_stripped(url: str) -> None:
    cleaned = web._strip_tracking(url)

    for parameter in ("utm_source", "utm_medium", "fbclid", "gclid", "msclkid"):
        assert parameter not in cleaned


def test_a_real_parameter_survives() -> None:
    assert web._strip_tracking("https://example.com/a?id=7&utm_source=x") == "https://example.com/a?id=7"


def test_an_advertising_host_is_not_a_source() -> None:
    async def searcher(query, limit):
        return [
            {"title": "Real", "url": "https://docs.python.org/3/", "snippet": "the docs"},
            {"title": "Ad", "url": "https://doubleclick.net/x", "snippet": "buy"},
        ]

    result = run(web.WebToolCall("web_search", {"query": "python 3.14"}), searcher=searcher)

    assert result.success is True
    assert len(result.data) == 1
    assert result.data[0]["domain"] == "docs.python.org"


# ---------------------------------------------------------------- 5. results


def test_a_search_returns_grounded_items_and_chips() -> None:
    async def searcher(query, limit):
        return [
            {"title": "Python 3.14 release", "url": "https://python.org/x?utm_source=a", "snippet": "Released today."}
        ]

    result = run(web.WebToolCall("web_search", {"query": "python release", "limit": 5}), searcher=searcher)

    assert result.success is True
    assert result.data[0]["domain"] == "python.org"
    assert "utm_source" not in result.data[0]["url"]
    labels = [chip["label"] for chip in result.action_chips]
    assert "Source: python.org" in labels
    assert "Search More" in labels


def test_an_empty_result_is_reported_not_invented() -> None:
    async def searcher(query, limit):
        return []

    result = run(web.WebToolCall("web_search", {"query": "nothing at all"}), searcher=searcher)

    assert result.success is False
    assert result.error_code == "NOT_FOUND"


def test_a_fetch_returns_clean_text_and_a_source_chip() -> None:
    async def fetcher(url):
        return "<html><body><script>x()</script><p>The page says this.</p></body></html>"

    result = run(
        web.WebToolCall("web_fetch_page", {"url": "https://example.com/page"}),
        fetcher=fetcher,
        resolver=PUBLIC,
    )

    assert result.success is True
    assert result.sanitized_output == "The page says this."
    assert result.action_chips[0]["label"] == "Source: example.com"


# ---------------------------------------------------------------- 6. failures and speed


@pytest.mark.parametrize(
    ("error", "code"),
    [("404 Not Found", "NOT_FOUND"), ("500 Internal Server Error", "UNAVAILABLE"), ("timed out", "TIMEOUT")],
)
def test_an_http_failure_becomes_a_result(error: str, code: str) -> None:
    async def fetcher(url):
        raise RuntimeError(error)

    result = run(
        web.WebToolCall("web_fetch_page", {"url": "https://example.com/x"}),
        fetcher=fetcher,
        resolver=PUBLIC,
    )

    assert result.success is False
    assert result.error_code == code


def test_an_empty_query_is_refused() -> None:
    result = run(web.WebToolCall("web_search", {"query": "   "}))

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENT"


def test_a_limit_is_clamped() -> None:
    arguments, _ = web.validate_arguments(web.WebToolCall("web_search", {"query": "x", "limit": 500}))

    assert arguments["limit"] == web.MAX_RESULTS


def test_the_perimeter_check_costs_under_its_budget() -> None:
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        web.is_safe_url("https://docs.python.org/3/", resolver=PUBLIC)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
