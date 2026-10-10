"""Intent detection and source formatting for keyless public web research."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.web_research.research import ResearchResult

_LIVE_RESEARCH_TERMS = (
    "latest", "current", "currently", "recent", "today", "news", "search",
    "research", "sources", "source links", "look up", "find online", "price",
    "cost", "availability", "weather", "update", "updates", "right now",
    "ಈಗಿನ", "ಇತ್ತೀಚಿನ", "ಇವತ್ತಿನ", "ಇಂದು", "ಈಗ", "ಸುದ್ದಿ", "ಹುಡುಕು",
    "ಹುಡುಕಿ", "ಬೆಲೆ", "ಹವಾಮಾನ", "ಮಾಹಿತಿ ಹುಡುಕು",
    "ivattina", "ivattu", "eegina", "eega", "hosa suddi", "suddi",
    "bele", "havamaana", "latest update",
)


def should_use_live_research(message: str) -> bool:
    """Search only when the user's wording signals a current/live lookup."""
    normalized = " ".join(message.casefold().split())
    if not normalized:
        return False
    return any(term in normalized for term in _LIVE_RESEARCH_TERMS)


def split_sources_footer(answer: str) -> tuple[str, list[ResearchResult]]:
    """Extract verified HTTP(S) source links from the answer's source footer."""
    marker = "\n\nSources:\n"
    marker_index = answer.rfind(marker)
    if marker_index < 0:
        return answer, []
    footer = answer[marker_index + len(marker):]
    pattern = re.compile(r"^- \[(?P<title>.+?)\]\((?P<url>https?://[^)\s]+)\)$")
    sources: list[ResearchResult] = []
    seen: set[str] = set()
    for line in footer.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        title = " ".join(match.group("title").split()).strip()[:500]
        url = match.group("url").strip()
        parsed = urlparse(url)
        if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        key = url.rstrip("/").casefold()
        if key not in seen:
            seen.add(key)
            sources.append(ResearchResult(title=title, url=url, snippet=""))
    return (answer[:marker_index].rstrip(), sources) if sources else (answer, [])


def format_sources_footer(answer: str, sources: list[ResearchResult]) -> str:
    """Append source links obtained from public providers, never invented URLs."""
    clean_answer = answer.rstrip()
    lines = [clean_answer, "", "Sources:"]
    for source in sources:
        title = " ".join(source.title.split()).strip()[:500]
        url = source.url.strip()
        parsed = urlparse(url)
        if title and parsed.scheme in {"http", "https"} and parsed.netloc:
            lines.append(f"- [{title}]({url})")
    return "\n".join(lines) if len(lines) > 3 else clean_answer
