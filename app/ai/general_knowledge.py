from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import httpx


DEFAULT_TIMEOUT_SECONDS = 6.0
MAX_SUMMARY_CHARS = 2_500


@dataclass(frozen=True)
class WikipediaAnswer:
    title: str
    url: str
    extract: str


class WikipediaKnowledgeProvider:
    """Small public-knowledge fallback for factual questions.

    This path is intentionally separate from the trained model so a slow or
    unavailable local inference runtime cannot turn basic factual questions
    into a generic failure response.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        self.timeout = timeout

    async def answer(self, query: str, language: str = "English") -> WikipediaAnswer | None:
        query = " ".join(query.strip().split())
        if not query:
            return None

        language_codes = {
            "English": "en",
            "Kannada": "kn",
            "Hindi": "hi",
            "Telugu": "te",
            "Tamil": "ta",
            "Malayalam": "ml",
            "Marathi": "mr",
            "Bengali": "bn",
            "Assamese": "as",
            "Gujarati": "gu",
            "Punjabi": "pa",
            "Odia": "or",
            "Urdu": "ur",
        }
        code = language_codes.get(language, "en")
        api_base = f"https://{code}.wikipedia.org"

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=False,
            headers={"Accept": "application/json"},
        ) as client:
            search_response = await client.get(
                f"{api_base}/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": "0",
                    "srlimit": "1",
                    "format": "json",
                    "formatversion": "2",
                },
            )
            search_response.raise_for_status()
            search_payload = search_response.json()

            search_items = search_payload.get("query", {}).get("search", [])
            if not isinstance(search_items, list) or not search_items:
                return None

            title = str(search_items[0].get("title", "")).strip()
            if not title:
                return None

            encoded_title = quote(title.replace(" ", "_"), safe="")
            summary_response = await client.get(
                f"{api_base}/api/rest_v1/page/summary/{encoded_title}",
            )
            summary_response.raise_for_status()
            summary_payload = summary_response.json()

        extract = str(summary_payload.get("extract", "")).strip()
        if not extract:
            return None

        page_url = (
            str(
                summary_payload.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
            ).strip()
        )
        if not page_url:
            page_url = f"{api_base}/wiki/{encoded_title}"

        return WikipediaAnswer(
            title=title,
            url=page_url,
            extract=extract[:MAX_SUMMARY_CHARS],
        )


_ENGLISH_PREFIXES = (
    "what is ",
    "what are ",
    "what was ",
    "what were ",
    "who is ",
    "who was ",
    "where is ",
    "where was ",
    "when was ",
    "when did ",
    "why is ",
    "why are ",
    "why does ",
    "why do ",
    "what causes ",
    "explain ",
    "define ",
    "difference between ",
)

_SCRIPT_PREFIXES = (
    "ಏನು",
    "ಯಾರು",
    "ಎಲ್ಲಿ",
    "ಯಾವಾಗ",
    "ಏಕೆ",
    "ವಿವರಿಸಿ",
    "ವ್ಯತ್ಯಾಸ",
    "क्या",
    "कौन",
    "कहाँ",
    "कहां",
    "कब",
    "क्यों",
    "समझाइए",
    "ఏమిటి",
    "ఏది",
    "ఎవరు",
    "ఎక్కడ",
    "ఎప్పుడు",
    "ఎందుకు",
    "వివరించండి",
    "என்ன",
    "யார்",
    "எங்கே",
    "எப்போது",
    "ஏன்",
    "விளக்குங்கள்",
    "എന്താണ്",
    "ആരാണ്",
    "എവിടെ",
    "എപ്പോൾ",
    "എന്തുകൊണ്ട്",
)


def is_general_knowledge_question(message: str) -> bool:
    normalized = " ".join(message.strip().casefold().split())
    if not normalized:
        return False

    if any(normalized.startswith(prefix) for prefix in _ENGLISH_PREFIXES):
        return True

    return any(message.strip().startswith(prefix) for prefix in _SCRIPT_PREFIXES)
