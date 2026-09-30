from __future__ import annotations

from dataclasses import dataclass
import os
import re
from urllib.parse import quote

import httpx

from app.ai.language_detection import SCRIPT_RANGES


DEFAULT_TIMEOUT_SECONDS = 6.0
MAX_SUMMARY_CHARS = 2_500
DEFAULT_USER_AGENT = "Indoone/1.0 (https://github.com/indooneteam/Indoone-Backend)"


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

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        configured_user_agent = (
            user_agent
            if user_agent is not None
            else os.getenv("INDOONE_WIKIMEDIA_USER_AGENT", "")
        ).strip()
        self.timeout = timeout
        self.user_agent = configured_user_agent or DEFAULT_USER_AGENT

    async def _summary(self, client: httpx.AsyncClient, api_base: str, title: str) -> WikipediaAnswer | None:
        normalized_title = " ".join(title.strip().split())
        if not normalized_title:
            return None

        encoded_title = quote(normalized_title.replace(" ", "_"), safe="")
        summary_response = await client.get(
            f"{api_base}/api/rest_v1/page/summary/{encoded_title}",
        )
        if summary_response.status_code == 404:
            return None
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

        canonical_title = str(summary_payload.get("title", "")).strip() or normalized_title
        return WikipediaAnswer(
            title=canonical_title,
            url=page_url,
            extract=extract[:MAX_SUMMARY_CHARS],
        )

    @staticmethod
    def _title_relevance_score(topic: str, title: str) -> int:
        topic_normalized = " ".join(topic.casefold().split()).strip()
        title_normalized = " ".join(title.casefold().split()).strip()
        if not topic_normalized or not title_normalized:
            return 0
        if topic_normalized == title_normalized:
            return 100
        if topic_normalized in title_normalized or title_normalized in topic_normalized:
            return 50
        topic_terms = {
            token
            for token in re.findall(r"[\w\u0080-\uffff]+", topic_normalized, flags=re.UNICODE)
            if len(token) > 1
        }
        title_terms = {
            token
            for token in re.findall(r"[\w\u0080-\uffff]+", title_normalized, flags=re.UNICODE)
            if len(token) > 1
        }
        return len(topic_terms & title_terms)

    async def _summary_candidates(
        self,
        client: httpx.AsyncClient,
        api_base: str,
        topic: str,
    ) -> WikipediaAnswer | None:
        normalized = " ".join(topic.strip().split())
        candidates = [normalized]
        variants = (
            re.sub(r"\s+are\s+there\s+", " ", normalized, flags=re.IGNORECASE),
            re.sub(r"\s+are\s+", " ", normalized, flags=re.IGNORECASE),
            re.sub(r"\s+is\s+", " ", normalized, flags=re.IGNORECASE),
        )
        for variant in variants:
            variant = " ".join(variant.split()).strip(" ?!.")
            if variant and variant.casefold() not in {item.casefold() for item in candidates}:
                candidates.append(variant)
        if normalized.casefold().startswith(("states ", "list ", "number of ")):
            list_variant = f"List of {normalized}"
            if list_variant.casefold() not in {item.casefold() for item in candidates}:
                candidates.append(list_variant)

        for candidate in candidates:
            try:
                result = await self._summary(client, api_base, candidate)
            except httpx.HTTPStatusError:
                continue
            if result is not None:
                return result
        return None

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
        # Romanized questions (for example, "gravity andre enu?") do not
        # reliably search localized Wikipedia editions by their Latin spelling.
        # Use English Wikipedia as the factual source when the query is Latin-only.
        has_native_script = any(
            pattern.search(query)
            for _, pattern in SCRIPT_RANGES
        )
        if language != "English" and re.search(r"[A-Za-z]", query) and not has_native_script:
            code = "en"
        api_base = f"https://{code}.wikipedia.org"
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
            "Api-User-Agent": self.user_agent,
        }
        search_query = _question_to_topic(query)

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            try:
                search_response = await client.get(
                    f"{api_base}/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": search_query,
                        "srnamespace": "0",
                        "srlimit": "5",
                        "format": "json",
                        "formatversion": "2",
                    },
                )
                search_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Wikimedia may reject search traffic independently of the
                # page-summary endpoint. Try the normalized topic directly so
                # a temporary/search-specific 403 does not force local-model
                # inference on the Render Free CPU budget.
                if exc.response.status_code != 403:
                    raise
                return await self._summary_candidates(client, api_base, search_query)

            search_payload = search_response.json()
            search_items = search_payload.get("query", {}).get("search", [])
            if not isinstance(search_items, list) or not search_items:
                return await self._summary_candidates(client, api_base, search_query)

            ranked_titles = [
                str(item.get("title", "")).strip()
                for item in search_items[:5]
                if str(item.get("title", "")).strip()
            ]
            if not ranked_titles:
                return await self._summary_candidates(client, api_base, search_query)

            title = max(
                enumerate(ranked_titles),
                key=lambda pair: (self._title_relevance_score(search_query, pair[1]), -pair[0]),
            )[1]
            return await self._summary(client, api_base, title)


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
    "how is ",
    "how many ",
    "how much ",
    "how does ",
    "how do ",
    "what causes ",
    "explain ",
    "define ",
    "meaning of ",
    "difference between ",
    "tell me about ",
    "teach me about ",
)

_NATIVE_FACTUAL_QUESTION_RE = re.compile(
    r"(?:ಏನು|ಏನಿದು|ಎಂದರೇನು|ಯಾರು|ಎಲ್ಲಿ|ಯಾವಾಗ|ಏಕೆ|ಯಾಕೆ|ಏನಕ್ಕೆ|ಹೇಗೆ|ಯಾವುದು|ಯಾವ|ಎಷ್ಟು|ಎಷ್ಟಿದೆ|"
    r"ಏಕೆಂದರೆ|ಬಗ್ಗೆ|ವಿವರಿಸಿ|ಅರ್ಥವೇನು|ಅರ್ಥ ಏನು)",
    flags=re.IGNORECASE,
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

_QUESTION_PREFIX_RE = re.compile(
    r"^(?:what\s+(?:is|are|was|were)|who\s+(?:is|was)|where\s+(?:is|was)|"
    r"when\s+(?:was|did)|why\s+(?:is|are|does|do)|how\s+(?:is|many|much|does|do)|"
    r"what\s+causes|explain|define|meaning\s+of|difference\s+between|"
    r"tell\s+me\s+about|teach\s+me\s+about)\s+",
    flags=re.IGNORECASE,
)

_ROMAN_KANNADA_QUESTION_RE = re.compile(
    r"(?:\bandre\s+(?:enu|yenu)\b|\b(?:enu|yenu|yaaru|yaake|hege)\b)",
    flags=re.IGNORECASE,
)

_NATIVE_KANNADA_QUESTION_TOKENS = (
    "ಎಂದರೇನು",
    "ಅರ್ಥವೇನು",
    "ಅರ್ಥ ಏನು",
    "ಎಷ್ಟು",
    "ಎಷ್ಟಿದೆ",
    "ಯಾವುದು",
    "ಯಾವಾಗ",
    "ಯಾವ",
    "ಯಾರು",
    "ಎಲ್ಲಿ",
    "ಏಕೆ",
    "ಯಾಕೆ",
    "ಏನಕ್ಕೆ",
    "ಯಾಕಾಗಿ",
    "ಏಕೆಗಾಗಿ",
    "ಯಾವ ಕಾರಣಕ್ಕೆ",
    "ಯಾವ ಕಾರಣದಿಂದ",
    "ಕಾರಣ ಏನು",
    "ಹೇಗೆ",
    "ಏನು",
    "ಬಗ್ಗೆ",
    "ವಿವರಿಸಿ",
)

_NATIVE_KANNADA_QUESTION_TOKENS = (
    "ಎಂದರೇನು",
    "ಅರ್ಥವೇನು",
    "ಅರ್ಥ ಏನು",
    "ಎಷ್ಟು",
    "ಎಷ್ಟಿದೆ",
    "ಯಾವುದು",
    "ಯಾವಾಗ",
    "ಯಾವ",
    "ಯಾರು",
    "ಎಲ್ಲಿ",
    "ಏಕೆ",
    "ಹೇಗೆ",
    "ಏನು",
    "ಬಗ್ಗೆ",
    "ವಿವರಿಸಿ",
)


def _question_to_topic(message: str) -> str:
    """Reduce natural-language factual questions to a useful article query."""
    normalized = " ".join(message.strip().split()).strip(" ?!.")
    if not normalized:
        return ""
    topic = _QUESTION_PREFIX_RE.sub("", normalized, count=1).strip(" ?!.")
    romanized_tails = (
        " andre enu", " andre yenu", " enu", " yenu", " yaaru", " yaake", " hege",
    )
    lowered_topic = topic.casefold()
    for tail in romanized_tails:
        if lowered_topic.endswith(tail) and len(topic) > len(tail):
            topic = topic[: -len(tail)].rstrip(" ?!.")
            break

    # Native Kannada questions often contain the interrogative in the middle
    # ("ಭೂಮಿ ಏಕೆ ತಿರುಗುತ್ತದೆ?") or at the end ("ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?").
    # Remove those question words before sending the query to Wikipedia so its
    # search index sees the actual topic instead of the full sentence.
    for marker in _NATIVE_KANNADA_QUESTION_TOKENS:
        topic = re.sub(rf"(?<!\S){re.escape(marker)}(?!\S)", " ", topic)
    topic = " ".join(topic.split()).strip(" ?!.")

    trailing_phrases = (
        " in simple words",
        " in simple terms",
        " in simple language",
        " briefly",
        " in short",
    )
    lowered = topic.casefold()
    for phrase in trailing_phrases:
        if lowered.endswith(phrase):
            topic = topic[: -len(phrase)].rstrip(" ?!.")
            break
    return topic or normalized


def is_general_knowledge_question(message: str) -> bool:
    normalized = " ".join(message.strip().casefold().split())
    if not normalized:
        return False

    freshness_markers = (
        "latest",
        "today",
        "current",
        "currently",
        "recent",
        "news",
        "right now",
        "this week",
        "this month",
        "this year",
        "price",
        "cost",
        "stock",
        "weather",
        "forecast",
        "score",
        "schedule",
        "now",
        "search",
        "look up",
        "research",
        "election",
        "president",
        "prime minister",
        "minister",
        "law",
        "regulation",
        "policy",
        "deadline",
        "release date",
        "version",
        "update",
    )
    if any(marker in normalized for marker in freshness_markers):
        return False

    if any(normalized.startswith(prefix) for prefix in _ENGLISH_PREFIXES):
        return True

    if any(marker in message.strip() for marker in _SCRIPT_PREFIXES):
        return True

    if _NATIVE_FACTUAL_QUESTION_RE.search(message) and "ನೀವು" not in message and "ನಾನು" not in message:
        return True

    return bool(_ROMAN_KANNADA_QUESTION_RE.search(normalized))


def is_question_like(message: str) -> bool:
    normalized = " ".join(message.strip().casefold().split())
    if not normalized:
        return False
    if normalized.endswith("?"):
        return True
    if any(normalized.startswith(prefix) for prefix in _ENGLISH_PREFIXES):
        return True
    if _NATIVE_FACTUAL_QUESTION_RE.search(message):
        return True
    return bool(_ROMAN_KANNADA_QUESTION_RE.search(normalized))