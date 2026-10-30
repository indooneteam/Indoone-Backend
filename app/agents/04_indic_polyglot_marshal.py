"""Agent 04 - the Indic Polyglot Marshal: names the language, the dialect and the tone.

Three questions a chat backend keeps asking and keeps answering badly:

* **Which script is this?** Counting code points answers it, and Devanagari needs one more
  step because it carries both Hindi and Marathi - so the words decide, not the range.
* **Is this English, or Indic typed in Latin letters?** "kya haal hai batao" is Hindi;
  "show me pull requests" is not, even though both are Latin. One shared word is not
  enough evidence, so the markers are scored and a floor is required.
* **How is the person speaking?** "ಮಾಡಿ" and "kijiye" are respectful; "ಮಾಡು" and "karo" are
  not; "git status" is neither, it is a command. The answer changes how a reply should be
  phrased, and it is cheaper to read it here than to have a model guess it.

It also canonicalises the verb, because the same request arrives spelled six ways
("maadi", "madi", "maadbeku", "madri") and anything that keys on the word has to see one
form. Nouns are never touched: "branch", "repo", "calendar" mean what they mean.

Standard library only, one pass over the text, pre-built lookups, sub-millisecond.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

# ---------------------------------------------------------------- scripts

_SCRIPTS: dict[str, tuple[int, int, str]] = {
    "kn": (0x0C80, 0x0CFF, "Knda"),
    "te": (0x0C00, 0x0C7F, "Telu"),
    "ta": (0x0B80, 0x0BFF, "Taml"),
    "ml": (0x0D00, 0x0D7F, "Mlym"),
    "hi": (0x0900, 0x097F, "Deva"),
    "bn": (0x0980, 0x09FF, "Beng"),
    "gu": (0x0A80, 0x0AFF, "Gujr"),
}

# Devanagari carries Hindi and Marathi. These are Marathi and never Hindi.
_MARATHI_MARKERS = frozenset({"आहे", "नाही", "करा", "मला", "तुमचा", "आणि", "कसे", "सांगा", "काय"})

# Romanised markers. A dialect needs at least DIALECT_FLOOR hits, so one borrowed word in an
# English sentence does not relabel it.
_DIALECTS: dict[str, frozenset[str]] = {
    "kanglish": frozenset(
        {"enu", "yenu", "beku", "bekithu", "madodu", "maadu", "maadi", "madi", "hege", "hegiddira",
         "nanna", "nanu", "illi", "kodra", "kodu", "helu", "hogu", "ide", "illa", "swalpa", "namaskara"}
    ),
    "hinglish": frozenset(
        {"kya", "kaise", "karo", "kijiye", "karna", "mujhe", "batao", "batayein", "hoga", "chahiye",
         "nahi", "hai", "haan", "acha", "thoda", "kripya", "dhanyavaad", "kaam", "bhai"}
    ),
    "tanglish": frozenset(
        {"enna", "eppadi", "sollu", "sollunga", "venum", "vendum", "illa", "naan", "ungal", "romba",
         "seri", "vanakkam", "nandri", "pannu", "pannunga"}
    ),
    "tenglish": frozenset(
        {"enti", "ela", "chey", "cheyandi", "kavali", "ledu", "meeru", "nenu", "chala", "baga",
         "namaskaram", "cheppandi"}
    ),
    "manglish": frozenset(
        {"enthu", "enganey", "parayu", "parayoo", "venam", "njan", "ningal", "valare", "sheri",
         "nanni", "cheyyu"}
    ),
}

DIALECT_FLOOR = 2

# Verb variants collapse to one stem. Nouns are deliberately absent.
_CANONICAL: dict[str, str] = {
    # Kannada
    "maadi": "maadu", "madi": "maadu", "maadbeku": "maadu", "madri": "maadu", "maadodu": "maadu",
    "helri": "helu", "helbeku": "helu", "heli": "helu",
    "kodi": "kodu", "kodri": "kodu",
    "nodri": "nodu", "nodi": "nodu", "nodbeku": "nodu",
    # Hindi
    "kijiye": "karo", "karna": "karo", "kariye": "karo", "kijiyega": "karo",
    "bataiye": "batao", "batayein": "batao", "batana": "batao",
    "dijiye": "do", "dena": "do",
    # Tamil
    "sollunga": "sollu", "sollunga": "sollu", "sollanum": "sollu",
    "pannunga": "pannu", "pannanum": "pannu",
    # Telugu
    "cheyandi": "chey", "cheyyandi": "chey",
    "cheppandi": "cheppu", "cheppu": "cheppu",
}

_CANONICAL_KEYS = frozenset(_CANONICAL)

_HONORIFIC = frozenset(
    {"ಮಾಡಿ", "ತಿಳಿಸಿ", "ಹೇಳಿ", "ಕೊಡಿ", "ದಯವಿಟ್ಟು", "kijiye", "batayein", "bataye", "dijiye",
     "kripya", "please", "kindly", "sollunga", "cheyandi", "cheppandi", "parayoo", "madam", "sir"}
)
_CASUAL = frozenset(
    {"ಮಾಡು", "ಹೇಳು", "ಕೊಡು", "karo", "batao", "do", "hey", "bro", "da", "maga", "sollu",
     "chey", "parayu", "yaar", "dude", "nanba"}
)

# A terse command with no pronouns and no politeness is a developer at work.
_TECHNICAL = re.compile(
    r"^\s*(git|docker|kubectl|npm|pnpm|yarn|pip|pytest|curl|ssh|systemctl|alembic|psql|uvicorn)\b"
    r"|\b(status|build|deploy|commit|merge|rebase|push|pull|logs?|restart)\b\s*$",
    re.IGNORECASE,
)

_WORD = re.compile(r"[a-z]+")
_SCRIPT_CHAR = re.compile(r"[\u0900-\u0dff]")


@dataclass(slots=True, frozen=True)
class Telemetry:
    execution_microseconds: int
    script_distribution: dict[str, float]


@dataclass(slots=True, frozen=True)
class MarshalVerdict:
    """What the text is, how it is spelled, and how the person is speaking."""

    primary_language: str
    script: str
    is_transliterated: bool
    dialect_family: str
    register_tone: str
    canonical_text: str
    telemetry: Telemetry


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_POLYGLOT_MARSHAL_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------- the analysis


def _script_distribution(text: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0

    for character in text:
        code = ord(character)

        # One range test first: every Indic script this agent knows lives between 0x0900 and
        # 0x0D7F, so the common case - a Latin letter - costs one comparison, not seven.
        if 0x0900 <= code <= 0x0D7F:
            for language, (low, high, _name) in _SCRIPTS.items():
                if low <= code <= high:
                    counts[language] = counts.get(language, 0) + 1
                    total += 1
                    break
        elif "a" <= character <= "z" or "A" <= character <= "Z":
            counts["en"] = counts.get("en", 0) + 1
            total += 1

    if not total:
        return {}

    return {language: round(count / total, 3) for language, count in counts.items()}


def _dialect(words: set[str]) -> tuple[str, int]:
    best, best_hits = "standard", 0

    for name, markers in _DIALECTS.items():
        hits = len(words & markers)
        if hits > best_hits:
            best, best_hits = name, hits

    return best, best_hits


def _canonicalise(text: str, words: set[str]) -> str:
    """Collapse verb spelling variants, leave every other word exactly as written."""
    if not words & _CANONICAL_KEYS:
        return text

    def replace(match: re.Match) -> str:
        word = match.group(0)
        return _CANONICAL.get(word.lower(), word)

    return _WORD.sub(replace, text)


def _register(text: str, words: set[str]) -> str:
    """Politeness first, then the shape of a command, then the neutral default.

    A message with no marker at all is labelled CASUAL because that is the least wrong of
    three: it is not respectful and it is not a command. Saying so here is better than
    letting the label imply something the text never said.
    """
    honorific = len(words & _HONORIFIC)
    casual = len(words & _CASUAL)

    # The native-script markers only need scanning when the text actually carries one.
    if _SCRIPT_CHAR.search(text):
        honorific += sum(1 for marker in _HONORIFIC if not marker.isascii() and marker in text)
        casual += sum(1 for marker in _CASUAL if not marker.isascii() and marker in text)

    if honorific > casual:
        return "HONORIFIC"

    # A tie goes to the respectful reading: when a message mixes "kijiye" with "batao", the
    # safer mistake is to answer politely.
    if honorific and honorific == casual:
        return "HONORIFIC"

    if casual > honorific:
        return "CASUAL"

    if _TECHNICAL.search(text.strip()):
        return "TECHNICAL"

    return "CASUAL"


def analyze(message: str) -> MarshalVerdict:
    """Read one message. Nothing here calls a model, a network or a file."""
    started = time.perf_counter()
    text = message or ""

    distribution = _script_distribution(text)

    # A native script wins outright: its presence is not ambiguous the way Latin is.
    native = {language: share for language, share in distribution.items() if language != "en"}

    primary_language = "en"
    script = "Latn"
    is_transliterated = False
    dialect = "standard"

    if native:
        primary_language = max(native, key=lambda name: native[name])
        script = _SCRIPTS[primary_language][2]

        if primary_language == "hi":
            devanagari_words = set(re.findall(r"[\u0900-\u097f]+", text))
            if devanagari_words & _MARATHI_MARKERS:
                primary_language = "mr"
    else:
        words = {word.lower() for word in _WORD.findall(text)}
        dialect, hits = _dialect(words)
        if hits >= DIALECT_FLOOR:
            is_transliterated = True
            primary_language = {
                "kanglish": "kn",
                "hinglish": "hi",
                "tanglish": "ta",
                "tenglish": "te",
                "manglish": "ml",
            }[dialect]
            script = "Latn"

    words = {word.lower() for word in _WORD.findall(text)}
    tone = _register(text, words)

    return MarshalVerdict(
        primary_language=primary_language,
        script=script,
        is_transliterated=is_transliterated,
        dialect_family=dialect if is_transliterated else ("technical" if tone == "TECHNICAL" else "standard"),
        register_tone=tone,
        canonical_text=_canonicalise(text, words),
        telemetry=Telemetry(
            execution_microseconds=int((time.perf_counter() - started) * 1_000_000),
            script_distribution=distribution,
        ),
    )


# ---------------------------------------------------------------- routing arbitration

TOOL_CONFIDENCE_THRESHOLD = 0.85


def route(text: str) -> dict[str, object]:
    """Whether a message may reach an external capability, and how sure we are.

    The marshal reads the language; the capability gate reads the intent. Neither alone is
    enough: a Kannada message can name a mailbox without asking for one, and an English one
    can ask for GitHub in four words. So the decision is taken here, once, against a stated
    threshold - below it the turn is conversation, and no tool is dispatched.
    """
    from app.agents import capability_gate

    gate = capability_gate()
    capability = gate.GITHUB if _names_github(text) else gate.MAIL if _names_mail(text) else gate.CONVERSATION
    decision = gate.decide(text, capability=capability)

    confidence = float(decision.confidence)
    # A deferral means this gate found nothing either way, so the router's own matcher
    # decides - not a low-confidence allow, and not a refusal.
    is_tool = bool(decision.is_capability) and (
        confidence >= TOOL_CONFIDENCE_THRESHOLD or getattr(decision, "deferred", False)
    )

    return {
        "capability": decision.capability,
        "confidence": confidence,
        "use_tool": is_tool,
        "reason": decision.reason,
    }


def _names_github(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:git|github|repo|repository|branch|branches|commit|commits|pr|prs|pull\s+request|"
            r"merge|diff|issue|issues)\b|ರೆಪೊ|ಶಾಖೆ|ಬ್ರಾಂಚ್|कमिट|ब्रांच",
            text or "",
            re.IGNORECASE,
        )
    )


def _names_mail(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:mail|email|emails|inbox|zoho|gmail|message|draft)\b|ಇಮೇಲ್|ಮೇಲ್|मेल|इमेल",
            text or "",
            re.IGNORECASE,
        )
    )
