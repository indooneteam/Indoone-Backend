"""Detect the language a reply should be written in.

This is the ONE place language detection lives. It recognises:

* explicit language names (``"answer in French"``, ``"हिंदी में बताओ"``),
* the writing script of the message, and
* romanized Indian-language hints (``"kannada bagge helu"``).

Script coverage is broad on purpose: it covers every script Google's models
can read and write, so the assistant can answer in any language the user writes
in rather than only the Indian languages. The Gemini chat path already mirrors
the user's language through its system instruction; this module provides the
same signal for the local-model path and for research providers.
"""
from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Writing scripts -> representative language
# --------------------------------------------------------------------------- #
# Order matters: the first matching script wins. Kana is listed before the CJK
# ideographs so Japanese text (which contains both) is not read as Chinese.
SCRIPT_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Indian scripts
    ("Kannada", re.compile(r"[\u0C80-\u0CFF]")),
    ("Telugu", re.compile(r"[\u0C00-\u0C7F]")),
    ("Tamil", re.compile(r"[\u0B80-\u0BFF]")),
    ("Malayalam", re.compile(r"[\u0D00-\u0D7F]")),
    ("Hindi", re.compile(r"[\u0900-\u097F]")),
    ("Bengali", re.compile(r"[\u0980-\u09FF]")),
    ("Gujarati", re.compile(r"[\u0A80-\u0AFF]")),
    ("Punjabi", re.compile(r"[\u0A00-\u0A7F]")),
    ("Odia", re.compile(r"[\u0B00-\u0B7F]")),
    ("Urdu", re.compile(r"[\u0600-\u06FF]")),
    ("Sinhala", re.compile(r"[\u0D80-\u0DFF]")),
    # Other major world scripts
    ("Greek", re.compile(r"[\u0370-\u03FF]")),
    ("Russian", re.compile(r"[\u0400-\u04FF]")),
    ("Armenian", re.compile(r"[\u0530-\u058F]")),
    ("Hebrew", re.compile(r"[\u0590-\u05FF]")),
    ("Thai", re.compile(r"[\u0E00-\u0E7F]")),
    ("Lao", re.compile(r"[\u0E80-\u0EFF]")),
    ("Tibetan", re.compile(r"[\u0F00-\u0FFF]")),
    ("Burmese", re.compile(r"[\u1000-\u109F]")),
    ("Georgian", re.compile(r"[\u10A0-\u10FF]")),
    ("Amharic", re.compile(r"[\u1200-\u137F]")),
    ("Japanese", re.compile(r"[\u3040-\u30FF]")),
    ("Korean", re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]")),
    ("Chinese", re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")),
)

# --------------------------------------------------------------------------- #
# Explicit language names (English + native)
# --------------------------------------------------------------------------- #
LANGUAGE_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Indian languages
    ("Kannada", ("kannada", "ಕನ್ನಡ", "ಕನ್ನಡದ")),
    ("Hindi", ("hindi", "हिंदी", "हिन्दी", "हिंदीमध्ये")),
    ("Telugu", ("telugu", "తెలుగు")),
    ("Tamil", ("tamil", "தமிழ்", "தமிழ")),
    ("Malayalam", ("malayalam", "മലയാളം")),
    ("Marathi", ("marathi", "मराठी", "मराठीमध्ये", "मराठीत")),
    ("Bengali", ("bengali", "bangla", "বাংলা", "বাঙালি", "বাংলায়")),
    ("Assamese", ("assamese", "অসমীয়া", "অসমিয়া", "অসমীয়াত", "অসমিয়াত")),
    ("Gujarati", ("gujarati", "ગુજરાતી", "ગુજરાતીમાં")),
    ("Punjabi", ("punjabi", "ਪੰਜਾਬੀ", "ਪੰਜਾਬੀ ਵਿੱਚ")),
    ("Odia", ("odia", "oriya", "ଓଡ଼ିଆ", "ଓଡିଆ", "ଓଡ଼ିଆରେ")),
    ("Urdu", ("urdu", "اردو")),
    # Other languages Google supports
    ("English", ("english",)),
    ("French", ("french", "français", "francais")),
    ("Spanish", ("spanish", "español", "espanol", "castellano")),
    ("German", ("german", "deutsch")),
    ("Italian", ("italian", "italiano")),
    ("Portuguese", ("portuguese", "português", "portugues")),
    ("Dutch", ("dutch", "nederlands")),
    ("Russian", ("russian", "русский")),
    ("Ukrainian", ("ukrainian", "українська")),
    ("Arabic", ("arabic", "العربية", "عربي")),
    ("Persian", ("persian", "farsi", "فارسی")),
    ("Hebrew", ("hebrew", "עברית")),
    ("Greek", ("greek", "ελληνικά")),
    ("Turkish", ("turkish", "türkçe", "turkce")),
    ("Chinese", ("chinese", "mandarin", "中文", "汉语", "漢語")),
    ("Japanese", ("japanese", "日本語")),
    ("Korean", ("korean", "한국어", "한국말")),
    ("Vietnamese", ("vietnamese", "tiếng việt", "tieng viet")),
    ("Thai", ("thai", "ภาษาไทย")),
    ("Indonesian", ("indonesian", "bahasa indonesia")),
    ("Polish", ("polish", "polski")),
)

# --------------------------------------------------------------------------- #
# Native-script hints (disambiguate languages that share a script)
# --------------------------------------------------------------------------- #
NATIVE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Marathi", ("मराठीमध्ये", "मराठीत", "मराठी", "सांगा", "सांग")),
    ("Assamese", ("অসমীয়াত", "অসমিয়াত", "অসমীয়া", "অসমিয়া", "কও", "কওক")),
    ("Bengali", ("বাংলায়", "বাংলা", "বল")),
    ("Hindi", ("हिंदी", "हिन्दी", "बताओ", "बताइए")),
    ("Gujarati", ("ગુજરાતીમાં", "ગુજરાતી", "કહો")),
    ("Punjabi", ("ਪੰਜਾਬੀ ਵਿੱਚ", "ਪੰਜਾਬੀ", "ਦੱਸੋ")),
    ("Odia", ("ଓଡ଼ିଆରେ", "ଓଡ଼ିଆ", "କହନ୍ତୁ")),
    ("Tamil", ("தமிழில்", "தமிழ்", "சொல்லு", "சொல்லுங்கள்")),
    ("Telugu", ("తెలుగులో", "తెలుగు", "చెప్పు", "చెప్పండి")),
    ("Malayalam", ("മലയാളത്തിൽ", "മലയാളം", "പറയൂ", "പറയുക")),
    ("Kannada", ("ಕನ್ನಡ", "ಹೇಳು", "ಹೇಳಿ")),
    ("Urdu", ("اردو", "بتائیں", "بتاؤ")),
)

# --------------------------------------------------------------------------- #
# Romanized Indian-language hints
# --------------------------------------------------------------------------- #
ROMANIZED_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("bagge", "helu", "heLi", "maadu", "maadi", "madbeku", "maadbeku", "enidu", "enu", "yenu", "yenide", "ivaga", "matte", "nanage", "nimage", "nanna", "namma", "ide", "illa", "agide", "beku")),
    ("Hindi", ("kya", "hai", "hain", "mujhe", "aap", "aapko", "batao", "bataiye", "kaise", "kaisa", "kahan", "kyun", "nahi", "abhi", "mera", "meri")),
    ("Telugu", ("enti", "ela", "cheppu", "cheppandi", "undi", "ledu", "nenu", "meeru", "naaku", "emiti", "enduku", "ippudu", "malli")),
    ("Tamil", ("enna", "epdi", "eppadi", "sollu", "sollunga", "irukku", "illa", "naan", "neenga", "enakku", "ippo", "yen")),
    ("Malayalam", ("entha", "engane", "parayu", "parayoo", "undu", "illa", "njan", "ningal", "enikku", "ippo")),
    ("Marathi", ("kay", "aahe", "mala", "tumhi", "sanga", "kasa", "kashi", "kuthe", "nahi", "aata", "madhe")),
    ("Bengali", ("ki", "ache", "ami", "apni", "bolo", "bolun", "amar", "keno", "nei")),
    ("Assamese", ("ase", "moi", "tumi", "kobo", "mur", "kio", "nai", "অসমীয়াত")),
    ("Gujarati", ("shu", "che", "chhe", "mane", "tame", "kaho", "kem", "nathi", "maru")),
    ("Punjabi", ("menu", "mainu", "tusi", "daso", "kive", "kiwe", "nahi", "mera", "sanu")),
    ("Odia", ("kana", "achhi", "mu", "tame", "kahantu", "kemiti", "nahi", "ebe")),
    ("Urdu", ("kya", "hai", "mujhe", "aap", "batao", "bataiye", "kaise", "kyun", "nahi")),
)


def detect_response_language(message: str) -> str:
    """Return the language the reply should use for ``message``.

    Falls back to ``"English"`` when nothing else is recognised, which is also
    the safe default for the local model (Latin script).
    """

    normalized = " ".join(message.casefold().split())

    for language, names in LANGUAGE_NAMES:
        for name in names:
            candidate = name.casefold()
            if candidate.isascii():
                if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized):
                    return language
            elif candidate in normalized:
                return language

    native_scores: dict[str, int] = {}
    for language, hints in NATIVE_HINTS:
        score = sum(1 for hint in hints if hint.casefold() in normalized)
        if score:
            native_scores[language] = score

    if native_scores:
        best_language, best_score = max(native_scores.items(), key=lambda item: item[1])
        tied = [language for language, score in native_scores.items() if score == best_score]
        if best_score >= 2 or len(tied) == 1:
            return best_language

    for language, pattern in SCRIPT_RANGES:
        if pattern.search(message):
            return language

    romanized_scores: dict[str, int] = {}
    for language, hints in ROMANIZED_HINTS:
        score = 0
        for hint in hints:
            if re.search(rf"(?<!\w){re.escape(hint.casefold())}(?!\w)", normalized):
                score += 1
        if score:
            romanized_scores[language] = score

    if romanized_scores:
        best_language, best_score = max(romanized_scores.items(), key=lambda item: item[1])
        tied = [language for language, score in romanized_scores.items() if score == best_score]
        if best_score >= 2 or len(tied) == 1:
            return best_language

    return "English"
