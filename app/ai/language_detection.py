from __future__ import annotations

import re


SCRIPT_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
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
)

LANGUAGE_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("kannada", "ಕನ್ನಡ", "ಕನ್ನಡದ")),
    ("Hindi", ("hindi", "हिंदी", "हिन्दी", "हिंदीमध्ये")),
    ("Telugu", ("telugu", "తెలుగు")),
    ("Tamil", ("tamil", "தமிழ்", "தமிழ")),
    ("Malayalam", ("malayalam", "മലയാളം")),
    ("Marathi", ("marathi", "मराठी", "मराठीमध्ये", "मराठीत")),
    ("Bengali", ("bengali", "bangla", "বাংলা", "বাঙালি", "বাংলায়")),
    ("Assamese", ("assamese", "অসমীয়া", "অসমিয়া", "অসমীয়াত", "অসমিয়াত")),
    ("Gujarati", ("gujarati", "ગુજરાતી", "ગુજરાતીમાં")),
    ("Punjabi", ("punjabi", "ਪੰਜਾਬੀ", "ਪੰਜਾਬੀ ਵਿੱਚ")),
    ("Odia", ("odia", "oriya", "ଓଡ଼ିଆ", "ଓଡିଆ", "ଓଡ଼ିଆରେ")),
    ("Urdu", ("urdu", "اردو")),
)

NATIVE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Marathi", ("मराठीमध्ये", "मराठीत", "मराठी", "सांगा", "सांग")),
    ("Assamese", ("অসমীয়াত", "অসমিয়াত", "অসমীয়া", "অসমিয়া", "কও", "কওক")),
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
    """Detect native-script or romanized Indian-language input."""

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
