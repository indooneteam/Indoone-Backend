import pytest

from app.ai.service import _detect_response_language


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("ಕನ್ನಡ ಬಗ್ಗೆ ಹೇಳು", "Kannada"),
        ("kannada bagge helu", "Kannada"),
        ("Hindi me batao", "Hindi"),
        ("हिंदी में बताओ", "Hindi"),
        ("telugu lo cheppu", "Telugu"),
        ("తెలుగు లో చెప్పు", "Telugu"),
        ("tamil la sollu", "Tamil"),
        ("தமிழில் சொல்லு", "Tamil"),
        ("malayalam parayu", "Malayalam"),
        ("മലയാളത്തിൽ പറയൂ", "Malayalam"),
        ("marathi madhe sanga", "Marathi"),
        ("मराठीमध्ये सांगा", "Marathi"),
        ("bangla te bolo", "Bengali"),
        ("বাংলায় বলো", "Bengali"),
        ("assamese kobo", "Assamese"),
        ("অসমীয়াত কও", "Assamese"),
        ("gujarati ma kaho", "Gujarati"),
        ("ગુજરાતીમાં કહો", "Gujarati"),
        ("punjabi vich daso", "Punjabi"),
        ("ਪੰਜਾਬੀ ਵਿੱਚ ਦੱਸੋ", "Punjabi"),
        ("odia re kahantu", "Odia"),
        ("ଓଡ଼ିଆରେ କହନ୍ତୁ", "Odia"),
        ("urdu mein batao", "Urdu"),
        ("اردو میں بتائیں", "Urdu"),
        ("What is Indoone?", "English"),
    ],
)
def test_detect_response_language(message: str, language: str) -> None:
    assert _detect_response_language(message) == language
