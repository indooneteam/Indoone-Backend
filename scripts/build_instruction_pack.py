from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BASE_EXAMPLES = [
    ("What should you do when a requirement is unclear?", "Ask for the specific missing detail that would materially change the solution.", "conversation"),
    ("How should two technical options be compared?", "Compare them against the same criteria and explain the trade-offs fairly.", "comparison"),
    ("What should you do when you are uncertain?", "State the uncertainty clearly, avoid inventing facts, and identify what would resolve it.", "honesty"),
    ("Why are unit tests useful?", "They check small pieces of code in isolation so regressions and incorrect behavior can be detected quickly.", "software"),
    ("Why should secrets stay out of Git?", "Committed secrets can be exposed through repository history and mirrors; store them in an appropriate secret-management system instead.", "security"),
    ("How should a difficult task be broken down?", "Define the goal, identify constraints and dependencies, split the work into smaller verifiable steps, and combine the results.", "planning"),
    ("Why should training and evaluation data be separate?", "A separate evaluation set gives a more meaningful estimate of performance on unseen data.", "development"),
]

LANGUAGE_GREETINGS = {
    "Kannada": "ಶುಭೋದಯ! ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
    "Hindi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
    "Telugu": "నమస్తే! నేను మీకు ఎలా సహాయం చేయగలను?",
    "Tamil": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவலாம்?",
    "Malayalam": "നമസ്കാരം! ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
    "Marathi": "नमस्कार! मी तुम्हाला कशी मदत करू शकतो?",
    "Bengali": "নমস্কার! আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
    "Gujarati": "નમસ્તે! હું તમને કેવી રીતે મદદ કરી શકું?",
    "Punjabi": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
    "Odia": "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
}


def make_examples(seed: int, target: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(instruction: str, response: str, category: str) -> None:
        key = (instruction.casefold(), response.casefold())
        if key not in seen:
            seen.add(key)
            examples.append({"instruction": instruction, "response": response, "category": category})

    for instruction, response, category in BASE_EXAMPLES:
        add(instruction, response, category)

    for a in range(1, 101):
        for b in (1, 2, 5, 7, 10):
            add(f"Calculate {a} + {b}.", f"{a} + {b} = {a + b}.", "math")
            if a >= b:
                add(f"Calculate {a} - {b}.", f"{a} - {b} = {a - b}.", "math")
            add(f"Calculate {a} × {b}.", f"{a} × {b} = {a * b}.", "math")

    for language, response in LANGUAGE_GREETINGS.items():
        for variant in ("Reply with a short greeting in {0}.", "Give me a simple welcome in {0}."):
            add(variant.format(language), response, "multilingual")

    translations = [
        ("Good morning", "ಶುಭೋದಯ.", "Kannada"),
        ("Thank you", "ಧನ್ಯವಾದಗಳು.", "Kannada"),
        ("How are you?", "ನೀವು ಹೇಗಿದ್ದೀರಿ?", "Kannada"),
        ("Good morning", "सुप्रभात.", "Hindi"),
        ("Thank you", "धन्यवाद.", "Hindi"),
        ("How are you?", "आप कैसे हैं?", "Hindi"),
        ("Good morning", "శుభోదయం.", "Telugu"),
        ("Thank you", "ధన్యవాదాలు.", "Telugu"),
        ("How are you?", "మీరు ఎలా ఉన్నారు?", "Telugu"),
    ]
    for source, response, language in translations:
        add(f"Translate '{source}' into {language}.", response, "translation")

    while len(examples) < target:
        instruction, response, category = rng.choice(BASE_EXAMPLES)
        add(f"{instruction} Give one practical example.", f"{response} A practical example is to apply the same rule to one small, testable case before scaling up.", category)

    rng.shuffle(examples)
    return examples[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic Indoone instruction training pack")
    parser.add_argument("--output", type=Path, default=Path("data/raw/indoone_generated_instructions.jsonl"))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be greater than zero")
    examples = make_examples(args.seed, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(example, ensure_ascii=False) for example in examples) + "\n",
        encoding="utf-8",
    )
    print({"examples": len(examples), "output": str(args.output), "seed": args.seed})


if __name__ == "__main__":
    main()
