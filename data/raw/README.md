# Core training data

`indoone_instructions.jsonl` is the seed instruction dataset used to validate Indoone's local training pipeline.

Each line contains an `instruction`, `response`, and `category`. The dataset is intentionally small and must be expanded with substantially more curated, licensed examples before any production model release.

## Email Safety

`indoone_email_safety_examples.jsonl` is the controlled seed for the Email Safety incremental capability.

Use only synthetic, public-domain, or appropriately licensed examples. Never add real private emails, passwords, OTPs, recovery codes, card numbers, API keys, or live malicious links.

The final Email Safety release is defined in `docs/ai/EMAIL_SAFETY_DATASET_SPEC.md` and covers all currently recognized Indoone languages plus English.
