# Indoone expanded training run

The next training run adds a generated instruction pack to the shared core corpus, curated instruction data, and targeted multilingual capability data.

## Build the expanded instruction pack

```bash
python -m scripts.build_instruction_pack \
  --source data/raw/indoone_instructions.jsonl \
  --output data/raw/indoone_generated_instructions.jsonl \
  --count 2000 \
  --seed 42
```

The builder is deterministic for the same source, count, and seed and removes exact instruction/response duplicates.

## Train

```bash
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
python -m app.ai.train \
  --corpus data/processed/train.txt \
  --validation data/processed/validation.txt \
  --instructions data/raw/indoone_instructions.jsonl \
  --multilingual-instructions data/raw/indoone_multilingual_examples.jsonl \
  --generated-instructions data/raw/indoone_generated_instructions.jsonl \
  --output models/indoone-small \
  --steps 2000 \
  --batch-size 16 \
  --checkpoint-interval 500 \
  --learning-rate 3e-4 \
  --seed 42
```

All instruction sources are merged with exact duplicate instruction/response pairs removed before tokenization.

This remains a baseline training run. The generated pack is a bootstrap for capability coverage, not a substitute for a much larger high-quality owned/licensed corpus.
