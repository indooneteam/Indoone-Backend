# Indoone first training run

This run is the first local-model baseline. It is a pipeline validation and capability baseline, not the final production-scale model.

## Data composition

The run keeps one shared knowledge corpus and adds targeted instruction/capability data:

- `data/raw/indoone_corpus.txt` — shared core corpus
- `data/raw/indoone_instructions.jsonl` — general instruction/reasoning examples
- `data/raw/indoone_multilingual_examples.jsonl` — multilingual capability examples only; do not duplicate the full knowledge corpus per language

The training code merges instruction sources and removes exact duplicate instruction/response pairs before creating the instruction corpus.

## Prepare the text splits

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
```

The dataset preparation step normalizes text, removes exact duplicate documents, creates train/validation/test splits, and rejects duplicates that cross splits.

## First baseline run

```bash
python -m app.ai.train \
  --corpus data/processed/train.txt \
  --validation data/processed/validation.txt \
  --instructions data/raw/indoone_instructions.jsonl \
  --multilingual-instructions data/raw/indoone_multilingual_examples.jsonl \
  --output models/indoone-small \
  --steps 2000 \
  --batch-size 16 \
  --checkpoint-interval 500 \
  --learning-rate 3e-4 \
  --seed 42
```

The trainer uses CUDA automatically when PyTorch reports a CUDA device; otherwise it runs on CPU.

## Artifacts expected

`models/indoone-small/` should contain at least:

- `indoone-small.pt`
- `tokenizer.json`
- `metadata.json`
- `training_history.json`
- `checkpoint.pt`

When validation is enabled and a best validation checkpoint is found, `best_checkpoint.pt` is also written.

## Acceptance gate

Before uploading a model to B2, verify:

1. the training command completes without errors;
2. `metadata.json` records both instruction data sources and their fingerprints;
3. validation is enabled and a best model is selected when validation data is usable;
4. the model artifact and tokenizer can be loaded by the backend;
5. multilingual spot checks answer in the requested language without requiring a duplicated full-language corpus.

Do not promote this baseline as the final ChatGPT/Gemini-level model. The next quality iteration requires substantially more high-quality, licensed/owned training data, stronger evaluation, and more compute.
