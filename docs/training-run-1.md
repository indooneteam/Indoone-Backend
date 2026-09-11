# Indoone First Training Run

This run is the first measurable Indoone local-model experiment after the curated instruction dataset expansion.

## Goals

- Keep the existing model architecture unchanged for the first baseline.
- Use the cleaned corpus plus curated instruction data.
- Keep validation data separate from parameter updates.
- Record configuration, loss history, and checkpoint metadata.
- Do not treat this first run as a production-quality model.

## Baseline command

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
python -m app.ai.train --corpus data/processed/train.txt --validation data/processed/validation.txt --instructions data/raw/indoone_instructions.jsonl --output models/indoone-small --steps 2000 --seed 42 --batch-size 16 --checkpoint-interval 500 --learning-rate 3e-4
```

## Artifacts

The run should produce:

- `models/indoone-small/indoone-small.pt`
- `models/indoone-small/tokenizer.json`
- `models/indoone-small/training_history.json`
- `models/indoone-small/metadata.json`
- `models/indoone-small/best_checkpoint.pt` when validation is enabled

## Evaluation

Before deployment, test the checkpoint on held-out validation/test material and on representative conversational prompts covering identity, math, reasoning, software, research, safety, translation, structured output, and follow-up context.

Only a checkpoint that passes the evaluation gate should be uploaded to B2 for deployment.
