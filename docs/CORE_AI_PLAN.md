# Indoone Core AI

The core AI is finished only when the local model pipeline is reproducible, evaluated, grounded, and safe enough to become the stable foundation for later integrations.

## Current milestone

Instruction-tuning data is now part of the training pipeline. Training can combine the prepared language corpus with curated JSONL instruction/response examples without introducing a hosted model dependency.

## Remaining core milestones

1. Expand the licensed and curated instruction corpus substantially.
2. Strengthen behavioral evaluation beyond simple topic matching.
3. Add repeatable candidate training/evaluation reports and promotion gates.
4. Improve model capacity and training quality based on measured evaluation results.
5. Freeze a validated Core AI release before building external integrations.

The seed instruction dataset in `data/raw/indoone_instructions.jsonl` is only a pipeline-validation starting point. It is not sufficient training data for a production-grade assistant.
