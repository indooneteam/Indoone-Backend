# Indoone dataset quality rules

Training data should grow in both quantity and quality. Before GPU training, the automatic pipeline now checks the curated instruction sources for:

- required instruction/response/category fields
- duplicate instruction/response pairs
- placeholder or incomplete responses
- excessive short answers
- category coverage

These checks complement the broader readiness audit, which covers corpus size, instruction-example count, multilingual-example count, and duplicate rate. A dataset must pass both gates before automatic model training is allowed.

The quality gate is intentionally conservative: it blocks questionable data instead of silently teaching the model low-quality behavior. As the corpus grows, the thresholds can be tightened or expanded through tests and the training policy.
