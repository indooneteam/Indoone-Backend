# Core AI development

## Core AI v1 status

Core AI v1 is now **frozen** as the reproducible local-model benchmark and candidate-promotion contract. The frozen contract is documented in [`CORE_AI_V1_FREEZE.md`](CORE_AI_V1_FREEZE.md).

The v1 path covers prepared language-corpus training, curated instruction/response data validation, repeatable language + behavioral evaluation, baseline/candidate comparison, reproducibility metadata, and guarded model promotion.

Production readiness is still separate from the v1 freeze. It requires a much larger licensed dataset, stronger behavioral coverage, and measured model-capacity improvements before broad deployment.

### v1 invariants

- benchmark schema/version remains `v1`
- candidate promotion requires behavioral-gate success and strict improvement in both loss and perplexity
- evaluation metrics must be finite and non-negative
- required model, tokenizer, and behavioral artifacts must exist
- active-model retirement and `parent_version` lineage are preserved
- v1 contract changes require a new compatibility version rather than an in-place schema change
