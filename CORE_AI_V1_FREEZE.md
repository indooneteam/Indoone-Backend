# Indoone Core AI v1 Freeze

Status: **Frozen**

Core AI v1 is the reproducible local-model evaluation and promotion contract currently used by the backend. Changes to the frozen contract require an explicit version bump (for example, v2) rather than silently changing v1 behavior.

## Frozen benchmark contract

- Benchmark version: `v1`
- Language metrics: `loss` and `perplexity`
- Behavioral result: `behavioral_gate`
- Final benchmark decision: boolean `overall_pass`
- Benchmark comparison requires matching benchmark versions.
- Only `v1` is accepted by the current promotion path.
- Candidate promotion requires behavioral-gate success and strict improvement in both loss and perplexity over the active model.

## Frozen promotion contract

A candidate model record contains:

- `version`
- `model_dir`
- `checkpoint`
- `tokenizer`
- `loss`
- `perplexity`
- `status`
- `behavioral_gate_passed`
- `parent_version`
- `benchmark_version`

The promotion path validates that metrics are finite and non-negative, the candidate version is non-empty and distinct from the active version, the benchmark version is supported and matches the active model, and required model/tokenizer/behavior artifacts exist.

The active model is retired when a candidate is promoted, and the promoted record stores the previous active version as `parent_version`.

## Reproducibility contract

Training/evaluation metadata and dataset fingerprints remain part of the model-development record. A model candidate must be evaluated through the repeatable benchmark/promotion path before it is considered active.

## Change policy

Do not modify the v1 report schema, benchmark semantics, promotion gates, or registry semantics in place. A behavioral or schema change must introduce a new compatibility version and corresponding regression coverage first.

This document freezes the current Core AI v1 contract; it does **not** claim production readiness or model-quality sufficiency for broad deployment.
