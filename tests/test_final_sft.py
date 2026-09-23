from pathlib import Path

import scripts.final_sft as sft


def test_final_sft_defaults_are_careful() -> None:
    assert sft.DEFAULT_STEPS == 5000
    assert sft.DEFAULT_BATCH_SIZE == 16
    assert sft.DEFAULT_LEARNING_RATE == 5e-5
    assert sft.DEFAULT_EVAL_INTERVAL == 100
    assert sft.DEFAULT_WEIGHT_DECAY == 0.01


def test_final_sft_paths_default_to_processed_train_and_validation_sets() -> None:
    assert sft.DEFAULT_TRAIN_INSTRUCTIONS == Path(
        "data/processed/curated_instructions.jsonl"
    )
    assert sft.DEFAULT_VALIDATION_INSTRUCTIONS == Path(
        "data/processed/instructions_validation.jsonl"
    )
