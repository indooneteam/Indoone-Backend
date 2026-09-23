import inspect

import scripts.run_training_pipeline as pipeline


def test_final_training_defaults_are_shared_and_curated_focused() -> None:
    source = inspect.getsource(pipeline.main)
    assert pipeline.DEFAULT_TRAINING_STEPS == 8000
    assert pipeline.DEFAULT_BATCH_SIZE == 16
    assert pipeline.DEFAULT_CHECKPOINT_INTERVAL == 500
    assert pipeline.DEFAULT_LEARNING_RATE == 1e-4
    assert pipeline.DEFAULT_INSTRUCTION_MIX_RATIO == 0.8
    assert pipeline.DEFAULT_SEED == 42
    assert "default=DEFAULT_TRAINING_STEPS" in source
    assert "default=DEFAULT_BATCH_SIZE" in source
    assert "default=DEFAULT_CHECKPOINT_INTERVAL" in source
    assert "default=DEFAULT_LEARNING_RATE" in source
    assert "default=DEFAULT_INSTRUCTION_MIX_RATIO" in source
    assert "default=DEFAULT_SEED" in source
    assert "validate_final_training_recipe.py" in source
    assert "curated_instructions.jsonl" in source
    assert "generated_multilingual_examples.jsonl" in source
