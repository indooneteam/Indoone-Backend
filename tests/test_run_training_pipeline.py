import inspect

import scripts.run_training_pipeline as pipeline


def test_final_instruction_mix_defaults_to_curated_focused_ratio() -> None:
    source = inspect.getsource(pipeline.main)
    assert "default=8000" in source
    assert "default=0.8" in source
    assert "validate_final_training_recipe.py" in source
    assert "curated_instructions.jsonl" in source
    assert "generated_multilingual_examples.jsonl" in source
