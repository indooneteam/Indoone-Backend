from pathlib import Path

import scripts.run_training_pipeline as pipeline


def test_pipeline_restores_only_readiness_augmented_raw_files() -> None:
    assert pipeline.RAW_TRAINING_FILES == (
        Path("data/raw/indoone_corpus.txt"),
        Path("data/raw/indoone_instructions.jsonl"),
        Path("data/raw/indoone_multilingual_examples.jsonl"),
    )


def test_final_instruction_mix_defaults_to_curated_focused_ratio() -> None:
    # Keep the final recipe strongly focused on supervised instruction data.
    import inspect

    source = inspect.getsource(pipeline.main)
    assert 'default=0.9' in source
    assert 'scripts.validate_training_manifest.py' in source
