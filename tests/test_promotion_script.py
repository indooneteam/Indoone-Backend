from pathlib import Path
import subprocess
import sys


def test_promotion_script_help_resolves_app_imports() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/promote_model.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Evaluate and promote an Indoone model candidate" in result.stdout
