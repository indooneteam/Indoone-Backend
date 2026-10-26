"""Keep the model layer from leaking between tests.

Measured, not assumed: running `tests/ai` as one process failed ten tests with
`can't start new thread`, `DefaultCPUAllocator: can't allocate memory` on allocations of a
couple of megabytes, and a torch/torchvision import mismatch. Every directory passes on its
own - training 57, local 40, gateway 63, execution 258 - so the failures were resource
exhaustion in a constrained container, not one test's model leaking into another.

Two things still belong here, because both make that pressure worse:

* a loaded checkpoint and a warm runtime survive a test that created them, so the next test
  pays for them again on top of everything already resident;
* torch defaults to one thread per core, and this suite spawns enough work that thread
  creation itself is what fails first.

The fixture below resets the model globals and caps torch to a single thread. It is not a
substitute for memory - it is the difference between a suite that fails for reasons of its
own and one that fails only when the container is genuinely out of room.
"""

from __future__ import annotations

import pytest


def _cap_torch_threads() -> None:
    """One thread, so thread creation stops being the thing that runs out first."""
    try:
        import torch  # noqa: PLC0415 - optional, and absent in a light install
    except Exception:  # noqa: BLE001 - torch is optional for most of this suite
        return

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001 - a second call raises; that is fine
        return


@pytest.fixture(autouse=True)
def _reset_model_state():
    """Forget the loaded runtime after every test in this tree."""
    _cap_torch_threads()

    yield

    try:
        from app.ai.gateway import service
    except Exception:  # noqa: BLE001 - a test may have broken the import on purpose
        return

    service._runtime = None
    service._NEXT_MODEL_LOAD_ATTEMPT = 0.0
    service._ARTIFACT_CHECK_COMPLETED = False
    service._model_answer_provider = service._LocalModelAnswerProvider()
