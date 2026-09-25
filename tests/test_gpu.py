"""Tests for the shared GPU helpers (OOM classification, retry, device pick)."""

import pytest

from patchworks._gpu import is_oom, retry_on_oom, visible_device_index


class _CupyStyleOOM(Exception):
    """cupy.cuda.memory.OutOfMemoryError is NOT a RuntimeError."""

    __name__ = "OutOfMemoryError"


def test_is_oom_covers_torch_and_cupy():
    """Both GPU stacks must be recognised, not just torch's RuntimeError."""
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))

    cupy_err = type("OutOfMemoryError", (Exception,), {})(
        "Out of memory allocating 1,073,741,824 bytes"
    )
    assert not isinstance(cupy_err, RuntimeError), "precondition for the bug"
    assert is_oom(cupy_err), "cupy OOM used to slip through uncaught"

    assert not is_oom(ValueError("shape mismatch"))


def test_retry_succeeds_after_transient_oom():
    """A tile that fails once then works must not fail the job."""
    calls = {"n": 0}
    released = {"n": 0}

    def _work():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    out = retry_on_oom(
        _work,
        on_release=lambda: released.__setitem__("n", released["n"] + 1),
        backoff=0,
    )
    assert out == "ok"
    assert calls["n"] == 2
    assert released["n"] == 1, "must drop its own device memory before waiting"


def test_retry_gives_up_and_reraises():
    def _always_oom():
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        retry_on_oom(_always_oom, retries=2, backoff=0)


def test_non_oom_errors_are_not_retried():
    """A real bug must surface at once, not after four backoffs."""
    calls = {"n": 0}

    def _broken():
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        retry_on_oom(_broken, backoff=0)
    assert calls["n"] == 1


def test_retry_disabled_on_cpu_path():
    calls = {"n": 0}

    def _work():
        calls["n"] += 1
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError):
        retry_on_oom(_work, enabled=False, backoff=0)
    assert calls["n"] == 1


def test_visible_device_index_follows_cuda_visible_devices(monkeypatch):
    """NVML enumerates every GPU, so index 0 is the wrong device under gres."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert visible_device_index() == 3

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert visible_device_index() == 2

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert visible_device_index() == 0

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    assert visible_device_index() == 0  # UUID form resolves separately


def test_oom_backoff_releases_the_failed_calls_memory(monkeypatch):
    """Nothing the failed call allocated may survive into the backoff.

    Releasing inside the except block kept the exception -- and through its
    traceback the failed frame's locals, i.e. its device buffers -- alive
    while caches were freed and the retry slept.
    """
    import weakref

    from patchworks import _gpu

    class OutOfMemoryError(Exception):
        pass

    class Buffer:
        pass

    refs, alive = [], []

    def call():
        buf = Buffer()  # noqa: F841 - held by the frame, like a tensor
        refs.append(weakref.ref(buf))
        if len(refs) == 1:
            raise OutOfMemoryError("CUDA out of memory")
        return "ok"

    monkeypatch.setattr(_gpu.time, "sleep", lambda s: alive.append(refs[0]()))
    assert (
        _gpu.retry_on_oom(
            call, on_release=lambda: alive.append(refs[0]()), backoff=0
        )
        == "ok"
    )
    assert alive == [None, None]
