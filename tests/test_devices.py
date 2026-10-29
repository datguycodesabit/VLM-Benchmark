from types import SimpleNamespace

import pytest

from vlm_bench.devices import DeviceUnavailableError, resolve_device


def _torch(*, mps=False, cuda=False, hip=None):
    return SimpleNamespace(
        __version__="test-torch",
        version=SimpleNamespace(cuda="12.4" if cuda and hip is None else None, hip=hip),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(is_available=lambda: cuda),
    )


def test_auto_device_preference_and_runtime_identity():
    mps = resolve_device("auto", torch_module=_torch(mps=True), system="Darwin")
    assert (mps.requested, mps.actual, mps.torch_device, mps.runtime) == (
        "auto",
        "mps",
        "mps",
        "mps",
    )

    cuda = resolve_device("auto", torch_module=_torch(cuda=True), system="Linux")
    assert (cuda.actual, cuda.torch_device, cuda.runtime) == ("cuda", "cuda", "cuda")

    rocm = resolve_device("auto", torch_module=_torch(cuda=True, hip="6.2"), system="Linux")
    assert (rocm.actual, rocm.torch_device, rocm.runtime) == ("rocm", "cuda", "rocm")
    assert rocm.to_dict(torch_module=_torch(cuda=True, hip="6.2"))["rocm_runtime_version"] == "6.2"

    cpu = resolve_device("auto", torch_module=_torch(), system="Linux")
    assert (cpu.actual, cpu.torch_device, cpu.runtime) == ("cpu", "cpu", "cpu")


@pytest.mark.parametrize("device", ["mps", "cuda", "rocm"])
def test_explicit_accelerator_selection_fails_without_matching_runtime(device):
    with pytest.raises(DeviceUnavailableError, match="explicitly requested"):
        resolve_device(device, torch_module=_torch(), system="Linux")


def test_cuda_and_rocm_explicit_requests_are_not_conflated():
    rocm_torch = _torch(cuda=True, hip="6.2")
    with pytest.raises(DeviceUnavailableError, match="ROCm runtime is active"):
        resolve_device("cuda", torch_module=rocm_torch, system="Linux")
    assert resolve_device("rocm", torch_module=rocm_torch, system="Linux").runtime == "rocm"

    with pytest.raises(DeviceUnavailableError, match="macOS PyTorch MPS runtime"):
        resolve_device("mps", torch_module=_torch(mps=False), system="Darwin")


def test_auto_fallback_can_only_be_recorded_for_auto_request():
    auto = resolve_device("auto", torch_module=_torch(), system="Linux")
    fallback = auto.with_fallback("cpu", "automatic accelerator placement failed")
    assert fallback.requested == "auto"
    assert fallback.actual == "cpu"
    assert fallback.fallback_reason

    explicit = resolve_device("cpu", torch_module=_torch(), system="Linux")
    with pytest.raises(DeviceUnavailableError, match="explicitly requested"):
        explicit.with_fallback("cpu", "not allowed")
