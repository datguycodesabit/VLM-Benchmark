"""Local PyTorch device selection, synchronization, and runtime diagnostics.

The module deliberately treats an explicitly requested accelerator as a hard
requirement. Only ``auto`` applies the documented accelerator-to-CPU fallback
policy. ROCm is selected through PyTorch's ``cuda`` device API and is identified
separately from NVIDIA CUDA using ``torch.version.hip``.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, replace
from typing import Any


class DeviceUnavailableError(RuntimeError):
    """A requested local runtime/device cannot be used in this process."""


@dataclass(frozen=True)
class DeviceResolution:
    """Requested device and its concrete PyTorch runtime selection."""

    requested: str
    actual: str
    torch_device: str
    runtime: str
    fallback_reason: str | None = None

    def to_dict(self, torch_module: Any | None = None) -> dict[str, Any]:
        """Return JSON-safe device and host/runtime identity fields."""
        return runtime_identity(self, torch_module=torch_module)

    def with_fallback(self, actual: str, reason: str) -> "DeviceResolution":
        """Record the only permitted fallback: an ``auto`` request to CPU."""
        if self.requested != "auto":
            raise DeviceUnavailableError(
                f"Cannot fall back from explicitly requested device {self.requested!r}"
            )
        if actual != "cpu":
            raise ValueError("Automatic device fallback must select CPU")
        return replace(
            self,
            actual="cpu",
            torch_device="cpu",
            runtime="cpu",
            fallback_reason=reason,
        )


def _torch(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except ImportError as exc:
        raise DeviceUnavailableError(
            "Local PyTorch adapters require PyTorch; install it before selecting a local model"
        ) from exc
    return torch


def _mps_available(torch: Any) -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _hip_version(torch: Any) -> str | None:
    try:
        value = getattr(getattr(torch, "version", None), "hip", None)
    except (AttributeError, RuntimeError):
        return None
    return value if isinstance(value, str) and value else None


def resolve_device(
    requested: str = "auto", *, torch_module: Any | None = None, system: str | None = None
) -> DeviceResolution:
    """Resolve ``auto|cpu|mps|cuda|rocm`` to a concrete PyTorch device.

    ``auto`` preference is MPS on macOS, then an available Linux CUDA/ROCm
    runtime, then CPU. If an automatically selected accelerator later fails to
    load a model, callers may record a fallback to CPU with
    :meth:`DeviceResolution.with_fallback`. An explicit accelerator never
    silently changes to CPU.
    """
    if not isinstance(requested, str) or requested not in {"auto", "cpu", "mps", "cuda", "rocm"}:
        raise ValueError("Device must be one of auto, cpu, mps, cuda, or rocm")
    torch = _torch(torch_module)
    host_os = system or platform.system()
    mps_ok = host_os == "Darwin" and _mps_available(torch)
    cuda_ok = host_os == "Linux" and _cuda_available(torch)
    hip = _hip_version(torch)

    if requested == "cpu":
        return DeviceResolution(requested, "cpu", "cpu", "cpu")
    if requested == "mps":
        if host_os != "Darwin" or not mps_ok:
            raise DeviceUnavailableError(
                "MPS was explicitly requested but a supported macOS PyTorch MPS runtime is unavailable"
            )
        return DeviceResolution(requested, "mps", "mps", "mps")
    if requested == "cuda":
        if host_os != "Linux":
            raise DeviceUnavailableError("CUDA selection is supported only on Linux")
        if hip:
            raise DeviceUnavailableError(
                "A ROCm runtime is active; request device 'rocm' to identify it explicitly"
            )
        if not cuda_ok:
            raise DeviceUnavailableError(
                "CUDA was explicitly requested but PyTorch reports no available NVIDIA CUDA device"
            )
        return DeviceResolution(requested, "cuda", "cuda", "cuda")
    if requested == "rocm":
        if host_os != "Linux":
            raise DeviceUnavailableError("ROCm selection is supported only on Linux")
        if not hip or not cuda_ok:
            raise DeviceUnavailableError(
                "ROCm was explicitly requested but PyTorch reports no available HIP/ROCm runtime"
            )
        return DeviceResolution(requested, "rocm", "cuda", "rocm")

    if mps_ok:
        return DeviceResolution("auto", "mps", "mps", "mps")
    if cuda_ok:
        runtime = "rocm" if hip else "cuda"
        return DeviceResolution("auto", runtime, "cuda", runtime)
    return DeviceResolution("auto", "cpu", "cpu", "cpu")


def synchronize(device: DeviceResolution | str, *, torch_module: Any | None = None) -> None:
    """Wait for queued accelerator work; CPU is already synchronous."""
    actual = device.torch_device if isinstance(device, DeviceResolution) else device
    if actual == "cpu":
        return
    torch = _torch(torch_module)
    if actual == "mps":
        torch.mps.synchronize()
        return
    if actual == "cuda" or actual.startswith("cuda:"):
        torch.cuda.synchronize(actual)
        return
    raise ValueError(f"Cannot synchronize unknown PyTorch device {actual!r}")


def runtime_identity(
    resolution: DeviceResolution, *, torch_module: Any | None = None
) -> dict[str, Any]:
    """Return a stable, credential-free environment identity for local runs."""
    torch = _torch(torch_module)
    result: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "requested_device": resolution.requested,
        "device": resolution.actual,
        "torch_device": resolution.torch_device,
        "runtime": resolution.runtime,
        "fallback_reason": resolution.fallback_reason,
        "pytorch_version": str(getattr(torch, "__version__", "unknown")),
    }
    version = getattr(torch, "version", None)
    cuda_version = getattr(version, "cuda", None)
    hip_version = getattr(version, "hip", None)
    if resolution.runtime == "cuda":
        result["cuda_runtime_version"] = cuda_version
    elif resolution.runtime == "rocm":
        result["rocm_runtime_version"] = hip_version
    if resolution.torch_device.startswith("cuda"):
        try:
            index = 0
            if ":" in resolution.torch_device:
                index = int(resolution.torch_device.partition(":")[2])
            result["device_name"] = torch.cuda.get_device_name(index)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            result["device_name"] = None
    return result


def memory_diagnostics(
    device: DeviceResolution | str, *, torch_module: Any | None = None
) -> dict[str, int | None]:
    """Best-effort allocated/peak accelerator memory in bytes."""
    actual = device.torch_device if isinstance(device, DeviceResolution) else device
    result: dict[str, int | None] = {
        "allocated_bytes": None,
        "peak_allocated_bytes": None,
    }
    if actual == "cpu":
        return result
    try:
        torch = _torch(torch_module)
        if actual == "mps":
            backend = torch.mps
            current = getattr(backend, "current_allocated_memory", None)
            result["allocated_bytes"] = int(current()) if callable(current) else None
            # PyTorch MPS has no portable peak-allocation API; leave it unknown.
            result["peak_allocated_bytes"] = None
        elif actual == "cuda" or actual.startswith("cuda:"):
            result["allocated_bytes"] = int(torch.cuda.memory_allocated(actual))
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(actual))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {"allocated_bytes": None, "peak_allocated_bytes": None}
    return result
