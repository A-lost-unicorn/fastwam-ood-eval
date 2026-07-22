"""Snapshot and restore process RNG state around shadow diagnostics.

Future diagnostics must not perturb the action policy's random stream.  This
module deliberately treats Python, NumPy, Torch CPU, and every visible CUDA
device as one state bundle and restores that bundle even when a probe raises.
NumPy and Torch are optional imports so the lightweight mock test environment
does not need the real Fast-WAM runtime installed.
"""

from __future__ import annotations

import copy
import importlib
import random
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any


_AUTO = object()


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _resolve_optional_module(value: Any, name: str) -> Any | None:
    if value is _AUTO:
        return _optional_module(name)
    return value


def _clone_state(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(value)


def _cuda_is_available(torch_module: Any | None) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(callable(is_available) and is_available())


@dataclass
class RngSnapshot:
    """Complete restorable RNG state for the libraries available at capture."""

    python_random_state: object
    numpy_random_state: object | None = None
    torch_cpu_rng_state: Any | None = None
    torch_cuda_rng_states: tuple[Any, ...] | None = None
    _numpy_module: Any | None = field(default=None, repr=False, compare=False)
    _torch_module: Any | None = field(default=None, repr=False, compare=False)

    @classmethod
    def capture(
        cls,
        *,
        numpy_module: Any = _AUTO,
        torch_module: Any = _AUTO,
    ) -> "RngSnapshot":
        """Capture all supported global RNGs without advancing any of them."""

        np = _resolve_optional_module(numpy_module, "numpy")
        torch = _resolve_optional_module(torch_module, "torch")

        numpy_state = None
        if np is not None:
            get_state = getattr(getattr(np, "random", None), "get_state", None)
            if callable(get_state):
                numpy_state = copy.deepcopy(get_state())

        torch_cpu_state = None
        get_rng_state = getattr(torch, "get_rng_state", None)
        if callable(get_rng_state):
            torch_cpu_state = _clone_state(get_rng_state())

        cuda_states = None
        if _cuda_is_available(torch):
            get_rng_state_all = getattr(torch.cuda, "get_rng_state_all", None)
            if callable(get_rng_state_all):
                cuda_states = tuple(_clone_state(state) for state in get_rng_state_all())

        return cls(
            python_random_state=random.getstate(),
            numpy_random_state=numpy_state,
            torch_cpu_rng_state=torch_cpu_state,
            torch_cuda_rng_states=cuda_states,
            _numpy_module=np,
            _torch_module=torch,
        )

    def restore(self) -> None:
        """Restore exactly the states captured by :meth:`capture`."""

        random.setstate(self.python_random_state)

        if self.numpy_random_state is not None and self._numpy_module is not None:
            set_state = getattr(getattr(self._numpy_module, "random", None), "set_state", None)
            if not callable(set_state):
                raise RuntimeError("NumPy RNG state was captured but numpy.random.set_state is unavailable")
            set_state(copy.deepcopy(self.numpy_random_state))

        if self.torch_cpu_rng_state is not None and self._torch_module is not None:
            set_rng_state = getattr(self._torch_module, "set_rng_state", None)
            if not callable(set_rng_state):
                raise RuntimeError("Torch CPU RNG state was captured but torch.set_rng_state is unavailable")
            set_rng_state(_clone_state(self.torch_cpu_rng_state))

        if self.torch_cuda_rng_states is not None and self._torch_module is not None:
            set_rng_state_all = getattr(getattr(self._torch_module, "cuda", None), "set_rng_state_all", None)
            if not callable(set_rng_state_all):
                raise RuntimeError(
                    "Torch CUDA RNG states were captured but torch.cuda.set_rng_state_all is unavailable"
                )
            set_rng_state_all([_clone_state(state) for state in self.torch_cuda_rng_states])


class RngIsolation(AbstractContextManager["RngIsolation"]):
    """Run diagnostic work under a seed, then restore every global RNG.

    ``diagnostic_seed=None`` is useful when isolation is needed without
    reseeding.  Passing modules explicitly is primarily useful for tests and
    for runtimes that expose Torch through an adapter.
    """

    def __init__(
        self,
        diagnostic_seed: int | None = None,
        *,
        numpy_module: Any = _AUTO,
        torch_module: Any = _AUTO,
    ) -> None:
        if isinstance(diagnostic_seed, bool):
            raise TypeError("diagnostic_seed must be an integer, not bool")
        self.diagnostic_seed = None if diagnostic_seed is None else int(diagnostic_seed)
        self._numpy_module_arg = numpy_module
        self._torch_module_arg = torch_module
        self.snapshot: RngSnapshot | None = None

    def __enter__(self) -> "RngIsolation":
        if self.snapshot is not None:
            raise RuntimeError("RngIsolation contexts cannot be re-entered")
        self.snapshot = RngSnapshot.capture(
            numpy_module=self._numpy_module_arg,
            torch_module=self._torch_module_arg,
        )
        if self.diagnostic_seed is not None:
            self._seed_diagnostic_streams(self.diagnostic_seed)
        return self

    def _seed_diagnostic_streams(self, seed: int) -> None:
        assert self.snapshot is not None
        random.seed(seed)

        np = self.snapshot._numpy_module
        if np is not None:
            numpy_seed = getattr(getattr(np, "random", None), "seed", None)
            if callable(numpy_seed):
                numpy_seed(seed % (2**32))

        torch = self.snapshot._torch_module
        if torch is not None:
            manual_seed = getattr(torch, "manual_seed", None)
            if callable(manual_seed):
                manual_seed(seed)
            if _cuda_is_available(torch):
                cuda_manual_seed_all = getattr(torch.cuda, "manual_seed_all", None)
                if callable(cuda_manual_seed_all):
                    cuda_manual_seed_all(seed)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.snapshot is None:
            raise RuntimeError("RngIsolation exited before entering")
        try:
            self.snapshot.restore()
        finally:
            self.snapshot = None
        return False


__all__ = ["RngIsolation", "RngSnapshot"]
