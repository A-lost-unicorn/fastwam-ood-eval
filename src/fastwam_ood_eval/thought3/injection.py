"""Scoped hook that injects the Adapter after ``action_encoder`` exactly once."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator

from torch import Tensor, nn

from fastwam_ood_eval.thought3.adapter import FutureToActionAdapter


class FutureInjectionError(RuntimeError):
    """Raised for missing, nested, stale, or repeated future contexts."""


@dataclass
class _ActiveFuture:
    latent: Tensor | None
    mask: Tensor | None
    expected_calls: int
    null_mask: bool = False
    calls: int = 0


class ActionEncoderFutureInjector:
    """Register one output hook while keeping future tensors request-scoped."""

    def __init__(
        self,
        action_encoder: nn.Module,
        adapter: FutureToActionAdapter,
    ) -> None:
        self.action_encoder = action_encoder
        self.adapter = adapter
        self._active: ContextVar[_ActiveFuture | None] = ContextVar(
            f"thought3_future_{id(self)}",
            default=None,
        )
        self._closed = False
        self._handle = action_encoder.register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> object:
        del module, inputs
        active = self._active.get()
        if active is None:
            return output
        if not isinstance(output, Tensor):
            raise FutureInjectionError(
                "action_encoder output must be a Tensor for Thought3 injection"
            )
        active.calls += 1
        if active.calls > active.expected_calls:
            raise FutureInjectionError(
                "action_encoder was called more often than the scoped contract"
            )
        if active.null_mask:
            if active.latent is not None or active.mask is not None:
                raise FutureInjectionError(
                    "formal null-mask context unexpectedly contains a tensor"
                )
            # Parameter-free null conditioning is an explicit request-scoped
            # identity at the injection boundary.  It does not construct a
            # zero latent and therefore gives a direct B0 parity control.
            return output
        if active.latent is None:
            raise FutureInjectionError(
                "non-null future context is missing its latent"
            )
        return self.adapter(output, active.latent, active.mask)

    @contextmanager
    def activate(
        self,
        future_latent: Tensor,
        future_mask: Tensor | None = None,
        *,
        expected_calls: int = 1,
    ) -> Iterator[None]:
        active = _ActiveFuture(
            latent=future_latent,
            mask=future_mask,
            expected_calls=expected_calls,
        )
        with self._activate(active):
            yield

    @contextmanager
    def activate_null(
        self,
        *,
        expected_calls: int = 1,
    ) -> Iterator[None]:
        """Activate a formal parameter-free null mask without a zero tensor."""

        active = _ActiveFuture(
            latent=None,
            mask=None,
            expected_calls=expected_calls,
            null_mask=True,
        )
        with self._activate(active):
            yield

    @contextmanager
    def _activate(self, active: _ActiveFuture) -> Iterator[None]:
        if self._closed:
            raise FutureInjectionError("injector is closed")
        if active.expected_calls <= 0:
            raise FutureInjectionError("expected_calls must be positive")
        if self._active.get() is not None:
            raise FutureInjectionError("nested future contexts are forbidden")
        token: Token[_ActiveFuture | None] = self._active.set(active)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self._active.reset(token)
            if not failed and active.calls != active.expected_calls:
                raise FutureInjectionError(
                    "action_encoder hook call mismatch: "
                    f"expected {active.expected_calls}, observed {active.calls}"
                )

    @property
    def has_active_context(self) -> bool:
        return self._active.get() is not None

    def close(self) -> None:
        if not self._closed:
            self._handle.remove()
            self._closed = True

    def __enter__(self) -> "ActionEncoderFutureInjector":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
