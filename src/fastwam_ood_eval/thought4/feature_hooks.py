"""Real-call hooks for frozen Fast-WAM Video/Action representations."""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence


class FeatureHookError(RuntimeError):
    """Raised for an invalid path, call count, tensor or replacement."""


HookLocation = Literal["input", "output"]
TensorTransform = Callable[[Any], Any]


def resolve_module(root: Any, module_path: str) -> Any:
    """Resolve an exact dot path, including integer ModuleList indices."""

    if not module_path or module_path.startswith(".") or module_path.endswith("."):
        raise FeatureHookError(f"invalid module path: {module_path!r}")
    current = root
    consumed: list[str] = []
    for component in module_path.split("."):
        consumed.append(component)
        if component.isdigit():
            try:
                current = current[int(component)]
            except (IndexError, KeyError, TypeError) as exc:
                raise FeatureHookError(
                    f"invalid module index at {'.'.join(consumed)}"
                ) from exc
        else:
            if not hasattr(current, component):
                raise FeatureHookError(
                    f"missing module at {'.'.join(consumed)}"
                )
            current = getattr(current, component)
    if not hasattr(current, "register_forward_hook"):
        raise FeatureHookError(f"path is not a torch module: {module_path}")
    return current


def validate_layer_indices(expert: Any, indices: Iterable[int], name: str) -> None:
    if not hasattr(expert, "blocks"):
        raise FeatureHookError(f"{name} has no blocks")
    count = len(expert.blocks)
    values = tuple(int(index) for index in indices)
    if not values:
        raise FeatureHookError(f"{name} layer selection is empty")
    if len(values) != len(set(values)):
        raise FeatureHookError(f"{name} layer selection contains duplicates")
    invalid = [index for index in values if index < 0 or index >= count]
    if invalid:
        raise FeatureHookError(
            f"{name} invalid layer indices {invalid}; actual block count={count}"
        )


def _first_tensor(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except FeatureHookError:
                continue
    if isinstance(value, dict):
        for key in sorted(value):
            try:
                return _first_tensor(value[key])
            except FeatureHookError:
                continue
    raise FeatureHookError("hook payload contains no Tensor")


def _replace_first_tensor(value: Any, replacement: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return replacement
    if isinstance(value, tuple):
        result = list(value)
        for index, item in enumerate(result):
            try:
                result[index] = _replace_first_tensor(item, replacement)
                return tuple(result)
            except FeatureHookError:
                continue
    if isinstance(value, list):
        result = list(value)
        for index, item in enumerate(result):
            try:
                result[index] = _replace_first_tensor(item, replacement)
                return result
            except FeatureHookError:
                continue
    if isinstance(value, dict):
        result = dict(value)
        for key in sorted(result):
            try:
                result[key] = _replace_first_tensor(result[key], replacement)
                return result
            except FeatureHookError:
                continue
    raise FeatureHookError("hook payload contains no Tensor")


@dataclass(frozen=True)
class HookSpec:
    name: str
    module_path: str
    location: HookLocation
    expected_calls: int | None = None

    def __post_init__(self) -> None:
        if self.location not in {"input", "output"}:
            raise FeatureHookError(f"invalid hook location: {self.location}")
        if self.expected_calls is not None and self.expected_calls <= 0:
            raise FeatureHookError("expected_calls must be positive")


@dataclass(frozen=True)
class VideoKVCacheSpec:
    """One actual current-frame cache tensor consumed by Action DiT."""

    layer_index: int
    kind: Literal["k", "v"]
    expected_calls: int | None = None

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise FeatureHookError("cache layer index must be non-negative")
        if self.kind not in {"k", "v"}:
            raise FeatureHookError("cache kind must be k or v")
        if self.expected_calls is not None and self.expected_calls <= 0:
            raise FeatureHookError("cache expected_calls must be positive")

    @property
    def name(self) -> str:
        return f"video_l{self.layer_index}_cache_{self.kind}"

    @property
    def module_path(self) -> str:
        return f"mot.video_kv_cache.{self.layer_index}.{self.kind}"


def _video_cache_from_kwargs(kwargs: Any) -> Any:
    if not isinstance(kwargs, dict) or "video_kv_cache" not in kwargs:
        raise FeatureHookError(
            "forward_action_with_video_cache must receive video_kv_cache by keyword"
        )
    cache = kwargs["video_kv_cache"]
    if not isinstance(cache, list) or not cache:
        raise FeatureHookError("video_kv_cache must be a non-empty list")
    return cache


def _cache_tensor(
    cache: Any,
    spec: VideoKVCacheSpec,
    *,
    validate_finite: bool = True,
) -> Any:
    import torch

    if spec.layer_index >= len(cache):
        raise FeatureHookError(
            f"cache layer {spec.layer_index} absent; cache length={len(cache)}"
        )
    row = cache[spec.layer_index]
    if not isinstance(row, dict) or spec.kind not in row:
        raise FeatureHookError(
            f"cache layer {spec.layer_index} lacks {spec.kind}"
        )
    tensor = row[spec.kind]
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise FeatureHookError("video cache entry must be a floating Tensor")
    if validate_finite and not bool(tensor.isfinite().all().item()):
        raise FeatureHookError("video cache entry contains NaN/Inf")
    return tensor


class ScopedVideoKVCacheCapture(
    AbstractContextManager["ScopedVideoKVCacheCapture"]
):
    """Capture the exact K/V list passed into every Action denoise call."""

    def __init__(
        self,
        mot: Any,
        specs: Sequence[VideoKVCacheSpec],
        *,
        to_cpu: bool = True,
    ) -> None:
        if not specs or len({spec.name for spec in specs}) != len(specs):
            raise FeatureHookError("cache capture specs must be non-empty/unique")
        self.mot = mot
        self.specs = tuple(specs)
        self.to_cpu = to_cpu
        self.captured: dict[str, list[Any]] = {spec.name: [] for spec in specs}
        self.calls = 0
        self._original: Any | None = None
        self._had_instance_attribute = False
        self._identities: dict[str, tuple[int, int, Any, Any]] = {}

    def __enter__(self) -> "ScopedVideoKVCacheCapture":
        if self._original is not None or not hasattr(
            self.mot, "forward_action_with_video_cache"
        ):
            raise FeatureHookError("invalid or re-entered MoT cache capture")
        self._had_instance_attribute = (
            "forward_action_with_video_cache" in vars(self.mot)
        )
        self._original = self.mot.forward_action_with_video_cache

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            cache = _video_cache_from_kwargs(kwargs)
            for spec in self.specs:
                tensor = _cache_tensor(
                    cache,
                    spec,
                    validate_finite=not bool(self.captured[spec.name]),
                )
                identity = (
                    int(tensor.data_ptr()),
                    int(getattr(tensor, "_version", 0)),
                    tensor.shape,
                    tensor.dtype,
                )
                prior = self._identities.setdefault(spec.name, identity)
                if prior != identity:
                    raise FeatureHookError(
                        f"video cache changed across denoise calls: {spec.name}"
                    )
                if not self.captured[spec.name]:
                    value = tensor.detach().clone()
                    if self.to_cpu:
                        value = value.cpu()
                    self.captured[spec.name].append(value)
            self.calls += 1
            return self._original(*args, **kwargs)

        setattr(self.mot, "forward_action_with_video_cache", wrapped)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original is not None:
            if self._had_instance_attribute:
                setattr(self.mot, "forward_action_with_video_cache", self._original)
            else:
                delattr(self.mot, "forward_action_with_video_cache")
            self._original = None
        if exc_type is None:
            for spec in self.specs:
                if len(self.captured[spec.name]) != 1:
                    raise FeatureHookError(f"cache capture missing: {spec.name}")
                if spec.expected_calls is not None and self.calls != spec.expected_calls:
                    raise FeatureHookError(
                        f"cache call mismatch: expected {spec.expected_calls}, "
                        f"observed {self.calls}"
                    )


class ScopedVideoKVCacheReplacement(
    AbstractContextManager["ScopedVideoKVCacheReplacement"]
):
    """Replace one actual cache tensor without mutating the frozen cache list."""

    def __init__(
        self,
        mot: Any,
        spec: VideoKVCacheSpec,
        transform: TensorTransform,
    ) -> None:
        self.mot = mot
        self.spec = spec
        self.transform = transform
        self.calls = 0
        self._original: Any | None = None
        self._had_instance_attribute = False

    def __enter__(self) -> "ScopedVideoKVCacheReplacement":
        if self._original is not None or not hasattr(
            self.mot, "forward_action_with_video_cache"
        ):
            raise FeatureHookError("invalid or re-entered MoT cache replacement")
        self._had_instance_attribute = (
            "forward_action_with_video_cache" in vars(self.mot)
        )
        self._original = self.mot.forward_action_with_video_cache

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            cache = _video_cache_from_kwargs(kwargs)
            original = _cache_tensor(cache, self.spec)
            replacement = self.transform(original)
            if replacement.shape != original.shape:
                raise FeatureHookError("cache replacement shape mismatch")
            if replacement.dtype != original.dtype or replacement.device != original.device:
                raise FeatureHookError(
                    "cache replacement must preserve dtype/device"
                )
            if not bool(replacement.isfinite().all().item()):
                raise FeatureHookError("cache replacement contains NaN/Inf")
            updated = list(cache)
            updated_row = dict(updated[self.spec.layer_index])
            updated_row[self.spec.kind] = replacement
            updated[self.spec.layer_index] = updated_row
            updated_kwargs = dict(kwargs)
            updated_kwargs["video_kv_cache"] = updated
            self.calls += 1
            return self._original(*args, **updated_kwargs)

        setattr(self.mot, "forward_action_with_video_cache", wrapped)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original is not None:
            if self._had_instance_attribute:
                setattr(self.mot, "forward_action_with_video_cache", self._original)
            else:
                delattr(self.mot, "forward_action_with_video_cache")
            self._original = None
        if exc_type is None:
            if self.calls == 0:
                raise FeatureHookError("cache replacement did not fire")
            if (
                self.spec.expected_calls is not None
                and self.calls != self.spec.expected_calls
            ):
                raise FeatureHookError(
                    f"cache replacement call mismatch: expected "
                    f"{self.spec.expected_calls}, observed {self.calls}"
                )


class ScopedFeatureCapture(AbstractContextManager["ScopedFeatureCapture"]):
    """Capture detached activations and remove all hooks on scope exit."""

    def __init__(
        self,
        root: Any,
        specs: Sequence[HookSpec],
        *,
        clone: bool = True,
        to_cpu: bool = False,
    ) -> None:
        if not specs:
            raise FeatureHookError("at least one hook spec is required")
        if len({spec.name for spec in specs}) != len(specs):
            raise FeatureHookError("hook spec names must be unique")
        self.root = root
        self.specs = tuple(specs)
        self.clone = clone
        self.to_cpu = to_cpu
        self.captured: dict[str, list[Any]] = {spec.name: [] for spec in specs}
        self._handles: list[Any] = []
        self._active = False

    def _save(self, name: str, payload: Any) -> None:
        tensor = _first_tensor(payload)
        if not tensor.is_floating_point():
            raise FeatureHookError(f"{name} activation must be floating point")
        if not bool(tensor.isfinite().all().item()):
            raise FeatureHookError(f"{name} activation contains NaN/Inf")
        detached = tensor.detach()
        if self.clone:
            detached = detached.clone()
        if self.to_cpu:
            detached = detached.cpu()
        self.captured[name].append(detached)

    def __enter__(self) -> "ScopedFeatureCapture":
        if self._active:
            raise FeatureHookError("feature capture cannot be nested/re-entered")
        for spec in self.specs:
            module = resolve_module(self.root, spec.module_path)
            if spec.location == "input":
                handle = module.register_forward_pre_hook(
                    lambda _module, args, name=spec.name: self._save(name, args)
                )
            else:
                handle = module.register_forward_hook(
                    lambda _module, _args, output, name=spec.name: self._save(
                        name, output
                    )
                )
            self._handles.append(handle)
        self._active = True
        return self

    def close(self, *, validate: bool = True) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._active = False
        if validate:
            for spec in self.specs:
                count = len(self.captured[spec.name])
                if count == 0:
                    raise FeatureHookError(
                        f"hook did not fire: {spec.name} ({spec.module_path})"
                    )
                if spec.expected_calls is not None and count != spec.expected_calls:
                    raise FeatureHookError(
                        f"hook call mismatch for {spec.name}: "
                        f"expected {spec.expected_calls}, observed {count}"
                    )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(validate=exc_type is None)


class ScopedFeatureReplacement(AbstractContextManager["ScopedFeatureReplacement"]):
    """Replace only the first Tensor at one real module boundary."""

    def __init__(
        self,
        root: Any,
        spec: HookSpec,
        transform: TensorTransform,
    ) -> None:
        self.root = root
        self.spec = spec
        self.transform = transform
        self.calls = 0
        self._handle: Any | None = None

    def _apply(self, payload: Any) -> Any:
        original = _first_tensor(payload)
        replacement = self.transform(original)
        if replacement is original:
            replacement = replacement.clone()
        if replacement.shape != original.shape:
            raise FeatureHookError(
                f"replacement shape mismatch: {replacement.shape} vs {original.shape}"
            )
        if replacement.dtype != original.dtype or replacement.device != original.device:
            raise FeatureHookError("replacement must preserve dtype and device")
        if not bool(replacement.isfinite().all().item()):
            raise FeatureHookError("replacement contains NaN/Inf")
        self.calls += 1
        return _replace_first_tensor(payload, replacement)

    def __enter__(self) -> "ScopedFeatureReplacement":
        module = resolve_module(self.root, self.spec.module_path)
        if self.spec.location == "input":
            self._handle = module.register_forward_pre_hook(
                lambda _module, args: self._apply(args)
            )
        else:
            self._handle = module.register_forward_hook(
                lambda _module, _args, output: self._apply(output)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if exc_type is None:
            if self.calls == 0:
                raise FeatureHookError(
                    f"replacement hook did not fire: {self.spec.module_path}"
                )
            if (
                self.spec.expected_calls is not None
                and self.calls != self.spec.expected_calls
            ):
                raise FeatureHookError(
                    f"replacement call mismatch: expected "
                    f"{self.spec.expected_calls}, observed {self.calls}"
                )


def freeze_backbone(module: Any) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def assert_backbone_frozen(module: Any) -> None:
    trainable = [name for name, value in module.named_parameters() if value.requires_grad]
    if trainable:
        raise FeatureHookError(
            f"frozen backbone contains trainable parameters: {trainable[:5]}"
        )


def assert_probe_only_trainable(backbone: Any, probe: Any) -> None:
    assert_backbone_frozen(backbone)
    probe_parameters = list(probe.parameters())
    if not probe_parameters or not all(value.requires_grad for value in probe_parameters):
        raise FeatureHookError("probe parameters must be the only trainable parameters")


def parameter_state_sha256(module: Any) -> str:
    """Thought3-compatible SHA over named parameters and raw bytes."""

    import torch

    digest = hashlib.sha256()
    for name, value in sorted(module.named_parameters(), key=lambda item: item[0]):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def video_hook_specs(
    layers: Sequence[int],
    *,
    include_kv: bool = False,
) -> tuple[HookSpec, ...]:
    specs: list[HookSpec] = []
    for layer in layers:
        specs.append(
            HookSpec(
                name=f"video_l{layer}_hidden",
                module_path=f"video_expert.blocks.{layer}.norm1",
                location="input",
            )
        )
        if include_kv:
            specs.extend(
                (
                    HookSpec(
                        name=f"video_l{layer}_projected_k",
                        module_path=f"video_expert.blocks.{layer}.self_attn.k",
                        location="output",
                    ),
                    HookSpec(
                        name=f"video_l{layer}_projected_v",
                        module_path=f"video_expert.blocks.{layer}.self_attn.v",
                        location="output",
                    ),
                )
            )
    return tuple(specs)


def video_kv_cache_specs(
    layers: Sequence[int],
    *,
    expected_calls: int | None = None,
) -> tuple[VideoKVCacheSpec, ...]:
    return tuple(
        VideoKVCacheSpec(int(layer), kind, expected_calls)
        for layer in layers
        for kind in ("k", "v")
    )


def action_hook_specs(
    module_paths: Sequence[str] | None = None,
) -> tuple[HookSpec, ...]:
    definitions = {
        "action_expert.action_encoder": ("action_input", "output"),
        "action_expert.blocks.15.norm1": ("action_middle", "input"),
        "action_expert.blocks.29.norm1": ("action_late", "input"),
        "action_expert.head": ("action_pre_head", "input"),
    }
    selected = tuple(module_paths) if module_paths is not None else tuple(definitions)
    unknown = [path for path in selected if path not in definitions]
    if unknown:
        raise FeatureHookError(f"unsupported Action hook paths: {unknown}")
    return tuple(
        HookSpec(definitions[path][0], path, definitions[path][1])
        for path in selected
    )
