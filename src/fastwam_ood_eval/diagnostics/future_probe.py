"""Shared tensor/frame helpers for shadow future probes.

The temporal Wan VAE encoder silently ignores incomplete four-frame chunks.
Consequently diagnostic frames are encoded *independently* through the same
single-frame VAE path used for the live observation.  These embeddings are
useful for approximate frame-wise comparison, but are not native temporal
future latents.
"""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.diagnostics.rng_isolation import RngIsolation


APPROXIMATE_REENCODED_EMBEDDING = "decoded frame re-encoded independently without temporal context"


def require_numpy(numpy_module: Any | None = None) -> Any:
    if numpy_module is not None:
        return numpy_module
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - real Fast-WAM installs NumPy
        raise RuntimeError("Future diagnostics require numpy in the Fast-WAM runtime") from exc


def _to_numpy(value: Any, np: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value)


def frame_to_rgb_uint8(
    frame: Any,
    *,
    expected_height: int | None = None,
    expected_width: int | None = None,
    numpy_module: Any | None = None,
) -> Any:
    """Convert one decoded/model-space frame to a validated HWC uint8 array."""

    np = require_numpy(numpy_module)
    if isinstance(frame, Mapping):
        panels = [
            frame_to_rgb_uint8(value, numpy_module=np)
            for value in frame.values()
        ]
        if not panels:
            raise ValueError("A frame mapping must contain at least one camera")
        heights = {int(panel.shape[0]) for panel in panels}
        if len(heights) != 1:
            raise ValueError(f"Camera panels must have one height, got {sorted(heights)}")
        array = np.concatenate(panels, axis=1)
    else:
        convert = getattr(frame, "convert", None)
        if callable(convert):
            frame = convert("RGB")
        array = _to_numpy(frame, np)

    if getattr(array, "ndim", None) == 4 and int(array.shape[0]) == 1:
        array = array[0]
    if getattr(array, "ndim", None) == 3 and int(array.shape[0]) in (1, 3, 4) and int(
        array.shape[-1]
    ) not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if getattr(array, "ndim", None) == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if getattr(array, "ndim", None) != 3:
        raise ValueError(f"Expected an image-like 3D value, got shape {getattr(array, 'shape', None)}")
    if int(array.shape[-1]) == 1:
        array = np.repeat(array, 3, axis=-1)
    elif int(array.shape[-1]) == 4:
        array = array[..., :3]
    if int(array.shape[-1]) != 3:
        raise ValueError(f"Expected RGB channels, got shape {tuple(array.shape)}")

    if expected_height is not None and int(array.shape[0]) != int(expected_height):
        raise ValueError(
            f"Frame height mismatch: got {int(array.shape[0])}, expected {int(expected_height)}"
        )
    if expected_width is not None and int(array.shape[1]) != int(expected_width):
        raise ValueError(
            f"Frame width mismatch: got {int(array.shape[1])}, expected {int(expected_width)}"
        )

    if not bool(np.isfinite(array).all()):
        raise ValueError("Frame contains NaN or infinite values")
    minimum = float(array.min()) if int(array.size) else 0.0
    maximum = float(array.max()) if int(array.size) else 0.0
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(f"Decoded frame values must be in [0,255], got [{minimum},{maximum}]")
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


def model_tensor_to_rgb_uint8(
    image: Any,
    *,
    expected_height: int,
    expected_width: int,
    numpy_module: Any | None = None,
) -> Any:
    """Invert official ``[-1,1]`` CHW preprocessing to an HWC uint8 frame."""

    np = require_numpy(numpy_module)
    value = image.detach() if hasattr(image, "detach") else image
    if hasattr(value, "float") and callable(value.float):
        value = value.float()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    array = _to_numpy(value, np)
    if getattr(array, "ndim", None) == 4:
        if int(array.shape[0]) != 1:
            raise ValueError(f"Model image batch must be one, got shape {tuple(array.shape)}")
        array = array[0]
    if getattr(array, "ndim", None) != 3 or int(array.shape[0]) != 3:
        raise ValueError(f"Model image must be CHW RGB, got shape {getattr(array, 'shape', None)}")
    array = np.transpose(array, (1, 2, 0))
    if tuple(int(v) for v in array.shape[:2]) != (int(expected_height), int(expected_width)):
        raise ValueError(
            "Model image geometry mismatch: "
            f"got {tuple(int(v) for v in array.shape[:2])}, "
            f"expected {(int(expected_height), int(expected_width))}"
        )
    if not bool(np.isfinite(array).all()):
        raise ValueError("Model image contains NaN or infinite values")
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < -1.0001 or maximum > 1.0001:
        raise ValueError(f"Model image values must be in [-1,1], got [{minimum},{maximum}]")
    array = (array + 1.0) * 127.5
    return np.ascontiguousarray(np.clip(array, 0.0, 255.0).astype(np.uint8))


def rgb_uint8_to_model_tensor(
    frame: Any,
    *,
    torch_module: Any,
    device: Any,
    dtype: Any,
    expected_height: int,
    expected_width: int,
    numpy_module: Any | None = None,
) -> Any:
    """Convert an already camera-concatenated RGB frame to official model space."""

    np = require_numpy(numpy_module)
    array = frame_to_rgb_uint8(
        frame,
        expected_height=expected_height,
        expected_width=expected_width,
        numpy_module=np,
    )
    tensor = torch_module.as_tensor(np.ascontiguousarray(array.copy()))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor * (2.0 / 255.0) - 1.0


def encode_frames_independently_with_first_frame_vae(
    frames: Sequence[Any],
    *,
    model: Any,
    torch_module: Any,
    device: Any,
    dtype: Any,
    expected_height: int,
    expected_width: int,
    tiled: bool = False,
    numpy_module: Any | None = None,
) -> Any:
    """Return ``[T,C,H',W']`` CPU embeddings via independent frame encodes.

    This intentionally calls ``_encode_input_image_latents_tensor`` once per
    frame and removes both its batch and singleton temporal dimensions.  The
    result is an approximate re-encoded frame embedding, never a native
    temporal future latent.
    """

    if len(frames) == 0:
        raise ValueError("encode_frame_embeddings requires at least one frame")
    np = require_numpy(numpy_module)
    inference_mode = getattr(torch_module, "inference_mode", None)
    inference_context = inference_mode() if callable(inference_mode) else nullcontext()
    embeddings = []
    with RngIsolation(None, numpy_module=np, torch_module=torch_module), inference_context:
        for frame in frames:
            image = rgb_uint8_to_model_tensor(
                frame,
                torch_module=torch_module,
                device=device,
                dtype=dtype,
                expected_height=expected_height,
                expected_width=expected_width,
                numpy_module=np,
            )
            latent = model._encode_input_image_latents_tensor(input_image=image, tiled=bool(tiled))
            shape = tuple(int(v) for v in latent.shape)
            if len(shape) != 5 or shape[0] != 1 or shape[2] != 1:
                raise RuntimeError(
                    "Fast-WAM first-frame VAE encoder must return [1,C,1,H,W], "
                    f"got {shape}"
                )
            embedding = latent[0, :, 0].detach()
            if hasattr(embedding, "float") and callable(embedding.float):
                embedding = embedding.float()
            embedding = embedding.cpu()
            embeddings.append(embedding)
    return torch_module.stack(embeddings, dim=0)


__all__ = [
    "APPROXIMATE_REENCODED_EMBEDDING",
    "encode_frames_independently_with_first_frame_vae",
    "frame_to_rgb_uint8",
    "model_tensor_to_rgb_uint8",
    "require_numpy",
    "rgb_uint8_to_model_tensor",
]
