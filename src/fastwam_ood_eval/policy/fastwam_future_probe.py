"""Strict shadow future probes backed by an already-loaded FastWAMAdapter."""

from __future__ import annotations

import copy
import inspect
import operator
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastwam_ood_eval.checkpoint import sha256_file
from fastwam_ood_eval.diagnostics.artifact_writer import action_chunk_hash, clone_action_chunk
from fastwam_ood_eval.diagnostics.future_probe import (
    APPROXIMATE_REENCODED_EMBEDDING,
    encode_frames_independently_with_first_frame_vae,
    frame_to_rgb_uint8,
    model_tensor_to_rgb_uint8,
    require_numpy,
)
from fastwam_ood_eval.diagnostics.protocol import FutureProbeOutput
from fastwam_ood_eval.diagnostics.rng_isolation import RngIsolation


# This is deliberately empty for the audited release.  Adding an entry is a
# source-reviewed trust decision, not a YAML override.  A future entry must be
# keyed by the checkpoint SHA-256 and bind the exact upstream commit, model
# config identity, and training recipe that established action-conditioned
# video training.
APPROVED_ACTION_CONDITIONED_CHECKPOINTS: dict[str, dict[str, str]] = {}


def _clone_observation_value(value: Any) -> Any:
    """Preserve observation value types while breaking all mutable storage."""

    if isinstance(value, Mapping):
        return {key: _clone_observation_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_observation_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_observation_value(item) for item in value)

    # Torch tensors expose clone(), while NumPy arrays and PIL images expose
    # copy().  Both paths preserve the value's runtime type and allocate
    # independent mutable storage without importing either heavy dependency.
    clone = getattr(value, "clone", None)
    if callable(clone):
        cloned = clone()
        if cloned is not value:
            return cloned
    copy_value = getattr(value, "copy", None)
    if callable(copy_value):
        try:
            copied = copy_value()
        except (TypeError, ValueError):
            pass
        else:
            if copied is not value:
                return copied
    return copy.deepcopy(value)


class FastWAMFutureProbe:
    """Generate a shadow future without changing executable actions.

    The object owns no model: it reuses the exact model, processor, official
    helper module, Hydra config, device, and task state of ``FastWAMAdapter``.
    ``unconditional_future`` is the release-compatible associational protocol;
    ``action_conditioned_future`` retains the stricter checkpoint and temporal
    dependency gates. Capability is validated at construction so an
    incompatible checkpoint fails before a benchmark episode or policy action.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        mode: str = "action_conditioned_future",
        numpy_module: Any | None = None,
        _mock_checkpoint_verifier: Any | None = None,
    ) -> None:
        if mode not in {"unconditional_future", "action_conditioned_future"}:
            raise ValueError(f"Unsupported future probe mode: {mode!r}")
        self.adapter = adapter
        self.mode = mode
        self.future_mode = mode
        self.model = adapter.model
        self.processor = adapter.processor
        self.official = adapter.official
        self.upstream_cfg = adapter.upstream_cfg
        self.device = adapter.device
        self.torch = adapter.torch
        self.input_h = int(adapter.input_h)
        self.input_w = int(adapter.input_w)
        self.action_horizon = int(adapter.action_horizon)
        self.action_video_freq_ratio = 0
        self.vae_temporal_downsample_factor = 0
        self.video_dit_temporal_patch_size = 0
        self.action_conditioning_latent_frame_count = 0
        self.action_conditioning_group_count = 0
        self.action_conditioning_group_size = 0
        self.video_attention_mask_mode = ""
        self.action_dependency_scope = ""
        self.required_executed_actions_for_first_future = 0
        self.control_horizon = 0
        self.checkpoint_verification: dict[str, Any] = {}
        self._checkpoint_verified_identity: tuple[int, int, int, str] | None = None
        self._numpy_module = numpy_module
        self._mock_checkpoint_verifier = _mock_checkpoint_verifier
        self.validate_capability()

    def validate_capability(self) -> None:
        """Validate the exact video semantics requested by ``self.mode``."""

        video_expert = getattr(self.model, "video_expert", None)
        action_conditioned = getattr(video_expert, "action_conditioned", None)
        if self.mode == "action_conditioned_future":
            if action_conditioned is not True:
                raise RuntimeError(
                    "FastWAMFutureProbe rejects libero_uncond: "
                    "model.video_expert.action_conditioned=false, so the checkpoint can only "
                    "produce an unconditional future. Action-conditioned future diagnostics "
                    "require action_conditioned=True."
                )
        elif action_conditioned is not False:
            raise RuntimeError(
                "unconditional_future requires a checkpoint whose video expert explicitly has "
                "action_conditioned=false; use action_conditioned_future for conditioned models"
            )
        if getattr(self.model, "training", False) is True:
            raise RuntimeError(
                "FastWAMFutureProbe requires model.training=false before diagnostics. "
                "model.infer_joint calls eval() internally, which would otherwise persistently "
                "change the executable policy's model mode."
            )

        infer_joint = getattr(self.model, "infer_joint", None)
        if not callable(infer_joint):
            raise RuntimeError("Action-conditioned future diagnostics require model.infer_joint")
        parameters = inspect.signature(infer_joint).parameters
        required_parameters = {
            "input_image",
            "num_video_frames",
            "action_horizon",
            "action",
            "proprio",
            "num_inference_steps",
            "seed",
            "test_action_with_infer_action",
        }
        missing = sorted(required_parameters - set(parameters))
        if missing:
            raise RuntimeError(f"model.infer_joint is missing required parameters: {missing}")

        self._validate_camera_concat()
        if self.mode == "action_conditioned_future":
            action_state_transforms = getattr(
                self.processor,
                "action_state_transforms",
                object(),
            )
            if action_state_transforms is not None:
                raise RuntimeError(
                    "FastWAM future diagnostics require processor.action_state_transforms is None. "
                    "The current LIBERO release uses no action-state transform; a non-null or "
                    "unknown transform cannot be skipped safely before official action "
                    "normalization."
                )
            self._expected_action_dim()
            self._validate_action_conditioning_alignment()
        else:
            self._validate_unconditional_alignment()
            self.checkpoint_verification = {
                "unconditional_video_architecture_verified": True,
                "video_expert_action_conditioned": False,
                "scientific_scope": (
                    "observation/language/proprio-conditioned future; protected policy actions "
                    "are not inputs to the video branch"
                ),
            }
            return
        checkpoint_path = str(
            getattr(getattr(self.adapter.cfg, "checkpoint", None), "path", "")
        )
        checkpoint_identity = (
            id(self.model),
            id(self.model.video_expert),
            id(getattr(self.model.video_expert, "action_embedding", None)),
            checkpoint_path,
        )
        if checkpoint_identity != self._checkpoint_verified_identity:
            self.checkpoint_verification = self._verify_action_conditioned_checkpoint()
            self._checkpoint_verified_identity = checkpoint_identity

    def _validate_unconditional_alignment(self) -> None:
        """Validate release video timing without inventing action dependencies."""

        self.action_video_freq_ratio = self._exact_positive_int(
            self.upstream_cfg.data.train.action_video_freq_ratio,
            "data.train.action_video_freq_ratio",
        )
        num_video_frames = self._exact_positive_int(
            self.official._get_num_video_frames(self.upstream_cfg),
            "official num_video_frames",
        )
        if num_video_frames <= 1:
            raise RuntimeError(
                "Unconditional future diagnostics require more than one video frame, "
                f"got {num_video_frames}"
            )
        vae = getattr(self.model, "vae", None)
        self.vae_temporal_downsample_factor = self._exact_positive_int(
            getattr(vae, "temporal_downsample_factor", None),
            "model.vae.temporal_downsample_factor",
        )
        tail_frames = num_video_frames - 1
        if tail_frames % self.vae_temporal_downsample_factor != 0:
            raise RuntimeError(
                "Official video length cannot be aligned to the actual VAE temporal factor: "
                f"num_video_frames={num_video_frames}, "
                f"temporal_downsample_factor={self.vae_temporal_downsample_factor}"
            )
        patch_size = getattr(self.model.video_expert, "patch_size", None)
        if not isinstance(patch_size, Sequence) or isinstance(patch_size, (str, bytes)):
            raise RuntimeError(
                "model.video_expert.patch_size must expose the actual three-dimensional DiT patch"
            )
        if len(patch_size) != 3:
            raise RuntimeError(
                "model.video_expert.patch_size must contain [temporal,height,width], "
                f"got {patch_size!r}"
            )
        self.video_dit_temporal_patch_size = self._exact_positive_int(
            patch_size[0],
            "model.video_expert.patch_size[0]",
        )
        self.control_horizon = self._exact_positive_int(
            self.adapter.cfg.benchmark.control_horizon,
            "benchmark.control_horizon",
        )
        self.action_conditioning_latent_frame_count = 0
        self.action_conditioning_group_count = 0
        self.action_conditioning_group_size = 0
        self.video_attention_mask_mode = str(
            getattr(self.model.video_expert, "video_attention_mask_mode", "")
        )
        self.action_dependency_scope = "not_applicable_unconditional"
        self.required_executed_actions_for_first_future = 0

    def _verify_action_conditioned_checkpoint(self) -> dict[str, Any]:
        """Prove both trusted provenance and actual action-embedding loading.

        Upstream loads ``mot`` with ``strict=False``.  Merely flipping the
        Hydra architecture flag could therefore leave a random action
        embedding in memory.  Real runs are accepted only when the checkpoint
        hash is in this module's reviewed allowlist *and* the live embedding is
        byte-for-byte equal (after dtype conversion) to the checkpoint value.
        """

        if self._mock_checkpoint_verifier is not None:
            backend = getattr(getattr(self.adapter.cfg, "benchmark", None), "backend", None)
            if backend != "mock":
                raise RuntimeError(
                    "The injected checkpoint verifier is restricted to the CPU mock backend"
                )
            result = dict(self._mock_checkpoint_verifier(self))
            if not (
                result.get("action_conditioning_parameters_loaded_verified") is True
                and result.get("action_conditioned_training_provenance_verified") is True
            ):
                raise RuntimeError("Mock checkpoint verifier did not return both verification flags")
            return result

        if not APPROVED_ACTION_CONDITIONED_CHECKPOINTS:
            raise RuntimeError(
                "No action-conditioned Fast-WAM checkpoint is approved in this repository. "
                "Runtime action_conditioned=true is insufficient because upstream loads mot "
                "with strict=False. Add a source-reviewed checkpoint SHA-256/config/training "
                "recipe record only after matched training provenance is available."
            )

        checkpoint_cfg = getattr(self.adapter.cfg, "checkpoint", None)
        checkpoint_path = Path(str(getattr(checkpoint_cfg, "path", "")))
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Cannot verify action-conditioned checkpoint: {checkpoint_path}")
        checkpoint_hash = sha256_file(checkpoint_path)
        trust = APPROVED_ACTION_CONDITIONED_CHECKPOINTS.get(str(checkpoint_hash))
        if trust is None:
            raise RuntimeError(
                "Action-conditioned checkpoint SHA-256 is not in the reviewed allowlist: "
                f"{checkpoint_hash}"
            )

        policy_cfg = getattr(self.adapter.cfg, "policy", None)
        recipe_id = getattr(policy_cfg, "training_recipe_id", None)
        if recipe_id in (None, "") or str(recipe_id) != trust.get("training_recipe_id"):
            raise RuntimeError(
                "Action-conditioned checkpoint training_recipe_id is missing or does not match "
                "the reviewed provenance record"
            )
        model_name = str(getattr(checkpoint_cfg, "model_name", ""))
        if model_name != trust.get("model_name"):
            raise RuntimeError(
                "Action-conditioned checkpoint model_name does not match reviewed provenance: "
                f"configured={model_name!r}, approved={trust.get('model_name')!r}"
            )
        config_path = Path(str(getattr(checkpoint_cfg, "config_path", "")))
        config_hash = sha256_file(config_path)
        if config_hash != trust.get("config_sha256"):
            raise RuntimeError(
                "Fast-WAM root config SHA-256 does not match reviewed action-conditioned "
                f"provenance: current={config_hash!r}, approved={trust.get('config_sha256')!r}"
            )
        current_fastwam_commit = self._fastwam_commit()
        if current_fastwam_commit != trust.get("fastwam_commit"):
            raise RuntimeError(
                "Pinned Fast-WAM commit does not match reviewed action-conditioned provenance: "
                f"current={current_fastwam_commit!r}, approved={trust.get('fastwam_commit')!r}"
            )

        loaded_keys = self._verify_live_action_embedding_matches_checkpoint(checkpoint_path)
        return {
            "checkpoint_sha256": checkpoint_hash,
            "training_recipe_id": str(recipe_id),
            "approved_model_name": model_name,
            "approved_config_sha256": config_hash,
            "approved_fastwam_commit": current_fastwam_commit,
            "action_embedding_checkpoint_keys": loaded_keys,
            "action_conditioning_parameters_loaded_verified": True,
            "action_conditioned_training_provenance_verified": True,
        }

    @staticmethod
    def _fastwam_commit() -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", "third_party/FastWAM", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() or None

    def _verify_live_action_embedding_matches_checkpoint(self, checkpoint_path: Path) -> list[str]:
        load = getattr(self.torch, "load", None)
        if not callable(load):
            raise RuntimeError("torch.load is unavailable for checkpoint verification")
        load_parameters = inspect.signature(load).parameters
        if "mmap" not in load_parameters or "weights_only" not in load_parameters:
            raise RuntimeError(
                "Checkpoint verification requires torch.load(mmap=True, weights_only=True); "
                "eager or unsafe fallback loading is forbidden"
            )
        try:
            payload = load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not memory-map the checkpoint for strict action-embedding verification"
            ) from exc
        try:
            if not isinstance(payload, Mapping) or not isinstance(payload.get("mot"), Mapping):
                raise RuntimeError("Checkpoint must contain a mapping payload['mot']")
            checkpoint_state = payload["mot"]
            action_embedding = getattr(self.model.video_expert, "action_embedding", None)
            state_dict = getattr(action_embedding, "state_dict", None)
            if not callable(state_dict):
                raise RuntimeError("Action-conditioned video expert has no action_embedding state")
            live_state = state_dict()
            if not isinstance(live_state, Mapping) or not live_state:
                raise RuntimeError("Action-conditioned action_embedding state is empty")

            prefix = "mixtures.video.action_embedding."
            expected_keys = {prefix + str(name) for name in live_state}
            checkpoint_keys = {
                str(name) for name in checkpoint_state if str(name).startswith(prefix)
            }
            if checkpoint_keys != expected_keys:
                raise RuntimeError(
                    "Checkpoint action_embedding keys do not exactly match the live model: "
                    f"checkpoint={sorted(checkpoint_keys)}, expected={sorted(expected_keys)}"
                )
            equal = getattr(self.torch, "equal", None)
            if not callable(equal):
                raise RuntimeError("torch.equal is unavailable for checkpoint verification")
            for local_name, live_value in live_state.items():
                full_name = prefix + str(local_name)
                checkpoint_value = checkpoint_state[full_name]
                if tuple(checkpoint_value.shape) != tuple(live_value.shape):
                    raise RuntimeError(
                        f"Checkpoint parameter shape mismatch for {full_name}: "
                        f"checkpoint={tuple(checkpoint_value.shape)}, live={tuple(live_value.shape)}"
                    )
                expected = checkpoint_value.to(device="cpu", dtype=live_value.dtype)
                actual = live_value.detach().to(device="cpu")
                if not bool(equal(actual, expected)):
                    raise RuntimeError(
                        f"Live parameter {full_name} does not equal the checkpoint value; "
                        "strict=False may have left an uninitialized conditioning parameter"
                    )
            return sorted(expected_keys)
        finally:
            # Releasing every reference closes the large mmap without ever
            # materializing the full checkpoint on CPU or GPU.
            del payload

    @staticmethod
    def _exact_positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool):
            raise RuntimeError(f"{label} must be a positive integer, got bool")
        try:
            parsed = operator.index(value)
        except TypeError as exc:
            raise RuntimeError(f"{label} must be a positive integer, got {value!r}") from exc
        parsed = int(parsed)
        if parsed <= 0:
            raise RuntimeError(f"{label} must be a positive integer, got {parsed}")
        return parsed

    def _validate_action_conditioning_alignment(self) -> None:
        """Derive the exact action groups created by WanVideoDiT.pre_dit."""

        self.action_video_freq_ratio = self._exact_positive_int(
            self.upstream_cfg.data.train.action_video_freq_ratio,
            "data.train.action_video_freq_ratio",
        )
        num_video_frames = self._exact_positive_int(
            self.official._get_num_video_frames(self.upstream_cfg),
            "official num_video_frames",
        )
        if num_video_frames <= 1:
            raise RuntimeError(
                "Action-conditioned future diagnostics require more than one video frame, "
                f"got {num_video_frames}"
            )

        vae = getattr(self.model, "vae", None)
        self.vae_temporal_downsample_factor = self._exact_positive_int(
            getattr(vae, "temporal_downsample_factor", None),
            "model.vae.temporal_downsample_factor",
        )
        tail_frames = num_video_frames - 1
        if tail_frames % self.vae_temporal_downsample_factor != 0:
            raise RuntimeError(
                "Official video length cannot be aligned to the actual VAE temporal factor: "
                f"num_video_frames={num_video_frames}, "
                f"temporal_downsample_factor={self.vae_temporal_downsample_factor}"
            )
        latent_frame_count = tail_frames // self.vae_temporal_downsample_factor + 1

        patch_size = getattr(self.model.video_expert, "patch_size", None)
        if not isinstance(patch_size, Sequence) or isinstance(patch_size, (str, bytes)):
            raise RuntimeError(
                "model.video_expert.patch_size must expose the actual three-dimensional DiT patch"
            )
        if len(patch_size) != 3:
            raise RuntimeError(
                "model.video_expert.patch_size must contain [temporal,height,width], "
                f"got {patch_size!r}"
            )
        self.video_dit_temporal_patch_size = self._exact_positive_int(
            patch_size[0],
            "model.video_expert.patch_size[0]",
        )
        if self.video_dit_temporal_patch_size != 1:
            raise RuntimeError(
                "Action-conditioned alignment currently requires temporal DiT patch size=1. "
                "A larger patch can mix the fixed first latent with future latents before "
                "action conditioning, so its decoded-frame dependency mapping is unproven."
            )
        if latent_frame_count % self.video_dit_temporal_patch_size != 0:
            raise RuntimeError(
                "VAE latent frames are not divisible by the actual temporal DiT patch size: "
                f"latent_frames={latent_frame_count}, "
                f"temporal_patch_size={self.video_dit_temporal_patch_size}"
            )

        dit_frame_count = latent_frame_count // self.video_dit_temporal_patch_size
        group_count = dit_frame_count - 1
        if group_count <= 0:
            raise RuntimeError(
                "WanVideoDiT action conditioning requires at least one post-initial temporal "
                f"group, got latent_frames={latent_frame_count}, DiT_frames={dit_frame_count}"
            )
        upstream_latent_groups = latent_frame_count - 1
        if self.action_horizon % upstream_latent_groups != 0:
            raise RuntimeError(
                "Action horizon violates WanVideoDiT's pre-patch latent grouping check: "
                f"action_horizon={self.action_horizon}, latent_groups={upstream_latent_groups}"
            )
        if self.action_horizon % group_count != 0:
            raise RuntimeError(
                "Action horizon is not divisible by WanVideoDiT's actual post-patch temporal "
                f"groups: action_horizon={self.action_horizon}, groups={group_count}"
            )

        try:
            configured_control_horizon = self.adapter.cfg.benchmark.control_horizon
        except AttributeError as exc:
            raise RuntimeError(
                "FastWAMFutureProbe requires adapter.cfg.benchmark.control_horizon to validate "
                "action-conditioned temporal groups"
            ) from exc
        self.control_horizon = self._exact_positive_int(
            configured_control_horizon,
            "benchmark.control_horizon",
        )
        self.action_conditioning_latent_frame_count = latent_frame_count
        self.action_conditioning_group_count = group_count
        self.action_conditioning_group_size = self.action_horizon // group_count
        self.video_attention_mask_mode = str(
            getattr(self.model.video_expert, "video_attention_mask_mode", "")
        )
        if self.video_attention_mask_mode == "per_frame_causal":
            self.action_dependency_scope = "causal_prefix"
            required_actions = self.action_conditioning_group_size
        elif self.video_attention_mask_mode in {"first_frame_causal", "bidirectional"}:
            self.action_dependency_scope = "all_future_groups"
            required_actions = self.action_horizon
        else:
            raise RuntimeError(
                "Unsupported or unproven video attention mask for action dependency closure: "
                f"{self.video_attention_mask_mode!r}"
            )
        self.required_executed_actions_for_first_future = required_actions
        if self.control_horizon < required_actions:
            raise RuntimeError(
                "Unsafe action-conditioned future alignment: "
                f"benchmark.control_horizon={self.control_horizon} is smaller than the proven "
                f"dependency closure for the first future frame={required_actions}. "
                f"video_attention_mask_mode={self.video_attention_mask_mode!r} allows the first "
                "future latent to depend directly or indirectly on actions that are never "
                "executed before replanning "
                f"(num_video_frames={num_video_frames}, "
                f"VAE temporal factor={self.vae_temporal_downsample_factor}, "
                f"direct_group_size={self.action_conditioning_group_size}, "
                f"action_horizon={self.action_horizon})."
            )

    def _validate_camera_concat(self) -> None:
        image_meta = list(self.processor.shape_meta["images"])
        camera_count = int(self.processor.num_output_cameras)
        concat_mode = str(self.upstream_cfg.data.train.get("concat_multi_camera", ""))
        if camera_count != 2 or len(image_meta) < 2 or concat_mode != "horizontal":
            raise RuntimeError(
                "FastWAM LIBERO future diagnostics require the official dual-camera horizontal "
                f"path; got num_output_cameras={camera_count}, image_meta={len(image_meta)}, "
                f"concat_multi_camera={concat_mode!r}"
            )

        shapes = [tuple(int(v) for v in meta["shape"]) for meta in image_meta[:2]]
        if any(len(shape) != 3 or shape[0] != 3 for shape in shapes):
            raise RuntimeError(f"Official dual-camera shapes must be [3,H,W], got {shapes}")
        if shapes[0][1] != shapes[1][1]:
            raise RuntimeError(f"Horizontal camera panels must have equal heights, got {shapes}")
        expected_hw = (shapes[0][1], shapes[0][2] + shapes[1][2])
        if expected_hw != (self.input_h, self.input_w):
            raise RuntimeError(
                "Official camera concatenation does not match model input size: "
                f"camera-derived={expected_hw}, configured={(self.input_h, self.input_w)}"
            )

    def _expected_action_dim(self) -> int:
        action_meta = list(self.processor.shape_meta["action"])
        if len(action_meta) != 1:
            raise RuntimeError(
                "FastWAM LIBERO future diagnostics require one official action field, "
                f"got {len(action_meta)}"
            )
        metadata_dim = int(action_meta[0]["shape"])
        model_dim = int(self.model.action_expert.action_dim)
        processor_dim = int(self.processor.action_output_dim)
        if len({metadata_dim, model_dim, processor_dim}) != 1:
            raise RuntimeError(
                "Action dimension mismatch across processor/model metadata: "
                f"shape_meta={metadata_dim}, processor={processor_dim}, model={model_dim}"
            )
        return model_dim

    def _resolve_num_video_frames(self, requested: int | None) -> int:
        derived = int(self.official._get_num_video_frames(self.upstream_cfg))
        value = derived if requested is None else int(requested)
        if value != derived:
            raise ValueError(
                "num_video_frames must match the official training-derived value: "
                f"requested={value}, derived={derived}"
            )
        if value <= 1 or (value - 1) % self.vae_temporal_downsample_factor != 0:
            raise ValueError(
                "num_video_frames must be >1 and satisfy "
                "(T - 1) % VAE temporal_downsample_factor == 0 "
                f"(T % {self.vae_temporal_downsample_factor} == 1), got {value}"
            )
        return value

    def _prepare_model_observation(self, observation: dict[str, Any]) -> tuple[Any, Any, Any, str]:
        prompt_template = self.official.DEFAULT_PROMPT
        prompt = str(prompt_template).format(task=self.adapter.task_description)
        shadow_observation = _clone_observation_value(observation)
        image, proprio, raw_images = self.official._obs_to_model_input(
            shadow_observation,
            cfg=self.upstream_cfg,
            processor=self.processor,
            width=self.input_w,
            height=self.input_h,
            device=self.device,
            dtype=self.model.torch_dtype,
        )
        image_shape = tuple(int(v) for v in image.shape)
        expected_shape = (1, 3, self.input_h, self.input_w)
        if image_shape != expected_shape:
            raise RuntimeError(f"Official observation preprocessing returned {image_shape}, expected {expected_shape}")
        if getattr(image, "dtype", None) != self.model.torch_dtype:
            raise TypeError(
                "Official observation dtype mismatch: "
                f"got {getattr(image, 'dtype', None)}, expected {self.model.torch_dtype}"
            )
        if not isinstance(raw_images, Mapping) or tuple(raw_images.keys())[:2] != (
            "image",
            "wrist_image",
        ):
            raise RuntimeError(
                "Official LIBERO preprocessing must preserve ordered image/wrist_image camera panels"
            )
        return image, proprio, raw_images, prompt

    def observation_to_model_frame(self, observation: dict[str, Any]) -> Any:
        """Return official dual-camera preprocessing as uint8 ``[H,W,3]``."""

        np = require_numpy(self._numpy_module)
        with RngIsolation(None, numpy_module=np, torch_module=self.torch):
            image, _, _, _ = self._prepare_model_observation(observation)
            return model_tensor_to_rgb_uint8(
                image,
                expected_height=self.input_h,
                expected_width=self.input_w,
                numpy_module=np,
            )

    def encode_frame_embeddings(self, frames: Sequence[Any]) -> Any:
        """Return approximate ``[T,C,H',W']`` CPU first-frame VAE embeddings.

        Frames are intentionally encoded one by one.  These values are decoded
        frame re-encodings without temporal context, not native video latents.
        """

        np = require_numpy(self._numpy_module)
        tiled = bool(self.upstream_cfg.EVALUATION.get("tiled", False))
        return encode_frames_independently_with_first_frame_vae(
            frames,
            model=self.model,
            torch_module=self.torch,
            device=self.device,
            dtype=self.model.torch_dtype,
            expected_height=self.input_h,
            expected_width=self.input_w,
            tiled=tiled,
            numpy_module=np,
        )

    def _normalize_executable_actions(self, actions: Any) -> tuple[Any, str]:
        original_hash = action_chunk_hash(actions)
        executable_copy = clone_action_chunk(actions)
        source_tensor = self.torch.as_tensor(executable_copy, device="cpu")
        is_floating_point = getattr(self.torch, "is_floating_point", None)
        if callable(is_floating_point):
            source_is_float = bool(is_floating_point(source_tensor))
        else:  # pragma: no cover - real Torch always exposes is_floating_point
            source_is_float = bool(getattr(getattr(source_tensor, "dtype", None), "is_floating_point", False))
        if not source_is_float:
            raise TypeError(
                "Executable actions must have a floating-point input dtype before conversion; "
                f"got {getattr(source_tensor, 'dtype', None)}"
            )
        tensor = source_tensor.to(device="cpu", dtype=self.torch.float32).clone()
        expected_shape = (self.action_horizon, self._expected_action_dim())
        actual_shape = tuple(int(v) for v in tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Executable action shape must be {expected_shape}, got {actual_shape}")
        if getattr(tensor, "dtype", None) != self.torch.float32:
            raise TypeError(f"Executable actions must convert to torch.float32, got {tensor.dtype}")

        # Official evaluator maps dataset gripper g to env gripper 1 - 2*g.
        # Invert that mapping on the private copy before dataset normalization.
        dataset_actions = tensor.clone()
        dataset_actions[..., -1] = (1.0 - dataset_actions[..., -1]) / 2.0

        action_meta = list(self.processor.shape_meta["action"])
        action_key = action_meta[0]["key"]
        try:
            field_normalizer = self.processor.normalizer.normalizers["action"][action_key]
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(f"Missing official action field normalizer for {action_key!r}") from exc
        normalized = field_normalizer.forward(dataset_actions)
        normalized = normalized.to(device="cpu", dtype=self.torch.float32).clone()
        if tuple(int(v) for v in normalized.shape) != expected_shape:
            raise RuntimeError(
                "Official action normalizer changed action shape: "
                f"got {tuple(int(v) for v in normalized.shape)}, expected {expected_shape}"
            )
        isfinite = getattr(self.torch, "isfinite", None)
        if callable(isfinite):
            finite = isfinite(normalized).all()
            finite_value = finite.item() if hasattr(finite, "item") else finite
            if not bool(finite_value):
                raise ValueError("Normalized action condition contains NaN or infinite values")
        if action_chunk_hash(actions) != original_hash:
            raise RuntimeError("Future probe mutated the executable action chunk while normalizing a copy")
        return normalized, original_hash

    def _cuda_available(self) -> bool:
        cuda = getattr(self.torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        return bool(callable(is_available) and is_available())

    def _synchronize_cuda(self) -> None:
        if not self._cuda_available():
            return
        synchronize = getattr(self.torch.cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(self.device)

    def _cuda_peak_bytes(self) -> int:
        if not self._cuda_available():
            return 0
        peak = getattr(self.torch.cuda, "max_memory_allocated", None)
        return int(peak(self.device)) if callable(peak) else 0

    def _reset_cuda_peak_memory_stats(self) -> bool:
        """Reset peak telemetry when supported, without making it a dependency."""

        if not self._cuda_available():
            return False
        reset_peak = getattr(self.torch.cuda, "reset_peak_memory_stats", None)
        if not callable(reset_peak):
            return False
        try:
            reset_peak(self.device)
        except (NotImplementedError, RuntimeError, TypeError):
            return False
        return True

    def predict_action_conditioned_future(
        self,
        observation: dict,
        actions: Any,
        *,
        diagnostic_seed: int,
        num_video_frames: int | None,
        num_inference_steps: int,
    ) -> FutureProbeOutput:
        """Generate a future conditioned on a private copy of executable actions."""

        if self.mode != "action_conditioned_future":
            raise RuntimeError(
                "predict_action_conditioned_future requires mode='action_conditioned_future'"
            )
        self.validate_capability()
        normalized_actions, executable_hash = self._normalize_executable_actions(actions)
        return self._generate_future(
            observation,
            actions,
            action_condition=normalized_actions,
            executable_hash=executable_hash,
            diagnostic_seed=diagnostic_seed,
            num_video_frames=num_video_frames,
            num_inference_steps=num_inference_steps,
        )

    def predict_unconditional_future(
        self,
        observation: dict,
        actions: Any,
        *,
        diagnostic_seed: int,
        num_video_frames: int | None,
        num_inference_steps: int,
    ) -> FutureProbeOutput:
        """Generate the release future without feeding policy actions to video."""

        if self.mode != "unconditional_future":
            raise RuntimeError(
                "predict_unconditional_future requires mode='unconditional_future'"
            )
        self.validate_capability()
        executable_hash = action_chunk_hash(actions)
        return self._generate_future(
            observation,
            actions,
            action_condition=None,
            executable_hash=executable_hash,
            diagnostic_seed=diagnostic_seed,
            num_video_frames=num_video_frames,
            num_inference_steps=num_inference_steps,
        )

    def _generate_future(
        self,
        observation: dict,
        actions: Any,
        *,
        action_condition: Any | None,
        executable_hash: str,
        diagnostic_seed: int,
        num_video_frames: int | None,
        num_inference_steps: int,
    ) -> FutureProbeOutput:
        """Run the official joint video path and discard its returned action."""

        if isinstance(diagnostic_seed, bool):
            raise TypeError("diagnostic_seed must be an integer, not bool")
        diagnostic_seed = int(diagnostic_seed)
        num_video_frames = self._resolve_num_video_frames(num_video_frames)
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")

        np = require_numpy(self._numpy_module)
        inference_mode = getattr(self.torch, "inference_mode", None)
        inference_context = inference_mode() if callable(inference_mode) else nullcontext()

        with RngIsolation(
            diagnostic_seed,
            numpy_module=np,
            torch_module=self.torch,
        ), inference_context:
            image, proprio, _, prompt = self._prepare_model_observation(observation)
            model_space_input = model_tensor_to_rgb_uint8(
                image,
                expected_height=self.input_h,
                expected_width=self.input_w,
                numpy_module=np,
            )

            evaluation_cfg = self.upstream_cfg.EVALUATION
            infer_kwargs = {
                "prompt": prompt,
                "input_image": image,
                "num_video_frames": num_video_frames,
                "action_horizon": self.action_horizon,
                "action": (
                    action_condition.clone()
                    if action_condition is not None
                    else None
                ),
                "proprio": proprio,
                "negative_prompt": str(evaluation_cfg.get("negative_prompt", "")),
                "text_cfg_scale": float(evaluation_cfg.get("text_cfg_scale", 1.0)),
                "num_inference_steps": num_inference_steps,
                "sigma_shift": (
                    None
                    if evaluation_cfg.get("sigma_shift") is None
                    else float(evaluation_cfg.get("sigma_shift"))
                ),
                "seed": diagnostic_seed,
                "rand_device": str(evaluation_cfg.get("rand_device", "cpu")),
                "tiled": bool(evaluation_cfg.get("tiled", False)),
                "test_action_with_infer_action": False,
            }

            self._synchronize_cuda()
            cumulative_peak_before_reset = self._cuda_peak_bytes()
            peak_was_reset = self._reset_cuda_peak_memory_stats()
            peak_before = self._cuda_peak_bytes()
            started = time.perf_counter()
            prediction = self.model.infer_joint(**infer_kwargs)
            self._synchronize_cuda()
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_after = self._cuda_peak_bytes()

        if not isinstance(prediction, Mapping):
            raise TypeError(f"model.infer_joint must return a mapping, got {type(prediction).__name__}")
        frames = prediction.get("video")
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise TypeError("model.infer_joint['video'] must be a frame sequence")
        if len(frames) != num_video_frames:
            raise RuntimeError(
                f"model.infer_joint returned {len(frames)} frames, expected {num_video_frames}"
            )
        model_frames = [
            frame_to_rgb_uint8(
                frame,
                expected_height=self.input_h,
                expected_width=self.input_w,
                numpy_module=np,
            )
            for frame in frames
        ]
        model_space_predicted_frames = np.stack(model_frames, axis=0)

        if action_chunk_hash(actions) != executable_hash:
            raise RuntimeError("Future probe mutated the executable action chunk")

        frame_control_offsets = [
            index * self.action_video_freq_ratio for index in range(num_video_frames)
        ]
        peak_mb = float(peak_after / 2**20)
        if peak_was_reset:
            peak_semantics = (
                "process max_memory_allocated after reset immediately before infer_joint "
                "through synchronized probe completion"
            )
        elif self._cuda_available():
            peak_semantics = (
                "process cumulative max_memory_allocated; reset_peak_memory_stats unavailable "
                "or unsuccessful"
            )
        else:
            peak_semantics = "CUDA unavailable; GPU peak memory reported as zero"
        action_conditioned = action_condition is not None
        metadata = {
            "future_kind": (
                "action_conditioned" if action_conditioned else "unconditional"
            ),
            "action_conditioned": action_conditioned,
            "protected_policy_action_used_as_video_condition": action_conditioned,
            "diagnostic_seed": diagnostic_seed,
            "num_video_frames": num_video_frames,
            "num_inference_steps": num_inference_steps,
            "action_horizon": self.action_horizon,
            "action_dim": self._expected_action_dim(),
            "normalized_action_shape": (
                list(action_condition.shape) if action_conditioned else None
            ),
            "normalized_action_dtype": (
                str(action_condition.dtype) if action_conditioned else None
            ),
            "executable_action_sha256": executable_hash,
            "frame_control_offsets": frame_control_offsets,
            "action_video_freq_ratio": self.action_video_freq_ratio,
            "vae_temporal_downsample_factor": self.vae_temporal_downsample_factor,
            "video_dit_temporal_patch_size": self.video_dit_temporal_patch_size,
            "action_conditioning_latent_frame_count": (
                self.action_conditioning_latent_frame_count
                if action_conditioned
                else None
            ),
            "action_conditioning_group_count": (
                self.action_conditioning_group_count if action_conditioned else None
            ),
            "action_conditioning_group_size": (
                self.action_conditioning_group_size if action_conditioned else None
            ),
            "video_attention_mask_mode": self.video_attention_mask_mode,
            "action_dependency_scope": self.action_dependency_scope,
            "required_executed_actions_for_first_future": (
                self.required_executed_actions_for_first_future
            ),
            "control_horizon": self.control_horizon,
            "input_hw": [self.input_h, self.input_w],
            "returned_action_discarded": "action" in prediction,
            "gpu_peak_memory_semantics": peak_semantics,
            "gpu_peak_memory_reset_before_probe": peak_was_reset,
            "gpu_peak_memory_cumulative_before_reset_mb": float(
                cumulative_peak_before_reset / 2**20
            ),
            "gpu_peak_memory_before_mb": float(peak_before / 2**20),
            "gpu_peak_memory_delta_mb": float(max(0, peak_after - peak_before) / 2**20),
            "native_future_latents_available": False,
            "frame_embedding_semantics": APPROXIMATE_REENCODED_EMBEDDING,
            "checkpoint_verification": dict(self.checkpoint_verification),
        }
        return FutureProbeOutput(
            predicted_frames=list(frames),
            model_space_input=model_space_input,
            model_space_predicted_frames=model_space_predicted_frames,
            predicted_latents=None,
            latency_ms=latency_ms,
            gpu_peak_memory_mb=peak_mb,
            metadata=metadata,
        )


__all__ = ["FastWAMFutureProbe"]
