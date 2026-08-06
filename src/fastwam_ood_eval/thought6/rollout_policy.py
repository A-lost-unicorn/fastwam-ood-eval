"""Online 20-step Action Flow policy with an externally frozen sigma gate."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

from fastwam_ood_eval.thought6 import ACTION_DENOISE_STEPS
from fastwam_ood_eval.thought6.future_modes import (
    FuturePayload,
    FusionDecision,
    FusionMode,
    decide_future_fusion,
)
from fastwam_ood_eval.thought6.schemas import Thought6Error, object_sha256, tensor_sha256
from fastwam_ood_eval.thought6.sigma_gate import validate_runtime_schedule


@dataclass
class _StepScope:
    decision: FusionDecision
    future_latent: Any | None
    future_mask: Any | None
    expected_calls: int
    calls: int = 0
    diagnostic: dict[str, Any] | None = None


class SigmaAwareFutureInjector:
    """Request-scoped action-encoder hook; the low-sigma branch is identity."""

    def __init__(self, action_encoder: Any, adapter: Any) -> None:
        self.action_encoder = action_encoder
        self.adapter = adapter
        self._active: ContextVar[_StepScope | None] = ContextVar(
            f"thought6_sigma_scope_{id(self)}", default=None
        )
        self._closed = False
        self._handle = action_encoder.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        import torch

        scope = self._active.get()
        if scope is None:
            return output
        if not isinstance(output, torch.Tensor):
            raise Thought6Error("action_encoder output must be a Tensor")
        if not bool(torch.isfinite(output).all()):
            raise Thought6Error("pre-fusion action hidden contains NaN/Inf")
        scope.calls += 1
        if scope.calls > scope.expected_calls:
            raise Thought6Error("action_encoder exceeded the scoped call contract")
        before = tensor_sha256(output)
        if not scope.decision.adapter_called:
            result = output
            contribution_rms = 0.0
        else:
            if scope.future_latent is None:
                raise Thought6Error("enabled future fusion is missing the K=1 latent")
            result = self.adapter(output, scope.future_latent, scope.future_mask)
            if not isinstance(result, torch.Tensor) or result.shape != output.shape:
                raise Thought6Error("Adapter output differs from action hidden shape")
            if not bool(torch.isfinite(result).all()):
                raise Thought6Error("Adapter output contains NaN/Inf")
            delta = result.detach().float() - output.detach().float()
            contribution_rms = float(torch.sqrt(torch.mean(delta.square())).cpu())
            if not math.isfinite(contribution_rms):
                raise Thought6Error("Adapter contribution RMS is non-finite")
        after = tensor_sha256(result)
        if not scope.decision.adapter_called and (result is not output or before != after):
            raise Thought6Error("disabled sigma gate did not preserve exact identity")
        scope.diagnostic = {
            **asdict(scope.decision),
            "adapter_output_rms": contribution_rms,
            "pre_fusion_hidden_sha256": before,
            "post_fusion_hidden_sha256": after,
            "identity_branch_bitwise": before == after,
        }
        return result

    @contextmanager
    def activate_step(
        self,
        decision: FusionDecision,
        *,
        future_latent: Any | None,
        future_mask: Any | None,
        expected_calls: int = 1,
    ) -> Iterator[_StepScope]:
        if self._closed or self._active.get() is not None:
            raise Thought6Error("invalid nested or closed sigma-aware scope")
        if expected_calls <= 0:
            raise Thought6Error("expected_calls must be positive")
        scope = _StepScope(decision, future_latent, future_mask, expected_calls)
        token: Token[_StepScope | None] = self._active.set(scope)
        failed = False
        try:
            yield scope
        except BaseException:
            failed = True
            raise
        finally:
            self._active.reset(token)
            if not failed and (
                scope.calls != expected_calls or scope.diagnostic is None
            ):
                raise Thought6Error(
                    f"action_encoder call mismatch: expected={expected_calls}, observed={scope.calls}"
                )

    def close(self) -> None:
        if not self._closed:
            self._handle.remove()
            self._closed = True

    def __enter__(self) -> "SigmaAwareFutureInjector":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def freeze_for_phase6(model: Any, adapter: Any) -> None:
    """Freeze both components and reject stale gradients; no optimizer exists."""

    model.requires_grad_(False)
    adapter.requires_grad_(False)
    model.eval()
    adapter.eval()
    parameters = list(model.parameters()) + list(adapter.parameters())
    if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
        raise Thought6Error("Phase 6 requires all parameters frozen and gradient-free")


def _future_for_decision(
    decision: FusionDecision,
    *,
    correct_future: Any | None,
    shuffled_future: Any | None,
) -> Any | None:
    if not decision.adapter_called:
        return None
    if decision.payload == "correct":
        value = correct_future
    elif decision.payload == "shuffle":
        value = shuffled_future
    else:
        raise Thought6Error("enabled fusion has an invalid payload")
    if value is None:
        raise Thought6Error(f"{decision.payload} future is absent")
    return value


def run_action_denoising(
    model: Any,
    adapter: Any,
    *,
    current_latent: Any,
    context: Any,
    context_mask: Any,
    mode: FusionMode | str,
    condition: str,
    action_seed: int,
    correct_future: Any | None = None,
    shuffled_future: Any | None = None,
    payload_override: FuturePayload | str | None = None,
    action_horizon: int = 32,
) -> tuple[Any, dict[str, Any]]:
    """Run exactly one frozen online action chunk and retain every gate event."""

    import torch
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
    )

    if action_horizon != 32:
        raise Thought6Error("Phase 6 freezes the release 32-action horizon")
    freeze_for_phase6(model, adapter)
    device = str(next(model.parameters()).device)
    dtype = model.torch_dtype
    current = current_latent.to(device=device, dtype=dtype)
    prompt = context.to(device=device, dtype=dtype)
    mask = context_mask.to(device=device, dtype=torch.bool)
    correct = None if correct_future is None else correct_future.to(device=device, dtype=dtype)
    shuffled = None if shuffled_future is None else shuffled_future.to(device=device, dtype=dtype)
    future_mask = torch.ones((1, 2, 14, 28), dtype=torch.bool, device=device)
    generator = torch.Generator(device="cpu").manual_seed(int(action_seed))
    action = torch.randn(
        (1, action_horizon, 7), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device=device, dtype=dtype)
    initial_action_sha256 = tensor_sha256(action)
    started = time.perf_counter()
    with torch.inference_mode():
        video_cache, attention_mask, video_seq_len = _prepare_video_cache(
            model, current, prompt, mask, action_seq_len=action_horizon
        )
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=ACTION_DENOISE_STEPS,
            device=device,
            dtype=action.dtype,
            shift_override=None,
        )
    schedule = validate_runtime_schedule(timesteps, deltas)
    rows: list[dict[str, Any]] = []
    adapter_calls = 0
    with SigmaAwareFutureInjector(model.action_expert.action_encoder, adapter) as injector:
        with torch.inference_mode():
            for schedule_row, step_t, step_delta in zip(
                schedule, timesteps, deltas, strict=True
            ):
                decision = decide_future_fusion(
                    mode,
                    condition=condition,
                    effective_sigma=schedule_row.effective_sigma,
                    payload_override=payload_override,
                )
                selected = _future_for_decision(
                    decision,
                    correct_future=correct,
                    shuffled_future=shuffled,
                )
                action_before = tensor_sha256(action)
                with injector.activate_step(
                    decision,
                    future_latent=selected,
                    future_mask=(future_mask if selected is not None else None),
                ) as scope:
                    timestep = step_t.reshape(1).to(device=device, dtype=dtype)
                    prediction = _action_from_video_cache(
                        model,
                        action,
                        timestep,
                        prompt,
                        mask,
                        video_cache,
                        attention_mask,
                        video_seq_len,
                    )
                action = model.infer_action_scheduler.step(prediction, step_delta, action)
                if not bool(torch.isfinite(action).all()):
                    raise Thought6Error("Action Flow state contains NaN/Inf")
                diagnostic = dict(scope.diagnostic or {})
                adapter_calls += int(decision.adapter_called)
                rows.append(
                    {
                        "denoising_step_index": schedule_row.step_index,
                        "raw_scheduler_sigma_fp32": schedule_row.raw_sigma_fp32,
                        "scheduler_timestep_bf16": schedule_row.scheduler_timestep_bf16,
                        "effective_scheduler_sigma": schedule_row.effective_sigma,
                        "gate": decision.external_gate,
                        "adapter_called": decision.adapter_called,
                        "adapter_output_rms": diagnostic["adapter_output_rms"],
                        "pre_fusion_hidden_sha256": diagnostic["pre_fusion_hidden_sha256"],
                        "post_fusion_hidden_sha256": diagnostic["post_fusion_hidden_sha256"],
                        "action_state_pre_sha256": action_before,
                        "action_state_sha256": tensor_sha256(action),
                    }
                )
    result = action[0].detach().cpu().float().contiguous()
    expected_calls = {
        FusionMode.B0: 0,
        FusionMode.F0: 20,
        FusionMode.FSIGMA: sum(row.gate for row in schedule),
        FusionMode.LABEL_ORACLE: 20 if condition.lower() == "camera" else 0,
        FusionMode.LABEL_ORACLE_FSIGMA: (
            sum(row.gate for row in schedule) if condition.lower() == "camera" else 0
        ),
        FusionMode.SHUFFLE_FSIGMA: sum(row.gate for row in schedule),
    }[FusionMode(mode)]
    if adapter_calls != expected_calls:
        raise Thought6Error(
            f"Adapter call count mismatch: expected={expected_calls}, observed={adapter_calls}"
        )
    return result, {
        "schema_version": "thought6.online_action_trace.v1",
        "mode": FusionMode(mode).value,
        "condition": condition.lower(),
        "initial_action_sha256": initial_action_sha256,
        "final_action_sha256": tensor_sha256(result),
        "adapter_call_count": adapter_calls,
        "adapter_activation_ratio": adapter_calls / ACTION_DENOISE_STEPS,
        "action_denoise_steps": ACTION_DENOISE_STEPS,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "steps": rows,
    }


def mock_online_contract() -> dict[str, Any]:
    """CPU-only gate contract used by dry-run; it loads no Fast-WAM code."""

    from fastwam_ood_eval.thought6.sigma_gate import build_inference_sigma_schedule

    schedule = build_inference_sigma_schedule()
    counts = {
        mode.value: sum(
            decide_future_fusion(
                mode, condition="camera", effective_sigma=row.effective_sigma
            ).adapter_called
            for row in schedule
        )
        for mode in (FusionMode.B0, FusionMode.F0, FusionMode.FSIGMA)
    }
    return {
        "status": "validated",
        "scientific_result": False,
        "model_loaded": False,
        "optimizer_created": False,
        "counts": counts,
        "expected": {"B0": 0, "F0": 20, "Fsigma": 17},
        "passed": counts == {"B0": 0, "F0": 20, "Fsigma": 17},
    }


def _run_thought3_formal_null(
    model: Any,
    adapter: Any,
    *,
    current_latent: Any,
    context: Any,
    context_mask: Any,
    action_seed: int,
    action_horizon: int = 32,
) -> Any:
    """Exact Thought3 formal-null identity path used only by Phase 6A parity."""

    import torch
    from fastwam_ood_eval.thought3.injection import ActionEncoderFutureInjector
    from fastwam_ood_eval.thought3.phase_c_smoke import (
        _action_from_video_cache,
        _prepare_video_cache,
    )

    device = str(next(model.parameters()).device)
    dtype = model.torch_dtype
    current = current_latent.to(device=device, dtype=dtype)
    prompt = context.to(device=device, dtype=dtype)
    mask = context_mask.to(device=device, dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(int(action_seed))
    action = torch.randn(
        (1, action_horizon, 7), generator=generator, device="cpu", dtype=torch.float32
    ).to(device=device, dtype=dtype)
    with torch.inference_mode():
        cache, attention_mask, video_seq_len = _prepare_video_cache(
            model, current, prompt, mask, action_seq_len=action_horizon
        )
        timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=ACTION_DENOISE_STEPS,
            device=device,
            dtype=dtype,
            shift_override=None,
        )
        validate_runtime_schedule(timesteps, deltas)
    with ActionEncoderFutureInjector(model.action_expert.action_encoder, adapter) as injector:
        with torch.inference_mode(), injector.activate_null(
            expected_calls=ACTION_DENOISE_STEPS
        ):
            for step_t, step_delta in zip(timesteps, deltas, strict=True):
                prediction = _action_from_video_cache(
                    model,
                    action,
                    step_t.reshape(1).to(device=device, dtype=dtype),
                    prompt,
                    mask,
                    cache,
                    attention_mask,
                    video_seq_len,
                )
                action = model.infer_action_scheduler.step(prediction, step_delta, action)
    return action[0].detach().cpu().float().contiguous()


def _loader_config(cfg: Any) -> Any:
    from dataclasses import replace
    from fastwam_ood_eval.thought4.config import load_thought4_config

    source = load_thought4_config("configs/thought4/phase4_geometry_action_smoke_v8.yaml")
    return replace(
        source,
        experiment=replace(
            source.experiment,
            name="thought6_phase6a_model_loader",
            output_dir=cfg.output_dir / "runtime" / "model_loader",
            seed=cfg.seed,
        ),
        runtime=replace(source.runtime, device=cfg.device, action_denoise_steps=20),
        backbone=replace(
            source.backbone,
            checkpoint_path=cfg.backbone_checkpoint,
            checkpoint_sha256=cfg.backbone_checkpoint_sha256,
            dataset_stats_path=cfg.dataset_stats_path,
            fastwam_commit=cfg.fastwam_commit,
            frozen_parameter_sha256=cfg.backbone_frozen_parameter_sha256,
        ),
    )


def _phase6a_observations(cfg: Any, task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render two exact Clean/Camera states, then release MuJoCo before model load."""

    import numpy as np
    from fastwam_ood_eval.envs.libero_plus_adapter import LiberoPlusAdapter
    from fastwam_ood_eval.evaluation.jobs import EvaluationJob
    from fastwam_ood_eval.thought4.paired_rendering import (
        array_sha256,
        camera_metadata,
        get_simulator_state,
    )
    from fastwam_ood_eval.thought6.paired_noise import validate_camera_only_pair

    camera_variant = task["selected_camera_variant"]
    if not camera_variant:
        raise Thought6Error("Phase 6A task has no frozen Camera variant")
    adapters = {
        condition: LiberoPlusAdapter(
            image_size=(224, 224),
            root=__import__("pathlib").Path("third_party/LIBERO-plus"),
            config_dir=cfg.output_dir / "runtime" / "phase6a_libero_plus",
        )
        for condition in ("clean", "camera")
    }
    output: list[dict[str, Any]] = []
    try:
        for state_index in range(2):
            values: dict[str, Any] = {}
            for condition, adapter in adapters.items():
                upstream_task_id = (
                    int(task["task_id"])
                    if condition == "clean"
                    else int(camera_variant["classification_id"]) - 1
                )
                job = EvaluationJob(
                    experiment_id="thought6_phase6a",
                    job_id=f"thought6a-{condition}-{state_index}",
                    suite=str(task["suite"]),
                    task_id=int(task["task_id"]),
                    task_name=str(task["task_name"]),
                    upstream_task_id=upstream_task_id,
                    upstream_task_name=(
                        str(task["task_name"])
                        if condition == "clean"
                        else str(camera_variant["name"])
                    ),
                    episode_index=state_index,
                    episode_seed=cfg.seed + state_index,
                    initial_state_index=state_index,
                    condition=condition,
                    perturbation_category=("camera" if condition == "camera" else None),
                    perturbation_level=("phase6_frozen" if condition == "camera" else None),
                    perturbation_parameters={"phase6": True},
                    policy_variant="technical_smoke",
                    test_time_future_imagination=False,
                    comparison_group=f"thought6a-state-{state_index}",
                )
                observation = adapter.reset(job)
                state = get_simulator_state(adapter.env)
                camera = camera_metadata(
                    adapter.env, camera_name="agentview", height=224, width=224
                )
                values[condition] = {
                    "observation": {
                        key: np.asarray(value).copy()
                        for key, value in observation.items()
                    },
                    "state_sha256": array_sha256(state),
                    "camera_sha256": camera.identity_sha256,
                    "task_description": adapter.task_description,
                }
            validate_camera_only_pair(
                clean_physical_state_sha256=values["clean"]["state_sha256"],
                camera_physical_state_sha256=values["camera"]["state_sha256"],
                clean_camera_sha256=values["clean"]["camera_sha256"],
                camera_camera_sha256=values["camera"]["camera_sha256"],
            )
            output.append(
                {
                    "state_index": state_index,
                    "physical_state_sha256": values["clean"]["state_sha256"],
                    "clean": values["clean"],
                    "camera": values["camera"],
                }
            )
    finally:
        for adapter in adapters.values():
            adapter.close()
    return output


def _prepare_phase6a_sample(runtime: Any, observation: Mapping[str, Any], task: str) -> dict[str, Any]:
    import torch
    from fastwam_ood_eval.thought3.real_training import preprocess_current_action_target

    official = runtime.official
    image, _proprio, _images = official._obs_to_model_input(
        dict(observation),
        cfg=runtime.upstream_cfg,
        processor=runtime.processor,
        width=runtime.input_width,
        height=runtime.input_height,
        device=str(next(runtime.model.parameters()).device),
        dtype=runtime.model.torch_dtype,
    )
    raw_state = torch.from_numpy(official._extract_sim_state(dict(observation)))
    _dummy_action, proprio, _padding = preprocess_current_action_target(
        torch.zeros((32, 7), dtype=torch.float32),
        raw_state,
        torch.zeros(32, dtype=torch.bool),
        processor=runtime.processor,
    )
    with torch.inference_mode():
        context, context_mask = runtime.model.encode_prompt(
            official.DEFAULT_PROMPT.format(task=task)
        )
        context, context_mask = runtime.model._append_proprio_to_context(
            context,
            context_mask,
            proprio.to(
                device=str(next(runtime.model.parameters()).device),
                dtype=runtime.model.torch_dtype,
            ),
        )
        current = runtime.model._encode_input_image_latents_tensor(image)
    return {
        "current_latent": current.detach().cpu(),
        "context": context.detach().cpu(),
        "context_mask": context_mask.detach().cpu(),
    }


def run_phase6a_smoke(cfg: Any) -> dict[str, Any]:
    """Confirmed real one-card technical smoke; never produces a paper result."""

    import json
    import torch
    from fastwam_ood_eval.thought3.future_sampler import VideoOnlyFutureSampler
    from fastwam_ood_eval.thought4.real_runtime import load_frozen_fastwam, release_fastwam
    from fastwam_ood_eval.thought5.checkpointing import frozen_parameter_sha256
    from fastwam_ood_eval.thought6.checkpoint_resolver import load_frozen_adapter
    from fastwam_ood_eval.thought6.schemas import file_sha256, seal_full_object, write_stage_json

    if cfg.stage != "phase6a" or cfg.device != "cuda:0":
        raise Thought6Error("real Phase 6A requires the frozen logical cuda:0 config")
    output = cfg.output_dir
    manifest_path = output / "task_selection_manifest.json"
    if not manifest_path.is_file():
        raise Thought6Error("run the Phase 6 audit before Phase 6A")
    prior = json.loads((output / "phase6a_smoke_results.json").read_text(encoding="utf-8"))
    if prior.get("status") == "complete":
        raise Thought6Error("completed Phase 6A artifact is immutable")
    tasks = json.loads(manifest_path.read_text(encoding="utf-8"))["selected_tasks"]
    task = next(row for row in tasks if row["suite"] == "libero_goal")
    print(json.dumps({"phase": "6A", "stage": "paired_render_started", "task": task["canonical_id"]}), flush=True)
    observations = _phase6a_observations(cfg, task)
    print(json.dumps({"phase": "6A", "stage": "model_load_started"}), flush=True)
    runtime = None
    try:
        runtime = load_frozen_fastwam(_loader_config(cfg))
        adapter, adapter_manifest = load_frozen_adapter(cfg, device=cfg.device)
        freeze_for_phase6(runtime.model, adapter)
        backbone_before = frozen_parameter_sha256(runtime.model.named_parameters())
        if backbone_before != cfg.backbone_frozen_parameter_sha256:
            raise Thought6Error("loaded backbone parameter SHA differs")

        class Velocity:
            def __init__(self, model: Any) -> None:
                self.model = model
                self.calls = 0

            def __call__(self, state: Any, timestep: Any, conditions: Mapping[str, object]) -> Any:
                self.calls += 1
                return self.model.video_expert(
                    x=state,
                    timestep=timestep.to(device=state.device, dtype=state.dtype),
                    context=conditions["context"],
                    context_mask=conditions["context_mask"],
                    action=None,
                    fuse_vae_embedding_in_latents=True,
                )

        velocity = Velocity(runtime.model)
        sampler = VideoOnlyFutureSampler(velocity, shift=5.0, num_train_timesteps=1000)
        prepared = [
            _prepare_phase6a_sample(runtime, row["camera"]["observation"], str(task["task_name"]))
            for row in observations
        ]
        futures = []
        for index, sample in enumerate(prepared):
            with torch.inference_mode():
                future = sampler.sample(
                    sample["current_latent"].to(cfg.device, dtype=runtime.model.torch_dtype),
                    initial_noise_seeds=(cfg.seed + 7000 + index,),
                    k=1,
                    conditions={
                        "context": sample["context"].to(cfg.device, dtype=runtime.model.torch_dtype),
                        "context_mask": sample["context_mask"].to(cfg.device, dtype=torch.bool),
                    },
                ).future_latent.detach().cpu()
            futures.append(future)
        if velocity.calls != 2:
            raise Thought6Error("Phase 6A K=1 sampler call count differs")
        sample_results = []
        all_checks: list[bool] = []
        for index, sample in enumerate(prepared):
            common = {
                "current_latent": sample["current_latent"],
                "context": sample["context"],
                "context_mask": sample["context_mask"],
                "condition": "camera",
                "action_seed": cfg.seed + 8000 + index,
            }
            b0, b0_trace = run_action_denoising(runtime.model, adapter, mode="B0", **common)
            formal = _run_thought3_formal_null(
                runtime.model,
                adapter,
                current_latent=sample["current_latent"],
                context=sample["context"],
                context_mask=sample["context_mask"],
                action_seed=common["action_seed"],
            )
            f0, f0_trace = run_action_denoising(
                runtime.model,
                adapter,
                mode="F0",
                correct_future=futures[index],
                **common,
            )
            f0_shuffle, f0_shuffle_trace = run_action_denoising(
                runtime.model,
                adapter,
                mode="F0",
                shuffled_future=futures[1 - index],
                payload_override="shuffle",
                **common,
            )
            fsigma, fsigma_trace = run_action_denoising(
                runtime.model,
                adapter,
                mode="Fsigma",
                correct_future=futures[index],
                **common,
            )
            fsigma_shuffle, fsigma_shuffle_trace = run_action_denoising(
                runtime.model,
                adapter,
                mode="Shuffle+Fsigma",
                shuffled_future=futures[1 - index],
                **common,
            )
            high = [row for row in fsigma_trace["steps"] if row["gate"] == 1]
            low = [row for row in fsigma_trace["steps"] if row["gate"] == 0]
            checks = {
                "b0_formal_null_bitwise": bool(torch.equal(b0, formal)),
                "b0_adapter_calls_zero": b0_trace["adapter_call_count"] == 0,
                "f0_adapter_calls_20": f0_trace["adapter_call_count"] == 20,
                "fsigma_adapter_calls_17": fsigma_trace["adapter_call_count"] == 17,
                "fsigma_low_sigma_exact_identity": all(
                    row["adapter_output_rms"] == 0.0
                    and row["pre_fusion_hidden_sha256"] == row["post_fusion_hidden_sha256"]
                    for row in low
                ),
                "fsigma_high_matches_f0": all(
                    left["post_fusion_hidden_sha256"] == right["post_fusion_hidden_sha256"]
                    for left, right in zip(high, f0_trace["steps"][: len(high)], strict=True)
                ),
                "paired_initial_action_noise": len(
                    {
                        trace["initial_action_sha256"]
                        for trace in (
                            b0_trace,
                            f0_trace,
                            f0_shuffle_trace,
                            fsigma_trace,
                            fsigma_shuffle_trace,
                        )
                    }
                )
                == 1,
                "all_outputs_finite": all(
                    bool(torch.isfinite(value).all())
                    for value in (b0, formal, f0, f0_shuffle, fsigma, fsigma_shuffle)
                ),
            }
            all_checks.extend(checks.values())
            sample_results.append(
                {
                    "state_index": index,
                    "physical_state_sha256": observations[index]["physical_state_sha256"],
                    "checks": checks,
                    "actions": {
                        "B0": tensor_sha256(b0),
                        "formal_null": tensor_sha256(formal),
                        "F0_correct": tensor_sha256(f0),
                        "F0_shuffle": tensor_sha256(f0_shuffle),
                        "Fsigma_correct": tensor_sha256(fsigma),
                        "Fsigma_shuffle": tensor_sha256(fsigma_shuffle),
                    },
                    "traces": {
                        "B0": b0_trace,
                        "F0": f0_trace,
                        "Fsigma": fsigma_trace,
                    },
                }
            )
        backbone_after = frozen_parameter_sha256(runtime.model.named_parameters())
        adapter_after = file_sha256(cfg.adapter_checkpoint_path / "adapter.safetensors")
        integrity = {
            "backbone_sha_before": backbone_before,
            "backbone_sha_after": backbone_after,
            "adapter_file_sha_before": adapter_manifest["adapter_file_sha256"],
            "adapter_file_sha_after": adapter_after,
            "all_parameters_frozen": not any(
                parameter.requires_grad or parameter.grad is not None
                for parameter in list(runtime.model.parameters()) + list(adapter.parameters())
            ),
            "optimizer_created": False,
        }
        all_checks.extend(
            [
                backbone_before == backbone_after == cfg.backbone_frozen_parameter_sha256,
                adapter_after == cfg.adapter_file_sha256,
                integrity["all_parameters_frozen"],
            ]
        )
        passed = all(all_checks)
        result = seal_full_object(
            {
                "schema_version": "thought6.phase6a.smoke_result.v1",
                "status": "complete" if passed else "failed",
                "scientific_result": False,
                "task": task["canonical_id"],
                "states": 2,
                "samples": sample_results,
                "execution_integrity": integrity,
                "contract_checks_passed": passed,
                "phase6b_unlocked": passed,
            }
        )
        write_stage_json(output / "phase6a_smoke_results.json", result)
        print(json.dumps({"phase": "6A", "stage": "smoke_complete", "passed": passed}), flush=True)
        return result
    finally:
        if runtime is not None:
            release_fastwam(runtime)
