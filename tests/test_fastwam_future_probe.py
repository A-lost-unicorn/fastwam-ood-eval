from __future__ import annotations

import copy
import random
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import fastwam_ood_eval.policy.fastwam_future_probe as probe_module
from fastwam_ood_eval.diagnostics.protocol import FutureProbeOutput, SupportsFutureProbe
from fastwam_ood_eval.policy.fastwam_future_probe import FastWAMFutureProbe


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


class _DType:
    def __init__(self, name: str, *, is_floating_point: bool):
        self.name = name
        self.is_floating_point = is_floating_point

    def __str__(self):
        return self.name


class _State:
    def __init__(self, value):
        self.value = copy.deepcopy(value)

    def clone(self):
        return _State(self.value)


class _Column:
    def __init__(self, values):
        self.values = list(values)

    def __rsub__(self, value):
        return _Column([value - item for item in self.values])

    def __truediv__(self, value):
        return _Column([item / value for item in self.values])


class _FakeTensor:
    def __init__(self, data, *, dtype, device="cpu"):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = device

    @property
    def shape(self):
        if not isinstance(self.data, list):
            return ()
        if not self.data:
            return (0,)
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def clone(self):
        return _FakeTensor(self.data, dtype=self.dtype, device=self.device)

    def detach(self):
        return self

    def cpu(self):
        return self.to(device="cpu")

    def to(self, *, device=None, dtype=None):
        return _FakeTensor(
            self.data,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )

    def tolist(self):
        return copy.deepcopy(self.data)

    def __getitem__(self, key):
        if key == (Ellipsis, -1):
            return _Column([row[-1] for row in self.data])
        return self.data[key]

    def __setitem__(self, key, value):
        if key != (Ellipsis, -1):
            raise AssertionError(f"Unexpected fake tensor assignment {key!r}")
        values = value.values if isinstance(value, _Column) else value
        for row, item in zip(self.data, values):
            row[-1] = item


class _FakeBoolTensor:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value


class _FakeCuda:
    def is_available(self):
        return False


class _TrackingCuda:
    def __init__(self, peak_readings_mb, *, supports_reset=True):
        self._peak_readings = [int(value * 2**20) for value in peak_readings_mb]
        self._last_peak = self._peak_readings[-1]
        self.reset_calls = []
        self.synchronize_calls = []
        if not supports_reset:
            self.reset_peak_memory_stats = None

    def is_available(self):
        return True

    def synchronize(self, device):
        self.synchronize_calls.append(device)

    def max_memory_allocated(self, device):
        if self._peak_readings:
            self._last_peak = self._peak_readings.pop(0)
        return self._last_peak

    def reset_peak_memory_stats(self, device):
        self.reset_calls.append(device)


class _FakeTorch:
    def __init__(self):
        self.float32 = _DType("torch.float32", is_floating_point=True)
        self.float64 = _DType("torch.float64", is_floating_point=True)
        self.int64 = _DType("torch.int64", is_floating_point=False)
        self.rng_state = _State("torch-initial")
        self.cuda = _FakeCuda()

    def as_tensor(self, value, *, device=None, dtype=None):
        if isinstance(value, _FakeTensor):
            tensor = value.clone()
            if device is not None or dtype is not None:
                tensor = tensor.to(device=device, dtype=dtype)
            return tensor
        flat = [item for row in value for item in row]
        inferred = self.float64 if any(isinstance(item, float) for item in flat) else self.int64
        return _FakeTensor(value, dtype=inferred if dtype is None else dtype, device=device or "cpu")

    def is_floating_point(self, tensor):
        return bool(tensor.dtype.is_floating_point)

    def isfinite(self, tensor):
        return _FakeBoolTensor(True)

    def get_rng_state(self):
        return self.rng_state.clone()

    def set_rng_state(self, state):
        self.rng_state = state.clone()

    def manual_seed(self, seed):
        self.rng_state = _State(("seeded", seed))

    def inference_mode(self):
        return nullcontext()


class _FakeNumpyRandom:
    def __init__(self):
        self.state = ("numpy-initial",)

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        self.state = copy.deepcopy(state)

    def seed(self, seed):
        self.state = ("seeded", seed)


class _FakeNumpy:
    def __init__(self):
        self.random = _FakeNumpyRandom()
        self.stack_calls = []

    def stack(self, values, axis=0):
        self.stack_calls.append((list(values), axis))
        return ("stacked", tuple(values), axis)


class _ImageTensor:
    shape = (1, 3, 2, 4)
    dtype = "bfloat16"


class _CloneBackedObservationTensor:
    def __init__(self, values):
        self.values = list(values)

    def clone(self):
        return _CloneBackedObservationTensor(self.values)


class _CopyBackedObservationArray:
    def __init__(self, values):
        self.values = list(values)

    def copy(self):
        return _CopyBackedObservationArray(self.values)


class _RecordingNormalizer:
    def __init__(self):
        self.seen = None

    def forward(self, tensor):
        self.seen = tensor.clone()
        result = tensor.clone()
        result.data = [[item * 2.0 for item in row] for row in result.data]
        return result


class _FakeOfficial:
    DEFAULT_PROMPT = "Move objects: {task}"

    def __init__(self, *, num_frames=5, mutate_observation=False):
        self.num_frames = num_frames
        self.mutate_observation = mutate_observation
        self.obs_calls = []

    def _get_num_video_frames(self, cfg):
        return self.num_frames

    def _obs_to_model_input(
        self,
        observation,
        *,
        cfg,
        processor,
        width,
        height,
        device,
        dtype,
    ):
        self.obs_calls.append(
            {
                "observation": observation,
                "cfg": cfg,
                "processor": processor,
                "width": width,
                "height": height,
                "device": device,
                "dtype": dtype,
            }
        )
        if self.mutate_observation:
            observation["robot0_eef_quat"].values[0] = 99.0
            observation["tensor_value"].values[0] = 88.0
            observation["nested"]["values"][0] = 77.0
        return (
            _ImageTensor(),
            "official-proprio",
            {"image": "agent-camera", "wrist_image": "wrist-camera"},
        )


class _FakeModel:
    torch_dtype = "bfloat16"

    def __init__(
        self,
        *,
        action_conditioned=True,
        action_dim=3,
        returned_frame_count=5,
        training=False,
        temporal_downsample_factor=4,
        temporal_patch_size=1,
        video_attention_mask_mode="per_frame_causal",
    ):
        self.video_expert = SimpleNamespace(
            action_conditioned=action_conditioned,
            patch_size=(temporal_patch_size, 2, 2),
            video_attention_mask_mode=video_attention_mask_mode,
        )
        self.action_expert = SimpleNamespace(action_dim=action_dim)
        self.vae = SimpleNamespace(temporal_downsample_factor=temporal_downsample_factor)
        self.returned_frame_count = returned_frame_count
        self.training = training
        self.infer_calls = []

    def infer_joint(
        self,
        *,
        prompt,
        input_image,
        num_video_frames,
        action_horizon,
        action,
        proprio,
        negative_prompt,
        text_cfg_scale,
        num_inference_steps,
        sigma_shift,
        seed,
        rand_device,
        tiled,
        test_action_with_infer_action,
    ):
        call = dict(locals())
        call.pop("self")
        self.infer_calls.append(call)
        return {
            "video": [f"decoded-{index}" for index in range(self.returned_frame_count)],
            "action": "must-be-discarded",
        }


def _make_adapter(
    *,
    action_conditioned=True,
    action_state_transforms=None,
    num_frames=5,
    returned_frame_count=5,
    action_video_freq_ratio=4,
    model_training=False,
    action_horizon=4,
    control_horizon=4,
    temporal_downsample_factor=4,
    temporal_patch_size=1,
    video_attention_mask_mode="per_frame_causal",
    mutate_observation=False,
):
    torch = _FakeTorch()
    normalizer = _RecordingNormalizer()
    processor = SimpleNamespace(
        shape_meta={
            "images": [{"shape": [3, 2, 2]}, {"shape": [3, 2, 2]}],
            "action": [{"shape": 3, "key": "action.main"}],
        },
        num_output_cameras=2,
        action_output_dim=3,
        action_state_transforms=action_state_transforms,
        normalizer=SimpleNamespace(
            normalizers={"action": {"action.main": normalizer}},
        ),
    )
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            train=_AttrDict(
                concat_multi_camera="horizontal",
                action_video_freq_ratio=action_video_freq_ratio,
            )
        ),
        EVALUATION=_AttrDict(
            negative_prompt="avoid errors",
            text_cfg_scale=3.5,
            sigma_shift=None,
            rand_device="cpu",
            tiled=False,
        ),
    )
    return SimpleNamespace(
        model=_FakeModel(
            action_conditioned=action_conditioned,
            returned_frame_count=returned_frame_count,
            training=model_training,
            temporal_downsample_factor=temporal_downsample_factor,
            temporal_patch_size=temporal_patch_size,
            video_attention_mask_mode=video_attention_mask_mode,
        ),
        processor=processor,
        official=_FakeOfficial(
            num_frames=num_frames,
            mutate_observation=mutate_observation,
        ),
        upstream_cfg=cfg,
        device="cuda:0",
        torch=torch,
        input_h=2,
        input_w=4,
        action_horizon=action_horizon,
        task_description="put the cup down",
        test_normalizer=normalizer,
        cfg=SimpleNamespace(
            benchmark=SimpleNamespace(control_horizon=control_horizon, backend="mock"),
        ),
    )


def _model_frame_converter(image, *, expected_height, expected_width, numpy_module):
    if image.shape != (1, 3, expected_height, expected_width):
        raise AssertionError("probe did not pass official model geometry")
    return "official-model-space-input"


def _decoded_frame_converter(frame, *, expected_height, expected_width, numpy_module):
    if frame == "decoded-bad":
        raise ValueError("Decoded frame values must be in [0,255]")
    return ("rgb-uint8", frame, expected_height, expected_width)


class FastWAMFutureProbeTests(unittest.TestCase):
    def setUp(self):
        self.fake_numpy = _FakeNumpy()
        self.patches = [
            patch.object(probe_module, "require_numpy", return_value=self.fake_numpy),
            patch.object(probe_module, "model_tensor_to_rgb_uint8", side_effect=_model_frame_converter),
            patch.object(probe_module, "frame_to_rgb_uint8", side_effect=_decoded_frame_converter),
        ]
        for active_patch in self.patches:
            active_patch.start()
            self.addCleanup(active_patch.stop)

    def _probe(self, adapter):
        return FastWAMFutureProbe(
            adapter,
            numpy_module=self.fake_numpy,
            _mock_checkpoint_verifier=lambda _: {
                "action_conditioning_parameters_loaded_verified": True,
                "action_conditioned_training_provenance_verified": True,
                "verification_scope": "cpu_mock_only",
            },
        )

    def test_action_conditioned_runtime_flag_without_approved_checkpoint_is_rejected(self):
        adapter = _make_adapter()

        with self.assertRaisesRegex(RuntimeError, "No action-conditioned Fast-WAM checkpoint"):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

    def test_constructor_rejects_unconditional_libero_checkpoint_before_inference(self):
        adapter = _make_adapter(
            action_conditioned=False,
            num_frames=9,
            returned_frame_count=9,
            action_horizon=32,
            control_horizon=10,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"libero_uncond.*action_conditioned=false.*unconditional future",
        ):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

        self.assertEqual(adapter.official.obs_calls, [])
        self.assertEqual(adapter.model.infer_calls, [])

    def test_unconditional_mode_uses_release_video_path_without_action_condition(self):
        adapter = _make_adapter(
            action_conditioned=False,
            num_frames=9,
            returned_frame_count=9,
            action_horizon=32,
            control_horizon=10,
        )
        probe = FastWAMFutureProbe(
            adapter,
            mode="unconditional_future",
            numpy_module=self.fake_numpy,
        )
        actions = [[0.0, 0.0, 1.0]] * 32
        original_actions = copy.deepcopy(actions)

        output = probe.predict_unconditional_future(
            {"pixels": "observation"},
            actions,
            diagnostic_seed=17,
            num_video_frames=9,
            num_inference_steps=2,
        )

        self.assertEqual(actions, original_actions)
        self.assertIsNone(adapter.model.infer_calls[0]["action"])
        self.assertFalse(output.metadata["action_conditioned"])
        self.assertFalse(
            output.metadata["protected_policy_action_used_as_video_condition"]
        )
        self.assertEqual(output.metadata["future_kind"], "unconditional")
        self.assertEqual(
            output.metadata["action_dependency_scope"],
            "not_applicable_unconditional",
        )
        self.assertTrue(
            probe.checkpoint_verification[
                "unconditional_video_architecture_verified"
            ]
        )

    def test_unconditional_mode_rejects_action_conditioned_video_expert(self):
        adapter = _make_adapter(action_conditioned=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"unconditional_future.*action_conditioned=false",
        ):
            FastWAMFutureProbe(
                adapter,
                mode="unconditional_future",
                numpy_module=self.fake_numpy,
            )

    def test_constructor_rejects_non_null_action_state_transforms(self):
        adapter = _make_adapter(action_state_transforms=["unsupported-transform"])

        with self.assertRaisesRegex(
            RuntimeError,
            r"action_state_transforms is None.*cannot be skipped safely",
        ):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

        self.assertEqual(adapter.official.obs_calls, [])
        self.assertEqual(adapter.model.infer_calls, [])

    def test_constructor_rejects_training_model_before_infer_joint_can_call_eval(self):
        adapter = _make_adapter(model_training=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"model.training=false.*persistently change",
        ):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

        self.assertTrue(adapter.model.training)
        self.assertEqual(adapter.official.obs_calls, [])
        self.assertEqual(adapter.model.infer_calls, [])

    def test_probe_uses_official_preprocessing_and_private_normalized_action_copy(self):
        adapter = _make_adapter()
        probe = self._probe(adapter)
        actions = [
            [0.1, 0.2, 1.0],
            [0.3, 0.4, -1.0],
            [0.5, 0.6, 0.0],
            [0.7, 0.8, 0.5],
        ]
        original_actions = copy.deepcopy(actions)

        output = probe.predict_action_conditioned_future(
            {"pixels": "observation"},
            actions,
            diagnostic_seed=123,
            num_video_frames=5,
            num_inference_steps=4,
        )

        self.assertIsInstance(output, FutureProbeOutput)
        self.assertIsInstance(probe, SupportsFutureProbe)
        self.assertEqual(probe.action_video_freq_ratio, 4)
        self.assertEqual(probe.vae_temporal_downsample_factor, 4)
        self.assertEqual(probe.action_conditioning_group_count, 1)
        self.assertEqual(probe.action_conditioning_group_size, 4)
        self.assertEqual(actions, original_actions)
        self.assertEqual(
            adapter.test_normalizer.seen.tolist(),
            [
                [0.1, 0.2, 0.0],
                [0.3, 0.4, 1.0],
                [0.5, 0.6, 0.5],
                [0.7, 0.8, 0.25],
            ],
        )
        call = adapter.model.infer_calls[0]
        self.assertEqual(call["action"].tolist(), [[item * 2.0 for item in row] for row in adapter.test_normalizer.seen.tolist()])
        self.assertIsNot(call["action"], adapter.test_normalizer.seen)
        self.assertEqual(call["prompt"], "Move objects: put the cup down")
        self.assertEqual(call["seed"], 123)
        self.assertEqual(call["num_video_frames"], 5)
        self.assertEqual(call["action_horizon"], 4)
        self.assertEqual(call["proprio"], "official-proprio")
        self.assertFalse(call["test_action_with_infer_action"])
        self.assertEqual(adapter.official.obs_calls[0]["width"], 4)
        self.assertEqual(adapter.official.obs_calls[0]["height"], 2)
        self.assertEqual(output.predicted_frames, [f"decoded-{index}" for index in range(5)])
        self.assertEqual(output.model_space_input, "official-model-space-input")
        self.assertEqual(output.model_space_predicted_frames[0], "stacked")
        self.assertIsNone(output.predicted_latents)
        self.assertTrue(output.metadata["returned_action_discarded"])
        self.assertEqual(output.metadata["frame_control_offsets"], [0, 4, 8, 12, 16])
        self.assertEqual(output.metadata["action_video_freq_ratio"], 4)
        self.assertEqual(output.metadata["vae_temporal_downsample_factor"], 4)
        self.assertEqual(output.metadata["action_conditioning_group_count"], 1)
        self.assertEqual(output.metadata["action_conditioning_group_size"], 4)
        self.assertEqual(output.metadata["future_kind"], "action_conditioned")
        self.assertGreaterEqual(output.latency_ms, 0.0)

    def test_non_positive_action_video_frequency_ratio_is_rejected_at_construction(self):
        adapter = _make_adapter(action_video_freq_ratio=0)

        with self.assertRaisesRegex(RuntimeError, "action_video_freq_ratio must be a positive"):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

        self.assertEqual(adapter.official.obs_calls, [])
        self.assertEqual(adapter.model.infer_calls, [])

    def test_integer_input_dtype_is_rejected_before_normalization_or_inference(self):
        adapter = _make_adapter()
        probe = self._probe(adapter)
        integer_actions = _FakeTensor(
            [[0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, -1]],
            dtype=adapter.torch.int64,
        )

        with self.assertRaisesRegex(TypeError, "floating-point input dtype"):
            probe.predict_action_conditioned_future(
                {},
                integer_actions,
                diagnostic_seed=0,
                num_video_frames=5,
                num_inference_steps=1,
            )

        self.assertIsNone(adapter.test_normalizer.seen)
        self.assertEqual(adapter.official.obs_calls, [])
        self.assertEqual(adapter.model.infer_calls, [])

    def test_action_shape_and_training_derived_frame_count_are_strict(self):
        adapter = _make_adapter()
        probe = self._probe(adapter)

        with self.assertRaisesRegex(ValueError, "Executable action shape"):
            probe.predict_action_conditioned_future(
                {},
                [[0.0, 0.0, 1.0]],
                diagnostic_seed=0,
                num_video_frames=5,
                num_inference_steps=1,
            )
        with self.assertRaisesRegex(ValueError, "training-derived"):
            probe.predict_action_conditioned_future(
                {},
                [[0.0, 0.0, 1.0]] * 4,
                diagnostic_seed=0,
                num_video_frames=9,
                num_inference_steps=1,
            )

    def test_invalid_temporal_length_and_wrong_output_count_are_rejected(self):
        bad_temporal_adapter = _make_adapter(num_frames=3, returned_frame_count=3)
        with self.assertRaisesRegex(RuntimeError, "cannot be aligned"):
            FastWAMFutureProbe(bad_temporal_adapter, numpy_module=self.fake_numpy)

        short_adapter = _make_adapter(returned_frame_count=4)
        short_probe = self._probe(short_adapter)
        with self.assertRaisesRegex(RuntimeError, "returned 4 frames, expected 5"):
            short_probe.predict_action_conditioned_future(
                {},
                [[0.0, 0.0, 1.0]] * 4,
                diagnostic_seed=0,
                num_video_frames=5,
                num_inference_steps=1,
            )

    def test_release_style_grouping_rejects_replan_before_first_group_is_executed(self):
        adapter = _make_adapter(
            num_frames=9,
            returned_frame_count=9,
            action_horizon=32,
            control_horizon=10,
            temporal_downsample_factor=4,
            temporal_patch_size=1,
            video_attention_mask_mode="first_frame_causal",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"control_horizon=10.*first future frame=32.*first_frame_causal.*never.*executed",
        ):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

        self.assertEqual(adapter.official.obs_calls, [])
        self.assertEqual(adapter.model.infer_calls, [])

    def test_release_style_group_properties_and_metadata_are_derived_not_guessed(self):
        adapter = _make_adapter(
            num_frames=9,
            returned_frame_count=9,
            action_horizon=32,
            control_horizon=32,
            temporal_downsample_factor=4,
            temporal_patch_size=1,
            video_attention_mask_mode="first_frame_causal",
        )
        probe = self._probe(adapter)

        self.assertEqual(probe.vae_temporal_downsample_factor, 4)
        self.assertEqual(probe.action_conditioning_latent_frame_count, 3)
        self.assertEqual(probe.action_conditioning_group_count, 2)
        self.assertEqual(probe.action_conditioning_group_size, 16)
        self.assertEqual(probe.video_attention_mask_mode, "first_frame_causal")
        self.assertEqual(probe.required_executed_actions_for_first_future, 32)
        output = probe.predict_action_conditioned_future(
            {},
            [[0.0, 0.0, 1.0]] * 32,
            diagnostic_seed=0,
            num_video_frames=9,
            num_inference_steps=1,
        )
        self.assertEqual(output.metadata["vae_temporal_downsample_factor"], 4)
        self.assertEqual(output.metadata["action_conditioning_latent_frame_count"], 3)
        self.assertEqual(output.metadata["action_conditioning_group_count"], 2)
        self.assertEqual(output.metadata["action_conditioning_group_size"], 16)
        self.assertEqual(output.metadata["control_horizon"], 32)
        self.assertEqual(output.metadata["action_dependency_scope"], "all_future_groups")
        self.assertEqual(output.metadata["required_executed_actions_for_first_future"], 32)

    def test_per_frame_causal_only_requires_the_first_direct_group(self):
        adapter = _make_adapter(
            num_frames=9,
            returned_frame_count=9,
            action_horizon=32,
            control_horizon=16,
            video_attention_mask_mode="per_frame_causal",
        )
        probe = self._probe(adapter)

        self.assertEqual(probe.action_conditioning_group_size, 16)
        self.assertEqual(probe.required_executed_actions_for_first_future, 16)
        self.assertEqual(probe.action_dependency_scope, "causal_prefix")

    def test_temporal_patch_larger_than_one_is_rejected_as_unproven(self):
        adapter = _make_adapter(temporal_patch_size=2)

        with self.assertRaisesRegex(RuntimeError, "temporal DiT patch size=1"):
            FastWAMFutureProbe(adapter, numpy_module=self.fake_numpy)

    def test_public_model_frame_and_approximate_embedding_helpers_reuse_model(self):
        adapter = _make_adapter()
        probe = self._probe(adapter)

        model_frame = probe.observation_to_model_frame({"pixels": "observation"})
        self.assertEqual(model_frame, "official-model-space-input")

        sentinel = object()
        with patch.object(
            probe_module,
            "encode_frames_independently_with_first_frame_vae",
            return_value=sentinel,
        ) as encoder:
            actual = probe.encode_frame_embeddings(["frame-0", "frame-1"])

        self.assertIs(actual, sentinel)
        encoder.assert_called_once_with(
            ["frame-0", "frame-1"],
            model=adapter.model,
            torch_module=adapter.torch,
            device="cuda:0",
            dtype="bfloat16",
            expected_height=2,
            expected_width=4,
            tiled=False,
            numpy_module=self.fake_numpy,
        )

    def test_official_preprocessing_cannot_mutate_control_observation_storage(self):
        adapter = _make_adapter(mutate_observation=True)
        probe = self._probe(adapter)
        observation = {
            "robot0_eef_quat": _CopyBackedObservationArray([1.25, 0.0, 0.0, 0.0]),
            "tensor_value": _CloneBackedObservationTensor([2.5, 3.5]),
            "nested": {"values": [4.5, 5.5]},
        }

        frame = probe.observation_to_model_frame(observation)

        self.assertEqual(frame, "official-model-space-input")
        self.assertEqual(observation["robot0_eef_quat"].values, [1.25, 0.0, 0.0, 0.0])
        self.assertEqual(observation["tensor_value"].values, [2.5, 3.5])
        self.assertEqual(observation["nested"]["values"], [4.5, 5.5])
        shadow = adapter.official.obs_calls[0]["observation"]
        self.assertIsNot(shadow, observation)
        self.assertIsInstance(shadow["robot0_eef_quat"], _CopyBackedObservationArray)
        self.assertIsInstance(shadow["tensor_value"], _CloneBackedObservationTensor)
        self.assertIsNot(shadow["robot0_eef_quat"], observation["robot0_eef_quat"])
        self.assertIsNot(shadow["tensor_value"], observation["tensor_value"])
        self.assertEqual(shadow["robot0_eef_quat"].values[0], 99.0)
        self.assertEqual(shadow["tensor_value"].values[0], 88.0)
        self.assertEqual(shadow["nested"]["values"][0], 77.0)

    def test_gpu_peak_is_reset_immediately_before_probe_when_supported(self):
        adapter = _make_adapter()
        adapter.torch.cuda = _TrackingCuda([900.0, 300.0, 700.0])
        probe = self._probe(adapter)

        output = probe.predict_action_conditioned_future(
            {},
            [[0.0, 0.0, 1.0]] * 4,
            diagnostic_seed=0,
            num_video_frames=5,
            num_inference_steps=1,
        )

        self.assertEqual(adapter.torch.cuda.reset_calls, ["cuda:0"])
        self.assertEqual(adapter.torch.cuda.synchronize_calls, ["cuda:0", "cuda:0"])
        self.assertEqual(output.gpu_peak_memory_mb, 700.0)
        self.assertTrue(output.metadata["gpu_peak_memory_reset_before_probe"])
        self.assertEqual(output.metadata["gpu_peak_memory_cumulative_before_reset_mb"], 900.0)
        self.assertEqual(output.metadata["gpu_peak_memory_before_mb"], 300.0)
        self.assertEqual(output.metadata["gpu_peak_memory_delta_mb"], 400.0)
        self.assertIn("reset immediately before infer_joint", output.metadata["gpu_peak_memory_semantics"])

    def test_gpu_peak_keeps_explicit_cumulative_semantics_without_reset_api(self):
        adapter = _make_adapter()
        adapter.torch.cuda = _TrackingCuda([900.0, 900.0, 950.0], supports_reset=False)
        probe = self._probe(adapter)

        output = probe.predict_action_conditioned_future(
            {},
            [[0.0, 0.0, 1.0]] * 4,
            diagnostic_seed=0,
            num_video_frames=5,
            num_inference_steps=1,
        )

        self.assertFalse(output.metadata["gpu_peak_memory_reset_before_probe"])
        self.assertEqual(output.gpu_peak_memory_mb, 950.0)
        self.assertIn("cumulative", output.metadata["gpu_peak_memory_semantics"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
