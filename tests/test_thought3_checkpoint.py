from __future__ import annotations

import torch

from fastwam_ood_eval.thought3.adapter import (
    FutureAdapterSpec,
    FutureToActionAdapter,
)
from fastwam_ood_eval.thought3.checkpointing import (
    find_latest_checkpoint,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)
from fastwam_ood_eval.thought3.schemas import AdapterCheckpointManifest


HASH = "a" * 64


def _adapter():
    return FutureToActionAdapter(
        FutureAdapterSpec(
            input_channels=4,
            action_hidden_dim=16,
            future_dim=8,
            attention_dim=16,
            num_heads=4,
            max_projected_grid=(2, 4, 4),
        )
    )


def _manifest(adapter, step=3):
    return AdapterCheckpointManifest(
        backbone_checkpoint_sha256=HASH,
        dataset_stats_sha256=HASH,
        fastwam_commit="b" * 40,
        adapter_fingerprint=adapter.spec.fingerprint,
        config_fingerprint=HASH,
        split_fingerprint=HASH,
        cache_fingerprint=HASH,
        variant="A1",
        k=1,
        train_seed=7,
        global_step=step,
        epoch=0,
        sample_cursor=6,
        trainable_parameter_count=adapter.trainable_parameter_count,
        trainable_parameter_names=tuple(
            f"adapter.{name}"
            for name, parameter in adapter.named_parameters()
            if parameter.requires_grad
        ),
        frozen_parameter_sha256=HASH,
        world_size=1,
    )


def test_adapter_only_checkpoint_round_trip_preserves_output(tmp_path):
    torch.manual_seed(19)
    adapter = _adapter()
    with torch.no_grad():
        adapter.gate.fill_(0.4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    action = torch.randn(2, 5, 16)
    future = torch.randn(2, 4, 2, 8, 8)
    expected = adapter(action, future).detach()
    root = tmp_path / "thought3" / "checkpoints"
    checkpoint = save_adapter_checkpoint(
        root / "step_000003",
        adapter=adapter,
        manifest=_manifest(adapter),
        optimizer=optimizer,
    )
    restored = _adapter()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    loaded = load_adapter_checkpoint(
        checkpoint,
        adapter=restored,
        optimizer=restored_optimizer,
        expected={"adapter_fingerprint": adapter.spec.fingerprint, "k": 1},
    )
    actual = restored(action, future).detach()
    assert torch.equal(actual, expected)
    assert loaded.global_step == 3
    assert set(path.name for path in checkpoint.iterdir()) == {
        "adapter.safetensors",
        "manifest.json",
        "optimizer.pt",
    }
    assert find_latest_checkpoint(root) == checkpoint
