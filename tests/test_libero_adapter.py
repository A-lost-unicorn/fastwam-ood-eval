from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fastwam_ood_eval.envs.libero_adapter import _load_trusted_init_states


def test_loads_numpy_init_states_with_explicit_legacy_pickle_mode(tmp_path: Path):
    root = tmp_path / "init_files"
    task_dir = root / "libero_spatial"
    task_dir.mkdir(parents=True)
    path = task_dir / "example.pruned_init"
    expected = np.arange(12, dtype=np.float64).reshape(3, 4)
    torch.save(expected, path)

    actual = _load_trusted_init_states(root, "libero_spatial", path.name)

    np.testing.assert_array_equal(actual, expected)


def test_rejects_init_state_path_outside_checkout_root(tmp_path: Path):
    root = tmp_path / "init_files"
    root.mkdir()
    outside = tmp_path / "outside.init"
    torch.save(np.arange(3), outside)

    with pytest.raises(RuntimeError, match="outside trusted root"):
        _load_trusted_init_states(root, "..", outside.name)


def test_rejects_unknown_init_state_extension(tmp_path: Path):
    root = tmp_path / "init_files"
    task_dir = root / "libero_spatial"
    task_dir.mkdir(parents=True)
    path = task_dir / "example.pt"
    torch.save(np.arange(3), path)

    with pytest.raises(RuntimeError, match="Unexpected LIBERO init-state extension"):
        _load_trusted_init_states(root, "libero_spatial", path.name)
