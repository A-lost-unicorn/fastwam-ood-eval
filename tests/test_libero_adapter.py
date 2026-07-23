from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastwam_ood_eval.envs.libero_adapter import (
    _load_trusted_init_states,
    configure_libero_package,
)


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


def test_backend_identity_accepts_symlink_alias_of_same_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_root = tmp_path / "real" / "LIBERO"
    module_path = real_root / "libero" / "libero" / "__init__.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("", encoding="utf-8")
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setitem(
        sys.modules,
        "libero",
        SimpleNamespace(__file__=str(alias_root / "libero" / "libero" / "__init__.py")),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    configured = configure_libero_package(real_root, tmp_path / "runtime")

    assert configured["package_root"] == real_root.resolve()


def test_backend_identity_accepts_namespace_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "LIBERO"
    namespace_path = root / "libero"
    namespace_path.mkdir(parents=True)
    monkeypatch.setitem(
        sys.modules,
        "libero",
        SimpleNamespace(__file__=None, __path__=[str(namespace_path)]),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))

    configured = configure_libero_package(root, tmp_path / "runtime")

    assert configured["package_root"] == root.resolve()
