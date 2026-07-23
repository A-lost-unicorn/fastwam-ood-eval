"""Adapter for the official clean LIBERO package."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from fastwam_ood_eval.envs.base import BaseBenchmarkEnv, StepResult
from fastwam_ood_eval.evaluation.jobs import EvaluationJob


def configure_libero_package(root: Path, config_dir: Path) -> dict[str, Path]:
    """Select one LIBERO checkout without constructing a simulator.

    Fast-WAM's official evaluator imports ``libero`` while the policy is being
    created. Shadow diagnostics intentionally validate the video capability
    before constructing an environment, so package selection must be a smaller
    operation than ``LiberoAdapter.__init__``. Calling this function repeatedly
    for the same checkout is safe; switching checkouts in one process is not.
    """

    package_root = Path(root).resolve()
    resolved_config_dir = Path(config_dir).resolve()
    benchmark_root = package_root / "libero" / "libero"
    bddl_root = (benchmark_root / "bddl_files").resolve()
    init_states_root = (benchmark_root / "init_files").resolve()
    resolved_config_dir.mkdir(parents=True, exist_ok=True)
    config_file = resolved_config_dir / "config.yaml"
    path_config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(bddl_root),
        "init_states": str(init_states_root),
        "datasets": str(package_root / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    temporary = config_file.with_name(f"{config_file.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(path_config, sort_keys=True), encoding="utf-8")
    temporary.replace(config_file)
    os.environ["LIBERO_CONFIG_PATH"] = str(resolved_config_dir)
    loaded = sys.modules.get("libero")
    if loaded is not None:
        locations: list[Path] = []
        loaded_file = getattr(loaded, "__file__", None)
        raw_locations = [loaded_file] if loaded_file not in (None, "") else []
        raw_locations.extend(list(getattr(loaded, "__path__", ()) or ()))
        for raw_location in raw_locations:
            try:
                locations.append(Path(str(raw_location)).resolve(strict=False))
            except (OSError, RuntimeError):
                continue
        if not locations or any(
            location != package_root and package_root not in location.parents
            for location in locations
        ):
            raise RuntimeError(
                "A different libero package is already loaded; run each backend in a fresh "
                f"process (expected_root={package_root}, loaded_locations={locations})"
            )
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    return {
        "package_root": package_root,
        "benchmark_root": benchmark_root,
        "bddl_root": bddl_root,
        "init_states_root": init_states_root,
        "config_dir": resolved_config_dir,
        "config_file": config_file,
    }


def _load_trusted_init_states(
    init_states_root: Path,
    problem_folder: str,
    init_states_file: str,
) -> Any:
    """Load an official LIBERO init-state file across PyTorch 2.6+.

    LIBERO init-state files contain NumPy arrays rather than a tensor-only
    state dict. PyTorch 2.6 changed ``torch.load`` to default to
    ``weights_only=True``, so the pinned upstream loader can no longer read
    them. Disabling that restriction can execute pickle payloads; keep the
    trust boundary narrow by resolving the file beneath this checkout's
    init-state directory and accepting only LIBERO's two known extensions.
    """

    trusted_root = init_states_root.resolve(strict=True)
    candidate = trusted_root / problem_folder / init_states_file
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LIBERO init-state file does not exist: {candidate}") from exc
    if trusted_root not in resolved.parents:
        raise RuntimeError(f"Refusing to load init-state file outside trusted root: {resolved}")
    if not resolved.is_file():
        raise RuntimeError(f"LIBERO init-state path is not a regular file: {resolved}")
    if not resolved.name.endswith((".init", ".pruned_init")):
        raise RuntimeError(f"Unexpected LIBERO init-state extension: {resolved}")

    import torch

    return torch.load(resolved, map_location="cpu", weights_only=False)


class LiberoAdapter(BaseBenchmarkEnv):
    def __init__(
        self,
        image_size: tuple[int, int],
        root: Path = Path("third_party/LIBERO"),
        config_dir: Path = Path("outputs/runtime/libero"),
    ) -> None:
        self.image_size = image_size
        self.root = root.resolve()
        self.config_dir = config_dir.resolve()
        self.env: Any = None
        self.task_description = ""
        self._success = False
        self._load_package()

    def _load_package(self) -> None:
        configured = configure_libero_package(self.root, self.config_dir)
        self.bddl_root = configured["bddl_root"]
        self.init_states_root = configured["init_states_root"]
        self.config_dir = configured["config_dir"]
        self.config_file = configured["config_file"]
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise RuntimeError(f"Cannot import clean LIBERO from {self.root}") from exc
        self.benchmark = benchmark
        self.get_libero_path = get_libero_path
        self.env_class = OffScreenRenderEnv

    def _make_suite(self, suite_name: str) -> Any:
        return self.benchmark.get_benchmark_dict()[suite_name]()

    def _resolve_init_state(self, task: Any) -> tuple[Path, bool]:
        return Path(task.problem_folder) / task.init_states_file, False

    def _load_task_init_states(self, task: Any) -> Any:
        relative_path, reshape_single = self._resolve_init_state(task)
        initial_states = _load_trusted_init_states(
            self.init_states_root,
            str(relative_path.parent),
            relative_path.name,
        )
        if reshape_single:
            initial_states = initial_states.reshape(1, -1)
        return initial_states

    def reset(self, job: EvaluationJob) -> dict[str, Any]:
        if self.env is not None:
            self.env.close()
        suite = self._make_suite(job.suite)
        task = suite.get_task(job.upstream_task_id)
        bddl = self.bddl_root / task.problem_folder / task.bddl_file
        self.env = self.env_class(
            bddl_file_name=str(bddl),
            camera_heights=self.image_size[0],
            camera_widths=self.image_size[1],
        )
        self.env.seed(job.episode_seed)
        self.env.reset()
        initial_states = self._load_task_init_states(task)
        obs = self.env.set_init_state(initial_states[job.initial_state_index % len(initial_states)])
        self.task_description = task.language
        self._success = False
        return obs

    def step(self, action: Any) -> StepResult:
        obs, reward, done, info = self.env.step(action)
        self._success = bool(done) or self._check_success()
        return StepResult(obs, float(reward), self._success, dict(info or {}))

    def _check_success(self) -> bool:
        if hasattr(self.env, "check_success"):
            return bool(self.env.check_success())
        inner = getattr(self.env, "env", self.env)
        return bool(inner._check_success())

    def is_success(self) -> bool:
        return self._success or self._check_success()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def runtime_config(self) -> dict[str, Any]:
        return {
            "backend_root": str(self.root),
            "libero_config_path": str(self.config_file),
            "libero_paths": yaml.safe_load(self.config_file.read_text(encoding="utf-8")),
        }
