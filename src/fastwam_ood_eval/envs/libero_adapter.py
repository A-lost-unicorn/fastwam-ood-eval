"""Adapter for the official clean LIBERO package."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

from fastwam_ood_eval.envs.base import BaseBenchmarkEnv, StepResult
from fastwam_ood_eval.evaluation.jobs import EvaluationJob


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
        package_root = self.root
        benchmark_root = package_root / "libero" / "libero"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.yaml"
        path_config = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(package_root / "libero" / "datasets"),
            "assets": str(benchmark_root / "assets"),
        }
        temporary = self.config_file.with_name(f"{self.config_file.name}.{os.getpid()}.tmp")
        temporary.write_text(yaml.safe_dump(path_config, sort_keys=True), encoding="utf-8")
        temporary.replace(self.config_file)
        # This is the upstream-supported switch and prevents an interactive ~/.libero prompt.
        os.environ["LIBERO_CONFIG_PATH"] = str(self.config_dir)
        loaded = sys.modules.get("libero")
        if loaded is not None and str(package_root) not in str(getattr(loaded, "__file__", "")):
            raise RuntimeError("A different libero package is already loaded; run each backend in a fresh process")
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise RuntimeError(f"Cannot import clean LIBERO from {self.root}") from exc
        self.benchmark = benchmark
        self.get_libero_path = get_libero_path
        self.env_class = OffScreenRenderEnv

    def reset(self, job: EvaluationJob) -> dict[str, Any]:
        if self.env is not None:
            self.env.close()
        suite = self.benchmark.get_benchmark_dict()[job.suite]()
        task = suite.get_task(job.upstream_task_id)
        bddl = Path(self.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env = self.env_class(
            bddl_file_name=str(bddl),
            camera_heights=self.image_size[0],
            camera_widths=self.image_size[1],
        )
        self.env.seed(job.episode_seed)
        self.env.reset()
        initial_states = suite.get_task_init_states(job.upstream_task_id)
        obs = self.env.set_init_state(initial_states[job.episode_index % len(initial_states)])
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
