"""Adapter for official pre-generated LIBERO-Plus task variants."""

from __future__ import annotations

from pathlib import Path

from fastwam_ood_eval.envs.libero_adapter import LiberoAdapter


class LiberoPlusAdapter(LiberoAdapter):
    def __init__(
        self,
        image_size: tuple[int, int],
        root: Path = Path("third_party/LIBERO-plus"),
        config_dir: Path = Path("outputs/runtime/libero_plus"),
    ) -> None:
        super().__init__(image_size=image_size, root=root, config_dir=config_dir)
