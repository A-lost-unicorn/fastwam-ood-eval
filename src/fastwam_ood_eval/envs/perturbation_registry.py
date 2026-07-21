"""Mapping from project-level perturbation names to official LIBERO-Plus labels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbationDefinition:
    name: str
    upstream_category: str
    levels: dict[str, tuple[int, ...]]


_LEVELS = {"easy": (1, 2), "medium": (3,), "hard": (4, 5)}

REGISTRY: dict[str, PerturbationDefinition] = {
    "camera_viewpoints": PerturbationDefinition("camera_viewpoints", "Camera Viewpoints", _LEVELS),
    "light_conditions": PerturbationDefinition("light_conditions", "Light Conditions", _LEVELS),
    "background_textures": PerturbationDefinition("background_textures", "Background Textures", _LEVELS),
    "robot_initial_states": PerturbationDefinition("robot_initial_states", "Robot Initial States", _LEVELS),
    "objects_layout": PerturbationDefinition("objects_layout", "Objects Layout", _LEVELS),
}


def get_perturbation(name: str) -> PerturbationDefinition:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown perturbation {name!r}; valid names: {sorted(REGISTRY)}") from exc

