"""Source-B Action feature extraction at fixed, real-call boundaries."""

from __future__ import annotations

from typing import Any

from fastwam_ood_eval.thought4.feature_hooks import (
    ScopedFeatureCapture,
    action_hook_specs,
    validate_layer_indices,
)


class ActionFeatureExtractor:
    """Capture input, middle, late and pre-head Action representations."""

    def __init__(self, model: Any) -> None:
        validate_layer_indices(model.action_expert, (15, 29), "Action DiT")
        self.model = model
        self.specs = action_hook_specs()

    def capture(self, forward: Any) -> dict[str, list[Any]]:
        with ScopedFeatureCapture(
            self.model, self.specs, clone=True, to_cpu=False
        ) as capture:
            forward()
        return capture.captured

    @property
    def manifest(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "source": "B",
                "name": spec.name,
                "module_path": spec.module_path,
                "location": spec.location,
            }
            for spec in self.specs
        )

