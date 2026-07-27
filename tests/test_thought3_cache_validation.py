from __future__ import annotations

import pytest

from fastwam_ood_eval.thought3.cache_builder import build_cache
from fastwam_ood_eval.thought3.cache_planner import (
    create_cache_plan,
    write_cache_plan,
)
from fastwam_ood_eval.thought3.cache_validator import validate_cache
from fastwam_ood_eval.thought3.config import load_thought3_config
from fastwam_ood_eval.thought3.future_cache import (
    CacheValidationError,
    FutureCacheReader,
)
from thought3_test_utils import write_thought3_config


def _built_cache(tmp_path):
    cfg = load_thought3_config(write_thought3_config(tmp_path))
    write_cache_plan(cfg)
    build_cache(cfg, resume=False)
    return cfg


def test_whole_cache_validates_and_reader_rejects_k_mismatch(tmp_path):
    cfg = _built_cache(tmp_path)
    report = validate_cache(cfg.cache.root)
    assert report["status"] == "valid"
    assert report["paired_k_valid"]
    entries, _, _ = create_cache_plan(cfg)
    base_id = entries[0].identity.base_sample_id
    reader = FutureCacheReader(
        cfg.cache.root,
        expected_cache_fingerprint=report["cache_fingerprint"],
    )
    latent, mask, row = reader.get(base_id, 1)
    assert latent.shape == (48, 2, 14, 28)
    assert mask.all()
    assert row["record"]["uses_ground_truth_future"] is False
    with pytest.raises(CacheValidationError, match="K mismatch"):
        reader.get(base_id, 3)


def test_checksum_detects_single_byte_corruption(tmp_path):
    cfg = _built_cache(tmp_path)
    tensor_path = sorted(cfg.cache.root.glob("k1/*.safetensors"))[0]
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 0x01
    tensor_path.write_bytes(bytes(payload))
    with pytest.raises(CacheValidationError, match="checksum mismatch"):
        validate_cache(cfg.cache.root)
