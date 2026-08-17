# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

from vllm_fl.patches import gdn_packed_decode


def _vulnerable_kernel():
    beta_val = "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
    return beta_val


def _fixed_kernel():
    beta_val = "tl.sigmoid(b_val)"
    return beta_val


def test_patch_replaces_vulnerable_kernel(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_vulnerable_kernel
    )
    monkeypatch.setattr(
        gdn_packed_decode.importlib, "import_module", lambda _module: target
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is True
    replacement = target.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert replacement is not _vulnerable_kernel
    assert replacement._fl_fp32_beta is True
    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False


def test_patch_preserves_already_fixed_upstream_kernel(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_fixed_kernel
    )
    monkeypatch.setattr(
        gdn_packed_decode.importlib, "import_module", lambda _module: target
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is _fixed_kernel


def test_patch_is_optional_when_symbol_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        gdn_packed_decode.importlib,
        "import_module",
        lambda _module: SimpleNamespace(),
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False
