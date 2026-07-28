# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False

# --------------------------------------------------
# When vllm is built with VLLM_TARGET_DEVICE=empty, the C extension modules
# are not compiled.  torch.ops._C_cache_ops.* are not registered, so the
# MetaX flash attention backend's reshape_and_cache_flash call path fails.
#
# vllm ships a pure-Triton implementation (triton_reshape_and_cache_flash)
# for platforms lacking the compiled extension.  Route _custom_ops through
# the Triton fallback when the C ops are missing.
#
# Safe for standard vllm wheels: the patch only fires when _C_cache_ops is
# absent, and every op in the dispatch chain is covered by the Triton kernel.
import torch

if not hasattr(torch.ops, "_C_cache_ops"):
    import vllm._custom_ops as _custom_ops
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    _custom_ops.reshape_and_cache_flash = triton_reshape_and_cache_flash
    _custom_ops.reshape_and_cache_flash_per_token_head_quant = (
        triton_reshape_and_cache_flash_per_token_head_quant
    )
