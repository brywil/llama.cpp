from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .qwen import Qwen2MoeModel


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1-BF16")
class Kolibri1Model(Qwen2MoeModel):
    """Aleph Alpha Kolibri 1: Qwen3-MoE with (1) a 4:1 sliding:full attention pattern where only the sliding layers
    use RoPE (full layers are NoPE), (2) sandwich norms, (3) top-k selected on `logits + expert_bias` but weighted by
    `sigmoid(logits)`, unnormalised, (4) one ungated shared expert, (5) per-head q/k norm.
    Reference: aleph_alpha_inference/kolibri1.py (the vLLM plugin)."""
    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams
        self.gguf_writer.add_expert_shared_count(1)
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID_LOGIT_BIAS)
        self.gguf_writer.add_expert_weights_norm(bool(hp.get("norm_topk_prob", False)))
        # layer_types must be a strict period: (period-1) sliding then 1 full, repeated -- that is what the C++ side
        # reconstructs from (sliding_window, sliding_window_pattern)
        lt = hp["layer_types"]
        period = lt.index("full_attention") + 1
        expect = (["sliding_attention"] * (period - 1) + ["full_attention"]) * (len(lt) // period)
        assert lt == expect, f"layer_types is not a strict {period}-period pattern"
        self.gguf_writer.add_sliding_window(hp["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(period)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Explicit names for the tensors whose HF names are ambiguous across families: Kolibri's
        # `post_attention_layernorm` is the PRE-FFN norm (Qwen naming), while Gemma-family mappings read that name as
        # the post-attention norm. Also: the shared expert must not fall into the parent's per-expert stacking path,
        # which matches any name containing "experts".
        explicit = {
            "input_layernorm.weight":          (gguf.MODEL_TENSOR.ATTN_NORM,       ".weight"),
            "post_attn_norm.weight":           (gguf.MODEL_TENSOR.ATTN_POST_NORM,  ".weight"),
            "post_attention_layernorm.weight": (gguf.MODEL_TENSOR.FFN_NORM,        ".weight"),
            "post_ffn_norm.weight":            (gguf.MODEL_TENSOR.FFN_POST_NORM,   ".weight"),
            "moe.router.expert_bias":          (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "mlp.gate.weight":                 (gguf.MODEL_TENSOR.FFN_GATE_INP,    ".weight"),
            "mlp.shared_experts.gate_proj.weight": (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "mlp.shared_experts.up_proj.weight":   (gguf.MODEL_TENSOR.FFN_UP_SHEXP,   ".weight"),
            "mlp.shared_experts.down_proj.weight": (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
        }
        if bid is not None:
            local = name.removeprefix(f"model.layers.{bid}.")
            if local in explicit:
                key, suffix = explicit[local]
                yield self.format_tensor_name(key, bid, suffix=suffix), data_torch
                return
        yield from super().modify_tensors(data_torch, name, bid)
