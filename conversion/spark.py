from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("Spark2_5ForCausalLM")
@ModelBase.example("XHToken/Spark-X2.5-4B")
class Spark2_5Model(TextModel):
    """XHToken Spark-X2.5.

    Three things differ from a stock llama-shaped decoder:

    1. TWO rope configurations, selected per layer type. The sliding layers rotate the whole
       head (partial_rotary_factor 1.0) at theta 10000; the full-attention layers rotate only
       a quarter of it (0.25 -> 64 of 256 dims) at theta 5000000 and pass the remaining 192
       dims through unrotated. Applying one rope to all layers yields a model that loads and
       produces fluent text while being wrong -- there is no error to catch it.
    2. A head-wise attention OUTPUT gate: g_proj is Linear(hidden, n_head), sigmoid, multiplying
       the attention result per head. 40,960 params per layer.
    3. q/k/v are fused into a single q_k_v_proj and must be split.
    """

    model_arch = gguf.MODEL_ARCH.SPARK2_5

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_context_length(hparams["max_position_embeddings"])
        self.gguf_writer.add_head_count(hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(hparams["num_key_value_heads"])
        self.gguf_writer.add_key_length(hparams["head_dim"])
        self.gguf_writer.add_value_length(hparams["head_dim"])
        self.gguf_writer.add_layer_norm_rms_eps(hparams["rms_norm_eps"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])

        # --- the part that must not be simplified ---------------------------------------
        # rope_parameters is keyed by layer type. Each entry has its own theta AND its own
        # partial_rotary_factor, so both the dimension count and the base differ between the
        # sliding and full-attention layers. llama.cpp already selects per layer via
        # hparams.n_rot(il) / get_rope_freq_base(cparams, il) -- it just needs both pairs.
        head_dim = hparams["head_dim"]
        rope = hparams["rope_parameters"]
        full = rope["full_attention"]
        swa = rope["sliding_attention"]

        n_rot_full = int(head_dim * full.get("partial_rotary_factor", 1.0))
        n_rot_swa = int(head_dim * swa.get("partial_rotary_factor", 1.0))

        self.gguf_writer.add_rope_freq_base(full["rope_theta"])
        self.gguf_writer.add_rope_dimension_count(n_rot_full)
        self.gguf_writer.add_rope_freq_base_swa(swa["rope_theta"])
        self.gguf_writer.add_rope_dimension_count_swa(n_rot_swa)

        # Layer types come straight from the checkpoint rather than being inferred from a
        # stride: the pattern is sssF but nothing guarantees that stays true across releases.
        layer_types = hparams["layer_types"]
        assert len(layer_types) == hparams["num_hidden_layers"], "layer_types length != n_layer"
        self.gguf_writer.add_sliding_window_pattern(
            [t == "sliding_attention" for t in layer_types]
        )

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Split the fused q_k_v_proj. Order in the checkpoint is [q | k | v], matching the
        # slicing in Spark2_5Attention.forward.
        if name.endswith("self_attn.q_k_v_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            n_kv = self.hparams["num_key_value_heads"]
            head_dim = self.hparams["head_dim"]
            q_dim = n_head * head_dim
            kv_dim = n_kv * head_dim
            assert data_torch.shape[0] == q_dim + 2 * kv_dim, (
                f"fused qkv is {data_torch.shape[0]} rows, expected {q_dim + 2 * kv_dim}"
            )
            q = data_torch[:q_dim]
            k = data_torch[q_dim:q_dim + kv_dim]
            v = data_torch[q_dim + kv_dim:]
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), q),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), k),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v),
            ]

        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # attn_gate is a COEFFICIENT, not a projection: Linear(2560, 16) = 40,960 params per
        # layer, 1.47 M for the whole model, ~5.9 MB at F32. Its sigmoid output MULTIPLIES the
        # attention result, so error there compounds instead of adding to one layer's output --
        # the case where a per-tensor KL probe under-reports the cost. Pin it on principle; at
        # this size the bits are free.
        if new_name.endswith("attn_gate.weight"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
