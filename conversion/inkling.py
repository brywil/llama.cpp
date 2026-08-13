from __future__ import annotations

import re
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, TextModel, gguf, logger


@ModelBase.register("InklingForConditionalGeneration")
class InklingModel(TextModel):
    model_arch = gguf.MODEL_ARCH.INKLING
    undo_permute = False
    supports_mtp_export = True

    # model.mtp.* is NOT skipped -- it is remapped onto blk.{n_layer + k}; see filter_tensors
    _SKIP_PREFIXES = ("model.visual.", "model.audio.")

    # set on the class in __init__ so the filter_tensors classmethod can see them
    # (same pattern as Cohere2MoeModel._n_main_layers)
    _n_main_layers: int | None = None
    _n_mtp: int = 0

    _MTP_RE = re.compile(r"^model\.mtp\.layers\.(\d+)\.(.*)$")
    # the three head-specific tensors; everything else in a head is an ordinary dense block
    _MTP_EXTRAS = {
        "embed_norm.weight":  "mtp_enorm.weight",
        "hidden_norm.weight": "mtp_hnorm.weight",
        "input_proj.weight":  "mtp_eh_proj.weight",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # explicit raises (not assert, stripped by python -O) guard the single supported variant
        hp = self.hparams

        # normalize keys renamed by HF-port re-saved configs back to checkpoint names
        if "dense_intermediate_size" not in hp and "moe_intermediate_size" in hp:
            hp["dense_intermediate_size"] = hp["intermediate_size"]
            hp["intermediate_size"] = hp["moe_intermediate_size"]

        # ...and then DISTRUST the result: measure the expert width from the tensor itself.
        #
        # load_hparams() goes through AutoConfig.from_pretrained().to_dict(), so we never see the
        # raw config.json. TML names the MoE expert width `intermediate_size` (2048 here) and the
        # dense width `dense_intermediate_size` (16384). InklingTextConfig reads the DENSE one
        # into `intermediate_size`, drops `dense_intermediate_size` entirely, and invents
        # `moe_intermediate_size` from its own default of 3072 -- so the branch above fires and
        # propagates a value that is in the checkpoint nowhere.
        #
        # This used to work: before transformers supported Inkling, AutoConfig raised, the loader
        # fell back to raw config.json, `dense_intermediate_size` was present and the branch was
        # skipped. transformers 5.14 adding native Inkling support is what broke it. Left
        # unfixed, expert_feed_forward_length ships as 3072 while the expert tensors are written
        # at their real 2048, and llama.cpp refuses to load the result.
        if (measured := self._measured_moe_ff()) is not None and measured != hp.get("intermediate_size"):
            logger.warning(
                f"expert intermediate_size {hp.get('intermediate_size')} disagrees with the "
                f"checkpoint; using the measured {measured} "
                f"(transformers' moe_intermediate_size default is not this checkpoint's value)"
            )
            hp["intermediate_size"] = measured
        if "sconv_kernel_size" not in hp and "conv_kernel_size" in hp:
            hp["sconv_kernel_size"] = hp["conv_kernel_size"]
        if "dense_mlp_idx" not in hp and hp.get("mlp_layer_types"):
            types = hp["mlp_layer_types"]
            hp["dense_mlp_idx"] = next((i for i, t in enumerate(types) if t != "dense"), len(types))

        if hp.get("gate_activation", "sigmoid") != "sigmoid":
            raise NotImplementedError(
                f"unsupported gate_activation {hp.get('gate_activation')!r}; only 'sigmoid' is implemented"
            )
        for flag, want in (
            ("norm_after_topk", True),
            ("shared_expert_sink", True),
            ("use_sconv", True),
            ("use_embed_norm", True),
            ("use_gate_bias", True),
            ("use_global_scale", True),
        ):
            if hp.get(flag, want) is not want:
                raise NotImplementedError(f"unsupported {flag}={hp.get(flag)!r}; only {want} is implemented")
        if hp.get("q_bias", False) is not False or hp.get("o_bias", False) is not False:
            raise NotImplementedError("attention q_bias / o_bias are not supported")
        if hp.get("final_logit_softcapping") not in (None, 0, 0.0):
            raise NotImplementedError(
                f"final_logit_softcapping={hp.get('final_logit_softcapping')!r} is not supported"
            )
        if hp["swa_head_dim"] != hp["head_dim"]:
            raise ValueError(f"swa_head_dim {hp['swa_head_dim']} must equal head_dim {hp['head_dim']}")
        if hp["swa_num_attention_heads"] != hp["num_attention_heads"]:
            raise ValueError(
                f"swa_num_attention_heads {hp['swa_num_attention_heads']} must equal "
                f"num_attention_heads {hp['num_attention_heads']}"
            )

        # context length comes from model_max_length per the design contract
        if (mml := hp.get("model_max_length")) is not None:
            self.hparams["max_position_embeddings"] = mml

        # checked by the base find_hparam list before the MoE "intermediate_size"
        self.hparams["prefix_dense_intermediate_size"] = hp["dense_intermediate_size"]

        # MTP: 8 chained heads, one per draft step, exported as blocks n_layer..n_layer+7.
        n_layer = hp["num_hidden_layers"]
        n_mtp = int((hp.get("mtp_config") or {}).get("num_nextn_predict_layers", 0) or 0)
        if self.no_mtp:
            n_mtp = 0
        self._n_mtp = n_mtp
        type(self)._n_main_layers = n_layer
        type(self)._n_mtp = n_mtp
        if n_mtp:
            # llama.cpp derives the trunk as n_layer() = n_layer_all - n_layer_nextn, so
            # block_count must cover the heads. tensor_map is built from block_count in
            # ModelBase.__init__ and MUST be rebuilt, or blk.>=n_layer names never resolve.
            self.block_count = n_layer + n_mtp
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        # full length (n_layer_all) -- the swa pattern describes the MTP heads too
        self._local_layer_flags = self._get_local_layer_flags()

        # head_count_kv: write a SCALAR when the local and global layers agree, which sidesteps a
        # length trap that cost a full conversion to find.
        #
        # The generic loader reads this key with `hparams.n_layer()`, which is
        # n_layer_all - n_layer_nextn -- but nextn_predict_layers is read LATER, by the
        # arch-specific hook, so at that moment n_layer_nextn is still 0 and n_layer() returns the
        # FULL block count (50), not the trunk (42). Reading the code without tracing that ordering
        # gives exactly the wrong answer, and get_key_or_arr throws on any mismatch:
        #     key inkling.attention.head_count_kv has wrong array length; expected 50, got 42
        # A scalar is broadcast to whatever length the loader asks for, so it is correct under
        # either reading and cannot rot if that ordering changes again.
        kv_swa, kv_glb = hp["swa_num_key_value_heads"], hp["num_key_value_heads"]
        if kv_swa == kv_glb:
            self.hparams["num_key_value_heads"] = kv_glb
        else:
            # Genuinely per-layer: emit the FULL n_layer_all length, matching what the generic
            # loader asks for. The MTP heads reuse the trunk's global value.
            self.hparams["num_key_value_heads"] = [
                kv_swa if is_local else kv_glb for is_local in self._local_layer_flags
            ]

    def index_tensors(self, remote_hf_model_id: str | None = None):
        """Set the MTP class attributes BEFORE any tensor is filtered.

        `ModelBase.__init__` calls `index_tensors()`, which runs `filter_tensors` over every
        tensor in the checkpoint. That happens BEFORE this subclass's `__init__` body executes,
        so setting `_n_mtp` there is too late: the guard in `filter_tensors` sees the class
        default of 0 and silently drops all 160 `model.mtp.*` tensors, while
        `set_gguf_parameters` -- which runs later, with `_n_mtp` correctly 8 -- still writes
        `nextn_predict_layers = 8` and `block_count = 50`.

        The result is a GGUF that ADVERTISES 8 MTP blocks and does not contain them. Nothing
        errors; the converter logs "MTP heads = 8 at blk.42..49" and reports success, and the
        491 GiB file is only revealed as broken when llama.cpp refuses it with "wrong number of
        tensors". This cost a full 72-minute conversion to discover.

        `Cohere2MoeModel` sets its own `_n_main_layers` from an `index_tensors` override for
        exactly this reason. Do not move this back into `__init__`.
        """
        hp = {**self.hparams, **(self.hparams.get("text_config") or {})}
        n_layer = hp.get("num_hidden_layers")
        n_mtp = int((hp.get("mtp_config") or {}).get("num_nextn_predict_layers", 0) or 0)
        if self.no_mtp:
            n_mtp = 0
        if n_layer is not None:
            type(self)._n_main_layers = int(n_layer)
            type(self)._n_mtp = n_mtp
            self._n_mtp = n_mtp
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def _measured_moe_ff(self) -> int | None:
        """Routed-expert intermediate size, read from a w13 tensor header (no tensor data).

        w13_weight is [n_expert, 2*I, n_embd] with gate and up interleaved, so I = dim(-2) // 2.
        Returns None if the checkpoint cannot be inspected, in which case the config value stands.
        """
        import json as _json
        import pathlib as _pl

        try:
            from safetensors import safe_open

            d = _pl.Path(self.dir_model)
            index = d / "model.safetensors.index.json"
            if index.is_file():
                wm = _json.loads(index.read_text())["weight_map"]
                name = next(k for k in sorted(wm) if k.endswith(".mlp.experts.w13_weight"))
                files = [(d / wm[name], name)]
            else:
                files = []
                for f in sorted(d.glob("*.safetensors")):
                    with safe_open(f, framework="pt") as h:
                        hit = next((k for k in sorted(h.keys()) if k.endswith(".mlp.experts.w13_weight")), None)
                    if hit:
                        files = [(f, hit)]
                        break
            if not files:
                return None
            path, name = files[0]
            with safe_open(path, framework="pt") as h:
                shape = list(h.get_slice(name).get_shape())
            return shape[-2] // 2
        except Exception as e:  # never let a probe failure block a conversion
            logger.warning(f"could not measure expert intermediate_size from the checkpoint: {e}")
            return None

    def _get_local_layer_flags(self) -> list[bool]:
        # local_layer_ids is authoritative; a round-tripped layer_types may be stale and must not override it
        n_layer = self.hparams["num_hidden_layers"]
        local_ids = self.hparams.get("local_layer_ids")
        if local_ids is None:
            # default: global at id % 6 == 5; omitted/null must not collapse to all-global (explicit [] does)
            local_ids = [i for i in range(n_layer) if i % 6 != 5]
        local_ids = set(local_ids)
        flags = [i in local_ids for i in range(n_layer)]

        # The MTP heads have their OWN local/global pattern, in mtp_config.local_layer_ids --
        # independently confirmed by attn.rel_logits_proj, which is [d_rel, 512] on a local head
        # and [d_rel, rel_extent] on a global one. The array is written for all n_layer_all
        # blocks, so leaving it trunk-length would silently mark every head global.
        if self._n_mtp:
            mtp_local = (self.hparams.get("mtp_config") or {}).get("local_layer_ids")
            if mtp_local is None:
                mtp_local = [k for k in range(self._n_mtp) if k % 6 != 5]
            mtp_local = set(mtp_local)
            flags += [k in mtp_local for k in range(self._n_mtp)]
        return flags

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens, toktypes, tokpre = super().get_vocab_base()
        # dedicated pre-type: o200k-family regex that keeps combining marks attached to base letters
        tokpre = "inkling"
        import gguf as _gguf
        n_vocab = self.hparams["vocab_size"]
        n_unpadded = self.hparams.get("unpadded_vocab_size") or n_vocab
        if len(tokens) != n_vocab:
            raise ValueError(f"Inkling tokenizer produced {len(tokens)} entries, expected {n_vocab}")
        # force-CONTROL special ids from added_tokens_decoder, else the trailing-60 convention
        try:
            import json as _json
            import pathlib as _pl
            tc = _json.loads((_pl.Path(self.dir_model) / "tokenizer_config.json").read_text())
            special_ids = sorted(int(i) for i, d in tc.get("added_tokens_decoder", {}).items() if d.get("special"))
        except Exception:
            special_ids = list(range(n_unpadded - 60, n_unpadded))
        for tid in special_ids:
            if 0 <= tid < n_vocab:
                toktypes[tid] = _gguf.TokenType.CONTROL
        if any(t != _gguf.TokenType.UNUSED for t in toktypes[n_unpadded:]):
            raise ValueError("real tokens found at/above unpadded_vocab_size; padded-vocab mask would hide them")
        return tokens, toktypes, tokpre

    def set_vocab(self):
        self._set_vocab_gpt2()
        eos_id = int(self.hparams.get("eos_token_id", 200006))
        if eos_id < 199998:
            # HF-port configs re-save generic bos/eos defaults; the real EOS lives at 199998+
            eos_id = 200006
        # 200006 is the SOLE end-of-generation token; <|end_message|> (200010) is an
        # intra-turn block separator and must NOT be registered eot/eog
        self.gguf_writer.add_eos_token_id(eos_id)
        # no BOS is ever prepended; pin bos to EOS so a stale base-tokenizer bos id never surfaces
        self.gguf_writer.add_bos_token_id(eos_id)
        self.gguf_writer.add_add_bos_token(False)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hp["intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hp["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hp["route_scale"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # sliding_window_size is canonical; explicit is-None fallback so a serialized 0 cannot bypass the mismatch check
        canonical_window = hp["sliding_window_size"]
        sliding_window = hp.get("sliding_window")
        if sliding_window is None:
            sliding_window = canonical_window
        elif sliding_window != canonical_window:
            raise ValueError(
                f"sliding_window {sliding_window} disagrees with sliding_window_size "
                f"{canonical_window!r}"
            )
        if sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        self.gguf_writer.add_sliding_window(sliding_window)
        # true = local (swa) layer
        self.gguf_writer.add_sliding_window_pattern(self._local_layer_flags)

        # no RoPE (arch-determined NONE); custom inkling.* keys per INKLING_DESIGN.md
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
        self.gguf_writer.add_uint32(f"{arch}.d_rel", hp["d_rel"])
        self.gguf_writer.add_uint32(f"{arch}.rel_extent", hp["rel_extent"])
        self.gguf_writer.add_uint32(f"{arch}.rel_extent_swa", sliding_window)
        self.gguf_writer.add_uint32(f"{arch}.shortconv_kernel", hp["sconv_kernel_size"])
        self.gguf_writer.add_uint32(f"{arch}.dense_block_count", hp["dense_mlp_idx"])
        self.gguf_writer.add_float32(f"{arch}.logit_scale_denom", hp["logits_mup_width_multiplier"])
        self.gguf_writer.add_uint32(f"{arch}.log_scaling_n_floor", int(hp.get("log_scaling_n_floor") or 0))
        self.gguf_writer.add_float32(f"{arch}.log_scaling_alpha", hp.get("log_scaling_alpha", 0.0))
        self.gguf_writer.add_uint32(f"{arch}.unpadded_vocab_size", hp["unpadded_vocab_size"])

        if self._n_mtp:
            self.gguf_writer.add_nextn_predict_layers(self._n_mtp)
            logger.info(
                f"gguf: (inkling) MTP heads = {self._n_mtp} at blk."
                f"{hp['num_hidden_layers']}..{hp['num_hidden_layers'] + self._n_mtp - 1}"
                f", block_count = {self.block_count}"
            )

        logger.info(f"gguf: (inkling) swa pattern (true=local) = {self._local_layer_flags}")
        logger.info(f"gguf: (inkling) unpadded_vocab_size = {hp['unpadded_vocab_size']}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith(cls._SKIP_PREFIXES):
            return None

        if (m := cls._MTP_RE.match(name)) is not None:
            # Each MTP head is enorm/hnorm/input_proj plus a full DENSE transformer_block. Remap
            # the block onto the trunk's own layer names at bid = n_layer + k so every existing
            # attn / sconv / dense-FFN rule applies unchanged, and give the three head-specific
            # tensors sentinel names that modify_tensors turns into blk.{bid}.nextn.*.
            if cls.no_mtp or not cls._n_mtp or cls._n_main_layers is None:
                return None
            k, tail = int(m.group(1)), m.group(2)
            if k >= cls._n_mtp:
                raise ValueError(
                    f"MTP head {k} exceeds mtp_config.num_nextn_predict_layers={cls._n_mtp} ({name!r})"
                )
            bid = cls._n_main_layers + k
            if tail.startswith("transformer_block."):
                name = f"model.layers.{bid}." + tail.removeprefix("transformer_block.")
            elif (sentinel := cls._MTP_EXTRAS.get(tail)) is not None:
                name = f"model.layers.{bid}.{sentinel}"
            else:
                # loud, not silent: an unrecognised head tensor would otherwise be dropped and
                # the drafter would load with a missing weight
                raise ValueError(f"unexpected Inkling MTP tensor {name!r}")
        elif cls.mtp_only:
            return None
        else:
            name = name.replace("model.llm.", "model.")

        # parameter has no ".weight"-style suffix in the checkpoint
        name = name.replace("rel_logits_proj.proj", "rel_logits_proj.weight")

        return super().filter_tensors((name, gen))

    @staticmethod
    def _deinterleave_w13(w13: Tensor) -> tuple[Tensor, Tensor]:
        # interleaved SwiGLU along the output rows: silu(z[..., ::2]) * z[..., 1::2]
        gate = w13[..., 0::2, :].contiguous()
        up   = w13[..., 1::2, :].contiguous()
        return gate, up

    # sentinel suffix (from filter_tensors) -> ggml nextn tensor
    _MTP_SENTINELS = {
        "mtp_enorm.weight":   gguf.MODEL_TENSOR.NEXTN_ENORM,
        "mtp_hnorm.weight":   gguf.MODEL_TENSOR.NEXTN_HNORM,
        "mtp_eh_proj.weight": gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
    }

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # MTP head extras. Checked before the generic rules because these names are synthetic and
        # map_tensor_name would not know them.
        for suffix, tensor in self._MTP_SENTINELS.items():
            if name.endswith(suffix):
                assert bid is not None
                return [(self.format_tensor_name(tensor, bid), data_torch)]

        # short convs: [C, 1, K] -> [C, K] (same layout as LFM2 shortconv.conv)
        if name.endswith("_sconv.weight"):
            data_torch = data_torch.squeeze(1)
            return [(self.map_tensor_name(name), data_torch)]

        if name.endswith(".mlp.w13_dn.weight"):
            assert bid is not None
            gate, up = self._deinterleave_w13(data_torch)
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), gate),
                (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP,   bid), up),
            ]

        if name.endswith(".mlp.global_scale") or name.endswith(".mlp.gate.global_scale"):
            assert bid is not None
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GSCALE, bid), data_torch.float())]

        if name.endswith(".mlp.gate.bias"):
            assert bid is not None
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, bid, ".bias"), data_torch.float())]

        if name.endswith(".mlp.experts.w13_weight"):
            assert bid is not None
            gate, up = self._deinterleave_w13(data_torch)
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), gate),
                (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP,   bid), up),
            ]
        if name.endswith(".mlp.experts.w2_weight"):
            assert bid is not None
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid), data_torch)]

        # shared experts stored stacked for mul_mat_id
        if name.endswith(".mlp.shared_experts.shared_w13_weight"):
            assert bid is not None
            gate, up = self._deinterleave_w13(data_torch)
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_SHEXP, bid), gate),
                (self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_SHEXP,   bid), up),
            ]
        if name.endswith(".mlp.shared_experts.shared_w2_weight"):
            assert bid is not None
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, bid), data_torch)]

        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        # used in fp32 rel-bias math; keep full precision
        if new_name.endswith("attn_rel_proj.weight"):
            return gguf.GGMLQuantizationType.F32
        # ggml_ssm_conv kernels are F32-only
        if ".shortconv_" in new_name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


@ModelBase.register("InklingForConditionalGeneration")
class InklingMmprojModel(MmprojModel):
    """Export Inkling's hMLP and dMel towers as one mtmd projector."""

    has_vision_encoder = True
    has_audio_encoder = True

    _IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _IMAGE_STD = [0.26862954, 0.2613026, 0.2757771]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.hparams_vision is not None
        hp = self.hparams_vision
        expected = {
            "vision_encoder_type": "hmlp",
            "patch_size": 40,
            "temporal_patch_size": 2,
            "n_channels": 3,
            "n_layers": 4,
            "use_vision_norm": True,
        }
        for key, want in expected.items():
            got = hp.get(key, want)
            if got != want:
                raise NotImplementedError(
                    f"Inkling mmproj requires vision_config.{key}={want!r}, got {got!r}"
                )
        # decoder_dmodel is the text model width, it varies per checkpoint
        if not hp.get("decoder_dmodel"):
            raise NotImplementedError("Inkling mmproj requires vision_config.decoder_dmodel")

        assert self.hparams_audio is not None
        ahp = self.hparams_audio
        audio_expected = {
            "audio_mode": "dmel",
            "n_mel_bins": 80,
            "mel_vocab_size": 16,
            "use_audio_norm": True,
        }
        for key, want in audio_expected.items():
            got = ahp.get(key, want)
            if got != want:
                raise NotImplementedError(
                    f"Inkling mmproj requires audio_config.{key}={want!r}, got {got!r}"
                )
        if not ahp.get("decoder_dmodel"):
            raise NotImplementedError("Inkling mmproj requires audio_config.decoder_dmodel")

    def set_gguf_parameters(self):
        hp = self.hparams_vision
        assert hp is not None

        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_vision_encoder(True)
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_vision_projector_type(gguf.VisionProjectorType.INKLING)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.INKLING)
        self.gguf_writer.add_vision_projection_dim(hp["decoder_dmodel"])

        # clip.cpp requires these common fields even though hMLP is not a ViT.
        self.gguf_writer.add_vision_image_size(hp["patch_size"])
        self.gguf_writer.add_vision_patch_size(hp["patch_size"])
        self.gguf_writer.add_vision_embedding_length(hp["n_channels"])
        self.gguf_writer.add_vision_feed_forward_length(0)
        self.gguf_writer.add_vision_block_count(hp["n_layers"])
        self.gguf_writer.add_vision_head_count(1)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_image_mean(self._IMAGE_MEAN)
        self.gguf_writer.add_vision_image_std(self._IMAGE_STD)

        ahp = self.hparams_audio
        assert ahp is not None
        self.gguf_writer.add_audio_projection_dim(ahp["decoder_dmodel"])
        self.gguf_writer.add_audio_embedding_length(ahp["decoder_dmodel"])
        self.gguf_writer.add_audio_feed_forward_length(0)
        self.gguf_writer.add_audio_block_count(0)
        self.gguf_writer.add_audio_head_count(1)
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_audio_num_mel_bins(ahp["n_mel_bins"])

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith(("model.visual.", "visual.", "model.audio.", "audio.")):
            return None
        return name, gen

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        del bid
        if name.startswith(("model.audio.", "audio.")):
            prefix = "model.audio." if name.startswith("model.audio.") else "audio."
            local = name.removeprefix(prefix)
            if local == "encoder.weight":
                yield "a.dmel.embedding.weight", data_torch
                return
            if local == "final_norm.weight":
                yield "a.dmel.final_norm.weight", data_torch
                return
            raise ValueError(f"unexpected Inkling audio tensor {name!r}")

        prefix = "model.visual." if name.startswith("model.visual.") else "visual."
        local = name.removeprefix(prefix)
        if local == "final_norm.weight":
            yield "v.hmlp.final_norm.weight", data_torch
            return

        parts = local.split(".")
        if len(parts) == 3 and parts[0] == "layers" and parts[2] == "weight":
            kind, sep, layer_s = parts[1].partition("_")
            if sep and kind in ("linear", "norm") and layer_s.isdigit():
                yield f"v.hmlp.{int(layer_s)}.{kind}.weight", data_torch
                return

        raise ValueError(f"unexpected Inkling vision tensor {name!r}")
