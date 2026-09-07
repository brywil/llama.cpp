#include "models.h"

// XHToken Spark-X2.5 (spark2_5).
//
// Two things make this arch different from a stock SWA decoder, and both fail SILENTLY if
// you get them wrong -- the model loads and produces fluent text either way:
//
//  1. TWO rope configurations, chosen per layer type. The sliding layers rotate the whole head
//     (256 of 256 dims) at theta 10000; the full-attention layers rotate only a quarter of it
//     (64 of 256) at theta 5000000 and pass the remaining 192 dims through untouched. So BOTH
//     the dimension count and the base differ per layer -- hence hparams.n_rot(il) and
//     model.get_rope_freq_base(cparams, il) below, NOT the graph-cached n_rot/freq_base.
//  2. A head-wise attention OUTPUT gate applied BEFORE the output projection, as in Step35.

void llama_model_spark2_5::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa, false);

    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

        // The checkpoint ships an explicit per-layer list; prefer it over a stride so a
        // future release that breaks the sssF pattern cannot be silently mis-read.
        uint32_t swa_period = 4;
        const auto res = ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, swa_period, false);
        if (res) {
            hparams.set_swa_pattern(swa_period);
        } else {
            ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());
        }

        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
        ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

        // Partial rotary differs per layer type too; n_rot_swa defaults to n_rot_full upstream.
        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    } else {
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    }

    switch (hparams.n_layer()) {
        case 36: type = LLM_TYPE_4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_spark2_5::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // tie_word_embeddings: the checkpoint carries no separate lm_head.
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // q_k_v_proj is fused in the checkpoint; the converter splits it into q/k/v.
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // head-wise attention output gate: Linear(n_embd, n_head), one scalar per head.
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_spark2_5::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.swa_type == LLAMA_SWA_TYPE_STANDARD) {
        return std::make_unique<graph<true>>(*this, params);
    }
    return std::make_unique<graph<false>>(*this, params);
}

template <bool iswa>
llama_model_spark2_5::graph<iswa>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // NOTE: deliberately no `n_embd_head == n_rot` assert -- partial rotary means the rotated
    // width is SMALLER than the head on the full-attention layers (64 of 256).

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<iswa, llm_graph_input_attn_kv_iswa, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn = nullptr;
    if constexpr (iswa) {
        inp_attn = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // Per-layer rope: BOTH the rotated width and the base vary by layer type.
            const int32_t n_rot_l    = (int32_t) hparams.n_rot(il);
            const float   freq_base_l = model.get_rope_freq_base(cparams, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                    n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                    n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // wo = nullptr: the gate multiplies the attention result BEFORE the projection.
            ggml_tensor * attn_out = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);
            cb(attn_out, "attn_out", il);

            // head-wise gate: attn_output * sigmoid(g_proj(x)), one scalar per head
            {
                ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur);
                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "attn_gate_sigmoid", il);

                ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head, n_head, n_tokens);
                ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,     1,           n_head, n_tokens);
                attn_3d  = ggml_mul(ctx0, attn_3d, gate_3d);
                attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head * n_head, n_tokens);
                cb(attn_out, "attn_gated", il);
            }

            cur = build_lora_mm(model.layers[il].wo, attn_out, model.layers[il].wo_s);
            cb(cur, "attn_proj", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // hidden_act is gelu in configuration_spark.py
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_spark2_5::graph<false>;
template struct llama_model_spark2_5::graph<true>;
