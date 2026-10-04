#ifndef LAYA_MODEL_H
#define LAYA_MODEL_H

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <map>

struct enc_layer {
    ggml_tensor * attn_norm = nullptr; 
    ggml_tensor * attn_qkv, * attn_out;
    ggml_tensor * ffn_norm, * ffn_up, * ffn_down;
    int32_t       window     = 0;
    float         rope_theta = 10000.0f;
};

struct head_layer {
    ggml_tensor * attn_q_w, * attn_q_b;
    ggml_tensor * attn_k_w, * attn_k_b;
    ggml_tensor * attn_v_w, * attn_v_b;
    ggml_tensor * attn_out_w, * attn_out_b;
    ggml_tensor * attn_norm_w, * attn_norm_b;
    ggml_tensor * ffn_norm_w, * ffn_norm_b;
    ggml_tensor * ffn_up_w, * ffn_up_b;
    ggml_tensor * ffn_down_w, * ffn_down_b;
};

struct laya_model {
    std::string name;

    // tokenizer (llama.cpp, vocab only)
    llama_model *       tok   = nullptr;
    const llama_vocab * vocab = nullptr;

    // weights: tensor metadata in ctx_w, data in buf_w (mmapped file or a CPU buffer)
    gguf_context *        gguf  = nullptr;
    ggml_context *        ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    void *                map_addr = nullptr;
    size_t                map_size = 0;

    int32_t n_embd       = 0;
    int32_t max_len      = 512;
    int32_t head_max_len = 192;

    // encoder
    int32_t                n_enc_head = 0;
    float                  enc_eps    = 1e-5f;
    ggml_tensor *          tok_embd = nullptr, * tok_norm = nullptr, * out_norm = nullptr;
    std::vector<enc_layer> enc;

    // decision head
    int32_t                 n_head   = 0;
    float                   norm_eps = 1e-5f;
    std::vector<head_layer> layers;
    ggml_tensor *           type_emb = nullptr;
    ggml_tensor *           scorer_norm_w = nullptr, * scorer_norm_b = nullptr;
    ggml_tensor *           scorer_fc_w = nullptr, * scorer_fc_b = nullptr;
    ggml_tensor *           scorer_out_w = nullptr, * scorer_out_b = nullptr;
    ggml_tensor *           act_fc_w = nullptr, * act_fc_b = nullptr, * act_out_w = nullptr, * act_out_b = nullptr;
    int32_t                 n_act = 2;

    bool        metaspace = false; 
    bool        nfc       = false; 
    std::string mask_text;

    std::vector<std::vector<std::pair<std::string, int32_t>>> added_by_byte;
    int32_t     tok_cls = -1, tok_sep = -1, tok_mask = -1, tok_pad = -1;

    float                        temperature[3] = { 1.0f, 1.0f, 1.0f };
    std::map<std::string, float> temperature_by_options;
};

int64_t find_value(const gguf_context * ctx, const char * key);

bool init_general_params(laya_model * model, const gguf_context * ctx);

bool init_encoder(laya_model * model, const gguf_context * ctx);

bool init_decision_head(laya_model * model, const gguf_context * ctx);

#endif // LAYA_MODEL_H
