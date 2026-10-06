#include "laya.h"
#include "laya-model.h"
#include "llama.h"
#include "tokenizer.h"

#include <vector>
#include <cmath>
#include <map>
#include <string.h>

constexpr uint32_t FORMAT_VERSION = 2;

int64_t find_value(const gguf_context * ctx, const char * key){
    const int64_t id = gguf_find_key(ctx, key);
    if(id<0){
        printf("[Error] failed to find key '%s' in the gguf context", key);
    }
    return id;  
}

static ggml_tensor * get_tensor(laya_model * model, const std::string & name, int64_t ne0, int64_t ne1 = 1, bool required = true){

    ggml_tensor *t = ggml_get_tensor(model->ctx_w, name.c_str());
    if(!t){
        if(required)
            printf("[Error] model file is missing tensor %s", name.c_str());
        return nullptr;
    }
    if (t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != 1 || t->ne[3] != 1) {
        printf("tensor '%s' has shape [%lld, %lld], expected [%lld, %lld]", name.c_str(), (long long) t->ne[0],
                  (long long) t->ne[1], (long long) ne0, (long long) ne1);
        return nullptr;
    }
    return t;
    
}

bool init_general_params(laya_model * model, const gguf_context * ctx){

    
    #define GET_U32(param, key)\
        id = find_value(ctx, key);\
        if(id<0) return false;\
        param = (int32_t) gguf_get_val_u32(ctx, id);
    #define GET_F32(param, key)\
        id = find_value(ctx, key);\
        if(id<0) return false;\
        param = gguf_get_val_f32(ctx, id);

    int64_t id;
    id = gguf_find_key(ctx, "laya.format_version");
    if(id<0 || gguf_get_val_u32(ctx, id) != FORMAT_VERSION) {
        printf("[Error] unsupported laya format version");
        return false;
    }
    id = gguf_find_key(ctx, "laya.name");
    if(id>0)
        model->name = gguf_get_val_str(ctx, id);
    
    int32_t n_enc_layers = 0, n_ff = 0, n_layers = 0;
    
    GET_U32(model->n_embd, "laya.embedding_length");
    GET_U32(n_enc_layers, "laya.encoder.layer_count");
    GET_U32(model->n_enc_head, "laya.encoder.head_count");
    GET_U32(n_ff, "laya.encoder.feed_forward_length");
    GET_F32(model->enc_eps, "laya.encoder.layer_norm_epsilon");
    GET_U32(model->n_head, "laya.head.head_count");
    GET_U32(n_layers, "laya.head.layer_count");
    GET_F32(model->norm_eps, "laya.head.layer_norm_epsilon");
    GET_U32(model->n_act, "laya.act.count");
    GET_U32(model->max_len, "laya.max_len");
    GET_U32(model->head_max_len, "laya.head_max_len");
    GET_U32(model->tok_cls, "laya.token.cls");
    GET_U32(model->tok_sep, "laya.token.sep");
    GET_U32(model->tok_mask, "laya.token.mask");
    GET_U32(model->tok_pad, "laya.token.pad");

    #undef GET_U32
    #undef GET_F32
    
    const int64_t n_embd = model->n_embd;
    if (n_embd <= 0 || model->n_enc_head <= 0 || n_embd % model->n_enc_head != 0 || model->n_head <= 0 || n_embd % model->n_head != 0) {
        printf("invalid embedding / head sizes in model file");
        return false;
    }

    id = find_value(ctx, "laya.tokenizer.pre");
    if(id<0) return false;
    model->metaspace = strcmp(gguf_get_val_str(ctx, id), "metaspace") == 0;
    
    id = find_value(ctx, "laya.tokenizer.nfc");
    if(id > 0){
        model->nfc = gguf_get_val_bool(ctx, id);
    }

    id = find_value(ctx, "laya.tokenizer.mask_text");
    if(id < 0) return false;
    model->mask_text = gguf_get_val_str(ctx, id);

    id = find_value(ctx, "laya.temperature");
    if(id<0) return false;
    if(gguf_get_arr_n(ctx, id) != 3){
        printf("[Error] invalid temperature array size");
        return false;
    }

    memcpy(model->temperature, gguf_get_arr_data(ctx, id), sizeof(model->temperature));
    id = find_value(ctx, "laya.temperature_by_options.count");
    if(id<0) return false;

    const int32_t n_temp = (int32_t) gguf_get_val_u32(ctx, id);
    const int64_t temp_keys = find_value(ctx, "laya.temperature_by_options.keys");
    const int64_t temp_vals = find_value(ctx, "laya.temperature_by_options.values");
    if(temp_keys<0 || temp_vals<0) return false;

    const float * temp_vals_ptr = (const float *) gguf_get_arr_data(ctx, temp_vals);
    for(int32_t i=0 ; i<n_temp; ++i){
        model->temperature_by_options[gguf_get_arr_str(ctx, temp_keys, i)] = temp_vals_ptr[i];
    }
    return true;
}

bool init_encoder(laya_model * model, const gguf_context * ctx){

    const int64_t win_id = find_value(ctx, "laya.encoder.attention_window");
    const int64_t rope_id = find_value(ctx,"laya.encoder.rope_theta");

    if(win_id<0 || rope_id<0) return false;
    int32_t n_enc_layers = 0, n_ff = 0;
    
    int64_t id = find_value(ctx, "laya.encoder.layer_count");
    if(id<0) return false;
    n_enc_layers = (int32_t) gguf_get_val_u32(ctx, id);

    id = find_value(ctx, "laya.encoder.feed_forward_length");
    if(id<0) return false;
    n_ff = (int32_t) gguf_get_val_u32(ctx, id);
    

    if((int32_t) gguf_get_arr_n(ctx, win_id) != n_enc_layers || (int32_t) gguf_get_arr_n(ctx, rope_id) != n_enc_layers){
        printf("[Error] laya.encoder.attention_window / rope_theta must have one entry per encoder layer");
        return false;
    }

    const int32_t * windows = (const int32_t *) gguf_get_arr_data(ctx, win_id);
    const float * thetas = (const float *) gguf_get_arr_data(ctx, rope_id);

    const int64_t n_vocab = ggml_get_tensor(model->ctx_w, "token_embd.weight") ? ggml_get_tensor(model->ctx_w, "token_embd.weight")->ne[1] : 0;
    
    const int64_t n_embd = model->n_embd;
    if (n_embd <= 0 || model->n_enc_head <= 0 || n_embd % model->n_enc_head != 0 || model->n_head <= 0 || n_embd % model->n_head != 0) {
        printf("invalid embedding / head sizes in model file");
        return false;
    }

    model->tok_embd = get_tensor(model, "token_embd.weight", n_embd, n_vocab);
    model->tok_norm = get_tensor(model, "token_embd_norm.weight", n_embd);
    model->out_norm = get_tensor(model, "output_norm.weight", n_embd);
    if(!model->tok_embd || !model->tok_norm || !model->out_norm) return false;

    model->enc.resize(n_enc_layers);
    
    for(int32_t i = 0; i<n_enc_layers; i++){
        enc_layer & L = model->enc[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        L.window = windows[i];
        L.rope_theta = thetas[i];
        L.attn_norm  = get_tensor(model, p + "attn_norm.weight", n_embd, 1, /*required*/ i != 0);
        if (i != 0 && !L.attn_norm) return false;
        if (!(L.attn_qkv = get_tensor(model, p + "attn_qkv.weight", n_embd, 3 * n_embd))) return false;
        if (!(L.attn_out = get_tensor(model, p + "attn_output.weight", n_embd, n_embd))) return false;
        if (!(L.ffn_norm = get_tensor(model, p + "ffn_norm.weight", n_embd))) return false;
        if (!(L.ffn_up = get_tensor(model, p + "ffn_up.weight", n_embd, 2 * n_ff))) return false;
        if (!(L.ffn_down = get_tensor(model, p + "ffn_down.weight", n_ff, n_embd))) return false;
    }
    return true;
}

bool init_decision_head(laya_model * model, const gguf_context * ctx){

    int32_t n_layers = 0;
    const int64_t n_embd = model->n_embd;
    int32_t id = find_value(ctx, "laya.head.layer_count");
    if(id<0) return false;
    n_layers = (int32_t) gguf_get_val_u32(ctx, id);
    
    const int64_t head_ff = ggml_get_tensor(model->ctx_w, "head.0.ffn_up.weight")?
                                ggml_get_tensor(model->ctx_w, "head.0.ffn_up.weight")->ne[1] : 4 * n_embd;;
    model->layers.resize(n_layers);
    for(int32_t i = 0; i<n_layers; ++i){
        head_layer & L = model->layers[i];
        const std::string p = "head." + std::to_string(i) + ".";
        struct {
            ggml_tensor ** dst;
            const char *   name;
            int64_t        ne0, ne1;
        } ts[] = {
            { &L.attn_q_w, "attn_q.weight", n_embd, n_embd }, { &L.attn_q_b, "attn_q.bias", n_embd, 1 },
            { &L.attn_k_w, "attn_k.weight", n_embd, n_embd }, { &L.attn_k_b, "attn_k.bias", n_embd, 1 },
            { &L.attn_v_w, "attn_v.weight", n_embd, n_embd }, { &L.attn_v_b, "attn_v.bias", n_embd, 1 },
            { &L.attn_out_w, "attn_out.weight", n_embd, n_embd }, { &L.attn_out_b, "attn_out.bias", n_embd, 1 },
            { &L.attn_norm_w, "attn_norm.weight", n_embd, 1 }, { &L.attn_norm_b, "attn_norm.bias", n_embd, 1 },
            { &L.ffn_norm_w, "ffn_norm.weight", n_embd, 1 }, { &L.ffn_norm_b, "ffn_norm.bias", n_embd, 1 },
            { &L.ffn_up_w, "ffn_up.weight", n_embd, head_ff }, { &L.ffn_up_b, "ffn_up.bias", head_ff, 1 },
            { &L.ffn_down_w, "ffn_down.weight", head_ff, n_embd }, { &L.ffn_down_b, "ffn_down.bias", n_embd, 1 },
        };
        for (const auto & t : ts) {
            if (!(*t.dst = get_tensor(model, p + t.name, t.ne0, t.ne1))) return false;
        }
    }
    if (!(model->type_emb = get_tensor(model, "type_emb.weight", n_embd, 3))) return false;
    if (!(model->scorer_norm_w = get_tensor(model, "scorer.norm.weight", n_embd))) return false;
    if (!(model->scorer_norm_b = get_tensor(model, "scorer.norm.bias", n_embd))) return false;
    if (!(model->scorer_fc_w = get_tensor(model, "scorer.fc.weight", n_embd, n_embd))) return false;
    if (!(model->scorer_fc_b = get_tensor(model, "scorer.fc.bias", n_embd))) return false;
    if (!(model->scorer_out_w = get_tensor(model, "scorer.out.weight", n_embd))) return false;
    if (!(model->scorer_out_b = get_tensor(model, "scorer.out.bias", 1))) return false;

    const int64_t act_hidden = ggml_get_tensor(model->ctx_w, "act.fc.weight") ?
                                   ggml_get_tensor(model->ctx_w, "act.fc.weight")->ne[1] : 0;
    if (!(model->act_fc_w = get_tensor(model, "act.fc.weight", n_embd + 4, act_hidden))) return false;
    if (!(model->act_fc_b = get_tensor(model, "act.fc.bias", act_hidden))) return false;
    if (!(model->act_out_w = get_tensor(model, "act.out.weight", act_hidden, model->n_act))) return false;
    if (!(model->act_out_b = get_tensor(model, "act.out.bias", model->n_act))) return false;
    if (model->type_emb->type != GGML_TYPE_F32) {
        printf("[Error] type_emb.weight must be f32");
        return false;
    }
    return true;
}

// y = W x + b; W is [n_in, n_out] in ggml order, x is [n_in, n]
static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, x, b) : x;
}

static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    if (w) {
        x = ggml_mul(ctx, x, w);
    }
    return b ? ggml_add(ctx, x, b) : x;
}

// q: [hd, nh, Lq], k/v: [hd, nh, Lk], mask: [Lk, Lq] additive or null -> [d, Lq]
static ggml_tensor * attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                               ggml_tensor * mask, int64_t d) {
    const int64_t hd = q->ne[0], Lq = q->ne[2];
    q = ggml_permute(ctx, q, 0, 2, 1, 3);                       // [hd, Lq, nh]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);                       // [hd, Lk, nh]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));       // [Lk, hd, nh]
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                 // [Lk, Lq, nh]
    kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f / sqrtf((float) hd), 0.0f);
    ggml_tensor * o = ggml_mul_mat(ctx, v, kq);                 // [hd, Lq, nh]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));       // [hd, nh, Lq]
    return ggml_reshape_2d(ctx, o, d, Lq);
}

ggml_cgraph * build_graph(laya_context * lc, int32_t L, int32_t k, int qtype, float T, graph_io & io) {
    const laya_model * m = lc->model;
    const int64_t      d = m->n_embd;

    const size_t buf_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
    lc->graph_buf.resize(buf_size);
    ggml_init_params ip = { buf_size, lc->graph_buf.data(), /*no_alloc*/ true };
    ggml_context *   ctx = ggml_init(ip);
    ggml_cgraph *    gf  = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    auto input = [&](ggml_tensor * t, const char * name) {
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    io.ids   = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L), "inp_ids");
    io.pos   = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L), "inp_pos");
    io.rows  = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1 + k), "inp_rows");
    io.act_k = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), "inp_act_k");
    io.masks.clear();
    for (const enc_layer & ly : m->enc) {
        if (ly.window > 0 && ly.window < L - 1 && !io.masks.count(ly.window)) {
            io.masks[ly.window] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, L), "inp_swa_mask");
        }
    }

    // ---- encoder (ModernBERT): pre-norm, fused QKV + NeoX RoPE, GeGLU (exact erf) FFN
    const int64_t nh = m->n_enc_head, hd = d / nh;
    ggml_tensor * x  = ggml_get_rows(ctx, m->tok_embd, io.ids); // [d, L]
    x                = layer_norm(ctx, x, m->tok_norm, nullptr, m->enc_eps);
    for (const enc_layer & ly : m->enc) {
        ggml_tensor * h   = ly.attn_norm ? layer_norm(ctx, x, ly.attn_norm, nullptr, m->enc_eps) : x;
        ggml_tensor * qkv = ggml_mul_mat(ctx, ly.attn_qkv, h); // [3d, L]
        const size_t  es  = ggml_element_size(qkv);
        ggml_tensor * q   = ggml_view_3d(ctx, qkv, hd, nh, L, hd * es, qkv->nb[1], 0);
        ggml_tensor * kk  = ggml_view_3d(ctx, qkv, hd, nh, L, hd * es, qkv->nb[1], d * es);
        ggml_tensor * v   = ggml_view_3d(ctx, qkv, hd, nh, L, hd * es, qkv->nb[1], 2 * d * es);
        q  = ggml_rope_ext(ctx, q, io.pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, ly.rope_theta, 1.0f, 0.0f, 1.0f,
                           0.0f, 0.0f);
        kk = ggml_rope_ext(ctx, kk, io.pos, nullptr, (int) hd, GGML_ROPE_TYPE_NEOX, 0, ly.rope_theta, 1.0f, 0.0f, 1.0f,
                           0.0f, 0.0f);
        const auto    mi   = io.masks.find(ly.window);
        ggml_tensor * mask = mi != io.masks.end() ? mi->second : nullptr;
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, ly.attn_out, attention(ctx, q, kk, v, mask, d)));

        ggml_tensor * f = layer_norm(ctx, x, ly.ffn_norm, nullptr, m->enc_eps);
        f               = ggml_geglu_erf(ctx, ggml_mul_mat(ctx, ly.ffn_up, f));
        x               = ggml_add(ctx, x, ggml_mul_mat(ctx, ly.ffn_down, f));
    }
    x = layer_norm(ctx, x, m->out_norm, nullptr, m->enc_eps);
    x = ggml_add(ctx, x, ggml_view_1d(ctx, m->type_emb, d, (size_t) qtype * m->type_emb->nb[1]));

    // ---- decision head: nn.TransformerEncoderLayer(norm_first=True, relu). The last layer only computes queries
    // and the FFN for the rows that are read ([CLS] + option markers); attention still sees every position.
    const int64_t hnh = m->n_head, hhd = d / hnh;
    const int     n_layers = (int) m->layers.size();
    for (int il = 0; il < n_layers; ++il) {
        const head_layer & ly   = m->layers[il];
        const bool         last = il == n_layers - 1;

        ggml_tensor * xn = layer_norm(ctx, x, ly.attn_norm_w, ly.attn_norm_b, m->norm_eps);
        ggml_tensor * K  = linear(ctx, xn, ly.attn_k_w, ly.attn_k_b);
        ggml_tensor * V  = linear(ctx, xn, ly.attn_v_w, ly.attn_v_b);
        ggml_tensor * xq = last ? ggml_get_rows(ctx, xn, io.rows) : xn;
        ggml_tensor * xr = last ? ggml_get_rows(ctx, x, io.rows) : x;
        ggml_tensor * Q  = linear(ctx, xq, ly.attn_q_w, ly.attn_q_b);
        const int64_t M  = Q->ne[1];

        ggml_tensor * a = attention(ctx, ggml_reshape_3d(ctx, Q, hhd, hnh, M), ggml_reshape_3d(ctx, K, hhd, hnh, L),
                                    ggml_reshape_3d(ctx, V, hhd, hnh, L), nullptr, d);
        x = ggml_add(ctx, xr, linear(ctx, a, ly.attn_out_w, ly.attn_out_b));

        ggml_tensor * f = layer_norm(ctx, x, ly.ffn_norm_w, ly.ffn_norm_b, m->norm_eps);
        f               = ggml_relu(ctx, linear(ctx, f, ly.ffn_up_w, ly.ffn_up_b));
        x               = ggml_add(ctx, x, linear(ctx, f, ly.ffn_down_w, ly.ffn_down_b));
    }
    if (n_layers == 0) {
        x = ggml_get_rows(ctx, x, io.rows);
    }
    // x: [d, 1 + k]; row 0 = [CLS], rows 1.. = option markers

    // ---- scorer: LayerNorm -> Linear -> GELU -> Linear
    ggml_tensor * s = layer_norm(ctx, x, m->scorer_norm_w, m->scorer_norm_b, m->norm_eps);
    s               = ggml_gelu_erf(ctx, linear(ctx, s, m->scorer_fc_w, m->scorer_fc_b));
    s               = linear(ctx, s, m->scorer_out_w, m->scorer_out_b); // [1, 1 + k]
    io.logits       = s;
    ggml_set_name(s, "logits");
    ggml_set_output(s);

    ggml_tensor * lg = ggml_cont(ctx, ggml_view_1d(ctx, s, k, s->nb[1])); // option logits [k]
    io.probs         = ggml_soft_max_ext(ctx, lg, nullptr, 1.0f / T, 0.0f);
    ggml_set_name(io.probs, "probs");
    ggml_set_output(io.probs);

    // ---- act head: [pooled [CLS], top1, top1 - top2, normalized entropy, k / 255] of the uncalibrated answer
    ggml_tensor * p      = ggml_soft_max(ctx, lg);
    ggml_tensor * sorted = ggml_get_rows(ctx, ggml_reshape_2d(ctx, p, 1, k), ggml_argsort(ctx, p, GGML_SORT_ORDER_DESC));
    sorted               = ggml_reshape_1d(ctx, sorted, k);
    ggml_tensor * top1   = ggml_view_1d(ctx, sorted, 1, 0);
    ggml_tensor * margin = k > 1 ? ggml_sub(ctx, top1, ggml_view_1d(ctx, sorted, 1, sorted->nb[0])) : top1;
    ggml_tensor * ent    = ggml_sum(ctx, ggml_mul(ctx, p, ggml_log(ctx, ggml_clamp(ctx, p, 1e-9f, INFINITY))));
    ent                  = ggml_scale(ctx, ent, -1.0f / logf((float) std::max(k, 2)));
    ggml_tensor * feats  = ggml_concat(ctx, ggml_concat(ctx, ggml_cont(ctx, top1), ggml_cont(ctx, margin), 0),
                                       ggml_concat(ctx, ent, io.act_k, 0), 0);
    ggml_tensor * pooled = ggml_view_1d(ctx, x, d, 0);
    ggml_tensor * a      = ggml_concat(ctx, ggml_cont(ctx, pooled), feats, 0); // [d + 4]
    a                    = ggml_gelu_erf(ctx, linear(ctx, a, m->act_fc_w, m->act_fc_b));
    io.act               = ggml_soft_max(ctx, linear(ctx, a, m->act_out_w, m->act_out_b));
    ggml_set_name(io.act, "act");
    ggml_set_output(io.act);

    ggml_build_forward_expand(gf, io.logits);
    ggml_build_forward_expand(gf, io.probs);
    ggml_build_forward_expand(gf, io.act);
    ggml_free(ctx);
    return gf;
}
