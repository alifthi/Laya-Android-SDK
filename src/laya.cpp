// llama.cpp is only used to load the tokenizer from the same file (vocab_only) and run its BPE.
#include "laya.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include <string.h>
struct laya_context {
    laya_model *         model  = nullptr;
    ggml_backend_t       cpu    = nullptr;
    ggml_gallocr_t       galloc = nullptr;

};

extern "C" {

/*
    * To silent ggml logs, except warnings and errors
*/
void silent_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

struct laya_context_params laya_context_default_params(void) {
    laya_context_params p;
    p.n_threads = 0;
    return p;
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

int load_params(laya_model * model, const char * path){

    gguf_init_params params = {true, model->ctx_w};
    
    model->gguf = gguf_init_from_file(path, params);
    if(!model->gguf) {
        printf("[Error] failed to load the gguf context from '%s'", path);
        return false;
    }

    const gguf_context * ctx = model->gguf;
    auto find_value = [&](const char * key) ->int64_t{
        const int64_t id = gguf_find_key(ctx, key);
        if(id<0){
            printf("[Error] failed to find key '%s' in the gguf context", key);
        }
        return id;  
    };
    
    #define GET_U32(param, key)\
        id = find_value(key);\
        if(id<0) return false;\
        param = (int32_t) gguf_get_val_u32(ctx, id);
    #define GET_F32(param, key)\
        id = find_value(key);\
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

    id = find_value("laya.tokenizer.pre");
    if(id<0) return false;
    model->metaspace = strcmp(gguf_get_val_str(ctx, id), "metaspace") == 0;
    
    id = find_value("laya.tokenizer.nfc");
    if(id > 0){
        model->nfc = gguf_get_val_bool(ctx, id);
    }

    id = find_value("laya.tokenizer.mask_text");
    if(id < 0) return false;
    model->mask_text = gguf_get_val_str(ctx, id);

    id = find_value("laya.temperature");
    if(id<0) return false;
    if(gguf_get_arr_n(ctx, id) != 3){
        printf("[Error] invalid temperature array size");
        return false;
    }

    memcpy(model->temperature, gguf_get_arr_data(ctx, id), sizeof(model->temperature));
    id = find_value("laya.temperature_by_options.count");
    if(id<0) return false;

    const int32_t n_temp = (int32_t) gguf_get_val_u32(ctx, id);
    const int64_t temp_keys = find_value("laya.temperature_by_options.key");
    const int64_t temp_vals = find_value("laya.temperature_by_options.values");
    if(temp_keys<0 || temp_vals<0) return false;

    const float * temp_vals_ptr = (const float *) gguf_get_arr_data(ctx, temp_vals);
    for(int32_t i=0 ; i<n_temp; ++i){
        model->temperature_by_options[gguf_get_arr_str(ctx, temp_keys, i)] = temp_vals_ptr[i];
    }

    // Encoder Loader

    const int64_t win_id = find_value("laya.encoder.attention_window");
    const int64_t rope_id = find_value("laya.encoder.rope_theta");

    if(win_id<0 || rope_id<0) return false;

    if((int32_t) gguf_get_arr_n(ctx, win_id) != n_enc_layers || (int32_t) gguf_get_arr_n(ctx, rope_id) != n_enc_layers){
        printf("[Error] laya.encoder.attention_window / rope_theta must have one entry per encoder layer");
        return false;
    }

    const int32_t * windows = (const int32_t *) gguf_get_arr_data(ctx, win_id);
    const float * thetas = (const float *) gguf_get_arr_data(ctx, rope_id);

    const int64_t n_vocab = ggml_get_tensor(model->ctx_w, "token_embd.weight") ? ggml_get_tensor(model->ctx_w, "token_embd.weight")->ne[1] : 0;

    model->token_embd = get_tensor(model, "token_embd.weight", n_embd, n_vocab);
    model->tok_norm = get_tensor(model, "token_embd_norm.weight", n_embd);
    model->out_norm = get_tensor(model, "output_norm.weight", n_embd);
    if(!model->token_embd || !model->tok_norm || !model->out_norm) return false;

    model->enc.resize(n_enc_layers);
    for(int32_t i = 0; i<n_enc_layers; ++i){
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

    //Decision Head

    const int64_t head_ff = ggml_get_tensor(model->ctx, "head.0.ffn_up.weight")?
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

int load_weights(laya_model * model, const char * path) {
    const gguf_context * ctx = model->gguf;
    const size_t offset = gguf_get_data_offset(ctx);
    const int64_t n = gguf_get_n_tensors(ctx);

    model->buf_w = ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_w, ggml_backend_cpu_buffer_type());
    if(!model->buf_w) {
        printf("[Error] failed to allocate the weight buffer");
        return false;
    }

    ggml_backend_buffer_set_uage(model->buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path, "rb");
    if(!f) {
        printf("[Error] failed to open '%s'", path);
        return false;
    }
    bool res = true;
    for(int i = 0; i < n; ++i){

        ggml_tensor * tens = ggml_get_tensor(model->ctx_w, gguf_get_tensor_name(ctx, i));
        const size_t off = offset + gguf_get_tensor_offset(ctx, i);
        
        res = fseeko(f, (off_t) off, SEEK_SET) == 0;
        res = fread(tens->data, 1, ggml_nbytes(tens), f) == ggml_nbytes(tens);

        if(!res) {
            printf("[Error] failed to read tensor '%s'", tens->name);
            break;
        }
    }
    fclose(f);
    return res;
}

laya_model * laya_model_load(const char * path) {
    
    llama_log_set(silent_log_callback, nullptr);
    ggml_log_set(silent_log_callback, nullptr);
    llama_backend_init();

    laya_model * model = new laya_model();
    
    if (!load_params(model, path) || !load_weights(model, path)) {
        laya_model_free(model);
        return nullptr;
    }

    return model;
}

} // extern "C"
