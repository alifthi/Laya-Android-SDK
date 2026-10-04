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
    
    if(!init_general_params(model, ctx)){
        printf("[Error] Failed to initialize general parameters.");
        return 1;
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

    model->tok_embd = get_tensor(model, "token_embd.weight", n_embd, n_vocab);
    model->tok_norm = get_tensor(model, "token_embd_norm.weight", n_embd);
    model->out_norm = get_tensor(model, "output_norm.weight", n_embd);
    if(!model->tok_embd || !model->tok_norm || !model->out_norm) return false;

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

int load_weights(laya_model * model, const char * path) {
    const gguf_context * ctx = model->gguf;
    const size_t offset = gguf_get_data_offset(ctx);
    const int64_t n = gguf_get_n_tensors(ctx);

    model->buf_w = ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_w, ggml_backend_cpu_buffer_type());
    if(!model->buf_w) {
        printf("[Error] failed to allocate the weight buffer");
        return false;
    }

    ggml_backend_buffer_set_usage(model->buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

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
    
    if(!load_params(model, path)){
        printf("[Error] Failed to load parameters.");
        laya_model_free(model);
        return nullptr;
    }
    if (!load_weights(model, path)) {
        printf("[Error] Failed to load weights.");
        laya_model_free(model);
        return nullptr;
    }

    return model;
}

} // extern "C"
