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


int load_params(laya_model * model, const char * path){

    gguf_init_params params = {true, &model->ctx_w};
    
    model->gguf = gguf_init_from_file(path, params);
    if(!model->gguf) {
        printf("[Error] failed to load the gguf context from '%s'", path);
        return false;
    }
    
    const gguf_context * ctx = model->gguf;
    
    if(!init_general_params(model, ctx)){
        printf("[Error] Failed to initialize general parameters.");
        return false;
    }

    if(!init_encoder(model, ctx)){
        printf("[Error] Failed to initialize Encoder.");
        return false;
    }
    
    if(!init_decision_head(model, ctx)){
        printf("[Error] Failed to initialize Decision Head.");
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

    if (!load_weights(model, path)){
        printf("[Error] Failed to load weights.");
        laya_model_free(model);
        return nullptr;
    }

    if (!load_tokenizer(model, path)){
        printf("[Error] Failed to load weights.");
        laya_model_free(model);
        return nullptr;
    }
    
    return model;
}

void laya_model_free(laya_model * model){
    if (!model) {
        return;
    }
    if (model->tok) {
        llama_model_free(model->tok);
    }
    if (model->buf_w) {
        ggml_backend_buffer_free(model->buf_w);
    }
    if (model->ctx_w) {
        ggml_free(model->ctx_w);
    }
    if (model->gguf) {
        gguf_free(model->gguf);
    }
    delete model;
}

} // extern "C"
