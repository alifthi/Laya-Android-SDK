// llama.cpp is only used to load the tokenizer from the same file (vocab_only) and run its BPE.
#include "laya.h"
#include "llama.h"
#include "ggml.h"
#include "laya-model.h"
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

int load_weights(laya_model * model, const char * path) {
    // Stub function to load weights.
    return 0;
}

laya_model * laya_model_load(const char * path) {
    
    llama_log_set(silent_log_callback, nullptr);
    ggml_log_set(silent_log_callback, nullptr);
    llama_backend_init();

    laya_model * model = new laya_model();
    
    if (!load_weights(model, path)) {
        laya_model_free(model);
        return nullptr;
    }

    return model;
}

} // extern "C"
