// llama.cpp is only used to load the tokenizer from the same file (vocab_only) and run its BPE.
#include "laya.h"
#include "laya-model.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "tokenizer.h"

#include <string.h>
#include <algorithm>
#include <cmath>
#include <thread>

struct laya_context {
    laya_model *         model  = nullptr;
    ggml_backend_t       cpu    = nullptr;
    ggml_gallocr_t       galloc = nullptr;

};


extern "C" {

const char * laya_model_name(const laya_model * m) { return m->name.c_str(); }
int32_t laya_model_max_len(const laya_model * m) { return m->max_len; }
int32_t laya_model_head_max_len(const laya_model * m) { return m->head_max_len; }
int32_t laya_model_n_embd(const laya_model * m) { return m->n_embd; }

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

laya_context * laya_create_context(laya_model * model, struct laya_context_params params){

    int n_threads = params.n_threads;
    if(n_threads < 0){
        n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    }

    laya_context * ctx = new laya_context();
    ctx->model = model;
    ctx->cpu = ggml_backend_cpu_init();
    if(ctx->cpu){
        printf("[Error] failed to initialize the CPU backend.");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->cpu, n_threads);
    ctx->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    return ctx;
}

void laya_context_free(laya_context * ctx){
    if(!ctx) return;
    if(ctx->galloc) ggml_gallocr_free(ctx->galloc);
    if(ctx->cpu) ggml_backend_free(ctx->cpu);
    delete ctx;
}

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return s;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

static bool render_options(const laya_question * q, std::vector<std::string> & opts) {
    opts.clear();
    auto desc = [&](int i) -> const char * {
        return q->descriptions ? q->descriptions[i] : nullptr;
    };
    switch (q->type) {
        case LAYA_QTYPE_CHOICE:
            if (q->n_options < 1 || !q->keys) {
                printf("[Error] choice question needs at least one option key");
                return false;
            }
            for (int i = 0; i < q->n_options; ++i) {
                if (!q->keys[i]) {
                    printf("[Error] choice option %d has no key", i);
                    return false;
                }
                const char * d = desc(i);
                opts.push_back(d && *d ? std::string(q->keys[i]) + ": " + d : std::string(q->keys[i]));
            }
            return true;
        case LAYA_QTYPE_SCORE:
            if (q->n_options < 1 || !q->descriptions) {
                printf("[Error] score question needs at least one level description");
                return false;
            }
            for (int i = 0; i < q->n_options; ++i) {
                const char * d = desc(i);
                opts.push_back("level " + std::to_string(i) + ": " + (d ? d : "None"));
            }
            return true;
        case LAYA_QTYPE_NOUL: {
            if (q->n_options != 0 && q->n_options != 2) {
                printf("[Error] noul question takes 0 or 2 descriptions (false, true), got %d", q->n_options);
                return false;
            }
            const char * f = q->n_options == 2 ? desc(0) : nullptr;
            const char * t = q->n_options == 2 ? desc(1) : nullptr;
            opts.push_back(std::string("false: ") + (f && *f ? f : "no, the statement does not hold"));
            opts.push_back(std::string("true: ") + (t && *t ? t : "yes, the statement holds"));
            return true;
        }
    }
    printf("[Error] unknown question type %d", (int) q->type);
    return false;
}

// [CLS] <type> question: instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]  (build_sequence)
static bool build_sequence(const laya_model * m, const char * state, const laya_question * q,
                           std::vector<int32_t> & ids, std::vector<int32_t> & markers) {
    std::vector<std::string> opts;
    if (!render_options(q, opts)) {
        return false;
    }
    const std::string & mt = m->mask_text;
    const std::string   ins = replace_all(q->instructions ? q->instructions : "", mt, " ");

    std::vector<int32_t> head_ids;
    const char * const QTYPE_NAMES[] = { "choice", "score", "noul" };
    if (!tokenize_text(m, std::string(QTYPE_NAMES[q->type]) + " question: " + ins, head_ids)) {
        return false;
    }

    std::vector<std::vector<int32_t>> opt_ids;
    std::vector<int32_t>              tmp;
    size_t                            total = 0;
    for (const auto & o : opts) {
        if (!tokenize_text(m, " " + replace_all(o, mt, " "), tmp)) {
            return false;
        }
        std::vector<int32_t> v{ m->tok_mask };
        v.insert(v.end(), tmp.begin(), tmp.begin() + std::min<size_t>(tmp.size(), 48));
        total += v.size();
        opt_ids.push_back(std::move(v));
    }
    int64_t opt_budget = (int64_t) m->head_max_len - (int64_t) total;
    if (opt_budget < 16) { // too many / too long options: shrink every option text evenly
        const int64_t per = std::max<int64_t>(4, (m->head_max_len - 16) / std::max<int64_t>(1, opt_ids.size()));
        total = 0;
        for (auto & o : opt_ids) {
            if ((int64_t) o.size() > per) {
                o.resize(per);
            }
            total += o.size();
        }
        opt_budget = (int64_t) m->head_max_len - (int64_t) total;
    }
    const int64_t head_keep = std::max<int64_t>(8, opt_budget);
    if ((int64_t) head_ids.size() > head_keep) {
        head_ids.resize(head_keep);
    }

    ids.clear();
    markers.clear();
    ids.push_back(m->tok_cls);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(m->tok_sep);
    for (const auto & o : opt_ids) {
        markers.push_back((int32_t) ids.size());
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(m->tok_sep);

    const int64_t room = std::max<int64_t>(0, (int64_t) m->max_len - (int64_t) ids.size() - 1);
    std::vector<int32_t> st;
    if (!tokenize_text(m, replace_all(state ? state : "", mt, " "), st)) {
        return false;
    }
    if ((int64_t) st.size() > room) {
            st.resize(room);
    }
    ids.insert(ids.end(), st.begin(), st.end());
    ids.push_back(m->tok_sep);
    if ((int64_t) ids.size() > m->max_len) {
        ids.resize(m->max_len);
    }
    markers.erase(std::remove_if(markers.begin(), markers.end(), [&](int32_t p) { return p >= m->max_len; }),
                  markers.end());
    if (markers.size() != opts.size()) {
        printf("[Error] options do not fit in head_max_len=%d tokens", m->head_max_len);
        return false;
    }
    return true;
}


int32_t laya_build_sequence(const laya_model * model, const char * state, const struct laya_question * q,
                            int32_t * tokens, int32_t n_max, int32_t * markers) {
    std::vector<int32_t> ids, mk;
    if (!build_sequence(model, state, q, ids, mk)) {
        return INT32_MIN;
    }
    if ((int32_t) ids.size() > n_max) {
        return -(int32_t) ids.size();
    }
    std::copy(ids.begin(), ids.end(), tokens);
    if (markers) {
        std::copy(mk.begin(), mk.end(), markers);
    }
    return (int32_t) ids.size();
}
} // extern "C"
