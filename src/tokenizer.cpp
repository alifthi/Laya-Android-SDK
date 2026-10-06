#include "tokenizer.h"
#include <sort>


static bool load_tokenizer(laya_model * model, const char * path){

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    model->tok = llama_load_model_from_file(path, params);

    if(!model->tok){
        printf("[Error] Failed to load tokenizer.");
        return false;
    }

    model->vocab = llama_model_get_vocab(model->tok);

    const int32_t n_vocab = llama_vocab_n_tokens(model->vocab);

    if (n_vocab > model->tok_embd->ne[1]) {
        printf("[Error] tokenizer has %d tokens but the embedding only %lld rows", n_vocab, (long long) m->tok_embd->ne[1]);
        return false;
    }

    model->added_by_byte.assign(256, {});
    for (int32_t id = 0; id < n_vocab; ++id) {
        const auto attr = llama_vocab_get_attr(model->vocab, id);
        if (attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
            const std::string t = llama_vocab_get_text(model->vocab, id);
            if (!t.empty()) {
                model->added_by_byte[(unsigned char) t[0]].emplace_back(t, id);
            }
        }
    }

    for (auto & b : model->added_by_byte) {
        std::stable_sort(b.begin(), b.end(), [](const auto & a, const auto & c) { return a.first.size() > c.first.size(); });
    }
    for (int32_t t : { model->tok_cls, model->tok_sep, model->tok_mask, model->tok_pad }) {
        if (t < 0 || t >= n_vocab) {
            printf("[Error] special token id %d is outside the vocabulary", t);
            return false;
        }
    }
    return true;

}