#include "tokenizer.h"
#include "laya-model.h"
#include "nfc.h"

#include <algorithm>


bool load_tokenizer(laya_model * model, const char * path){

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
        printf("[Error] tokenizer has %d tokens but the embedding only %lld rows", n_vocab, (long long) model->tok_embd->ne[1]);
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

static bool tokenize_piece(const laya_model * m, const std::string & text, std::vector<int32_t> & out) {
    if (text.empty()) {
        return true;
    }
    std::vector<llama_token> buf(text.size() + 8);
    int32_t n = llama_tokenize(m->vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(),
                               /*add_special*/ false, /*parse_special*/ false);
    if (n < 0) {
        buf.resize(-n);
        n = llama_tokenize(m->vocab, text.data(), (int32_t) text.size(), buf.data(), (int32_t) buf.size(), false, false);
    }
    if (n < 0) {
        printf("[Error] tokenization failed");
        return false;
    }
    out.insert(out.end(), buf.begin(), buf.begin() + n);
    return true;
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

static bool tokenize_segment(const laya_model * m, const std::string & seg, std::vector<int32_t> & out) {
    if (seg.empty()) {
        return true;
    }
    if (!m->metaspace) {
        return tokenize_piece(m, m->nfc ? laya::nfc(seg) : seg, out);
    }
    const char * METASPACE = "\xE2\x96\x81";
    std::string s = replace_all(seg, " ", METASPACE);
    if (s.compare(0, 3, METASPACE) != 0) {
        s = METASPACE + s;
    }
    size_t start = 0;
    while (start < s.size()) {
        size_t next = s.find(METASPACE, start + 3);
        if (next == std::string::npos) {
            next = s.size();
        }
        if (!tokenize_piece(m, s.substr(start, next - start), out)) {
            return false;
        }
        start = next;
    }
    return true;
}

// Equivalent of `tok(text, add_special_tokens=False)["input_ids"]`
bool tokenize_text(const laya_model * m, const std::string & text, std::vector<int32_t> & out) {
    out.clear();
    size_t seg_start = 0, i = 0;
    while (i < text.size()) {
        const auto & cands = m->added_by_byte[(unsigned char) text[i]];
        int32_t      match = -1;
        size_t       len   = 0;
        for (const auto & c : cands) {
            if (text.compare(i, c.first.size(), c.first) == 0) {
                match = c.second;
                len   = c.first.size();
                break; 
            }
        }
        if (match < 0) {
            ++i;
            continue;
        }
        if (!tokenize_segment(m, text.substr(seg_start, i - seg_start), out)) {
            return false;
        }
        out.push_back(match);
        i += len;
        seg_start = i;
    }
    return tokenize_segment(m, text.substr(seg_start), out);
}

int32_t laya_tokenize(const laya_model * model, const char * text, int32_t * tokens, int32_t n_max){
        std::vector<int32_t> ids;
    if (!tokenize_text(model, text ? text : "", ids)) {
        return INT32_MIN;
    }
    if ((int32_t) ids.size() > n_max) {
        return -(int32_t) ids.size();
    }
    std::copy(ids.begin(), ids.end(), tokens);
    return (int32_t) ids.size();
}
