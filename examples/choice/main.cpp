#include "laya.h"
#include <cstdio>

int main() {
    const char * path = "laya.gguf";
    laya_model *   model = laya_model_load(path);
    laya_context * ctx   = laya_context_new(model, laya_context_default_params());

    const char * state   = "Hi, my refund still has not arrived after two weeks.";
    const char * keys[]  = { "billing", "technical", "sales" };
    laya_question q      = { LAYA_QTYPE_CHOICE, "Which team should handle this message?", keys, nullptr, 3 };

    float       probs[3];
    laya_answer a;
    laya_run_inference(ctx, state, &q, probs, nullptr, &a);

    printf("answer: %s (p=%.3f)\n", keys[a.best], probs[a.best]);

    laya_context_free(ctx);
    laya_model_free(model);
}
