#ifndef LAYA_H
#define LAYA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct laya_model   laya_model;
typedef struct laya_context laya_context;

enum laya_qtype {
    LAYA_QTYPE_CHOICE = 0,
    LAYA_QTYPE_SCORE  = 1,
    LAYA_QTYPE_NOUL   = 2,
};

struct laya_context_params {
    int32_t n_threads;
};

struct laya_question {
    enum laya_qtype      type;
    const char *         instructions;
    const char * const * keys;
    const char * const * descriptions;
    int32_t              n_options;
};

struct laya_answer {
    int32_t n_options;
    int32_t best;            // index of the most probable option
    float   value;           // choice: p[best]; score: expected level sum(i * p[i]); noul: P(true)
    float   confidence;      // 1 - normalized entropy of the calibrated distribution
};

#ifdef __cplusplus
}
#endif

#endif // LAYA_H
