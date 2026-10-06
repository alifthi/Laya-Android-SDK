#ifndef TOKENIZER_H
#define TOKENIZER_H

struct laya_model;


bool load_tokenizer(laya_model * model, const char * path);

int32_t laya_tokenize(const laya_model * model, const char * text, int32_t * tokens, int32_t n_max);

#endif // TOKENIZER_H