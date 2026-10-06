#ifndef TOKENIZER_H
#define TOKENIZER_H
#include <stdint.h>
#include <string>
#include <vector>

struct laya_model;


bool load_tokenizer(laya_model * model, const char * path);

int32_t laya_tokenize(const laya_model * model, const char * text, int32_t * tokens, int32_t n_max);

bool tokenize_text(const laya_model * m, const std::string & text, std::vector<int32_t> & out);
#endif // TOKENIZER_H