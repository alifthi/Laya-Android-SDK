#include "laya.h"
#include "llama.h"
#include <vector>
#include <cmath>
#include <map>
#include <string.h>


constexpr uint32_t FORMAT_VERSION = 2;

int64_t find_value(const gguf_context * ctx, char * key){
    const int64_t id = gguf_find_key(ctx, key);
    if(id<0){
        printf("[Error] failed to find key '%s' in the gguf context", key);
    }
    return id;  
}

bool init_general_params(laya_model * model, const gguf_context * ctx){

    
    #define GET_U32(param, key)\
        id = find_value(ctx, key);\
        if(id<0) return false;\
        param = (int32_t) gguf_get_val_u32(ctx, id);
    #define GET_F32(param, key)\
        id = find_value(ctx, key);\
        if(id<0) return false;\
        param = gguf_get_val_f32(ctx, id);

    int64_t id;
    id = gguf_find_key(ctx, "laya.format_version");
    if(id<0 || gguf_get_val_u32(ctx, id) != FORMAT_VERSION) {
        printf("[Error] unsupported laya format version");
        return false;
    }
    id = gguf_find_key(ctx, "laya.name");
    if(id>0)
        model->name = gguf_get_val_str(ctx, id);
    
    int32_t n_enc_layers = 0, n_ff = 0, n_layers = 0;
    
    GET_U32(model->n_embd, "laya.embedding_length");
    GET_U32(n_enc_layers, "laya.encoder.layer_count");
    GET_U32(model->n_enc_head, "laya.encoder.head_count");
    GET_U32(n_ff, "laya.encoder.feed_forward_length");
    GET_F32(model->enc_eps, "laya.encoder.layer_norm_epsilon");
    GET_U32(model->n_head, "laya.head.head_count");
    GET_U32(n_layers, "laya.head.layer_count");
    GET_F32(model->norm_eps, "laya.head.layer_norm_epsilon");
    GET_U32(model->n_act, "laya.act.count");
    GET_U32(model->max_len, "laya.max_len");
    GET_U32(model->head_max_len, "laya.head_max_len");
    GET_U32(model->tok_cls, "laya.token.cls");
    GET_U32(model->tok_sep, "laya.token.sep");
    GET_U32(model->tok_mask, "laya.token.mask");
    GET_U32(model->tok_pad, "laya.token.pad");

    #undef GET_U32
    #undef GET_F32
    
    const int64_t n_embd = model->n_embd;
    if (n_embd <= 0 || model->n_enc_head <= 0 || n_embd % model->n_enc_head != 0 || model->n_head <= 0 || n_embd % model->n_head != 0) {
        printf("invalid embedding / head sizes in model file");
        return false;
    }

    id = find_value(ctx, "laya.tokenizer.pre");
    if(id<0) return false;
    model->metaspace = strcmp(gguf_get_val_str(ctx, id), "metaspace") == 0;
    
    id = find_value(ctx, "laya.tokenizer.nfc");
    if(id > 0){
        model->nfc = gguf_get_val_bool(ctx, id);
    }

    id = find_value(ctx, "laya.tokenizer.mask_text");
    if(id < 0) return false;
    model->mask_text = gguf_get_val_str(ctx, id);

    id = find_value(ctx, "laya.temperature");
    if(id<0) return false;
    if(gguf_get_arr_n(ctx, id) != 3){
        printf("[Error] invalid temperature array size");
        return false;
    }

    memcpy(model->temperature, gguf_get_arr_data(ctx, id), sizeof(model->temperature));
    id = find_value(ctx, "laya.temperature_by_options.count");
    if(id<0) return false;

    const int32_t n_temp = (int32_t) gguf_get_val_u32(ctx, id);
    const int64_t temp_keys = find_value(ctx, "laya.temperature_by_options.key");
    const int64_t temp_vals = find_value(ctx, "laya.temperature_by_options.values");
    if(temp_keys<0 || temp_vals<0) return false;

    const float * temp_vals_ptr = (const float *) gguf_get_arr_data(ctx, temp_vals);
    for(int32_t i=0 ; i<n_temp; ++i){
        model->temperature_by_options[gguf_get_arr_str(ctx, temp_keys, i)] = temp_vals_ptr[i];
    }
    return true;
}
