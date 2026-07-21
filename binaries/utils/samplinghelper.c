#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef int32_t llama_token;
typedef struct llama_token_data
{
    llama_token id;
    float logit;
    float p;
} llama_token_data;

typedef struct llama_token_data_array
{
    llama_token_data *data;
    size_t size;
    int64_t selected;
    bool sorted;
} llama_token_data_array;

llama_token_data_array *logits_to_curp(float *logits, size_t length)
{
    llama_token_data *data = malloc(sizeof(llama_token_data) * length);
    for (size_t i = 0; i < length; i++)
    {
        data[i].id = i;
        data[i].logit = logits[i];
        data[i].p = 0;
    }
    llama_token_data_array *cur_p = malloc(sizeof(llama_token_data_array));
    cur_p->data = data;
    cur_p->selected = -1;
    cur_p->size = length;
    cur_p->sorted = false;
    return cur_p;
}

llama_token curp_to_token(llama_token_data_array *cur_p)
{
    llama_token token = cur_p->data[cur_p->selected].id;
    free(cur_p->data);
    free(cur_p);
    return token;
}

//