#include <stdint.h>
#include <stdlib.h>
#include <math.h>

double entropy_of_logits(float *logits, size_t length)
{
    double divisor = 0;
    for (size_t i = 0; i < length; i++)
    {
        divisor += exp(logits[i]);
    }
    float *p = malloc(sizeof(float) * length);
    for (size_t i = 0; i < length; i++)
    {
        p[i] = exp(logits[i]) / divisor;
    }
    double entropy = 0;
    for (size_t i = 0; i < length; i++)
    {
        entropy += -p[i] * log(p[i]);
    }
    free(p);
    return entropy;
}

//