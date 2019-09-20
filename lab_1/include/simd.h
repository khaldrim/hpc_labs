#include <x86intrin.h>
#ifndef _SIMD_H_
#define _SIMD_H_

typedef struct {
    int **data __attribute__ ((aligned (16)));
    int **output __attribute__ ((aligned (16)));
} DataSimd;

void es_simd(char *inputFile, char *outputSimd, int nflag, int dflag);
void applySimdKernel(int dimension, int **data, int **output);
void print128_num(__m128i var);
int** copyDataMatrix(int **data, int **output, int dimension);

#endif
