#include <smmintrin.h> /* _mm_max_epi32 (Use SSE4.2 Flag, includes SSE2) */
#include <stdint.h> /* uint32_t type */
#include <stdio.h> /* printf() */
#ifndef _SIMD_H_
#define _SIMD_H_

void es_simd(int **data, int **output, int dimension);
void print128_num(__m128i var);

#endif
