#include <xmmintrin.h>
#include <stdio.h>

#ifndef _SIMD_H_
#define _SIMD_H_

typedef struct {
  __m128 __attribute__ ((packed)) top;
  __m128 __attribute__ ((aligned (16))) center;
  __m128 __attribute__ ((aligned (16))) down;
} Kernel;

void es_simd();
Kernel* createKernelSIMD(Kernel *k);

#endif
