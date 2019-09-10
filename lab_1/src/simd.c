#include "../include/simd.h"
#include "../include/functions.h"
#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>

void es_simd() {
  Kernel *kernel = NULL;
  float **data = NULL;
  FILE *fp = NULL;

  kernel = createKernelSIMD(kernel);
  fp = fopen("8_8_example.raw", "rb");
  if (fp == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data = createDataMatrix(data, 8);
  data = readImageValues(fp, data, 8);
  fclose(fp);

  int i = 0, j = 0;

  __m128 v;
  for (i = 1; i < 8 - 2; i++) {
    printf("fila %i ", i);
    for (j = 0; j < 8 - 2; j+=4) {
      printf("columna %i \n", j);

        if (j == 4) {
          printf("hola\n");
          // float __attribute__ ((aligned (16))) partial[4] = {data[i][j], data[i][j+1], -100, 100};
          // v = _mm_load_ps(partial);
          // printf("v: %f %f %f %f\n", v[0], v[1], v[2], v[3]);
        } else {
          // printf("%f\n", data[i][j]);
          printf("%i %i \n", i, j);
          v = _mm_load_ps(&data[i][j]);
          // partial = _mm_add_ps(kernel->top, v);
          printf("v: %f %f %f %f\n", v[0], v[1], v[2], v[3]);
        }

      // printf("top: %f %f %f %f\n", partial[0], partial[1], partial[2], partial[3]);
    }
  }

  printf("\n\n");
  printResult(8, data);
}

Kernel* createKernelSIMD(Kernel *k) {
  float x[4] __attribute__ ((aligned (16)))= {0, 1, 0, 0};
  float y[4] __attribute__ ((aligned (16)))= {1, 1, 1, 0};

  k = (Kernel*)malloc(sizeof(Kernel));

  k->top    = _mm_load_ps(x);
  k->center = _mm_load_ps(y);
  k->down    = _mm_load_ps(x);

  return k;
}
