#include "../include/simd.h"
#include <stdlib.h>
#include <stdio.h>
#include <xmmintrin.h>

void es_simd() {
  Kernel *kernel = NULL;
  int **data = NULL;
  FILE *fp = NULL;

  kernel = createKernelSIMD(kernel);
  fp = fopen("8_8_example.raw", "rb");
  if (fp == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data = createPackedDataMatrix(8);
  data = readImageValues(fp, data, 8);
  fclose(fp);

  int i = 0, j = 0;

  for (i = 0; i < 8; i++) {
    printf("fila %i\n", i);
    for (j = 0; j < 8; j+=4) {
      printf("data de 4 en 4: %i %i %i %i\n", data[i][j], data[i][j+1], data[i][j+2], data[i][j+3]);
    }
  }

}

int** readImageValues(FILE *fp, int **data, int dimension) {
  int i = 0;
  for (i = 0; i < dimension; i++)
    fwrite(data[i], sizeof(int), dimension, fp);

  return data;
}

int** createPackedDataMatrix(int dimension) {
  int i = 0;
  int **data = (int**)calloc(dimension, sizeof(int*));
  for (i = 0; i < dimension; i++)
    data[i] = (int*)calloc(dimension, sizeof(int));

  return data;
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
