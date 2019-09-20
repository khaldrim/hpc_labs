#include "simd.h"
#include "functions.h"
#include <stdlib.h>
#include <stdio.h>

// #include <immintrin.h>
// #include <smmintrin.h> /* _mm_max_epi32() */
// #include <emmintrin.h>
#include <stdint.h>

void es_simd(char *inputFile, char *outputSimd, int nflag, int dflag) {
  FILE *image_raw = NULL;

  image_raw = fopen(inputFile, "rb");
  if (image_raw == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }
  
  DataSimd *d = (DataSimd*)malloc(sizeof(DataSimd));

  d->data   = createDataMatrix(d->data, nflag);
  d->output = createDataMatrix(d->output, nflag);
  d->output = copyDataMatrix(d->data, d->output, nflag);

  d->data = readImageValues(image_raw, d->data, nflag);
  fclose(image_raw);
 

  printf("Printing image\n");
  printResult(nflag, d->data);
  printf("\n\n");

  applySimdKernel(nflag, d->data, d->output);
  
  printf("Printing image simd\n");
  printResult(nflag, d->output);
  printf("\n\n");


}

void applySimdKernel(int dimension, int **data, int **output) {
  int i, j;
  
  for (i = 1; i < dimension - 2; i++)
  {
    for (j = 1; j < dimension - 2; j+=4)
    {
      __m128i p1 = _mm_set1_epi32(0);
      __m128i p2 = _mm_set1_epi32(0);
      __m128i p3 = _mm_set1_epi32(0);
      __m128i total = _mm_set1_epi32(0);
      
      if ((dimension - j) < 4) {
        //Case where you can't load 4 data at the same time because seg. fault.

        __m128i top    = _mm_set_epi32(0, data[i-1][j+2], data[i-1][j+1], data[i-1][j]);
        __m128i down   = _mm_set_epi32(0, data[i+1][j+2], data[i+1][j+1], data[i+1][j]);
        
        __m128i right  = _mm_set_epi32(0, 0, data[i][j+2], data[i][j+1]);
        __m128i left   = _mm_set_epi32(data[i][j+2], data[i][j+1], data[i][j], data[i][j-1]);
        __m128i center = _mm_set_epi32(0, data[i][j+2], data[i][j+1], data[i][j]);

        p1 = _mm_max_epi32(top, left);
        p2 = _mm_max_epi32(down, right);
        p3 = _mm_max_epi32(p1, center);
        total = _mm_max_epi32(p3, p2);

        uint32_t *final = (uint32_t*) &total;
        if(final[0] > 0)
          output[i][j] = 255;
        if (final[1] > 0)
          output[i][j+1] = 255;
        if (final[2] > 0)
          output[i][j+2] = 255;
        if (final[3] > 0)
          output[i][j+3] = 255;

      } else {
        //Load data
        __m128i top = _mm_set_epi32(data[i-1][j+3], data[i-1][j+2], data[i-1][j+1], data[i-1][j]);
        __m128i down = _mm_set_epi32(data[i+1][j+3], data[i+1][j+2], data[i+1][j+1], data[i+1][j]);

        __m128i right = _mm_set_epi32(data[i][j+4], data[i][j+3], data[i][j+2], data[i][j+1]);
        __m128i left   = _mm_set_epi32(data[i][j+2], data[i][j+1], data[i][j], data[i][j-1]);
        __m128i center = _mm_set_epi32(data[i][j+3], data[i][j+2], data[i][j+1], data[i][j]);

        p1 = _mm_max_epi32(top, left);
        p2 = _mm_max_epi32(down, right);
        p3 = _mm_max_epi32(p1, center);
        total = _mm_max_epi32(p3, p2);

        uint32_t *final = (uint32_t*) &total;
        if(final[0] > 0)
          output[i][j] = 255;
        if (final[1] > 0)
          output[i][j+1] = 255;
        if (final[2] > 0)
          output[i][j+2] = 255;
        if (final[3] > 0)
          output[i][j+3] = 255;
      }
        // printf("Fila: %i Columna: %i\n", i, j);
        // printResult(dimension, output);
        // if(i == 7)
        //   exit(1);
    }
  }
}


void print128_num(__m128i var)
{
    uint32_t *val = (uint32_t*) &var;
    printf("Numerical: %i %i %i %i\n", 
           val[0], val[1], val[2], val[3]);
}

int** copyDataMatrix(int **data, int **output, int dimension) {
  int i, j;
  for (i = 0; i < dimension; i++)
    for (j = 0; j < dimension; j++)
      output[i][j] = data[i][j];

  return output;
}