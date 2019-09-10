#include "../include/functions.h"
#include "../include/secuential.h"
#include "../include/simd.h"

#include <stdlib.h>
#include <stdio.h>

void start(char *inputFile, char *outputSecuential, char *output_simd, int nflag, int dflag) {
  // es_secuencial(inputFile, outputSecuential, nflag, dflag, kernel);
  es_simd();
}

float **readImageValues(FILE *fp, float **data, int dimension) {
  int i = 0, col = 0, value = 0;
  int total_dimension = dimension * dimension;
  while(i < total_dimension) {
    fread(&value, sizeof(float), 1, fp);

    if (value <= 0)
      value = 0;
    else
      value = 255;

    data[i%dimension][col] = value;
    col++;
    i++;

    if (col > dimension)
      col = 0;
  }

  return data;
}

float **createDataMatrix(float **data, int dimension) {
  int i = 0;
  data = (float**)calloc(dimension, sizeof(float*));
  for (i = 0; i < dimension; i++)
    data[i] = (float*)calloc(dimension, sizeof(float));

  return data;
}

void printResult(int dimension, float **data) {
  /* Permite imprimir por pantalla la imagen resultante;
   * donde el valor 0 se reemplaza por un 0 y el valor
   * 255 por un 1.
   */

  int row, col = 0;
  for (row = 0; row < dimension; row++) {
    for (col = 0; col < dimension; col++) {
      if(data[row][col] == 0) {
        printf("0");
      } else {
        printf("1");
      }
    }
    printf("\n");
  }
}
