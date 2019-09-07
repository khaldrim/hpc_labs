#include "../include/functions.h"
#include <stdlib.h>
#include <stdio.h>

void start(char *inputFile, char *outputSecuential, char *output_simd, int nflag, int dflag) {
  int **kernel = NULL;

  kernel = createKernel();

  es_secuencial(inputFile, outputSecuential, nflag, dflag, kernel);
  // es_simd();
}

void es_secuencial(char *inputFile, char *outputSecuential, int nflag, int dflag, int **kernel) {
  /* Se aplica el elemento de estructuración (ES) de forma
   * sencuencial.
   *
   */

  FILE *image_raw;
  int **data, **outputDataSecuential;
  int i = 0, col = 0, value = 0;
  int total_dimension = nflag * nflag;

  image_raw = fopen(inputFile, "rb");
  if (image_raw == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data = createDataMatrix(nflag);
  outputDataSecuential = createDataMatrix(nflag);

  while(i < total_dimension) {
    fread(&value, sizeof(int), 1, image_raw);

    if (value <= 0)
      value = 0;
    else
      value = 255;

    data[i%nflag][col] = value;
    col++;
    i++;

    if (col > nflag)
      col = 0;
  }

  fclose(image_raw);
  outputDataSecuential = copyDataMatrix(data, outputDataSecuential, nflag);

  printf("Printing secuential image normal: \n");
  printResult(nflag, data);
  printf("\n\n");

  outputDataSecuential = secuential_kernel(nflag, data, outputDataSecuential, kernel);

  if (nflag) {
    printf("Printing secuential image with dilation: \n");
    printResult(nflag, outputDataSecuential);
  }

  writeResult(nflag, outputDataSecuential, outputSecuential);
}

void es_simd() {
  printf("simd\n");
}

int** secuential_kernel(int dimension, int **data, int **output, int **kernel) {
  int i, j = 1;

  for (i = 1; i < dimension - 2; i++) {
    for (j = 1; j < dimension - 2; j++) {
      // centro || arriba || derecha || abajo || izquierda
      if ((data[i][j] * kernel[1][1] == 255) || (data[i-1][j] * kernel[0][1] == 255) || (data[i][j+1] * kernel[1][2] == 255) || (data[i+1][j] * kernel[2][1] == 255) || (data[i][j-1] * kernel[1][0] == 255))
        output[i][j] = 255;
    }
  }

  return output;
}

void writeResult(int nflag, int **data, char *name) {
  FILE *fp = NULL;
  fp = fopen(name, "wb");
  if (fp == NULL) {
    printf("No se puede escribir el archivo: %s", name);
    exit(EXIT_FAILURE);
  }

  // int i = 0, j = 0;
  // for (i = 0; i < nflag; i++)
  //   for (j = 0; j < nflag; j++)
  //     fwrite(data[i][j], sizeof(int), 1, fp);
  //
  // fwrite(data, sizeof(data), 1, fp);
  int i = 0;
  for (i = 0; i < nflag; i++)
    fwrite(data[i], sizeof(int), nflag, fp);
  fclose(fp);
}

int** createDataMatrix(int dimension) {
  int i = 0;
  int **data = (int **)calloc(dimension, sizeof(int*));
  for (i=0; i<dimension; i++) {
    data[i] = (int*)calloc(dimension, sizeof(int));
  }

  return data;
}

int** copyDataMatrix(int **data, int **output, int dimension) {
  int i = 0, j = 0;
  for (i = 0; i<dimension; i++)
    for (j = 0; j<dimension; j++)
      output[i][j] = data[i][j];

  return output;
}

int** createKernel() {
  int i = 0, rowsize = 3, colsize = 3;

  int **kernel = (int **)malloc(sizeof(int*)*rowsize);
  for (i=0; i<rowsize; i++)
    kernel[i] = (int*)malloc(sizeof(int)*colsize);

  /*  kernel shape
   *  0 1 0
   *  1 1 1
   *  0 1 0
   */

  kernel[0][1] = kernel[1][0] = kernel[1][1] = kernel[1][2] = kernel[2][1] = 1;
  return kernel;
}

void printResult(int dimension, int **data) {
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
