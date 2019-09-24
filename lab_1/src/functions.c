#include "functions.h"
#include "secuential.h"
#include "simd.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct {
  int **originalData __attribute__ ((aligned (16)));
  int **outputSecuential __attribute__ ((aligned (16)));
  int **outputSimd __attribute__ ((aligned (16)));
} Data;

void start(char *inputFile, char *nameSecuential, char *nameSimd, int nflag, int dflag) {
  FILE *fp;
  clock_t tsec, tsimd;
  int **kernel = NULL;
  
  Data *data = (Data*)malloc(sizeof(Data));


  fp = fopen(inputFile, "rb");
  if (fp == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data->originalData = createDataMatrix(data->originalData, nflag);
  data->outputSecuential = createDataMatrix(data->outputSecuential, nflag);
  data->outputSimd = createDataMatrix(data->outputSimd, nflag);

  /* copy the original data to non modified output matrixs */
  // data->outputSecuential = copyDataMatrix(data->originalData, data->outputSecuential, nflag);
  // data->outputSimd = copyDataMatrix(data->originalData, data->outputSimd, nflag);

  kernel = createDataMatrix(kernel, 3);
  kernel[0][1] = 1, kernel[1][0] = 1, kernel[1][1] = 1, kernel[1][2] = 1, kernel[2][1] = 1;

  data->originalData = readImageValues(fp, data->originalData, nflag);
  fclose(fp);

  tsimd = clock();
  es_simd(data->originalData, data->outputSimd, nflag);
  tsimd = clock() - tsimd;
  double tsimd_taken = ((double) tsimd) / CLOCKS_PER_SEC;

  tsec = clock();
  data->outputSecuential = es_secuencial(data->originalData, data->outputSecuential, kernel, nflag);
  tsec = clock() - tsec;
  double tsec_taken = ((double) tsec) / CLOCKS_PER_SEC;
  
  /* write output files */
  writeResult(nflag, data->outputSecuential, nameSecuential);
  writeResult(nflag, data->outputSimd, nameSimd);

  if (dflag) {
    printf("\n Printing original image: \n");
    printResult(nflag, data->originalData);
    
    printf("\nPrinting secuential image with dilation: \n");
    printResult(nflag, data->outputSecuential);
  
    printf("\nPrinting simd image with dilation: \n");
    printResult(nflag, data->outputSimd);
  } 

  printf("\n ------ Time ------- \n");
  printf("Secuential (segundos): %f\n", tsec_taken);
  printf("Simd (segundos): %f\n", tsimd_taken);
}

int **createDataMatrix(int **data, int dimension) {
  int i = 0;
  data = (int**)calloc(dimension, sizeof(int*));
  for (i = 0; i < dimension; i++)
    data[i] = (int*)calloc(dimension, sizeof(int));

  return data;
}

int** copyDataMatrix(int **data, int **output, int dimension) {
  int i, j;
  for (i = 0; i < dimension; i++)
    for (j = 0; j < dimension; j++)
      output[i][j] = data[i][j];

  return output;
}

int **readImageValues(FILE *fp, int **data, int dimension) {
  int i = 0, j = 0, value = 0;

  fseek(fp, 0, SEEK_SET);

  for (i = 0; i < dimension; i++) {
    for (j = 0; j < dimension; j++) {
      fread(&value, sizeof(int), 1, fp);
      
      if (value == 0)
        data[i][j] = 0;
      else
        data[i][j] = 255;
    }
  }
  
  return data;
}

void writeResult(int nflag, int **data, char *name) {
  FILE *fp = NULL;
  int i = 0;

  fp = fopen(name, "wb");
  if (fp == NULL) {
    printf("No se puede escribir el archivo: %s", name);
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < nflag; i++)
    fwrite(data[i], sizeof(int), nflag, fp);
  fclose(fp);
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
        printf(". ");
      } else {
        printf("# ");
      }
    }
    printf("\n");
  }
}
