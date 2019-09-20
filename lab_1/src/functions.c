#include "functions.h"
#include "secuential.h"
#include "simd.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//TODO abrir el archivo y crear las matrices aqui, luego mandarlas a las funciones correspondientes
void start(char *inputFile, char *outputSecuential, char *outputSimd, int nflag, int dflag) {
  // es_secuencial(inputFile, outputSecuential, nflag, dflag);
  es_simd(inputFile, outputSimd, nflag, dflag);
}

int **createDataMatrix(int **data, int dimension) {
  int i = 0;
  data = (int**)calloc(dimension, sizeof(int*));
  for (i = 0; i < dimension; i++)
    data[i] = (int*)calloc(dimension, sizeof(int));

  return data;
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
