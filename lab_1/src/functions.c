#include "../include/functions.h"
#include <stdlib.h>
#include <stdio.h>

void start(char *input_file, char *output_secuential, char *output_simd, int nflag, int dflag) {
  es_secuencial(input_file, output_secuential, nflag, dflag);
  es_simd();
}

void es_secuencial(char *input_file, char *output_secuential, int nflag, int dflag) {
  /* Se aplica el elemento de estructuración (ES) de forma
   * sencuencial.
   *
   */

  FILE *image_raw;
  int **data;
  int i = 0;
  int col = 0;
  int value = 0;
  int total_dimension = nflag * nflag;

  image_raw = fopen(input_file, "rb");
  if (image_raw == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data = createDataMatrix(nflag);
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

  //TODO falta aplicar el kernel a la imagen
  secuential_kernel();

  if (nflag) {
    printf("Printing secuential image: \n");
    printResult(nflag, data);
  }
}

void es_simd() {
  printf("simd\n");
}

// //recibe el nombre de la ima
// FILE* readImage() {
// }

int** createDataMatrix(int dimension) {
  int i;
  int **data = (int **)malloc(sizeof(int*) * dimension);
  for (i=0; i<dimension; i++) {
    data[i] = (int*)malloc(sizeof(int)*dimension);
  }

  return data;
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
