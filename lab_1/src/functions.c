#include "../include/functions.h"
#include "../include/secuential.h"
#include "../include/simd.h"

#include <stdlib.h>
#include <stdio.h>

void start(char *inputFile, char *outputSecuential, char *output_simd, int nflag, int dflag) {
  // es_secuencial(inputFile, outputSecuential, nflag, dflag, kernel);
  es_simd();
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
