#include <stdlib.h>
#include <stdio.h>
#include "secuential.h"
#include "functions.h"


void es_secuencial(char *inputFile, char *outputSecuential, int nflag, int dflag) {
  /* Se aplica el elemento de estructuración (ES) de forma
   * sencuencial.
   *
   */

  FILE *image_raw;

  /* Arreglo de NxN */
  int **data = NULL, **output = NULL, **kernel = NULL;

  image_raw = fopen(inputFile, "rb");
  if (image_raw == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data   = createDataMatrix(data, nflag);
  output = createDataMatrix(output, nflag);
  kernel = createDataMatrix(kernel, 3);
  kernel[0][1] = 1, kernel[1][0] = 1, kernel[1][1] = 1, kernel[1][2] = 1, kernel[2][1] = 1;

  data = readImageValues(image_raw, data, nflag);
  fclose(image_raw);

  // printf("Printing image\n");
  // printResult(nflag, data);
  // printf("\n\n");

  applySecuentialKernel(nflag, data, output, kernel);

  if (nflag) {
    printf("Printing secuential image with dilation: \n");
    printResult(nflag, output);
  }

    writeResult(nflag, output, outputSecuential);
}


void applySecuentialKernel(int dimension, int **data, int **output, int **kernel) {
  int i, j;

  for (i = 1; i < dimension - 2; i++) {
    for (j = 1; j < dimension - 2; j++) {
      // centro || arriba || derecha || abajo || izquierda
      if (
        ((data[i][j] * kernel[1][1]) == 255) ||
        ((data[i-1][j] * kernel[0][1]) == 255) ||
        ((data[i][j+1] * kernel[1][2]) == 255) ||
        ((data[i+1][j] * kernel[2][1]) == 255) ||
        ((data[i][j-1] * kernel[1][0]) == 255)
      ) {
        output[i][j] = 255;
      }
    }
  }
}
