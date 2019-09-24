#include <stdlib.h>
#include <stdio.h>
#include "secuential.h"
#include "functions.h"


/* Se aplica el elemento de estructuración (ES) de forma
  * sencuencial.
  *
  */
int** es_secuencial(int **data, int **output, int **kernel, int dimension) {
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

  return output;
}