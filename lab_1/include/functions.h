#include <stdio.h>

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

/* functions file */
void start(char *input_file, char *output_secuential, char *output_simd, int nflag, int dflag);
void printResult(int dimension, float **data);
float **readImageValues(FILE *fp, float **data, int dimension);
float **createDataMatrix(float **data, int dimension);

#endif
