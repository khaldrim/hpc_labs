#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

/* functions file */
void start(char *input_file, char *output_secuential, char *output_simd, int nflag, int dflag);
void es_secuencial(char *input_file, char *output_secuential, int nflag, int dflag);
void es_simd();
int** createDataMatrix(int dimension);
void printResult(int dimension, int **data);

#endif
