#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

/* functions file */
void start(char *input_file, char *output_secuential, char *output_simd, int nflag, int dflag);
void es_secuencial(char *input_file, char *output_secuential, int nflag, int dflag, int **kernel);
void es_simd();
int** createDataMatrix(int dimension);
int** createKernel();
void printResult(int dimension, int **data);
int** secuential_kernel(int dimension, int **data, int **output, int **kernel);
int** copyDataMatrix(int **data, int **output, int dimension);
void writeResult(int nflag, int **data, char *name);

#endif
