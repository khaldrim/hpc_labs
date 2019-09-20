#ifndef _SECUENTIAL_H_
#define _SECUENTIAL_H_

void es_secuencial(char *inputFile, char *outputSecuential, int nflag, int dflag);
void applySecuentialKernel(int dimension, int **data, int **output, int **kernel);
void writeResult(int nflag, int **data, char *name);
int** copyDataMatrix(int **data, int **output, int dimension);

#endif
