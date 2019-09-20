#include <stdio.h>

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

/* functions file */
void start(char *inputFile, char *outputSecuential, char *outputSimd, int nflag, int dflag);
void printResult(int dimension, int **data);
int **readImageValues(FILE *fp, int **data, int dimension);
int **createDataMatrix(int **data, int dimension);
int **readImageValues(FILE *fp, int **data, int dimension);
void writeResult(int nflag, int **data, char *name);

#endif
