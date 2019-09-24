#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

typedef struct {
  int **originalData __attribute__ ((aligned (16)));
  int **outputSecuential __attribute__ ((aligned (16)));
  int **outputSimd __attribute__ ((aligned (16)));
} Data;


void start(char *inputFile, char *outputSecuential, char *outputSimd, int nflag, int dflag);
void printResult(int dimension, int **data);
int **readImageValues(FILE *fp, int **data, int dimension);
int **createDataMatrix(int **data, int dimension);
// int** copyDataMatrix(int **data, int **output, int dimension);
int **readImageValues(FILE *fp, int **data, int dimension);
void writeResult(int nflag, int **data, char *name);

#endif
