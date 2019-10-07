#ifndef _MANDELBROTP_H_
#define _MANDELBROTP_H_

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

/* structs */

typedef struct  {
    int column, row;
    double **display;
} Data;

/* functions */
void start(int depth, double a, double b, double c, double d, double s, char *fileName, int t);
Data* initDataStructure(Data *data, double a, double b, double c, double d, double s);
void mandelbrot(Data *data, int i, int j, int depth, double a, double d, double s);
void writeData(Data *data, char *fileName);
void clean(Data *data);

#endif