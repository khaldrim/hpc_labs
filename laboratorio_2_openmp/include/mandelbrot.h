#ifndef _MANDELBROT_H_
#define _MANDELBROT_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <complex.h>
#include <math.h>

typedef struct  {
    int column, row;
    double **display;
} Data;

void start(int depth, double a, double b, double c, double d, double s, char *fileName);
Data* initDataStructure(Data *data, double a, double b, double c, double d, double s);
Data* mandelbrot(Data *data, int depth, double a, double d, double s);
void writeData(Data *data, char *fileName);

#endif