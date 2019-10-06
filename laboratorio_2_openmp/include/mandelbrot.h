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
    float **display;
} Data;

void start(int depth, float a, float b, float c, float d, float s, char *fileName);
Data* initDataStructure(Data *data, float a, float b, float c, float d, float s);
Data* mandelbrot(Data *data, int depth, float a, float d, float s);
void writeData(Data *data, char *fileName);

#endif