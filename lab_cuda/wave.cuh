#ifndef _WAVE_H_
#define _WAVE_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>
#include <cuda.h>

/*structs*/

/*functions*/
float** createDataMatrix(float **data, int N);
float** initalMatrixState(float **data, int N);
void start(int N, int x, int y, int T, int t, char *fileName);
void printMatrix(float **data, int N);
void writeData(float *data, int N, char *fileName);
__global__ void firstWaveIteration(int N, float *data, float *data_t1);
__global__ void waveIterations(int N, float *data, float *data_t1, float *data_t2);

#endif
