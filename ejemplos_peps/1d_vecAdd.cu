#include <stdio.h>
#include <math.h>

#define N 2000  /* Cantidad de elementos */
#define B 1024  /* Tamaño del bloque en CUDA */

/* 
	Para compilar: nvcc 1d_vecAdd.cu -o vecAdd
*/

__global__ void vecAdd(float *a, float *b, float *c) {
    /* 
        Como es de una dimension, puedo calcular mi ID global y operar sobre ese dato en específico
    */

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < N)
        c[tid] = a[tid] + b[tid];
    
    printf("Thread: %d -> (%f + %f) = %f\n",tid,a[tid],b[tid],c[tid]);
}

__host__ int main() {
    /* Se declaran las variables de vectores a utilizar en el host */
    float *a = (float *)malloc(N*sizeof(float));
    float *b = (float *)malloc(N*sizeof(float));
    float *c = (float *)malloc(N*sizeof(float));

    /* Agregamos algunos datos para trabajar */
    int i;
    for(i=0;i<N;i++) {
        a[i] = sin(i)*sin(i);
        b[i] = cos(i)*cos(i);
    }

    /* 
        Declaramos las variables que utilizaremos en el device
        y alojamos memoria para ellas en el device (cudaMalloc)
    */
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, N*sizeof(float));
    cudaMalloc((void **) &d_b, N*sizeof(float));
    cudaMalloc((void **) &d_c, N*sizeof(float));

    /* Copiamos los datos hacia el device */
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);

    /*
        Ahora calculamos el tamaño de la grilla y los bloques a utilizar,
        como tenemos un vector 1d de N elementos, primero debemos definir el 
        tamaño de nuestro bloque en potencias de 2.

        Luego, definimos el tamaño de la grilla en funcion del tamaño del bloque
        que deseamos.
    */
    int blockSize, gridSize;
    blockSize = B;
    gridSize = (int) ceil((float)N/blockSize);

	/* 
		Llama a la funcion que se ejecutara en el device (GPU).
		Se lanza con un tamaño de grilla y bloque definidos.
    */
    vecAdd<<<gridSize, blockSize>>>(d_a,d_b,d_c);
    
    /* Libera la mem pedida en el device */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);
	return 0;
}