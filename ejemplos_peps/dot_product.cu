#include <stdio.h>
#include <math.h>

#define N 2000  /* Cantidad de elementos */
#define T 512  /* Tamaño del bloque en CUDA */

/* 
	Para compilar: nvcc dot_product.cu -o vecAdd
*/

__global__ void dotProduct(float *a, float *b, float *s) {
    /* 
        Como es de una dimension, puedo calcular mi ID global y operar sobre ese dato en específico
    */

    __shared__ float temp[N];

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    temp[threadIdx.x] = a[tid] * b[tid];
    
    /* Esperamos a que todas las hebras finalicen su multiplicacion */
    __syncthreads();
    /* Asignamos a un if para que solo 1 hebra ejecute esta porcion de codigo, que suma los resultados */
    if(threadIdx.x == 0) {
        float sum = 0;
        int i;
        for (i=0;i<T;i++) {
            sum += temp[i];
        }

        atomicAdd(s,sum); /* atomicAdd utilizado para modificar la mem asignada globalmente en el host*/
    }

}

__host__ int main() {
    /* Se declaran las variables de vectores a utilizar en el host */
    float *a = (float *)malloc(N*sizeof(float));
    float *b = (float *)malloc(N*sizeof(float));
    float c = 0.0;

    /* Agregamos algunos datos para trabajar */
    int i;
    for(i=0;i<N;i++) {
        // a[i] = sin(i)*sin(i);
        // b[i] = cos(i)*cos(i);
        a[i] = b[i] = 2.0;
    }

    /* 
        Declaramos las variables que utilizaremos en el device
        y alojamos memoria para ellas en el device (cudaMalloc)
    */
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, N*sizeof(float));
    cudaMalloc((void **) &d_b, N*sizeof(float));
    cudaMalloc((void **) &d_c, sizeof(float));

    /* Copiamos los datos hacia el device */
    cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(float), cudaMemcpyHostToDevice);

    /*
        Ahora calculamos el tamaño de la grilla y los bloques a utilizar,
        como tenemos un vector 1d de N elementos, primero debemos definir el 
        tamaño de nuestro bloque en potencias de 2.

        Luego, definimos el tamaño de la grilla en funcion del tamaño del bloque
        que deseamos.
    */
    int blockSize, gridSize;
    blockSize = T;
    gridSize = (int) ceil((float)N/blockSize);

	/* 
		Llama a la funcion que se ejecutara en el device (GPU).
		Se lanza con un tamaño de grilla y bloque definidos.
    */
    dotProduct<<<gridSize, blockSize>>>(d_a,d_b,d_c);

    cudaMemcpy(&c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    printf("La suma fue: %f\n", c);
    /* Libera la mem pedida en el device */
    cudaFree(d_a);
    cudaFree(d_b);

    free(a);
    free(b);
	return 0;
}