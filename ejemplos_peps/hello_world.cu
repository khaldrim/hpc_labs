#include <stdio.h>

/* 
	Para compilar: nvcc hello_world.cu -o hw
*/

__global__ void hello2D() {
	/*
		blockDim.x  => entrega dimension del bloque en X (Lo mismo para y/z)
		gridDim.x   => entrega dimension de la grilla en el eje X (Mismo para y/z)
		blockIdx.x  => entrega el Id del bloque en el cual se ejecuta la hebra (lo mismo para y/z)
		threadIdx.x => entrega posicion de la hebra en el eje X, esto quiere decir en la columna 
					   que se encuentra la hebra dentro del bloque.
	*/

	int blockSize = blockDim.y * blockDim.x;
	int blockId   = gridDim.x * blockIdx.y + blockIdx.x;
	int tid 	  = blockId * blockSize + blockDim.x*threadIdx.y + threadIdx.x;

	printf("I'm thread (%d, %d) in block (%d, %d). My Global Id is: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, tid);
}

__host__ int main() {
	/* Dimensiones de grilla y bloque ha utilizar */
	dim3 blockSize;
	dim3 gridSize;

	gridSize.x  = 2;
	gridSize.y  = 3;
	blockSize.x = 4; 
	blockSize.y = 3;

	/* 
		Llama a la funcion que se ejecutara en el device (GPU).
		Se lanza con un tamaño de grilla y bloque definidos.
	*/
	hello2D<<<gridSize, blockSize>>>();

	/* Blocks until the device has completed all preceding requested tasks */
	cudaDeviceSynchronize();
	return 0;
}