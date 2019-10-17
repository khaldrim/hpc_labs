#include <stdio.h>
#include <math.h>

#define a_row 4
#define a_col 8

#define b_row 8
#define b_col 4

#define c_row 4
#define c_col 4

__global__ void matrixMul(int *a, int *b, int *c, int N);

__host__ int** initMatrix(int** A, int row, int col) {
    int i;

    A = (int**)malloc(row * sizeof(int*));
    for(i=0;i<row;i++)
        A[i] = (int*)malloc(col * sizeof(int));
    
    return A;
}

__host__ void freeMatrix(int **M, int row) {
    int i;
    for(i=0;i<row;i++)
        free(M[i]);
}

__host__ int main() {
    /* define host matrix */
    int** A;
    int** B;
    int** C;

    A = initMatrix(A, a_row, a_col);
    B = initMatrix(B, b_row, b_col);
    C = initMatrix(C, c_row, c_col);

    int i,j;
    for(i=0;i<a_row;i++)
        for(j=0;j<a_col;j++)
            A[i][j] = log(i*j) + 1;

    for(i=0;i<b_row;i++)
        for(j=0;j<b_col;j++)
            B[i][j] = log(i*j) + 1;
    
    for(i=0;i<c_row;i++)
        for(j=0;j<c_col;j++)
            C[i][j] = 0;
    
    int **d_A, **d_B, **d_C;
    /* we've to "flatten" our 2d matrix */
    cudaMalloc((void **) &d_A, sizeof(int) * a_row * a_col);
    cudaMalloc((void **) &d_B, sizeof(int) * b_row * b_col);
    cudaMalloc((void **) &d_C, sizeof(int) * c_row * c_col);

    dim3 blocksPerGrid(8, 1, 1);
    dim3 threadsPerBlock(4, 1, 1);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    freeMatrix(A, a_row);
    freeMatrix(B, b_row);
    freeMatrix(C, c_row);

    return 0;
}


