#include "wave.cuh"

#define C 1.0
#define dt 0.1
#define dd 2.0

/* Kernels */
__global__ void firstWaveIteration(int N, float *data, float *data_t1) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && column < N) {
        int tid = row * N + column;

        /* check border cases */
        if (row == 0 || column == 0 || row == N-1 || column == N-1) {
            data[tid] = 0;
        }
        
        //printf("tid: %d | row, col: %d, %d\n", tid, row, column);

        data_t1[tid] = data[tid];

        int up    = (row - 1) * N + column;
        int down  = (row + 1) * N + column;
        int left  = row * N + (column - 1);
        int right = row * N + (column + 1);

        data[tid] = data_t1[tid] + (((C^2)/2)*((dt/dd)^2)) * ((data_t1[up] + data_t1[down] + data_t1[left] + data_t1[right]) - 4*data_t1[tid]);
    }
}

__global__ void waveIterations(int N, float *data, float *data_t1, float *data_t2) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && column < N) {
        int tid = row * N + column;

        /* check border cases */
        if (row == 0 || column == 0 || row == N-1 || column == N-1) {
            data[tid] = 0;
        }
        
        data_t2[tid] = data_t1[tid];
        data_t1[tid] = data[tid];

        int up    = (row-1) * N + column;
        int down  = (row+1) * N + column;
        int left  = row * N + (column - 1);
        int right = row * N + (column + 1);

        data[tid] = (2*data_t1[tid]) - data_t2[tid] + (((C^2)/2)*((dt/dd)^2)) * ((data_t1[up] + data_t1[down] + data_t1[left] + data_t1[right]) - 4*data_t1[tid]); 
    }
}

/*
 * Entrada: Puntero a Data, Puntero a char con nombre del archivo ingresado por parámetro
 * Salida: Vacía
 * Obj: Primero se crea un puntero al archivo de salida, luego se itera sobre data->display
 *      para escribir por fila en el archivo de salida. Finalmente se cierra el archivo.
*/
void writeData(float *data, int N, char *fileName) {
    FILE *fp = NULL;
    int i = 0;

    fp = fopen(fileName, "wb");
    if (fp == NULL) {
        printf("No se puede escribir el archivo: %s", fileName);
        exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < N; i++)
        fwrite(data[i], sizeof(float), N, fp);
    
    fclose(fp);
}

void printMatrix(float **data, int N) {
    int i, j;
    for(i=0;i<N;i++) {
        for(j=0;j<N;j++) {
            printf("%.0f  ", data[i][j]);
        }
        printf("\n");
    }
}

/*
 * Entrada:
 * Salida:
 * Obj:
*/
float** initalMatrixState(float **data, int N) {
    int i,j;
    float x_limit_left, x_limit_right, y_limit_up, y_limit_down;

    x_limit_left  = 0.4 * N;
    x_limit_right = 0.6 * N;

    y_limit_up   = 0.4 * N;
    y_limit_down = 0.6 * N;

    printf("x_left: %f | x_right: %f\n", x_limit_left, x_limit_right);
    printf("y_up:   %f | y_down:  %f\n", y_limit_up, y_limit_down);

    for(i=0;i<N;i++) {
        for(j=0;j<N;j++) {
            if((x_limit_left < i && i < x_limit_right) && (y_limit_up < j && j < y_limit_down))
              data[i][j] = 20;
            else
              data[i][j] = 0;
        }
    }

    return data;
}

/*
 * Entrada:
 * Salida:
 * Obj:
*/
float** createDataMatrix(float **data, int N) {
    int i;
    data = (float**)calloc(N, sizeof(float*));
    if (data != NULL) {
        for (i = 0; i < N; i++) {
            data[i] = (float*)calloc(N, sizeof(float));
            if (data[i] == NULL) {
                perror("Error allocating memory for the display matrix: ");
                exit(EXIT_FAILURE);
            }
        }
    } else {
        perror("Error allocating memory for the display matrix: ");
        exit(EXIT_FAILURE);
    }
    return data;
}

/*
 * Entrada: Recibe parámetros ingresados mediante getopt
 * Salida: Vacía
 * Obj:
*/
void start(int N, int x, int y, int T, int t, char *fileName) {
    /* Vars */
    float **data = NULL;
    float **data_t1 = NULL;
    float **data_t2 = NULL;

    int iterations = 1;

    /* Cuda Vars*/
    float *dev_data, *dev_data_t1, *dev_data_t2;

    data = createDataMatrix(data, N);
    data_t1 = createDataMatrix(data_t1, N);
    data_t2 = createDataMatrix(data_t2, N);

    data = initalMatrixState(data, N);

    /* Malloc to device */
    cudaMalloc((void **) &dev_data, (N*N)*sizeof(float));
    cudaMalloc((void **) &dev_data_t1, (N*N)*sizeof(float));
    
    /* Copy from host to device */
    cudaMemcpy(dev_data, data, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data_t1, data_t1, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

    /* Kernel config */
    dim3 blockSize(x, y);
    dim3 gridSize(ceil((N + blockSize.x - 1)/blockSize.x), ceil((N + blockSize.y - 1)/blockSize.y));

    printf("blockSize (x, y): (%d, %d)\n", blockSize.x, blockSize.y);
    printf("gridSize  (x, y): (%d, %d)\n", gridSize.x, gridSize.y);

    /* Fisrt Iteration in Kernel */
    firstWaveIteration<<<gridSize, blockSize>>>(N, dev_data, dev_data_t1);

    /* Rest of the iterations */
    for (iterations; iterations < t; iterations++)
        waveIterations<<<gridSize, blockSize>>>(N,dev_data, dev_data_t1, dev_data_t2);

    /* Check CUDA errors */
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    /* getting the results from device */
    cudaMemcpy(data, (dev_data), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_t1, (dev_data_t1), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(data_t2, (dev_data_t2), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

    /* writing output */

}

/*
 *  Bloque main, recibe los parametros mediante getopt:
 *      -N -> tamaño grilla
 *      -x -> tamaño bloque x
 *      -y -> tamaño bloque y
 *      -T -> num pasos
 *      -t -> iteración salida
 *      -f -> archivo de salida en formato .raw
 *
 *
*/
int main(int argc, char **argv) {
    int N, x, y, T, t;
    char *f;

    int opt;
    extern int optopt;
    extern char* optarg;

    while((opt = getopt(argc, argv, "N:x:y:T:t:f:")) != -1) {
        switch (opt) {
        case 'N':
            sscanf(optarg, "%d", &N);
            if (N <= 0) {
                printf("La bandera -N no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'x':
            sscanf(optarg, "%d", &x);
            break;
        case 'y':
            sscanf(optarg, "%d", &y);
            break;
        case 'T':
            sscanf(optarg, "%d", &T);
            break;
        case 't':
            sscanf(optarg, "%d", &t);
            break;
        case 'f':
            f = optarg;
            break;
        case '?':
            if(optopt == 'c')
            fprintf(stderr, "Opción -%c requiere un argumento.\n", optopt);
            else if (isprint(optopt))
            fprintf(stderr, "Opción desconocida '-%c'.\n", optopt);
            else
            fprintf(stderr, "Opción con caracter desconocido. '\\x%x'.\n", optopt);
            exit(EXIT_FAILURE);
        default:
            printf("Faltan argumentos.\n");
            exit(EXIT_FAILURE);
        }
    }

    start(N,x,y,T,t,f);
    return EXIT_SUCCESS;
}
