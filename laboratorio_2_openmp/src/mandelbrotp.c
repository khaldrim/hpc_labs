#include "mandelbrotp.h"

void clean(Data *data) {
    int i;
    for (i = 0; i < data->row; i++) {
        double *currentPtr = data->display[i];
        free(currentPtr);
    }
}

void writeData(Data *data, char *fileName) {
    FILE *fp = NULL;
    int i = 0;
    
    fp = fopen(fileName, "wb");
    if (fp == NULL) {
        printf("No se puede escribir el archivo: %s", fileName);
        exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < data->row; i++)
        fwrite(data->display[i], sizeof(double), data->column, fp);
    
    fclose(fp);
}

void mandelbrot(Data *data, int i, int j, int depth, double a, double d, double s) {
    int n = 1;

    double c_real = a + (j*s);
    double c_imag = d - (i*s);

    double z_real = 0.0;
    double z_imag = 0.0;

    z_real = 0.0 + c_real;
    z_imag = 0.0 + c_imag;
    do {
        double temp = ((pow(z_real, 2.0)) - (pow(z_imag, 2.0))) + c_real;
        z_imag = (2 * z_real * z_imag) + c_imag;
        z_real = temp;
        n++;
    } while(((pow(z_real, 2.0) + pow(z_imag, 2.0)) < 4.0) && n < depth);
    
    data->display[i][j] = log((double) n);
}

Data* initDataStructure(Data *data, double a, double b, double c, double d, double s) {
    int i;
    data = (Data*)malloc(sizeof(Data));

    /* calculate dimension of complex grid NxM */
    data->column = (int) ((c - a)/s) + 1;
    data->row    = (int) ((d - b)/s) + 1;
    printf("Dimension: %d filas %d columnas \n", data->row, data->column);
    
    data->display = (double**)calloc(data->column, sizeof(double*));
    if (data->display != NULL) {
        for (i = 0; i < data->row; i++) {
            data->display[i] = (double*)calloc(data->column, sizeof(double));
            if (data->display[i] == NULL) {
                perror("Error allocating memory for the display matrix: ");
                exit(EXIT_FAILURE);
            }
        }
    } else {
        perror("Error allocating memory for the display matrix: ");
        exit(EXIT_FAILURE);
    }

    printf("matrix created \n");
    return data;
}

void start(int depth, double a, double b, double c, double d, double s, char *fileName, int t) {
    /* initialize var */
    Data *data = NULL;
    
    /* set number of threads */
    omp_set_num_threads(t);

    data = initDataStructure(data, a, b, c, d, s);
    
    int i = 0;
    while (i < data->row) {
        #pragma omp parallel
        {
            int j = 0;
            int tid = omp_get_thread_num();
            #pragma omp parallel for
            for(j= omp_get_thread_num(); j < data->column; j += t) {
                // printf("fila: %d, columna: %d | tid: %d \n", i, (j+tid), tid);
                if ((j + tid) < data->column)
                    mandelbrot(data, i, (j+tid), depth, a, d, s);
            }

            #pragma omp barrier

            #pragma omp single 
            {
                i++;
            }
        }
    }

    /* 2da version me enrrede !
    #pragma omp parallel shared(data, depth, a, d, s)
    {
        int i = 0, j = 0;
        int tid = omp_get_thread_num(); 
        j = j + tid;
        #pragma omp parallel for shared(data, a, d, s) private(i, j, tid)  
        for(i = 0; i < data->row; i++) {
            while (j < data->column) {
                mandelbrot(data, i, j, depth, a, d, s);
                j += omp_get_thread_num();    
                j = j + tid;
            }
        }
    }
    */

    /* 1era version, se cae en matrices muy grandes, creo que es porque se generan demasiados task
    #pragma omp parallel shared(data, depth, a, d, s)
    {
        int i = 0,j = 0;
        #pragma omp single private(i, j) 
        {
            for (i = 0; i < data->row; i++) {
                for (j = 0; j < data->column; j++) {
                    #pragma omp task
                    {
                        mandelbrot(data, i, j, depth, a, d, s);
                    }
                }
            }
        }
    }
    */

    writeData(data, fileName);
    clean(data);
}

int main(int argc, char **argv) {
    int i = 1, t = 1;
    double s = 0.1, a = -1.0, b = -1.0, c = 1.0, d = 1.0;
    char *f;

    int opt;
    extern int optopt;
    extern char* optarg;
    
    while((opt = getopt(argc, argv, "i:a:b:c:d:s:f:t:")) != -1) {
        switch (opt) {
        case 'i':
            sscanf(optarg, "%d", &i);
            if (i <= 0) {
                printf("La bandera -i no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'a':
            sscanf(optarg, "%lf", &a);
            // if (a >= 0) {
            //     printf("La bandera -a no puede ser menor o igual a cero.\n");
            //     exit(EXIT_FAILURE);
            // }
            break;
        case 'b':
            sscanf(optarg, "%lf", &b);
            // if (b >= 0) {
            //     printf("La bandera -b no puede ser menor o igual a cero.\n");
            //     exit(EXIT_FAILURE);
            // }
            break;
        case 'c':
            sscanf(optarg, "%lf", &c);
            // if (c <= 0) {
            //     printf("La bandera -c no puede ser menor o igual a cero.\n");
            //     exit(EXIT_FAILURE);
            // }
            break;
        case 'd':
            sscanf(optarg, "%lf", &d);
            // if (d <= 0) {
            //     printf("La bandera -d no puede ser menor o igual a cero.\n");
            //     exit(EXIT_FAILURE);
            // }
            break;
        case 's':
            sscanf(optarg, "%lf", &s);
            if (s <= 0.0) {
                printf("La bandera -s no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'f':
            f = optarg;
            break;
        case 't':
            sscanf(optarg, "%d", &t);
            if (t <= 0) {
                printf("La bandera -d no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
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

    /* comprobar cotas */

    start(i,a,b,c,d,s,f,t);
    return EXIT_SUCCESS;
}