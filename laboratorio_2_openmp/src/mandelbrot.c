#include "mandelbrot.h"

void writeData(Data *data, char *fileName) {
    FILE *fp = NULL;
    int i = 0;
    
    fp = fopen(fileName, "wb");
    if (fp == NULL) {
        printf("No se puede escribir el archivo: %s", fileName);
        exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < data->row; i++)
        fwrite(data->display[i], sizeof(float), data->column, fp);
    
    fclose(fp);
}


Data* mandelbrot(Data *data, int depth, float a, float d, float s) {
    int i,j;

    /* desde el punto complejo (-1, 1) hasta (1, )*/

    for (i = 0; i < data->row; i++) {
        for (j = 0; j < data->column; j++) {
            /* calculate complex values */
            int n = 1;

            double c_real = a + (j*s);
            double c_imag = d + (i*s);

            double z_real = 0.0;
            double z_imag = 0.0;

            z_real = 0.0 + c_real;
            z_imag = 0.0 + c_imag;
            while((((z_real * z_real) + (z_imag * z_imag)) < 4.0) && n < depth) {
                double temp = ((pow(z_real, 2.0)) - (pow(z_imag, 2.0))) + c_real;
                z_imag = (2 * z_real * z_imag) + c_imag;
                z_real = temp;
                n++;
            }
            
            data->display[i][j] = log(n) + 1.0; 
        }
    }

    return data;
}

Data* initDataStructure(Data *data, float a, float b, float c, float d, float s) {
    int i;
    data = (Data*)malloc(sizeof(Data));

    /* calculate dimension of complex grid NxM */
    data->column = (int) ceil((abs(a)/s) + (c/s));
    data->row    = (int )ceil((abs(b)/s) + (d/s));
    printf("Dimension: %d filas %d columnas \n", data->row, data->column);

    data->display = (float**)calloc(data->column, sizeof(float*));
    if (data->display != NULL) {
        for (i = 0; i < data->row; i++) {
            data->display[i] = (float*)calloc(data->column, sizeof(float));
            if (data->display[i] == NULL) {
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


void start(int depth, float a, float b, float c, float d, float s, char *fileName) {
    Data *data = NULL;

    data = initDataStructure(data, a, b, c, d, s); 
    data = mandelbrot(data, depth, a, d, s);
    writeData(data, fileName);
}

int main(int argc, char **argv) {
    int i = 1;
    double s = 0.1, a = -1.0, b = -1.0, c = 1.0, d = 1.0;
    char *f;

    int opt;
    extern int optopt;
    extern char* optarg;
    
    while((opt = getopt(argc, argv, "i:a:b:c:d:s:f:")) != -1) {
        switch (opt) {
        case 'i':
            sscanf(optarg, "%d", &i);
            if (i <= 0) {
                printf("La bandera -i no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'a':
            sscanf(optarg, "%f", &a);
            if (a >= 0) {
                printf("La bandera -a no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'b':
            sscanf(optarg, "%f", &b);
            if (b >= 0) {
                printf("La bandera -b no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'c':
            sscanf(optarg, "%f", &c);
            if (c <= 0) {
                printf("La bandera -c no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'd':
            sscanf(optarg, "%f", &d);
            if (d <= 0) {
                printf("La bandera -d no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 's':
            sscanf(optarg, "%f", &s);
            if (s <= 0.0) {
                printf("La bandera -s no puede ser menor o igual a cero.\n");
                exit(EXIT_FAILURE);
            }
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

    printf("Values: \n");
    printf("    i: %d\n", i);
    printf("    a: %f\n", a);
    printf("    b: %f\n", b);
    printf("    c: %f\n", c);
    printf("    d: %f\n", d);
    printf("    s: %f\n", s);
    printf("    f: %s\n\n", f);

    start(i,a,b,c,d,s,f);

    return EXIT_SUCCESS;
}