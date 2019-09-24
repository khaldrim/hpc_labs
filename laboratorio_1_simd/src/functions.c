#include "functions.h"
#include "secuential.h"
#include "simd.h"

/* 
  * Recibe el nombre del archivo de entrada, los de salida, la dimensión de la matriz y la opcion de debug.
  * Primero se habre el archivo, luego se inicializan los arreglos que contendrán los datos de entrada, y las matrices
  * de salida para el algoritmo secuencial y simd. Luego se crea el kernel que se aplica en el algoritmo secuencial.
  * Luego, leemos los datos de entrada, inicializamos el reloj, y llamamos a la funcion que realiza el algoritmo aplicando SIMD,
  * despues se llama a la funcion que realiza el algoritmo secuencial donde también calculamos el tiempo de ejecución.
  * 
  * Escribimos los resultados obtenidos y si está activa la opcion de debug, se imprime por pantalla la matriz de datos original,
  * la matriz de datos del algoritmo secuencial y la matriz de datos del algoritmo SIMD. Luego mostramos por stdout los tiempos
  * obtenidos para cada algoritmo.
  */
void start(char *inputFile, char *nameSecuential, char *nameSimd, int nflag, int dflag) {
  FILE *fp;
  clock_t tsec, tsimd;
  int **kernel = NULL;
  
  Data *data = (Data*)malloc(sizeof(Data));


  fp = fopen(inputFile, "rb");
  if (fp == NULL) {
    printf("No se encontró el archivo de entrada.\n");
    exit(EXIT_FAILURE);
  }

  data->originalData = createDataMatrix(data->originalData, nflag);
  data->outputSecuential = createDataMatrix(data->outputSecuential, nflag);
  data->outputSimd = createDataMatrix(data->outputSimd, nflag);

  kernel = createDataMatrix(kernel, 3);
  kernel[0][1] = 1, kernel[1][0] = 1, kernel[1][1] = 1, kernel[1][2] = 1, kernel[2][1] = 1;

  data->originalData = readImageValues(fp, data->originalData, nflag);
  fclose(fp);

  tsimd = clock();
  es_simd(data->originalData, data->outputSimd, nflag);
  tsimd = clock() - tsimd;
  double tsimd_taken = ((double) tsimd) / CLOCKS_PER_SEC;

  tsec = clock();
  data->outputSecuential = es_secuencial(data->originalData, data->outputSecuential, kernel, nflag);
  tsec = clock() - tsec;
  double tsec_taken = ((double) tsec) / CLOCKS_PER_SEC;
  
  /* write output files */
  writeResult(nflag, data->outputSecuential, nameSecuential);
  writeResult(nflag, data->outputSimd, nameSimd);

  if (dflag) {
    printf("\n Printing original image: \n");
    printResult(nflag, data->originalData);
    
    printf("\nPrinting secuential image with dilation: \n");
    printResult(nflag, data->outputSecuential);
  
    printf("\nPrinting simd image with dilation: \n");
    printResult(nflag, data->outputSimd);
  } 

  printf("\n ------ Time ------- \n");
  printf("Secuential (segundos): %f\n", tsec_taken);
  printf("Simd (segundos): %f\n", tsimd_taken);
}

/* 
  * Recibe la dimension de la matriz, la matriz con los datos y el nombre del archivo de salida.
  * Permite escribir una imagen binaria dependiendo del nombre de archivo pasado por parámetro.
  * Si no es posible escribir en algún directorio o el archivo falla al inicializarse, el programa
  * finaliza su ejecución.
  */
int **createDataMatrix(int **data, int dimension) {
  int i = 0;
  data = (int**)calloc(dimension, sizeof(int*));
  for (i = 0; i < dimension; i++)
    data[i] = (int*)calloc(dimension, sizeof(int));

  return data;
}

/* funcion no utilizada finalmente.
int** copyDataMatrix(int **data, int **output, int dimension) {
  int i, j;
  for (i = 0; i < dimension; i++)
    for (j = 0; j < dimension; j++)
      output[i][j] = data[i][j];

  return output;
}
*/

/* 
  * Recibe el puntero al archivo a leer, la matriz con los datos y la dimension de la matriz.
  * Permite leer los valores de un archivo apuntado por fp. Primero se busca la posición 0 del archivo
  * y luego se itera sobre sus datos con un doble ciclo, leyendo valor por valor, de tipo int, y se decide
  * dar valor de 0 a valores iguales o menores a 0 y 255 para cualquier otro valor positivo.
  * Finalmente se devuelve la matriz de datos.
  */
int **readImageValues(FILE *fp, int **data, int dimension) {
  int i = 0, j = 0, value = 0;
  fseek(fp, 0, SEEK_SET);
  for (i = 0; i < dimension; i++) {
    for (j = 0; j < dimension; j++) {
      fread(&value, sizeof(int), 1, fp);
      if (value == 0)
        data[i][j] = 0;
      else
        data[i][j] = 255;
    }
  }
  return data;
}

/* 
  * Recibe la dimension de la matriz, la matriz con los datos y el nombre del archivo de salida.
  * Permite escribir una imagen binaria dependiendo del nombre de archivo pasado por parámetro.
  * Si no es posible escribir en algún directorio o el archivo falla al inicializarse, el programa
  * finaliza su ejecución.
  */
void writeResult(int nflag, int **data, char *name) {
  FILE *fp = NULL;
  int i = 0;

  fp = fopen(name, "wb");
  if (fp == NULL) {
    printf("No se puede escribir el archivo: %s", name);
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < nflag; i++)
    fwrite(data[i], sizeof(int), nflag, fp);
  fclose(fp);
}

/* 
  * Recibe la dimension de la matriz y la matriz con los datos.
  * Permite imprimir por pantalla la imagen resultante;
  * donde el valor 0 se reemplaza por un . y el valor
  * 255 por un &.
  */
void printResult(int dimension, int **data) {
  int row, col = 0;
  for (row = 0; row < dimension; row++) {
    for (col = 0; col < dimension; col++) {
      if(data[row][col] == 0) {
        printf(". ");
      } else {
        printf("& ");
      }
    }
    printf("\n");
  }
}
