#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include "functions.h"

int main(int argc, char **argv) {
  int nflag = 0;
  int dflag = 0;
  char *input_file, *output_secuential, *output_simd;

  int opt;
  extern int optopt;
  extern char* optarg;

  while((opt = getopt(argc, argv, "i:s:p:N:D:")) != -1) {
    switch (opt) {
      case 'i':
        input_file = optarg;
        break;
      case 's':
        output_secuential = optarg;
        break;
      case 'p':
        output_simd = optarg;
        break;
      case 'N':
        sscanf(optarg, "%d", &nflag);
        if(nflag <=0) {
          printf("La bandera -N no puede ser menor o igual a cero.\n");
          exit(EXIT_FAILURE);
        }
        break;
      case 'D':
        sscanf(optarg, "%d", &dflag);
        if(dflag != 0 && dflag != 1) {
          printf("La bandera -D solo puede tener valores 0 o 1.\n");
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

  start(input_file, output_secuential, output_simd, nflag, dflag);
  return 0;
}
