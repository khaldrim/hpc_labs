#include "simd.h"

/*
 * Se declaran las variables a utilizar __m128i, se recorre la matriz de datos
 * desde la posicion [1,1] hasta la posicion [N-2, N-2]. Como la funcion _mm_loadu
 * carga de a 4 registros inmediatamente, consideramos que los casos en que la posicion
 * de la columna sea mayor a 4 para evitar error de segmento. Entonces cargamos los datos
 * en los registros, formando la cruz, y se realiza la operacion _mm_max() que extrae los valores maximos
 * entre dos variables _m128i. Estas variables se comportan como vectores, por lo que se realiza la comparacion
 * inmediata de los 4 elementos. 
 * 
 * Primero obtenemos el residuo de la comparacion entre r0 (top) y r1 (right), tambien se realiza la comparacion
 * entre r2 (down) y r3 (left). Estos resultados se comparan inmediatamente y se guardan en la variable partial
 * finalmente en total, se realiza la comparacion entre r4 (center) y partial. Aplicando la Estructura requerida.
 *  
 * Para guardar los valores en la matriz de salida, se convierte a un vector de 4 enteros llamado final, y
 * se realiza la asignacion a las posiciones correspondientes.
*/
void es_simd(int **data, int **output, int dimension) {
  int i, j;  
  
  /* r0 - top | r1 - right | r2 - down | r3 - left | r4 - center*/
  __m128i r0, r1, r2, r3, r4, total, partial;

  for (i = 1; i < dimension - 2; i++) {
    for (j = 1; j < dimension - 2; j+=4) {
      
      if ((dimension - j) > 4) {
        r0 = _mm_loadu_si128((__m128i*)&data[i-1][j]);
        r1 = _mm_loadu_si128((__m128i*)&data[i][j+1]);
        r2 = _mm_loadu_si128((__m128i*)&data[i+1][j]);
        r3 = _mm_loadu_si128((__m128i*)&data[i][j-1]);
        r4 = _mm_loadu_si128((__m128i*)&data[i][j]);

        partial = _mm_max_epu16(_mm_max_epi16(r0, r1), _mm_max_epi16(r2, r3));
        total = _mm_max_epi16(r4, partial);
        
        uint32_t *final = (uint32_t*) &total;
        output[i][j] = final[0];
        output[i][j+1] = final[1];
        output[i][j+2] = final[2];
        output[i][j+3] = final[3];
      }
    }
  }
}

/*
 * Recibe una variable de tipo __m128i e imprime sus valores correspondientes.
 * Esta funcion fue creada para comprobar la funcion es_simd() funcionace correctamente.
 * 
*/
void print128_num(__m128i var)
{
    uint32_t *val = (uint32_t*) &var;
    printf("Numerical: %i %i %i %i\n", 
           val[0], val[1], val[2], val[3]);
}