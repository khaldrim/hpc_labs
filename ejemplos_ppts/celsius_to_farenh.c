#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cpuid.h>
#include <mmintrin.h>
#include <xmmintrin.h>

void test_sse1() {
  __m128 celsiusvector, fahrenheitvector, partialvector, coeffvector;
  const __m128 thirtytwovector = {32.0, 32.0, 32.0, 32.0};
  float celsiusarray[] __attribute__ ((aligned (16))) = {-100.0, -80.0, -40.0, 0.0};
  float fahrenheitarray[4] __attribute__ ((aligned (16)));
  int i;


  /*
    __m128 _mm_set1_ps (float a)
    Broadcast single-presicion (32bit) floating-point value a to all elements of dst

    O sea, aquí cargo 1.8 en los 4 registros de coeffvector
  */
  coeffvector = _mm_set1_ps(9.0 / 5.0);

  /*

    _mm_load_ps (float a)
    Load 128-bits (composed of 4 packed single-precision (32-bit) floating-point elements)
    from memory into dst. mem_addr must be aligned on a 16-byte boundary or a general-protection exception may be generated.

    Aqui carga el arreglo celsiusarray (tomando los 4 elementos al mismo tiempo) a celsiusvector
    esto es posible porque celsiusarray esta alineado a 16
  */
  celsiusvector = _mm_load_ps(celsiusarray);

  /*

  */
  partialvector = _mm_mul_ps(celsiusvector, coeffvector);
  fahrenheitvector = _mm_add_ps(partialvector, thirtytwovector);

  _mm_store_ps(fahrenheitarray, fahrenheitvector);

  for (i = 0; i < 4; i++) {
    printf("%f celsius is %f farenheit.\n", celsiusarray[i], fahrenheitarray[i]);
  }
}

int main (int argc, char *argv[]) {
  unsigned int eax, ebx, ecx, edx;
  unsigned int extensions, sig;
  int result, sse1_available;

  // result = __get_cpuid (FUNC_FEATURES, &eax, &ebx, &ecx, &edx);
  // sse1_available = (bit_SSE & edx);
  // if (0 == sse1_available) {
  //   fprintf(stderr, "Error: SSE1 features not available.\n");
  //   exit(-1);
  // } else {
  //   fprintf(stderr, "SSE1 features ARE available.\n");
  // }

  test_sse1();

  return 0;
}
