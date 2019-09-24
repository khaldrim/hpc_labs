#include "simd.h"

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

void print128_num(__m128i var)
{
    uint32_t *val = (uint32_t*) &var;
    printf("Numerical: %i %i %i %i\n", 
           val[0], val[1], val[2], val[3]);
}