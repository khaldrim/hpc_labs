#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define f(n) (sqrt(1.0-(pow(n,2))))
#define N 10000
#define threads 10

int main() {
    double sum = 0.0, sum_partial = 0, x = 0, pi = 0;
    int i;

    int i_am_private = 10;
    int i_am_private_shared = 10;
    
    omp_set_nested(1);

    #pragma omp parallel private(i_am_private) shared(i_am_private_shared) num_threads(4) if(x == 0)
    {   
    omp_set_num_threads(threads);
        i_am_private = 10;
        i_am_private += 1;

        // #pragma omp critical
        // printf("init shared: %d\n", i_am_private_shared);


        #pragma omp critical
        i_am_private_shared += 1;
        
        printf("what am i: %d\n", omp_get_thread_num());

        /* race condition */

        // printf("what am i shared: %d\n", i_am_private_shared);
    }

    #pragma omp parallel num_threads(3)
    {
        printf("soy: %d, en nivel 1\n", omp_get_thread_num());
        #pragma omp parallel num_threads(2)
        {
            printf("soy: %d, en nivel 2\n", omp_get_thread_num());
        }
    }

    

    // printf("shared final: %d\n", i_am_private_shared);

    return 0;
}
