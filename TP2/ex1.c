 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>

# define N 10000000
#define custom int

int main() {

    //using multiple unrolling factors 1 2 4 8 16 32
    printf("unrolling factor,Time ms\n");
    int summ = 0;

    //unrolling using 1
    custom *a = malloc(N * sizeof(custom));
    custom sum = 0.0;
    double start, end;
    for (int i = 0; i < N; i++)a[i] = 1.0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i++) sum += a[i];
    end = (double)clock() / CLOCKS_PER_SEC;
    summ += sum;
    printf("%d,%f\n", 1, (end - start) * 1000);
    free(a);



    //unrolling using 2
    a = malloc(N * sizeof(custom));
    sum = 0.0;
    for (int i = 0; i < N; i++)a[i] = 1.0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i+=2) sum += a[i] + a[i+1];
    end = (double)clock() / CLOCKS_PER_SEC;
    summ += sum;
    printf("%d,%f\n", 2, (end - start) * 1000);
    free(a);





    //unrolling using 4
    a = malloc(N * sizeof(custom));
    sum = 0.0;
    for (int i = 0; i < N; i++)a[i] = 1.0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i+=4) sum += a[i] + a[i+1] + a[i+2] + a[i+3];
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("%d,%f\n", 4, (end - start) * 1000);
    summ += sum;
    free(a);


    //unrolling using 8
    a = malloc(N * sizeof(custom));
    sum = 0.0;
    for (int i = 0; i < N; i++)a[i] = 1.0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i+=8) sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7];
    end = (double)clock() / CLOCKS_PER_SEC;
    printf("%d,%f\n", 8, (end - start) * 1000);
    summ += sum;


    free(a);


    //unrolling using 16
    a = malloc(N * sizeof(custom));
    sum = 0.0;
    for (int i = 0; i < N; i++)a[i] = 1.0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i+=16) sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7] + a[i+8] + a[i+9] + a[i+10] + a[i+11] + a[i+12] + a[i+13] + a[i+14] + a[i+15];
    end = (double)clock() / CLOCKS_PER_SEC;
    summ += sum;
    printf("%d,%f\n", 16, (end - start) * 1000);
    free(a);



    //unrolling using 32
    a = malloc(N * sizeof(custom));
    sum = 0.0;
    for (int i = 0; i < N; i++)a[i] = 1.0;
    start = (double)clock() / CLOCKS_PER_SEC;
    for (int i = 0; i < N; i+=32) sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7] + a[i+8] + a[i+9] + a[i+10] + a[i+11] + a[i+12] + a[i+13] + a[i+14] + a[i+15] + a[i+16] + a[i+17] + a[i+18] + a[i+19] + a[i+20] + a[i+21] + a[i+22] + a[i+23] + a[i+24] + a[i+25] + a[i+26] + a[i+27] + a[i+28] + a[i+29] + a[i+30] + a[i+31];
    end = (double)clock() / CLOCKS_PER_SEC;
    summ += sum;
    printf("%d,%f\n", 32, (end - start) * 1000);
    free(a);

    printf("%d",summ);
    return 0;
 }