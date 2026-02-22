#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "utils.h"

#define DEBUG_FLAG 0
#define SIZE 2


// Function to perform matrix multiplication
void multiply(float* mat1, float* mat2, float* res, int size) {

    // Outer product accumulation: for each k, add A[:,k] * B[k,:]
    for (int k = 0; k < size; ++k) {
        for (int i = 0; i < size; ++i) {
            float a_ik = mat1[i * size + k];        // A[i][k] — element of column k in row i
            for (int j = 0; j < size; ++j) {
                float b_kj = mat2[k * size + j];    // B[k][j] — element of row k in column j
                res[i * size + j] += a_ik * b_kj;  // Accumulate outer product
            }
        }
    }
}

int main(int argc, char* argv[]) {

    printf("Testing matrix \n");

    int size = SIZE;
    int input_case = RANDOM_INPUT;
    int DEBUG_PRINT_IO = 0;

    if(IS_HELP){
        printf("options:\n");
        printf("> SIZE\n> INPUT_CASE\n> DEBUG_PRINT_IO\n\n");
        printf("default values:\n");
        printf("> size: %d x %d \n> input_case:%d (%s)\n", 
            size, size,
            input_case, input_case_name(input_case)
        );
        exit(0);
    }


    if( ARG("DEBUG_PRINT_IO") ){
        printf("> passing DEBUG_PRINT_IO");
        DEBUG_PRINT_IO = atoi( ARG("DEBUG_PRINT_IO") );
        printf(" %d\n", DEBUG_PRINT_IO);        
    }
    if( ARG("SIZE") ){
        printf("> passing SIZE");
        size = atoi( ARG("SIZE") );
        printf(" %d\n", size);
    }
    if( ARG("INPUT_CASE") ){
        printf("> passing INPUT_CASE");
        input_case = atoi( ARG("INPUT_CASE") );
        printf(" %d\n", input_case);        
    }


    printf("size: %d x %d     input_case:%d (%s)\n", 
            size, size,
            input_case, input_case_name(input_case)
    );
    
    // Allocate memory for matrices
    float *A = (float*)malloc(size * size * sizeof(float));
    float *B = (float*)malloc(size * size * sizeof(float));
    float *C = (float*)malloc(size * size * sizeof(float));

    // Init matrix values pseudorandom

    // set initial seed for rand, 1 if debug-mode
    srand( DEBUG_FLAG ? 1 : time(NULL) );

    init_matrix_input(input_case, A, B, size, size, size);

    if(DEBUG_PRINT_IO){
        printf("A");
        print_matrixf32(A, size, size, 0);
        printf("B");
        print_matrixf32(B, size, size, 0);
    }

    // Start timer
    clock_t start_time = clock();

    // Perform matrix multiplication (GEMM)
    multiply(A, B, C, size);

    // Stop timer
    clock_t end_time = clock();

    if(DEBUG_PRINT_IO){
        print_matrixf32(C, size, size, 0);
    }

    // Calculate and print execution time
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", execution_time);

    // line to grep results in benchmark phase
    printf("> BENCHMARK_RECORD : version=%s, time=%f, size=%d\n", version(argv[0]), execution_time, size);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
