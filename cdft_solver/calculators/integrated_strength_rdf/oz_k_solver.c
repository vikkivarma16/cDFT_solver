#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>

void solve_linear_system(int *N_ptr, int *M_ptr, double *A, double *B) {
    int N = *N_ptr;  // size of matrix
    int M = *M_ptr;  // number of RHS columns
    int info;
    
    int *ipiv = (int*)malloc(N * sizeof(int));
    if (!ipiv) {
        printf("Memory allocation failed!\n");
        return;
    }

    // Solve A*X = B
    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, M, A, N, ipiv, B, N);
    if (info != 0) {
        printf("LAPACKE_dgesv failed: info=%d\n", info);
    }

    free(ipiv);
}

