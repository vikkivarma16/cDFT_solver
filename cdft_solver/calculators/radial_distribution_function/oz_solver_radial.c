#include <math.h>
#include <fftw3.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>
#include <stdlib.h>
#include <stdio.h>
//gcc -O3 -fPIC -shared oz_solver_radial.c -o liboz_radial.so -llapacke -lopenblas -lfftw3
void hankel_forward_dst(int N, const double *r, const double *f_r, double *k, double *F_k) {
    double dr = r[1] - r[0];
    double Rmax = (N + 1) * dr;

    double *x = fftw_malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        k[i] = M_PI * (i + 1) / Rmax;
        x[i] = r[i] * f_r[i];
    }

    fftw_plan plan = fftw_plan_r2r_1d(N, x, x, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < N; i++)
        F_k[i] = (2.0 * M_PI * dr / k[i]) * x[i];

    fftw_destroy_plan(plan);
    fftw_free(x);
}

void hankel_inverse_dst(int N, const double *r, const double *k, const double *F_k, double *f_r) {
    double dr = r[1] - r[0];
    double Rmax = N * dr;
    double dk = M_PI / Rmax;

    double *y = fftw_malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) y[i] = k[i] * F_k[i];

    fftw_plan plan = fftw_plan_r2r_1d(N, y, y, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan);

    for (int i = 0; i < N; i++)
        f_r[i] = (dk / (4.0 * M_PI * M_PI * r[i])) * y[i];

    fftw_destroy_plan(plan);
    fftw_free(y);
}

void solve_oz_matrix(int N, int Nr, const double *r, const double *densities, double *c_r, double *gamma_r) {
    const double eps = 1e-8;
    int Nk = Nr;

    // allocate on heap
    double *c_k = (double *)malloc(N*N*Nk*sizeof(double));
    double *gamma_k = (double *)malloc(N*N*Nk*sizeof(double));
    double *k = (double *)malloc(Nk*sizeof(double));

    double *Ck = (double *)malloc(N*N*sizeof(double));
    double *A = (double *)malloc(N*N*sizeof(double));
    double *num = (double *)malloc(N*N*sizeof(double));
    int *ipiv = (int *)malloc(N*sizeof(int));

    if (!c_k || !gamma_k || !k || !Ck || !A || !num || !ipiv) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    // Hankel transform each (a,b)
   
    
    
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < N; b++) {

            // row-major flattening: index = a*Nr*N + b*Nr
            int idx = a * Nr * N + b * Nr;
            hankel_forward_dst(
                Nr,
                r,
                &c_r[idx],
                k,
                &c_k[a * Nk * N + b * Nk]
            );
            
        }
    }
    
    
    

    // OZ solve in k-space
    for (int ik = 0; ik < Nk; ik++) {
        // extract Ck for this k
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Ck[i*N + j] = c_k[i*Nk*N + j*Nk + ik];

        // build A = I - C*rho + eps*I
        memset(A, 0, N*N*sizeof(double));
        for (int i = 0; i < N; i++) A[i*N + i] = 1.0 + eps;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i*N + j] -= Ck[i*N + j] * densities[j];

        // num = Ck * rho * Ck
        double *rho = (double *)malloc(N*N*sizeof(double));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                rho[i*N + j] = (i==j) ? densities[i] : 0.0;

        double *tmp = malloc(N*N*sizeof(double));

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N,
            1.0,
            Ck,  N,
            rho, N,
            0.0,
            tmp, N
        );

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N,
            1.0,
            tmp, N,
            Ck,  N,
            0.0,
            num, N
        );

        free(tmp);
        free(rho);
        //double norm = 0.0;
        //for (int i = 0; i < N*N; i++) norm += fabs(num[i]);
        //printf("||num|| = %e\n", norm);


        // solve A * gamma_k = num
        int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, N, A, N, ipiv, num, N);
        if (info != 0) {
            printf("dgesv failed at ik=%d, info=%d\n", ik, info);
        }


        // copy to gamma_k
        for (int i = 0; i < N; i++){
        
            for (int j = 0; j < N; j++){
                gamma_k[i*Nk*N + j*Nk + ik] = num[i*N + j];
                
                //printf ("%lf  %lf \n", num[i*N + j], densities[i]);
            }    
        }
    }
   
    // inverse Hankel
    for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++)
            hankel_inverse_dst(
                Nr,
                r,
                k,
                &gamma_k[a*Nk*N + b*Nk],
                &gamma_r[a*Nr*N + b*Nr]
            );

    // free memory
    free(c_k);
    free(gamma_k);
    free(k);
    free(Ck);
    free(A);
    free(num);
    free(ipiv);
}

