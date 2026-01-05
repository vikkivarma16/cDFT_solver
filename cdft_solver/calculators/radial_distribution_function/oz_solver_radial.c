#include <math.h>
#include <fftw3.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>

void hankel_forward_dst(
    int N,
    const double *r,
    const double *f_r,
    double *k,
    double *F_k
){
    double dr = r[1] - r[0];
    double Rmax = (N + 1) * dr;

    double *x = fftw_malloc(sizeof(double) * N);

    for (int i = 0; i < N; i++) {
        k[i] = M_PI * (i + 1) / Rmax;
        x[i] = r[i] * f_r[i];
    }

    fftw_plan plan = fftw_plan_r2r_1d(
        N, x, x, FFTW_RODFT00, FFTW_ESTIMATE
    );
    fftw_execute(plan);

    for (int i = 0; i < N; i++) {
        F_k[i] = (2.0 * M_PI * dr / k[i]) * x[i];
    }

    fftw_destroy_plan(plan);
    fftw_free(x);
}



void hankel_inverse_dst(
    int N,
    const double *r,
    const double *k,
    const double *F_k,
    double *f_r
){
    double dr = r[1] - r[0];
    double Rmax = (N + 1) * dr;
    double dk = M_PI / Rmax;

    double *y = fftw_malloc(sizeof(double) * N);

    for (int i = 0; i < N; i++)
        y[i] = k[i] * F_k[i];

    fftw_plan plan = fftw_plan_r2r_1d(
        N, y, y, FFTW_RODFT00, FFTW_ESTIMATE
    );
    fftw_execute(plan);

    for (int i = 1; i < N; i++)
        f_r[i] = (dk / (4.0 * M_PI * M_PI * r[i])) * y[i];

    f_r[0] = f_r[1];

    fftw_destroy_plan(plan);
    fftw_free(y);
}





void solve_oz_matrix(
    int N,
    int Nr,
    const double *r,
    const double *densities,
    const double *c_r,
    double *gamma_r
){

    printf("I am running till here !!!");
    const double eps = 1e-12;
    int Nk = Nr;

    double c_k[N*N*Nk];
    double gamma_k[N*N*Nk];
    double k[Nk];

    // Hankel transform each (a,b)
    for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++)
            hankel_forward_dst(
                Nr,
                r,
                &c_r[(a*N + b)*Nr],
                k,
                &c_k[(a*N + b)*Nk]
            );

    // OZ solve
    for (int ik = 0; ik < Nk; ik++) {
        double Ck[N*N];
        double A[N*N];
        double num[N*N];

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Ck[i*N + j] = c_k[(i*N + j)*Nk + ik];

        memset(A, 0, sizeof(A));
        for (int i = 0; i < N; i++)
            A[i*N + i] = 1.0 + eps;

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i*N + j] -= Ck[i*N + j] * densities[j];

        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N,
            1.0, Ck, N,
            Ck, N,
            0.0, num, N
        );

        int ipiv[N];
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, N, A, N, ipiv, num, N);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                gamma_k[(i*N + j)*Nk + ik] = num[i*N + j];
    }

    // Inverse Hankel
    for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++)
            hankel_inverse_dst(
                Nr,
                r,
                k,
                &gamma_k[(a*N + b)*Nk],
                &gamma_r[(a*N + b)*Nr]
            );
}

