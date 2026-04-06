#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <stdlib.h>
#include <stdio.h>

// -------------------------------------------------
// Build J0 kernel
// -------------------------------------------------
void build_J0(int Nr, int Nk, const double *r, const double *k, double *J0) {
    for (int ik = 0; ik < Nk; ik++) {
        for (int ir = 0; ir < Nr; ir++) {
            J0[ik * Nr + ir] = j0(k[ik] * r[ir]);
        }
    }
}

// -------------------------------------------------
// Forward Hankel (2D)
// -------------------------------------------------
void hankel_forward_2d(
    int Nr, int Nk,
    const double *r,
    const double *k,
    const double *f_r,
    double *F_k
) {
    double dr = r[1] - r[0];

    double *J0 = malloc(Nk * Nr * sizeof(double));
    build_J0(Nr, Nk, r, k, J0);

    for (int ik = 0; ik < Nk; ik++) {
        double sum = 0.0;
        for (int ir = 0; ir < Nr; ir++) {
            sum += J0[ik*Nr + ir] * r[ir] * f_r[ir];
        }
        F_k[ik] = 2.0 * M_PI * sum * dr;
    }

    free(J0);
}

// -------------------------------------------------
// Inverse Hankel (2D)
// -------------------------------------------------
void hankel_inverse_2d(
    int Nr, int Nk,
    const double *r,
    const double *k,
    const double *F_k,
    double *f_r
) {
    double dk = k[1] - k[0];

    double *J0 = malloc(Nk * Nr * sizeof(double));
    build_J0(Nr, Nk, r, k, J0);

    for (int ir = 0; ir < Nr; ir++) {
        double sum = 0.0;
        for (int ik = 0; ik < Nk; ik++) {
            sum += J0[ik*Nr + ir] * k[ik] * F_k[ik];
        }
        f_r[ir] = sum * dk / (2.0 * M_PI);
    }

    free(J0);
}

// -------------------------------------------------
// OZ solver (2D)
// -------------------------------------------------
void solve_oz_matrix(
    int N, int Nr,
    const double *r,
    const double *densities,
    double *c_r,
    double *gamma_r
) {
    int Nk = Nr;

    double *k = malloc(Nk * sizeof(double));
    double Rmax = r[Nr-1];

    for (int i = 0; i < Nk; i++)
        k[i] = (i + 1) * M_PI / Rmax;

    double *c_k = malloc(N * N * Nk * sizeof(double));
    double *gamma_k = malloc(N * N * Nk * sizeof(double));

    double *Ck = malloc(N * N * sizeof(double));
    double *A = malloc(N * N * sizeof(double));
    double *num = malloc(N * N * sizeof(double));
    int *ipiv = malloc(N * sizeof(int));

    // -----------------------------
    // Forward transform
    // -----------------------------
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < N; b++) {

            int idx = (a*N + b)*Nr;

            hankel_forward_2d(
                Nr, Nk,
                r, k,
                &c_r[idx],
                &c_k[(a*N + b)*Nk]
            );
        }
    }

    // -----------------------------
    // OZ solve in k-space
    // -----------------------------
    for (int ik = 0; ik < Nk; ik++) {

        // extract C(k)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Ck[i*N + j] = c_k[(i*N + j)*Nk + ik];

        // A = I - C rho
        for (int i = 0; i < N*N; i++) A[i] = 0.0;
        for (int i = 0; i < N; i++) A[i*N + i] = 1.0;

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i*N + j] -= Ck[i*N + j] * densities[j];

        // num = C rho C
        for (int i = 0; i < N*N; i++) num[i] = 0.0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int m = 0; m < N; m++) {
                    num[i*N + j] += Ck[i*N + m] * densities[m] * Ck[m*N + j];
                }
            }
        }

        // solve
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, N, N, A, N, ipiv, num, N);

        for (int i = 0; i < N*N; i++)
            gamma_k[i*Nk + ik] = num[i];
    }

    // -----------------------------
    // Inverse transform
    // -----------------------------
    for (int a = 0; a < N; a++) {
        for (int b = 0; b < N; b++) {

            hankel_inverse_2d(
                Nr, Nk,
                r, k,
                &gamma_k[(a*N + b)*Nk],
                &gamma_r[(a*N + b)*Nr]
            );
        }
    }

    free(k);
    free(c_k);
    free(gamma_k);
    free(Ck);
    free(A);
    free(num);
    free(ipiv);
}
