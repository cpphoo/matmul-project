const char* dgemm_desc = "My awesome dgemm.";
#include <immintrin.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#define min(a,b) (((a)<(b))?(a):(b))
#pragma GCC optimize ("O3")


/*
 * Copy n_row-by-n_col submatrix from source matrix of size src_size
 * to destination matrix of size dest_size. Both matrices are square.
 * Zero padding if n_row or n_col does not fill entire block_size.
 */
void matrix_copy_aligned(double* restrict destination, const double* restrict source, int dest_size, int src_size, int n_row, int n_col, int block_size) {
    if (n_row < block_size || n_col < block_size) {
        memset(destination, 0, block_size * block_size * sizeof(double));
    }
    for (int j = 0; j < n_col; ++j) {
        memcpy(&destination[j*dest_size], &source[j*src_size], n_row * sizeof(double));
    }
}

/*
 * Performs block matrix multiplication of
 *  C := C + A * B
 * where C is MM-by-NN, A is MM-by-KK, and B is KK-by-NN
 */
void naive_matrix_block_multiply(double* restrict A, double* restrict B, double* restrict C, int block_size, int MM, int NN, int KK) {
    int i, j, k;
    double sum;
    for (i = 0; i < MM; ++i) {
        for (j = 0; j < NN; ++j) {
            sum = C[i+j*block_size];
            for (k = 0; k < KK; ++k) {
                sum += A[k*block_size+i] * B[j*block_size+k];
            }
            C[i+j*block_size] = sum;
        }
    }
}

/*
 */
void square_dgemm(const int N, const double *A, const double *B, double *C)
{
    double* subA = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
    double* subB = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
    double* subC = (double*) _mm_malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double), 64);
    int i, j, k;
    for (j = 0; j < N; j += BLOCK_SIZE) {
        int dim_j = min (BLOCK_SIZE, (N-j));
        #pragma GCC ivdep
        for (i = 0; i < N; i += BLOCK_SIZE) {
            int dim_i = min (BLOCK_SIZE, (N-i));
            matrix_copy_aligned(subC, &C[i + j * N], BLOCK_SIZE, N, dim_i, dim_j, BLOCK_SIZE);
            for (k = 0; k < N; k += BLOCK_SIZE)
            {
                int dim_k = min (BLOCK_SIZE, (N-k));
                matrix_copy_aligned(subA,&A[i + k * N], BLOCK_SIZE, N, dim_i, dim_k, BLOCK_SIZE);
                matrix_copy_aligned(subB,&B[k + j * N], BLOCK_SIZE, N, dim_k, dim_j, BLOCK_SIZE);
                naive_matrix_block_multiply(subA, subB, subC, BLOCK_SIZE, dim_i, dim_j, dim_k);
            }
            matrix_copy_aligned(&C[i + j * N], subC, N, BLOCK_SIZE, dim_i, dim_j, 0);
        }
    }

    _mm_free(subA);
    _mm_free(subB);
    _mm_free(subC);
}