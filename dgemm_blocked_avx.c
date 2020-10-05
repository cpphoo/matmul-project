const char* dgemm_desc = "My awesome dgemm.";
#include <immintrin.h>
#include <string.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
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
 * using AVX instructions.
 * TODO: Implement outer product trick introduced in http://web.mit.edu/neboat/www/6.S898-sp17/mm.pdf.
 */
void AVX_matrix_block_multiply(double* restrict A, double* restrict B, double* restrict C, int MM, int NN, int KK, int block_size) {
    int i, j, k;
    __m256d c[8], left[2], right[4];
    for (i = 0; i < MM; i += 8)
    {
        for (j = 0; j < NN; j += 4)
        {
            c[0] = _mm256_load_pd(&C[i+j*block_size]);
            c[1] = _mm256_load_pd(&C[i+(j+1)*block_size]);
            c[2] = _mm256_load_pd(&C[i+(j+2)*block_size]);
            c[3] = _mm256_load_pd(&C[i+(j+3)*block_size]);
            c[4] = _mm256_load_pd(&C[i+4+j*block_size]);
            c[5] = _mm256_load_pd(&C[i+4+(j+1)*block_size]);
            c[6] = _mm256_load_pd(&C[i+4+(j+2)*block_size]);
            c[7] = _mm256_load_pd(&C[i+4+(j+3)*block_size]);

            for (k = 0; k < KK; ++k) {
                left[0] = _mm256_load_pd(&A[i+k*block_size]);
                left[1] = _mm256_load_pd(&A[i+4+k*block_size]);

                right[0] = _mm256_broadcast_sd(&B[k+j*block_size]);
                right[1] = _mm256_broadcast_sd(&B[k+(j+1)*block_size]);
                right[2] = _mm256_broadcast_sd(&B[k+(j+2)*block_size]);
                right[3] = _mm256_broadcast_sd(&B[k+(j+3)*block_size]);

                c[0] = _mm256_fmadd_pd(left[0], right[0], c[0]);
                c[1] = _mm256_fmadd_pd(left[0], right[1], c[1]);
                c[2] = _mm256_fmadd_pd(left[0], right[2], c[2]);
                c[3] = _mm256_fmadd_pd(left[0], right[3], c[3]);
                c[4] = _mm256_fmadd_pd(left[1], right[0], c[4]);
                c[5] = _mm256_fmadd_pd(left[1], right[1], c[5]);
                c[6] = _mm256_fmadd_pd(left[1], right[2], c[6]);
                c[7] = _mm256_fmadd_pd(left[1], right[3], c[7]);
            }
            _mm256_storeu_pd(&C[i+j*block_size], c[0]);
            _mm256_storeu_pd(&C[i+(j+1)*block_size], c[1]);
            _mm256_storeu_pd(&C[i+(j+2)*block_size], c[2]);
            _mm256_storeu_pd(&C[i+(j+3)*block_size], c[3]);
            _mm256_storeu_pd(&C[i+4+j*block_size], c[4]);
            _mm256_storeu_pd(&C[i+4+(j+1)*block_size], c[5]);
            _mm256_storeu_pd(&C[i+4+(j+2)*block_size], c[6]);
            _mm256_storeu_pd(&C[i+4+(j+3)*block_size], c[7]);
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
    for (i = 0; i < N; i += BLOCK_SIZE)
    {
        int dim_i = min(BLOCK_SIZE, (N - i));
        #pragma GCC ivdep
        for (j = 0; j < N; j += BLOCK_SIZE)
        {
            int dim_j = min(BLOCK_SIZE, (N - j));
            matrix_copy_aligned(subC, &C[i + j * N], BLOCK_SIZE, N, dim_i, dim_j, BLOCK_SIZE);
            for (k = 0; k < N; k += BLOCK_SIZE)
            {
                int dim_k = min (BLOCK_SIZE, (N-k));
                matrix_copy_aligned(subA,&A[i + k * N], BLOCK_SIZE, N, dim_i, dim_k, BLOCK_SIZE);
                matrix_copy_aligned(subB,&B[k + j * N], BLOCK_SIZE, N, dim_k, dim_j, BLOCK_SIZE);
                AVX_matrix_block_multiply(subA, subB, subC, dim_i, dim_j, dim_k, BLOCK_SIZE);
            }
            matrix_copy_aligned(&C[i + j * N], subC, N, BLOCK_SIZE, dim_i, dim_j, 0);
        }
    }

    _mm_free(subA);
    _mm_free(subB);
    _mm_free(subC);
}