const char* dgemm_desc = "My awesome dgemm.";
#include <immintrin.h>
#include <string.h>

/* Block sizes */
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 256

#define min(a,b) (((a)<(b))?(a):(b))
#pragma GCC optimize ("O3")


/*
 * Copy (num_row, num_col) sub-matrix from source matrix of dim (src_m, src_n)
 * to destination matrix of dim (dest_m, dest_n).
 * Zero padding if (num_row, num_col) does not fill entire destination and not in reverse mode.
 */
void matrix_copy_aligned(double* restrict destination, const double* restrict source, int dest_m, int dest_n,
                         int src_m, int src_n, int num_row, int num_col, int reverse) {
    if (!reverse && (num_row < dest_m || num_col < dest_n)) {
        memset(destination, 0, dest_m * dest_n * sizeof(double));
    }
    #pragma GCC unroll 4
    for (int j = 0; j < num_col; ++j) {
        memcpy(&destination[j*dest_m], &source[j*src_m], num_row * sizeof(double));
    }
}


/*
 * Performs block matrix multiplication of
 *  C := C + A * B
 * where C is MM-by-NN, A is MM-by-KK, and B is KK-by-NN
 * using AVX instructions.
 */
void AVX_matrix_BLOCK_Multiply(double* restrict A, double* restrict B, double* restrict C, int MM, int NN, int KK) {
    int i, j, k;
    __m256d c[8], left[2], right[4];
    #pragma GCC unroll 2
    for (i = 0; i < MM; i += 8)
    {
        #pragma GCC unroll 2
        for (j = 0; j < NN; j += 4)
        {
            c[0] = _mm256_load_pd(&C[i+j*BLOCK_M]);
            c[1] = _mm256_load_pd(&C[i+(j+1)*BLOCK_M]);
            c[2] = _mm256_load_pd(&C[i+(j+2)*BLOCK_M]);
            c[3] = _mm256_load_pd(&C[i+(j+3)*BLOCK_M]);
            c[4] = _mm256_load_pd(&C[i+4+j*BLOCK_M]);
            c[5] = _mm256_load_pd(&C[i+4+(j+1)*BLOCK_M]);
            c[6] = _mm256_load_pd(&C[i+4+(j+2)*BLOCK_M]);
            c[7] = _mm256_load_pd(&C[i+4+(j+3)*BLOCK_M]);

            #pragma GCC unroll 8
            for (k = 0; k < KK; ++k) {
                left[0] = _mm256_load_pd(&A[i+k*BLOCK_M]);
                left[1] = _mm256_load_pd(&A[i+4+k*BLOCK_M]);

                right[0] = _mm256_broadcast_sd(&B[k+j*BLOCK_K]);
                right[1] = _mm256_broadcast_sd(&B[k+(j+1)*BLOCK_K]);
                right[2] = _mm256_broadcast_sd(&B[k+(j+2)*BLOCK_K]);
                right[3] = _mm256_broadcast_sd(&B[k+(j+3)*BLOCK_K]);

                c[0] = _mm256_fmadd_pd(left[0], right[0], c[0]);
                c[1] = _mm256_fmadd_pd(left[0], right[1], c[1]);
                c[2] = _mm256_fmadd_pd(left[0], right[2], c[2]);
                c[3] = _mm256_fmadd_pd(left[0], right[3], c[3]);
                c[4] = _mm256_fmadd_pd(left[1], right[0], c[4]);
                c[5] = _mm256_fmadd_pd(left[1], right[1], c[5]);
                c[6] = _mm256_fmadd_pd(left[1], right[2], c[6]);
                c[7] = _mm256_fmadd_pd(left[1], right[3], c[7]);
            }
            _mm256_storeu_pd(&C[i+j*BLOCK_M], c[0]);
            _mm256_storeu_pd(&C[i+(j+1)*BLOCK_M], c[1]);
            _mm256_storeu_pd(&C[i+(j+2)*BLOCK_M], c[2]);
            _mm256_storeu_pd(&C[i+(j+3)*BLOCK_M], c[3]);
            _mm256_storeu_pd(&C[i+4+j*BLOCK_M], c[4]);
            _mm256_storeu_pd(&C[i+4+(j+1)*BLOCK_M], c[5]);
            _mm256_storeu_pd(&C[i+4+(j+2)*BLOCK_M], c[6]);
            _mm256_storeu_pd(&C[i+4+(j+3)*BLOCK_M], c[7]);
        }
    }
}


/*
 * TODO 1: Try different block sizes (not necessarily square) for A, B and C to increase L1/L2 cache hits.
 * TODO 2: Transpose A/B to row major.
 * TODO 3: Loop unrolling.
 */
void square_dgemm(const int dim, const double *A, const double *B, double *C)
{
    double* subA = (double*) _mm_malloc(BLOCK_M * BLOCK_K * sizeof(double), 64);
    double* subB = (double*) _mm_malloc(BLOCK_K * BLOCK_N * sizeof(double), 64);
    double* subC = (double*) _mm_malloc(BLOCK_M * BLOCK_N * sizeof(double), 64);
    int i, j, k;
    int dim_i, dim_j, dim_k;
    for (i = 0; i < dim; i += BLOCK_M)
    {
        dim_i = min(BLOCK_M, (dim - i));
        
        #pragma GCC ivdep
        #pragma GCC unroll 2
        for (j = 0; j < dim; j += BLOCK_N)
        {
            dim_j = min(BLOCK_N, (dim - j));
            matrix_copy_aligned(subC, &C[i + j * dim], BLOCK_M, BLOCK_N, dim, dim, dim_i, dim_j, 0);
            for (k = 0; k < dim; k += BLOCK_K)
            {
                dim_k = min (BLOCK_K, (dim-k));
                matrix_copy_aligned(subA,&A[i + k * dim], BLOCK_M, BLOCK_K, dim, dim, dim_i, dim_k, 0);
                matrix_copy_aligned(subB, &B[k + j * dim], BLOCK_K, BLOCK_N, dim, dim, dim_k, dim_j, 0);
                AVX_matrix_BLOCK_Multiply(subA, subB, subC, dim_i, dim_j, dim_k);
            }
            matrix_copy_aligned(&C[i + j * dim], subC, dim, dim, BLOCK_M, BLOCK_N, dim_i, dim_j, 1);
        }
    }

    _mm_free(subA);
    _mm_free(subB);
    _mm_free(subC);
}
