const char* dgemm_desc = "My awesome dgemm.";
#include <immintrin.h>
#include <string.h>

/* L2 Block sizes */
#define L2_M 64
#define L2_N 64
#define L2_K 256

/* L1 Block sizes */
#define L1_M 32
#define L1_N 32
#define L1_K 64

/* Register Block sizes */
#define REG_M 4
#define REG_N 2
#define REG_K 256

#define min(a,b) (((a)<(b))?(a):(b))
#pragma GCC optimize ("O3")


/*
 * Copy (num_row, num_col) submatrix from source matrix of dim (src_m, src_n)
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
 * TODO: Implement outer product trick introduced in http://web.mit.edu/neboat/www/6.S898-sp17/mm.pdf.
 */
void AVX_matrix_block_multiply(double* restrict A, double* restrict B, double* restrict C, int MM, int NN, int KK) {
    int i, j, k;
    __m256d c[8], left[2], right[4];
    #pragma GCC unroll 2
    for (j = 0; j < NN; j+=4) {
        #pragma GCC unroll 2
        for (i = 0; i < MM; i+=8) {
            c[0] = _mm256_load_pd(&C[i+j*L2_M]);
            c[1] = _mm256_load_pd(&C[i+(j+1)*L2_M]);
            c[2] = _mm256_load_pd(&C[i+(j+2)*L2_M]);
            c[3] = _mm256_load_pd(&C[i+(j+3)*L2_M]);
            c[4] = _mm256_load_pd(&C[i+4+j*L2_M]);
            c[5] = _mm256_load_pd(&C[i+4+(j+1)*L2_M]);
            c[6] = _mm256_load_pd(&C[i+4+(j+2)*L2_M]);
            c[7] = _mm256_load_pd(&C[i+4+(j+3)*L2_M]);

            #pragma GCC unroll 8
            for (k = 0; k < KK; ++k) {
                left[0] = _mm256_load_pd(&A[i+k*L2_M]);
                left[1] = _mm256_load_pd(&A[i+4+k*L2_M]);

                right[0] = _mm256_broadcast_sd(&B[k+j*L2_K]);
                right[1] = _mm256_broadcast_sd(&B[k+(j+1)*L2_K]);
                right[2] = _mm256_broadcast_sd(&B[k+(j+2)*L2_K]);
                right[3] = _mm256_broadcast_sd(&B[k+(j+3)*L2_K]);

                c[0] = _mm256_fmadd_pd(left[0], right[0], c[0]);
                c[1] = _mm256_fmadd_pd(left[0], right[1], c[1]);
                c[2] = _mm256_fmadd_pd(left[0], right[2], c[2]);
                c[3] = _mm256_fmadd_pd(left[0], right[3], c[3]);
                c[4] = _mm256_fmadd_pd(left[1], right[0], c[4]);
                c[5] = _mm256_fmadd_pd(left[1], right[1], c[5]);
                c[6] = _mm256_fmadd_pd(left[1], right[2], c[6]);
                c[7] = _mm256_fmadd_pd(left[1], right[3], c[7]);
            }
            _mm256_storeu_pd(&C[i+j*L2_M], c[0]);
            _mm256_storeu_pd(&C[i+(j+1)*L2_M], c[1]);
            _mm256_storeu_pd(&C[i+(j+2)*L2_M], c[2]);
            _mm256_storeu_pd(&C[i+(j+3)*L2_M], c[3]);
            _mm256_storeu_pd(&C[i+4+j*L2_M], c[4]);
            _mm256_storeu_pd(&C[i+4+(j+1)*L2_M], c[5]);
            _mm256_storeu_pd(&C[i+4+(j+2)*L2_M], c[6]);
            _mm256_storeu_pd(&C[i+4+(j+3)*L2_M], c[7]);
        }
    }
}


void matrix_block_multiply(double* restrict A, double* restrict B, double* restrict C, int MM, int NN, int KK) {
    int i, j, k;
    int dim_i, dim_j, dim_k;
    for (j = 0; j < NN; j += L1_N) {
        dim_j = min (L1_N, (NN-j));
        for (k = 0; k < KK; k += L1_K) {
            dim_k = min (L1_K, (KK-k));
            for (i = 0; i < MM; i += L1_M) {
                dim_i = min (L1_M, (MM-i));
                AVX_matrix_block_multiply(&A[i + k * L2_M], &B[k + j * L2_K], &C[i + j * L2_M], dim_i, dim_j, dim_k);
            }
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
    double* subA = (double*) _mm_malloc(L2_M * L2_K * sizeof(double), 64);
    double* subB = (double*) _mm_malloc(L2_K * L2_N * sizeof(double), 64);
    double* subC = (double*) _mm_malloc(L2_M * L2_N * sizeof(double), 64);
    int i, j, k;
    int dim_i, dim_j, dim_k;
    for (j = 0; j < dim; j += L2_N) {
        dim_j = min (L2_N, (dim-j));
        #pragma GCC ivdep
        #pragma GCC unroll 2
        for (i = 0; i < dim; i += L2_M) {
            dim_i = min (L2_M, (dim-i));
            matrix_copy_aligned(subC, &C[i + j * dim], L2_M, L2_N, dim, dim, dim_i, dim_j, 0);
            for (k = 0; k < dim; k += L2_K)
            {
                dim_k = min (L2_K, (dim-k));
                matrix_copy_aligned(subA,&A[i + k * dim], L2_M, L2_K, dim, dim, dim_i, dim_k, 0);
                matrix_copy_aligned(subB, &B[k + j * dim], L2_K, L2_N, dim, dim, dim_k, dim_j, 0);
                matrix_block_multiply(subA, subB, subC, dim_i, dim_j, dim_k);
            }
            matrix_copy_aligned(&C[i + j * dim], subC, dim, dim, L2_M, L2_N, dim_i, dim_j, 1);
        }
    }

    _mm_free(subA);
    _mm_free(subB);
    _mm_free(subC);
}
