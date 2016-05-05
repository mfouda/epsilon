#ifndef EPSILON_LINEAR_LAPACK_H
#define EPSILON_LINEAR_LAPACK_H
extern "C" {

void dgemv_(
    char* transa, int* m, int* n, double* alpha, double* A, int* lda, double* x,
    int* incx, double* beta, double* y, int* incy);

void dgemm_(
    char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *A, int *lda, double *B, int *ldb,
    double *beta, double *C, int *ldc);

}  // extern "C"


#endif  // EPSILON_LINEAR_LAPACK_H
