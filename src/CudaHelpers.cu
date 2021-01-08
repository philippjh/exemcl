#ifndef EXEMCL_CUDAHELPERS_H_
#define EXEMCL_CUDAHELPERS_H_

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>

/*
 * Function / macro definitions.
 */
static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#define CUBLAS_CHECK_RETURN(value) CheckCuBlasErrorAux(__FILE__, __LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit the application if the call has failed.
 */
static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line;
    exit(1);
}

/**
 * Check the return value of the cuSolver API call and exit the application if the call has failed.
 */
static void CheckCuBlasErrorAux(const char* file, unsigned line, const char* statement, cublasStatus_t err) {
    if (err == CUBLAS_STATUS_SUCCESS)
        return;
    std::cerr << statement << " returned error code " << err << " at " << file << ":" << line;
    exit(1);
}

/**
 * A struct to communicate kernel configurations.
 */
struct KernelConfiguration {
    dim3 blockDim;
    dim3 gridDim;
    unsigned int sharedMemory;
};

#endif /* EXEMCL_CUDAHELPERS_H_ */
