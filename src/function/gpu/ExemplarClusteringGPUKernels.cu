#ifndef EXEMCL_EXEMPLARCLUSTERINGGPUKERNELS_CU
#define EXEMCL_EXEMPLARCLUSTERINGGPUKERNELS_CU

template<typename DeviceDataType>
__global__ void exemplarClusteringKernel(const DeviceDataType* vMatrix, const int nV, const DeviceDataType* summaryMatrix, const int maxS, const int* summarySizes,
                                         const int nS_multi, const int dim, DeviceDataType* resultMatrix) {
    // Create a variable, which represents the current v and S to work on.
    int vJob = blockDim.x * blockIdx.x + threadIdx.x;
    int sJob = blockDim.y * blockIdx.y + threadIdx.y;

    // Check, whether we have a valid V job.
    if (vJob < nV) {
        // Load the current v into shared memory.
        extern __shared__ unsigned char _vShared[];
        auto* vShared = reinterpret_cast<DeviceDataType*>(_vShared);

        if (threadIdx.y == 0) {
            for (int d = 0; d < dim; d++) {
                vShared[threadIdx.x * dim + d] = vMatrix[d * nV + vJob];
            }
        }

        // Ensure, that every block has loaded "their" v vectors into shared memory.
        __syncthreads();

        // Check, whether we have a valid S job.
        if (sJob < nS_multi) {
            // Compute the minimum distance for v and all vectors in S.
            DeviceDataType minDistance = std::numeric_limits<DeviceDataType>::max();

            // Iterate over all vectors in S.
            for (int i = 0; i < summarySizes[sJob]; i++) {
                DeviceDataType distance = 0.0;
                for (int d = 0; d < dim; d++) {
                    if constexpr (std::is_same<DeviceDataType, float>::value)
                        distance += powf(vShared[threadIdx.x * dim + d] - summaryMatrix[i * nS_multi + d * maxS * nS_multi + sJob], 2.0f);
                    else
                        distance += pow(vShared[threadIdx.x * dim + d] - summaryMatrix[i * nS_multi + d * maxS * nS_multi + sJob], 2.0);
                }
                minDistance = minDistance > distance ? distance : minDistance;
            }

            // Write the min distance for this summary into memory.
            resultMatrix[(long) vJob * (long) nS_multi + (long) sJob] = minDistance / (DeviceDataType) nV;
        }
    }
}

__global__ void exemplarClusteringKernel(const __half* vMatrix, const int nV, const __half* summaryMatrix, const int maxS, const int* summarySizes, const int nS_multi,
                                         const int dim, float* resultMatrix) {
#define V_ACCESS(dim_idx) vSharedHalf[threadIdx.x * dim + (dim_idx)]
#define SMAT_ACCESS(dim_idx) summaryMatrix[i * nS_multi + (dim_idx) *maxS * nS_multi + sJob]
    // Create a variable, which represents the current v and S to work on.
    int vJob = blockDim.x * blockIdx.x + threadIdx.x;
    int sJob = blockDim.y * blockIdx.y + threadIdx.y;

    // Check, whether we have a valid V job.
    if (vJob < nV) {
        // Load the current v into shared memory.
        extern __shared__ __half vSharedHalf[];
        if (threadIdx.y == 0) {
            for (int d = 0; d < dim; d++) {
                vSharedHalf[threadIdx.x * dim + d] = vMatrix[d * nV + vJob];
            }
        }

        // Ensure, that every block has loaded "their" v vectors into shared memory.
        __syncthreads();

        // Check, whether we have a valid S job.
        if (sJob < nS_multi) {
            // Compute the minimum distance for v and all vectors in S.
            __half minDistance = 0x7BFF; // fp16 max

            // Iterate over all vectors in S.
            for (int i = 0; i < summarySizes[sJob]; i++) {
                __half2 distance2(0.0, 0.0);
                int d = 0;
                while (d < dim) {
                    __half2 vData(V_ACCESS(d), V_ACCESS(d + 1));
                    __half2 sData(SMAT_ACCESS(d), SMAT_ACCESS(d + 1));
                    __half2 diffData = __hsub2(vData, sData);
                    __half2 powData = __hmul2(diffData, diffData);
                    distance2 += powData;

                    // Increase `d`.
                    d += 2;
                }
                __half distance = distance2.x + distance2.y;
                if (dim % 2 == 1) {
                    __half diff = V_ACCESS(dim - 1) - SMAT_ACCESS(dim - 1);
                    distance += __hmul(diff, diff);
                }

                minDistance = minDistance > distance ? distance : minDistance;

                // Write the min distance for this summary into memory.
                resultMatrix[(long) vJob * (long) nS_multi + (long) sJob] = minDistance / (__half) nV;
            }
        }
    }
}

#endif // EXEMCL_EXEMPLARCLUSTERINGGPUKERNELS_CU
