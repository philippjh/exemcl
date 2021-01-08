#ifndef EXEMCL_EXEMPLARCLUSTERINGSUBMODULARFUNCTION_CUH
#define EXEMCL_EXEMPLARCLUSTERINGSUBMODULARFUNCTION_CUH

#include "ExemplarClusteringGPUKernels.cu"
#include <cublas.h>
#include <cuda_fp16.h>
#include <src/CudaHelpers.cu>
#include <src/function/SubmodularFunction.h>

namespace exemcl::gpu {
    /**
     * This class provides a GPU implementation of the submodular function of Exemplar-based clustering.
     */
    template<typename DeviceDataType = float, typename HostDataType = float>
    class ExemplarClusteringSubmodularFunction : public SubmodularFunction<HostDataType> {
    public:
        using SubmodularFunction<HostDataType>::operator();
        using SubmodularFunction<HostDataType>::_workerCount;
        static_assert((std::is_same<HostDataType, float>::value && std::is_same<DeviceDataType, __half>::value)
                          || (std::is_same<HostDataType, float>::value && std::is_same<DeviceDataType, float>::value)
                          || (std::is_same<HostDataType, double>::value && std::is_same<DeviceDataType, double>::value),
                      "<DeviceDataType, HostDataType> needs to be either <__half, float>, <float, float> or <double, double>.");
        using HostOpDataType = typename std::conditional<std::is_same<DeviceDataType, __half>::value, Eigen::half, HostDataType>::type;

        // Disable copy constructor.
        ExemplarClusteringSubmodularFunction(const ExemplarClusteringSubmodularFunction&) = delete;

        /**
         * Instantiates the submodular function of Exemplar-based clustering.
         * @param V The ground set to operate on.
         * @param workerCount The number of workers to employ (defaults to -1, i.e. all available cores).
         */
        explicit ExemplarClusteringSubmodularFunction(const exemcl::MatrixX<HostDataType, Eigen::ColMajor>& V, int workerCount = -1) {
            // Store V shape.
            _vShape[0] = V.rows();
            _vShape[1] = V.cols();

            // Copy the vMatrix on the GPU.
            _gpuVMatrixMemoryAllocated = V.size() * sizeof(DeviceDataType);
            CUDA_CHECK_RETURN(cudaMalloc((void**) &_vMatrix, _gpuVMatrixMemoryAllocated));
            if constexpr (std::is_same<DeviceDataType, __half>::value) {
                auto VCasted = std::make_unique<exemcl::MatrixX<HostOpDataType, Eigen::ColMajor>>(V.template cast<HostOpDataType>());
                CUDA_CHECK_RETURN(cudaMemcpy(_vMatrix, VCasted->data(), _gpuVMatrixMemoryAllocated, cudaMemcpyHostToDevice));
            } else {
                CUDA_CHECK_RETURN(cudaMemcpy(_vMatrix, V.data(), _gpuVMatrixMemoryAllocated, cudaMemcpyHostToDevice));
            }

            // Configure shared memory bank size accordingly.
            if constexpr (std::is_same<HostDataType, float>::value)
                CUDA_CHECK_RETURN(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));
            else if constexpr (std::is_same<HostDataType, double>::value)
                CUDA_CHECK_RETURN(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

            // Evaluate the zero vec value.
            auto* accuArray = new HostDataType[V.rows()];
#pragma omp parallel for num_threads(_workerCount)
            for (unsigned int i = 0; i < V.rows(); i++)
                accuArray[i] = (-1 * V.row(i)).squaredNorm();

            _zeroVecValue = 0.0;
#pragma omp simd reduction(+ : _zeroVecValue)
            for (unsigned int i = 0; i < V.rows(); i++)
                _zeroVecValue += accuArray[i];
            delete[] accuArray;
            _zeroVecValue /= V.rows();

            // Create cuBLAS handle.
            cublasCreate(&_handle);
        };

        /**
         * Evaluates the function value for a set of sets, yielding a function value for every set in `S_multi`.
         * @param S_multi The set of sets, which should be evaluated for their respective function value.
         * @return A list of function values one for each set in `S_multi`.
         */
        std::vector<HostDataType> operator()(const std::vector<MatrixX<HostDataType>>& S_multi) const override {
            auto S_multi_copy = std::make_unique<std::vector<MatrixX<HostDataType>>>(S_multi);

            // Add the zero vector to all summaries.
            unsigned long maxS = 0;
            for (auto& S : *S_multi_copy) {
                S.conservativeResize(S.rows() + 1, S.cols());
                S.row(S.rows() - 1).setZero();
                maxS = std::max(maxS, (unsigned long) S.rows());
            }

            if (maxS > 0) {
                // Acquire memory information from GPU.
                size_t freeGPUMemory;
                size_t totalGPUMemory;
                CUDA_CHECK_RETURN(cudaMemGetInfo(&freeGPUMemory, &totalGPUMemory));
                freeGPUMemory *= 0.95; // We will assume, that only 95% of free GPU memory is actually available (for robustness).

                // Take a possibly set gpu memory limit into account.
                if (_gpuMemoryLimit > 0)
                    freeGPUMemory = freeGPUMemory < _gpuMemoryLimit ? freeGPUMemory : _gpuMemoryLimit;

                // Check, whether we need chunking.
                auto chunking = calculateProblemDependentChunking(freeGPUMemory, S_multi_copy->size(), (*S_multi_copy)[0].cols(), maxS);
                size_t totalGPUMemoryReq = std::get<0>(chunking);

                // Calculate the function values.
                std::vector<HostDataType> multiSummaryValues;
                if (freeGPUMemory >= totalGPUMemoryReq) {
                    // If enough memory is available, we will just compute the function for every summary.
                    multiSummaryValues = L(*S_multi_copy);
                } else {
                    // Otherwise, we will split the problem into smaller chunks.
                    unsigned long chunkSize = std::get<1>(chunking);
                    unsigned long totalChunks = std::get<2>(chunking);
#ifndef NDEBUG
                    std::cout << "Problem has been splitted into " << totalChunks << " chunks of size " << chunkSize << "!" << std::endl;
#endif

                    // Evaluate function for every chunk.
                    multiSummaryValues.reserve(S_multi_copy->size());
                    auto it = S_multi_copy->begin();
                    for (unsigned int i = 0; i < totalChunks; i++) {
                        auto itLimit = std::distance(S_multi_copy->begin(), it + chunkSize) > S_multi_copy->size() ? std::distance(it, S_multi_copy->end()) : chunkSize;
                        auto S_multi_chunked = std::make_unique<std::vector<MatrixX<HostDataType>>>(it, it + itLimit);
                        auto multiChunkSummaryValues = L(*S_multi_chunked);
                        multiSummaryValues.insert(multiSummaryValues.end(), multiChunkSummaryValues.begin(), multiChunkSummaryValues.end());
                        it = it + itLimit;
                    }
                }

                // Subtract the zero vec value.
#pragma omp parallel for num_threads(_workerCount)
                for (int i = 0; i < S_multi_copy->size(); i++)
                    multiSummaryValues[i] = _zeroVecValue - multiSummaryValues[i];

                // Return the result.
                return multiSummaryValues;
            } else
                return {};
        };

        /**
         * Evaluates the function value for a set of sets, yielding a function value for every set in `S_multi`.
         * @param S_multi The set of sets, which should be evaluated for their respective function value.
         * @return A list of function values one for each set in `S_multi`.
         */
        std::vector<HostDataType> operator()(const std::vector<MatrixX<HostDataType>>& S_multi) override {
            return ((const ExemplarClusteringSubmodularFunction*) (this))->operator()(S_multi);
        };

        /**
         * Evaluates a single set for its function value.
         * @param S The dataset to evaluate for its function value.
         * @return The function value.
         */
        HostDataType operator()(const MatrixX<HostDataType>& S) const override {
            if (S.rows() > 0) {
                MatrixX<HostDataType> S_copy = S;

                // Create the zero vector and add it to the summary to evaluate.
                S_copy.conservativeResize(S_copy.rows() + 1, S_copy.cols());
                S_copy.row(S_copy.rows() - 1).setZero();

                return _zeroVecValue - L(S_copy);
            } else
                return 0.0;
        };

        /**
         * Evaluates a single set for its function value.
         * @param S The dataset to evaluate for its function value.
         * @return The function value.
         */
        HostDataType operator()(const MatrixX<HostDataType>& S) override {
            return ((const ExemplarClusteringSubmodularFunction*) (this))->operator()(S);
        };

        /**
         * Sets a limit regarding the used GPU memory by this class. Please note, that this restriction only affects additionally allocated memory by specific function evaluations.
         * Permanently allocated memory (like ground set information) is not being limited in any form.
         *
         * @param memoryLimit GPU memory limit (in bytes).
         */
        void setGPUMemoryLimit(long memoryLimit) {
            // Subtract the number of bytes we allocated for the V matrix, which, of course, is not available anymore.
            long newMemoryLimit = memoryLimit - _gpuVMatrixMemoryAllocated;
            if (newMemoryLimit < 0)
                throw std::runtime_error("ExemplarClusteringSubmodularFunction::setGPUMemoryLimit: Inadequate memory limit set. No more memory for function evaluations left. "
                                         "Please set a higher memory limit.");
            else
                _gpuMemoryLimit = newMemoryLimit;
        }

        /**
         * Destructor, which effectively releases GPU memory and destroys the cuBLAS handle.
         */
        virtual ~ExemplarClusteringSubmodularFunction() {
            // Release GPU memory.
            CUDA_CHECK_RETURN(cudaFree(_vMatrix));

            // Destory cuBLAS handle.
            cublasDestroy(_handle);
        };

    private:
        HostDataType _zeroVecValue;
        std::array<unsigned long, 2> _vShape;

        // CuBLAS
        cublasHandle_t _handle;

        // GPU memory.
        DeviceDataType* _vMatrix;
        size_t _gpuVMatrixMemoryAllocated;
        long _gpuMemoryLimit = -1; // allows to limit the usable GPU memory (-1 = no limit).

        /**
         * Evaluates the `L` function value, which is an important subtask to find the function value.
         * @param S_multi The set of sets, which should be evaluated for their respective `L` function value.
         * @return A list of `L` function values one for each set in `S_multi`.
         */
        std::vector<HostDataType> L(std::vector<MatrixX<HostDataType>>& S_multi) const {
            // Build the summary matrix.
            Eigen::Matrix<HostOpDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>* summaryMatrix;
            int* summarySizes;
            int maxS;
            buildSummary(S_multi, &maxS, &summarySizes, &summaryMatrix);

            // Calculate kernel configuration.
            KernelConfiguration kernelConf = calculateKernelConfiguration(S_multi);

            // Allocate memory for the summaryMatrix and the results vector.
            DeviceDataType* gpuSummaryMatrix;
            int* gpuSummarySizes;
            HostDataType* gpuResultMatrix;

            CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuSummaryMatrix, summaryMatrix->rows() * summaryMatrix->cols() * sizeof(DeviceDataType)));
            CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuSummarySizes, S_multi.size() * sizeof(int)));
            CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuResultMatrix, S_multi.size() * _vShape[0] * sizeof(HostDataType)));

            // Copy the matrices on the GPU.
            CUDA_CHECK_RETURN(cudaMemcpy(gpuSummaryMatrix, summaryMatrix->data(), summaryMatrix->rows() * summaryMatrix->cols() * sizeof(DeviceDataType), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(gpuSummarySizes, summarySizes, S_multi.size() * sizeof(int), cudaMemcpyHostToDevice));

            // Invoke kernel.
            if constexpr (std::is_same<DeviceDataType, __half>::value) {
                exemplarClusteringKernel<<<kernelConf.gridDim, kernelConf.blockDim, kernelConf.sharedMemory>>>(_vMatrix, (int) _vShape[0], gpuSummaryMatrix, maxS, gpuSummarySizes,
                                                                                                               (int) S_multi.size(), (int) _vShape[1], gpuResultMatrix);
            } else {
                exemplarClusteringKernel<DeviceDataType><<<kernelConf.gridDim, kernelConf.blockDim, kernelConf.sharedMemory>>>(
                    _vMatrix, (int) _vShape[0], gpuSummaryMatrix, maxS, gpuSummarySizes, (int) S_multi.size(), (int) _vShape[1], gpuResultMatrix);
            }

            // Copy auxiliary one vector to GPU as long as we are waiting for the result matrix to arrive.
            VectorX<HostDataType> oneVector = VectorX<HostDataType>::Ones(_vShape[0]);
            HostDataType* gpuOneVector;
            CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuOneVector, _vShape[0] * sizeof(HostDataType)));
            CUDA_CHECK_RETURN(cudaMemcpy(gpuOneVector, oneVector.data(), _vShape[0] * sizeof(HostDataType), cudaMemcpyHostToDevice));

            // Allocate memory for the result of the row reduction by sum.
            HostDataType* gpuRowReductionVector;
            CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuRowReductionVector, S_multi.size() * sizeof(HostDataType)));

            // Wait for the results to arrive.
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
            CUDA_CHECK_RETURN(cudaPeekAtLastError());

            // Copy results back.
            if constexpr (std::is_same<HostDataType, float>::value) {
                float alpha = 1.0;
                float beta = 0.0;
                CUBLAS_CHECK_RETURN(
                    cublasSgemv(_handle, CUBLAS_OP_N, S_multi.size(), _vShape[0], &alpha, gpuResultMatrix, S_multi.size(), gpuOneVector, 1, &beta, gpuRowReductionVector, 1));
            } else if constexpr (std::is_same<HostDataType, double>::value) {
                double alpha = 1.0;
                double beta = 0.0;
                CUBLAS_CHECK_RETURN(
                    cublasDgemv(_handle, CUBLAS_OP_N, S_multi.size(), _vShape[0], &alpha, gpuResultMatrix, S_multi.size(), gpuOneVector, 1, &beta, gpuRowReductionVector, 1));
            }

            std::vector<HostDataType> finalResultVector(S_multi.size(), 0.0);
            CUDA_CHECK_RETURN(cudaMemcpy(finalResultVector.data(), gpuRowReductionVector, S_multi.size() * sizeof(HostDataType), cudaMemcpyDeviceToHost));

            // Release data.
            delete[] summarySizes;
            delete summaryMatrix;
            CUDA_CHECK_RETURN(cudaFree(gpuSummaryMatrix));
            CUDA_CHECK_RETURN(cudaFree(gpuSummarySizes));
            CUDA_CHECK_RETURN(cudaFree(gpuResultMatrix));
            CUDA_CHECK_RETURN(cudaFree(gpuOneVector));
            CUDA_CHECK_RETURN(cudaFree(gpuRowReductionVector));

            return finalResultVector;
        };

        HostDataType L(const MatrixX<HostDataType>& S) const {
            std::vector<MatrixX<HostDataType>> S_multi = {S};
            return L(S_multi)[0];
        };

        /**
         * Builds the summary matrix, which is necessary for this particular GPU computation.
         *
         * This function will dynamically allocate memory at retSummarySizes and retSummaryMatrix, which manually has to be freed in order to prevent memory leakage.
         *
         * @param S_multi The set of sets for which the summary matrix should be constructed.
         * @param retMaxS Pointer, to which the maximal cardinality of sets in `S_multi` should be written to.
         * @param retSummarySizes Pointer pointing to a location, to which the sizes of every set in `S_multi` should be written to.
         * @param retSummaryMatrix Pointer pointng to a location, to which the Eigen summary matrix should be written to.
         */
        void buildSummary(const std::vector<MatrixX<HostDataType>>& S_multi, int* retMaxS, int** retSummarySizes,
                          Eigen::Matrix<HostOpDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>** retSummaryMatrix) const {
            // Allocate memory for summary sizes.
            int* summarySizes = new int[S_multi.size()];

            // Check common dim, find max |S|.
            long dim = _vShape[1];
            unsigned long maxS = 0;
            for (unsigned int i = 0; i < S_multi.size(); i++) {
                if (dim != S_multi[i].cols()) {
                    delete[] summarySizes;
                    throw std::runtime_error("ExemplarClusteringGPUSubmodularFunction::buildSummary: The dimensionalities in S do not match (" + std::to_string(dim) + " vs. "
                                             + std::to_string(S_multi[i].cols()) + ")");
                }
                maxS = std::max(maxS, (unsigned long) (S_multi[i].rows()));

                // Assign summary sizes.
                summarySizes[i] = S_multi[i].rows();
            }

            // Create summary matrix in round-robin fashion.
            auto summaryMatrix = new Eigen::Matrix<HostOpDataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(maxS * S_multi.size(), dim);
            unsigned long k = 0;
            for (unsigned long i = 0; i < maxS; i++) {
                for (unsigned long j = 0; j < S_multi.size(); j++) {
                    const MatrixX<HostDataType>& S = S_multi[j];
                    if (i < S.rows())
                        summaryMatrix->row(k) = S.row(i).template cast<HostOpDataType>();
                    k++;
                }
            }

            // Assign return values.
            *retMaxS = maxS;
            *retSummarySizes = summarySizes;
            *retSummaryMatrix = summaryMatrix;
        };

        /**
         * Calculate the kernel configuration for a particular problem described by `S_multi`.
         * @param S_multi The set of sets for which the kernel configuration should be calculated.
         * @return The kernel configuration which solves the problem.
         */
        KernelConfiguration calculateKernelConfiguration(const std::vector<MatrixX<HostDataType>>& S_multi) const {
            const unsigned int threadCount = 1024;
            int deviceNo;
            CUDA_CHECK_RETURN(cudaGetDevice(&deviceNo));
            int sharedMemorySize;
            CUDA_CHECK_RETURN(cudaDeviceGetAttribute(&sharedMemorySize, cudaDevAttrMaxSharedMemoryPerBlock, deviceNo));
            const unsigned int memoryPerV = static_cast<unsigned int>(_vShape[1] * sizeof(DeviceDataType));

            const unsigned int S_multi_size_2powfloored = static_cast<unsigned int>(pow(2.0, floor(log2(static_cast<HostDataType>(S_multi.size())))));

            const unsigned int blockDimY = std::min({threadCount, S_multi_size_2powfloored});
            const unsigned int blockDimX = std::min({static_cast<unsigned int>(floor((HostDataType) threadCount / (HostDataType) blockDimY)),
                                                     static_cast<unsigned int>(floor((HostDataType) sharedMemorySize / (HostDataType) memoryPerV))});
            const unsigned int gridDimX = static_cast<unsigned int>(ceil(static_cast<HostDataType>(_vShape[0]) / static_cast<HostDataType>(blockDimX)));
            const unsigned int gridDimY = static_cast<unsigned int>(ceil(static_cast<HostDataType>(S_multi.size()) / static_cast<HostDataType>(blockDimY)));

            // Create kernel configuration object.
            KernelConfiguration kernelConf;
            kernelConf.blockDim = dim3(blockDimX, blockDimY);
            kernelConf.gridDim = dim3(gridDimX, gridDimY);
            kernelConf.sharedMemory = blockDimX * memoryPerV;

            return kernelConf;
        };

        /**
         * This function calculates how a given problem should be chunked in order to solve it completely, when no sufficient GPU memory is available.
         * @param freeGPUMemory Free GPU memory in bytes.
         * @param summaryCount The number of summaries to evaluate for their function value.
         * @param dim The dimensionality of every data point.
         * @param maxS The cardinality of the greatest summary to evaluate.
         * @return A three-valued tuple consisting of total memory requirements for this problem (in bytes), the size of every chunk (i.e. number of summaries) to evaluate on the
         * GPU and the number of chunks in total.
         */
        std::tuple<size_t, unsigned long, unsigned long> calculateProblemDependentChunking(size_t freeGPUMemory, unsigned long summaryCount, unsigned long dim,
                                                                                           unsigned long maxS) const {
            // Calculate total GPU memory requirements for problem solution.
            size_t memoryReqSummaryMatrix = maxS * summaryCount * dim * sizeof(DeviceDataType);
            size_t memoryReqSummarySizes = summaryCount * sizeof(int);
            size_t memoryReqResultMatrix = summaryCount * _vShape[0] * sizeof(HostDataType);
            size_t memoryRequirements = memoryReqSummaryMatrix + memoryReqSummarySizes + memoryReqResultMatrix;

            // Calculate chunking.
            unsigned long chunkSize = static_cast<unsigned long>(
                floor(static_cast<HostDataType>(freeGPUMemory) / static_cast<HostDataType>(maxS * dim * sizeof(DeviceDataType) + sizeof(int) + _vShape[0] * sizeof(HostDataType))));
            if (chunkSize == 0) {
                throw std::runtime_error("ExemplarClusteringSubmodularFunction::calculateProblemDependentChunking: No memory left on GPU to evaluate the function.");
            }
            unsigned long totalChunks = static_cast<unsigned long>(ceil((HostDataType) summaryCount / (HostDataType) chunkSize));

            return std::make_tuple(memoryRequirements, chunkSize, totalChunks);
        };
    };
}

#endif // EXEMCL_EXEMPLARCLUSTERINGSUBMODULARFUNCTION_CUH
