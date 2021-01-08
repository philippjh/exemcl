#ifndef EXEMCL_FUNCTION_CPU
#define EXEMCL_FUNCTION_CPU

#include <src/function/SubmodularFunction.h>
#include <utility>

namespace exemcl::cpu {
    /**
     * This class provides a CPU implementation of the submodular function of exemplar-based clustering.
     */
    template<typename HostDataType = float>
    class ExemplarClusteringSubmodularFunction : public SubmodularFunction<HostDataType> {
    public:
        using SubmodularFunction<HostDataType>::operator();

        /**
         * Constructs the exemplar clustering submodular function using a ground set V.
         *
         * @param V The ground set V.
         */
        explicit ExemplarClusteringSubmodularFunction(const MatrixX<HostDataType>& V, int workerCount = -1) : SubmodularFunction<HostDataType>(workerCount), _V(V) {
            MatrixX<HostDataType> zeroVec = VectorX<HostDataType>::Zero(_V.cols()).transpose();
            _zeroVecValue = L(zeroVec);
        };

        /**
         * Evaluates the exemplar cluster-submodular function.
         *
         * @param S The set to evaluate.
         * @return The submodular function value.
         */
        HostDataType operator()(const MatrixX<HostDataType>& S) override {
            return ((const ExemplarClusteringSubmodularFunction*) (this))->operator()(S);
        };

        /**
         * Evaluates the exemplar cluster-submodular function.
         *
         * @param S The set to evaluate.
         * @return The submodular function value.
         */
        HostDataType operator()(const MatrixX<HostDataType>& S) const override {
            auto S_copy = std::make_unique<MatrixX<HostDataType>>(S);

            // Add zero vector to data copy.
            S_copy->conservativeResize(S_copy->rows() + 1, Eigen::NoChange_t());
            S_copy->row(S_copy->rows() - 1).setZero();

            // Make calculations.
            HostDataType L_2 = L(*S_copy);

            return _zeroVecValue - L_2;
        };

        /**
         * Returns a reference to the ground set V.
         * @return As stated above.
         */
        const MatrixX<HostDataType>& getV() const {
            return _V;
        };

    private:
        HostDataType _zeroVecValue;
        const MatrixX<HostDataType>& _V;

        /**
         * Calculates the L function.
         *
         * @param S_inner Set of data to calculate the L function for.
         * @return L function value.
         */
        HostDataType L(const MatrixX<HostDataType>& S_inner) const {
            auto* accuArray = new HostDataType[_V.rows()];

            for (unsigned int i = 0; i < _V.rows(); i++) {
                auto min_val = std::numeric_limits<HostDataType>::max();
                for (unsigned int j = 0; j < S_inner.rows(); j++)
                    min_val = std::min((_V.row(i) - S_inner.row(j)).squaredNorm(), min_val);
                accuArray[i] = min_val;
            }

            HostDataType accu = 0.0;
#pragma omp simd reduction(+ : accu)
            for (unsigned int i = 0; i < _V.rows(); i++)
                accu += accuArray[i];

            delete[] accuArray;
            return accu / static_cast<HostDataType>(_V.rows());
        };
    };
}

#endif // EXEMCL_FUNCTION_CPU
