#ifndef EXEMCL_SUBM_FUNCTION_H
#define EXEMCL_SUBM_FUNCTION_H

#include <src/io/DataTypes.h>
#include <thread>
#include <utility>

namespace exemcl {
    /**
     * Submodular functions represent a special kind of set function, which map subsets (usually denoted as \f$S\f$) of some ground set
     * (denoted by \f$V\f$) to a positive real value (sometimes called the "utility"), whilst maintaining a property of diminishing returns.
     *
     * Formally, we have the ground set \f$V \subset \mathbb{R} \f$ and a function \f$f: \mathcal{P}(V) \rightarrow \mathbb{R}^+\f$.
     * The function \f$f\f$ is now submodular, iff \f$\Delta_f(e \mid A) \geq \Delta_f(e \mid B)\f$ for arbitrary \f$A \subseteq B \subseteq V\f$ and \f$e \in V \setminus B\f$.
     * The vector \f$e\f$ is sometimes called the "marginal element". \f$\Delta_f\f$ represents the discrete derivative \f$\Delta_f(e | S) := f(S \cup \left\lbrace e \right\rbrace
     * - f(S))\f$.
     *
     * This (abstract) class provides an interface to implementing submodular functions of any kind.
     */
    class SubmodularFunction {
    public:
        /**
         * Provides a base constructor, which updates the worker count of the submodular function.
         * @param workerCount The number of workers to employ (defaults to -1, i.e. all available cores).
         */
        SubmodularFunction(int workerCount = -1) {
            setWorkerCount(workerCount);
        }

        /**
         * Calculates the submodular function value for a set.
         *
         * @param S Set of vectors, to calculate the submodular function for.
         * @return The submodular function value \f$f(S)\f$.
         */
        virtual double operator()(const MatrixX<double>& S) const = 0;

        /**
         * Calculates the submodular function value for a set.
         *
         * @param S Set of vectors, to calculate the submodular function for.
         * @return The submodular function value \f$f(S)\f$.
         */
        virtual double operator()(const MatrixX<double>& S) = 0;

        /**
         * Calculates the marginal gain of the submodular function, w.r.t to \f$S\f$ and a marginal element \f$e\f$.
         * @param S Set of vectors, to calculate the submodular function for.
         * @param elem A marginal element.
         * @return The marginal gain of the \f$f(S) - f(S \cup \left\{elem\right\})\f$
         */
        virtual double operator()(const MatrixX<double>& S, VectorXRef<double> elem) const {
            if (S.cols() == elem.size()) {
                std::unique_ptr<MatrixX<double>> S_elem = std::make_unique<MatrixX<double>>(S);
                S_elem->conservativeResize(S.rows() + 1, Eigen::NoChange_t());
                S_elem->row(S.rows()) << elem.transpose();

                // Call subsequent operators.
                auto marginalGain = operator()(*S_elem) - operator()(S);

                // Return value.
                return marginalGain;
            } else
                throw std::runtime_error("SubmodularFunction::operator(): The number of columns in matrix `S` and the number of elements in vector `elem` do not match ("
                                         + std::to_string(S.cols()) + " vs. " + std::to_string(elem.size()) + ").");
        }

        /**
         * Calculates the marginal gain of the submodular function, w.r.t to \f$S\f$ and a marginal element \f$e\f$.
         * @param S Set of vectors, to calculate the submodular function for.
         * @param elem A marginal element.
         * @return The marginal gain of the \f$f(S) - f(S \cup \left\{elem\right\})\f$
         */
        virtual double operator()(const MatrixX<double>& S, VectorXRef<double> elem) {
            return ((const SubmodularFunction*) (this))->operator()(S, elem);
        }

        /**
         * Calculates the submodular function for more than one set.
         *
         * @param S_multi A set of sets \f$ S = \left\{S_1, ..., S_n\right\}\f$, which should be evaluated using the submodular function.
         * @return A set of utility values \f$\left\{f(S_1), ..., f(S_n)\right\}\f$.
         */
        virtual std::vector<double> operator()(const std::vector<MatrixX<double>>& S_multi) const {
            // Construct vector for storing utilities.
            std::vector<double> utilities;
            utilities.resize(S_multi.size());

            // Calculate utilities.
#pragma omp parallel for num_threads(_workerCount)
            for (unsigned long i = 0; i < S_multi.size(); i++)
                utilities[i] = operator()(S_multi[i]);

            // Return value.
            return utilities;
        };

        /**
         * Calculates the submodular function for more than one set.
         *
         * @param S_multi A set of sets \f$ S = \left\{S_1, ..., S_n\right\}\f$, which should be evaluated using the submodular function.
         * @return A set of utility values \f$\left\{f(S_1), ..., f(S_n)\right\}\f$.
         */
        virtual std::vector<double> operator()(const std::vector<MatrixX<double>>& S_multi) {
            return ((const SubmodularFunction*) (this))->operator()(S_multi);
        };

        /**
         * Calculates the marginal gain for a multi set and a marginal element.
         *
         * @param S_multi A set of sets \f$ S = \left\{S_1, ..., S_n\right\} \f$, which should be evaluated using the submodular function.
         * @param elem A marginal element \f$ e \f$.
         * @return A set of marginal gain values \f$\Delta_f(e | S_1), ..., \Delta_f(e | S_n) \f$.
         */
        virtual std::vector<double> operator()(const std::vector<MatrixX<double>>& S_multi, VectorXRef<double> elem) const {
            auto S_multi_elem = std::make_unique<std::vector<MatrixX<double>>>();
            S_multi_elem->reserve(S_multi.size());

            // Create a new S_multi set, but include the marginal vector.
            for (auto& S_elem : S_multi) {
                S_multi_elem->push_back(S_elem);
                S_multi_elem->back().conservativeResize(S_elem.rows() + 1, Eigen::NoChange_t());
                S_multi_elem->back().row(S_elem.rows()) << elem.transpose();
            }

            // Evaluate S_multi_elem and S_multi.
            auto utilityS_multi = operator()(S_multi);
            auto utilityS_multi_elem = operator()(*S_multi_elem);

            // Calculate the difference between the utilities of S_multi_elem and S_multi.
            std::vector<double> marginalGains;
            marginalGains.resize(S_multi.size());

            for (unsigned long i = 0; i < utilityS_multi.size(); i++)
                marginalGains[i] = utilityS_multi_elem[i] - utilityS_multi[i];

            return marginalGains;
        }

        /**
         * Calculates the marginal gain for a multi set and a marginal element.
         *
         * @param S_multi A set of sets \f$ S = \left\{S_1, ..., S_n\right\} \f$, which should be evaluated using the submodular function.
         * @param elem A marginal element \f$ e \f$.
         * @return A set of marginal gain values \f$\Delta_f(e | S_1), ..., \Delta_f(e | S_n) \f$.
         */
        virtual std::vector<double> operator()(const std::vector<MatrixX<double>>& S_multi, VectorXRef<double> elem) {
            return ((const SubmodularFunction*) (this))->operator()(S_multi, elem);
        }

        /**
         * Calculates the marginal gain for a single set \f$S\f$ and a set of marginal vectors.
         *
         * @param S Set of vectors, which should be used to calculate the marginal value in conjunction with `elems`.
         * @param elems A set of marginal vectors \f$ \left\{e_1, ..., e_n \right\}\f$.
         * @return A set of marginal gain values \f$\Delta_f(e_1 | S), ..., \Delta_f(e_n | S) \f$.
         */
        virtual std::vector<double> operator()(const MatrixX<double>& S, std::vector<VectorXRef<double>> elems) const {
            // Create a vector, which will hold {S u e_1}, ..., {S u e_n}
            auto S_elems = std::make_unique<std::vector<MatrixX<double>>>(elems.size(), S);

            // Build {S u e_1}, ..., {S u e_n}.
            for (unsigned int i = 0; i < elems.size(); i++) {
                auto& elem = elems[i];
                (*S_elems)[i].conservativeResize(S.rows() + 1, Eigen::NoChange_t());
                (*S_elems)[i].row(S.rows()) << elem.transpose();
            }

            // Evaluate S.
            auto S_funcValue = operator()(S);

            // Evaluate all S with elems.
            auto S_elems_funcValue = operator()(*S_elems);

            // Create a result vector.
            std::vector<double> gains;
            gains.resize(elems.size());

            // Fill the results.
            for (unsigned int i = 0; i < elems.size(); i++)
                gains[i] = S_elems_funcValue[i] - S_funcValue;

            return gains;
        }

        /**
         * Calculates the marginal gain for a single set \f$S\f$ and a set of marginal vectors.
         *
         * @param S Set of vectors, which should be used to calculate the marginal value in conjunction with `elems`.
         * @param elems A set of marginal vectors \f$ \left\{e_1, ..., e_n \right\}\f$.
         * @return A set of marginal gain values \f$\Delta_f(e_1 | S), ..., \Delta_f(e_n | S) \f$.
         */
        virtual std::vector<double> operator()(const MatrixX<double>& S, std::vector<VectorXRef<double>> elems) {
            return ((const SubmodularFunction*) (this))->operator()(std::move(S), std::move(elems));
        }

        /**
         * Returns the worker count, which is currently assigned to this submodular function.
         * @return Worker count.
         */
        virtual unsigned int getWorkerCount() const {
            return _workerCount;
        };

        /**
         * Updates the worker count for the submodular function. If the supplied value is below one,
         * the function will try to update the worker count to the number of cores available to the
         * program.
         *
         * @param workerCount New worker count.
         */
        virtual void setWorkerCount(int workerCount) {
            if (workerCount >= 1)
                _workerCount = workerCount;
            else {
                auto suggestedThreads = std::thread::hardware_concurrency();
                _workerCount = suggestedThreads > 0 ? suggestedThreads : 1;
            }
        }

        /**
         * Limits the usable memory by this class. Must be overridden by the implementing class and yields an exception otherwise.
         * @param memoryLimit Memory limit (in byte).
         */
        virtual void setMemoryLimit(long memoryLimit) {
            throw std::runtime_error("SubmodularFunction::setMemoryLimit: Not implemented.");
        }

        /**
         * Destructor.
         */
        virtual ~SubmodularFunction() = default;

    protected:
        unsigned int _workerCount = 1;
    };
}

#endif // EXEMCL_SUBM_FUNCTION_H