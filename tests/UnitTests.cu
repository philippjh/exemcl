#include <Eigen/Eigen>
#include <gtest/gtest.h>
#include <src/function/SubmodularFunction.h>
#include <src/function/cpu/ExemplarClusteringSubmodularFunction.h>
#include <src/function/gpu/ExemplarClusteringSubmodularFunction.cuh>
#include <tests/CSVFile.h>

#ifndef EXEMCL_TESTFILES_DIR
#error No testfile directory supplied. Compilation aborted.
#else
std::string TESTFILES_ROOT(EXEMCL_TESTFILES_DIR);
#endif

struct SubmodularTestData {
    exemcl::MatrixX<double> groundSet;
    std::vector<exemcl::MatrixX<double>> subsets;

    exemcl::VectorX<double> fValuesExpected;
    exemcl::VectorX<double> marginalsExpected;

    exemcl::VectorX<double> marginal;
};

SubmodularTestData loadSubmodularTestData(const std::string& subdir) {
    SubmodularTestData testData;

    // Load test data.
    testData.groundSet = CSVFile::readCSVFile(TESTFILES_ROOT + subdir + "/ground_set.csv", ',')->asMatrix().cast<double>();
    exemcl::MatrixX<double> subsetsMatrix = CSVFile::readCSVFile(TESTFILES_ROOT + subdir + "/subsets.csv", ',')->asMatrix().cast<double>();
    std::vector<std::vector<exemcl::VectorX<double>>> subsetVectors;
    for (unsigned int i = 0; i < subsetsMatrix.rows(); i++) {
        exemcl::VectorX<double> matrixRow = subsetsMatrix.row(i);
        unsigned int subsetIdx = matrixRow[subsetsMatrix.cols() - 1];
        if (subsetIdx >= subsetVectors.size())
            subsetVectors.emplace_back();
        subsetVectors[subsetIdx].push_back(matrixRow.head(matrixRow.rows() - 1));
    }

    // Create subsets.
    for (auto& subset : subsetVectors) {
        exemcl::MatrixX<double> subsetMatrix(subset.size(), testData.groundSet.cols());
        for (unsigned int i = 0; i < subset.size(); i++)
            subsetMatrix.row(i) = subset[i];
        testData.subsets.push_back(subsetMatrix);
    }

    testData.fValuesExpected = CSVFile::readCSVFile(TESTFILES_ROOT + subdir + "/f_values.csv", ',')->asMatrix().col(0).cast<double>();
    testData.marginalsExpected = CSVFile::readCSVFile(TESTFILES_ROOT + subdir + "/marginal_values.csv", ',')->asMatrix().col(0).cast<double>();
    testData.marginal = testData.groundSet.row(testData.groundSet.rows() - 1);

    return testData;
}

void testSubmodularFunction(exemcl::SubmodularFunction& submodularFunction, SubmodularTestData& testData, double tolerancy) {
    // Test for correct individual evaluation.
    for (unsigned long i = 0; i < testData.subsets.size(); i++)
        EXPECT_NEAR(testData.fValuesExpected(i), submodularFunction(testData.subsets[i]), tolerancy);

    // Test for correct joint evaluation.
    auto fValuesComputedJoint = submodularFunction(testData.subsets);
    for (unsigned long i = 0; i < testData.subsets.size(); i++)
        EXPECT_NEAR(testData.fValuesExpected(i), fValuesComputedJoint[i], tolerancy);

    // Test for correct individual gains.
    for (unsigned long i = 0; i < testData.subsets.size(); i++)
        EXPECT_NEAR(testData.marginalsExpected(i), submodularFunction(testData.subsets[i], testData.marginal), tolerancy);

    // Test for correct joint gains.
    auto marginalsComputedJoint = submodularFunction(testData.subsets, testData.marginal);
    for (unsigned long i = 0; i < testData.subsets.size(); i++)
        EXPECT_NEAR(testData.marginalsExpected(i), marginalsComputedJoint[i], tolerancy);

    // Test for correct multiple marginals.
    exemcl::MatrixX<double> emptySet(0, testData.groundSet.cols());
    for (auto& S : testData.subsets) {
        std::vector<double> individualGains;
        std::vector<exemcl::VectorXRef<double>> marginalsTested;
        for (unsigned int i = 0; i < S.rows(); i++) {
            exemcl::VectorXRef<double> marginal = S.row(i);
            individualGains.push_back(submodularFunction(emptySet, marginal));
            marginalsTested.push_back(marginal);
        }

        std::vector<double> totalGains = submodularFunction(emptySet, marginalsTested);

        EXPECT_EQ(individualGains.size(), totalGains.size());
        for (unsigned int i = 0; i < individualGains.size(); i++)
            EXPECT_NEAR(individualGains[i], totalGains[i], tolerancy);
    }
}

#define FP16_ERROR_TOLERANCY 0.01f
#define FP32_ERROR_TOLERANCY 0.001f
#define FP64_ERROR_TOLERANCY 0.000000000001

using DeviceDataTypes = ::testing::Types<__half, float, double>;
template<typename T>
class GPUTests : public ::testing::Test { };
TYPED_TEST_SUITE(GPUTests, DeviceDataTypes);

TYPED_TEST(GPUTests, ExemplarClusteringST) {
    if constexpr (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, double>::value) {
        // Load test data.
        SubmodularTestData testData = loadSubmodularTestData("exem");

        // Create submodular function.
        exemcl::gpu::ExemplarClusteringSubmodularFunction<TypeParam, TypeParam> submodularFunction(testData.groundSet.cast<TypeParam>(), 1);

        // Run the test function.
        if constexpr (std::is_same<TypeParam, float>::value)
            testSubmodularFunction(submodularFunction, testData, FP32_ERROR_TOLERANCY);
        else
            testSubmodularFunction(submodularFunction, testData, FP64_ERROR_TOLERANCY);
    } else if constexpr (std::is_same<TypeParam, __half>::value) {
        // Load test data.
        SubmodularTestData testData = loadSubmodularTestData("exem");

        // Create submodular function.
        exemcl::gpu::ExemplarClusteringSubmodularFunction<TypeParam, float> submodularFunction(testData.groundSet.cast<float>(), 1);

        // Run the test function.
        testSubmodularFunction(submodularFunction, testData, FP16_ERROR_TOLERANCY);
    }
}

TYPED_TEST(GPUTests, ExemplarClusteringMT) {
    if constexpr (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, double>::value) {
        // Load test data.
        SubmodularTestData testData = loadSubmodularTestData("exem");

        // Create submodular function.
        exemcl::gpu::ExemplarClusteringSubmodularFunction<TypeParam, TypeParam> submodularFunction(testData.groundSet.cast<TypeParam>(), -1);

        // Run the test function.
        if constexpr (std::is_same<TypeParam, float>::value)
            testSubmodularFunction(submodularFunction, testData, FP32_ERROR_TOLERANCY);
        else
            testSubmodularFunction(submodularFunction, testData, FP64_ERROR_TOLERANCY);
    } else if constexpr (std::is_same<TypeParam, __half>::value) {
        // Load test data.
        SubmodularTestData testData = loadSubmodularTestData("exem");

        // Create submodular function.
        exemcl::gpu::ExemplarClusteringSubmodularFunction<TypeParam, float> submodularFunction(testData.groundSet.cast<float>(), -1);

        // Run the test function.
        testSubmodularFunction(submodularFunction, testData, FP16_ERROR_TOLERANCY);
    }
}

TYPED_TEST(GPUTests, ExemplarClusteringChunked) {
    if constexpr (std::is_same<TypeParam, float>::value || std::is_same<TypeParam, double>::value) {
        // Load test data.
        SubmodularTestData testData = loadSubmodularTestData("exem");

        // Create submodular function.
        exemcl::gpu::ExemplarClusteringSubmodularFunction<TypeParam, TypeParam> submodularFunction(testData.groundSet.cast<TypeParam>(), 1);
        submodularFunction.setGPUMemoryLimit(100 * 1024); // 100 KiB

        // Run the test function.
        if constexpr (std::is_same<TypeParam, float>::value)
            testSubmodularFunction(submodularFunction, testData, FP32_ERROR_TOLERANCY);
        else
            testSubmodularFunction(submodularFunction, testData, FP64_ERROR_TOLERANCY);
    } else if constexpr (std::is_same<TypeParam, __half>::value) {
        // Load test data.
        SubmodularTestData testData = loadSubmodularTestData("exem");

        // Create submodular function.
        exemcl::gpu::ExemplarClusteringSubmodularFunction<TypeParam, float> submodularFunction(testData.groundSet.cast<float>(), 1);
        submodularFunction.setGPUMemoryLimit(100 * 1024); // 100 KiB

        // Run the test function.
        testSubmodularFunction(submodularFunction, testData, FP16_ERROR_TOLERANCY);
    }
}

using HostDataTypes = ::testing::Types<float, double>;
template<typename T>
class CPUTests : public ::testing::Test { };
TYPED_TEST_SUITE(CPUTests, HostDataTypes);

TYPED_TEST(CPUTests, ExemplarClusteringST) {
    // Load test data.
    SubmodularTestData testData = loadSubmodularTestData("exem");

    // Create submodular function.
    exemcl::cpu::ExemplarClusteringSubmodularFunction<TypeParam> submodularFunction(testData.groundSet.cast<TypeParam>(), 1);

    // Run the test function.
    if constexpr (std::is_same<TypeParam, float>::value)
        testSubmodularFunction(submodularFunction, testData, FP32_ERROR_TOLERANCY);
    else
        testSubmodularFunction(submodularFunction, testData, FP64_ERROR_TOLERANCY);
}

TYPED_TEST(CPUTests, ExemplarClusteringMT) {
    // Load test data.
    SubmodularTestData testData = loadSubmodularTestData("exem");

    // Create submodular function.
    exemcl::cpu::ExemplarClusteringSubmodularFunction<TypeParam> submodularFunction(testData.groundSet.cast<TypeParam>(), -1);

    // Run the test function.
    if constexpr (std::is_same<TypeParam, float>::value)
        testSubmodularFunction(submodularFunction, testData, FP32_ERROR_TOLERANCY);
    else
        testSubmodularFunction(submodularFunction, testData, FP64_ERROR_TOLERANCY);
}

int main(int argc, char** argv) {
    std::cout << "Reading testfiles from: " << TESTFILES_ROOT << std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}