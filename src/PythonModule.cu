#ifndef EXEMCL_PYTHONBINDING_H
#define EXEMCL_PYTHONBINDING_H

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <src/function/SubmodularFunction.h>
#include <src/function/cpu/ExemplarClusteringSubmodularFunction.h>
#include <src/function/gpu/ExemplarClusteringSubmodularFunction.cuh>

namespace py = pybind11;
using namespace exemcl;

std::shared_ptr<SubmodularFunction> constructFunction(const MatrixX<double>& V, const std::string& precision, const std::string& dev, int workerCount) {
    if (precision == "fp16") {
        // -----------------
        // FP16 CONSTRUCTION
        // -----------------
        if (dev == "gpu")
            return std::shared_ptr<SubmodularFunction>(new gpu::ExemplarClusteringSubmodularFunction<__half, float>(V.cast<float>(), workerCount));
        else if (dev == "cpu")
            throw std::runtime_error("ExemCl: Construction failed. FP16 precision is not available on CPUs.");
        else
            throw std::runtime_error("ExemCl: Construction failed. Unknown device '" + dev + "' provided. Choose either 'gpu' or 'cpu'.");
    } else if (precision == "fp32") {
        // -----------------
        // FP32 CONSTRUCTION
        // -----------------
        if (dev == "gpu")
            return std::shared_ptr<SubmodularFunction>(new gpu::ExemplarClusteringSubmodularFunction<float, float>(V.cast<float>(), workerCount));
        else if (dev == "cpu")
            return std::shared_ptr<SubmodularFunction>(new cpu::ExemplarClusteringSubmodularFunction<float>(V.cast<float>(), workerCount));
        else
            throw std::runtime_error("ExemCl: Construction failed. Unknown device '" + dev + "' provided. Choose either 'gpu' or 'cpu'.");
    } else if (precision == "fp64") {
        // -----------------
        // FP64 CONSTRUCTION
        // -----------------
        if (dev == "gpu")
            return std::shared_ptr<SubmodularFunction>(new gpu::ExemplarClusteringSubmodularFunction<double, double>(V, workerCount));
        else if (dev == "cpu")
            return std::shared_ptr<SubmodularFunction>(new cpu::ExemplarClusteringSubmodularFunction<double>(V, workerCount));
        else
            throw std::runtime_error("ExemCl: Construction failed. Unknown device '" + dev + "' provided. Choose either 'gpu' or 'cpu'.");
    } else
        throw std::runtime_error("ExemCl: Construction failed. Unknown precision '" + precision + "' provided. Choose either 'fp16', 'fp32' or 'fp64'.");
}

PYBIND11_MODULE(exemcl, m) {
    m.doc() = "exemcl python plugin";

    py::class_<SubmodularFunction, std::shared_ptr<SubmodularFunction>>(m, "ExemplarClustering")
        .def(py::init<>(&constructFunction), py::arg("ground_set"), py::arg("precision") = "fp32", py::arg("device") = "gpu", py::arg("worker_count") = -1)
        .def("__call__", py::overload_cast<const MatrixX<double>&>(&SubmodularFunction::operator()), py::arg("S"))
        .def("__call__", py::overload_cast<const std::vector<MatrixX<double>>&>(&SubmodularFunction::operator()), py::arg("S_multi"));
}

#endif // EXEMCL_PYTHONBINDING_H
