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

template<typename HostDataType, typename DeviceDataType>
std::shared_ptr<SubmodularFunction<HostDataType>> constructFunction(const MatrixX<HostDataType>& V, const std::string& dev) {
    if (dev == "gpu")
        return std::shared_ptr<SubmodularFunction<HostDataType>>(new gpu::ExemplarClusteringSubmodularFunction<DeviceDataType, HostDataType>(V));
    else if (dev == "cpu")
        return std::shared_ptr<SubmodularFunction<HostDataType>>(new cpu::ExemplarClusteringSubmodularFunction<HostDataType>(V));
    else
        throw std::runtime_error("ExemCl: Construction failed. Unknown device '" + dev + "' provided. Choose either 'gpu' or 'cpu'.");
}

PYBIND11_MODULE(exemcl, m) {
    m.doc() = "exemcl python plugin";

    auto m_fp16 = m.def_submodule("fp16");
    auto m_fp32 = m.def_submodule("fp32");
    auto m_fp64 = m.def_submodule("fp64");

    py::class_<exemcl::SubmodularFunction<float>, std::shared_ptr<SubmodularFunction<float>>>(m_fp16, "ExemplarClustering")
        .def(py::init<>(&constructFunction<float, __half>), py::arg("ground_set"), py::arg("device") = "gpu");
    py::class_<SubmodularFunction<float>, std::shared_ptr<SubmodularFunction<float>>>(m_fp32, "ExemplarClustering")
        .def(py::init<>(&constructFunction<float, float>), py::arg("ground_set"), py::arg("device") = "gpu")
        .def("__call__", py::overload_cast<const MatrixX<float>&>(&SubmodularFunction<float>::operator()), py::arg("S"));
    py::class_<SubmodularFunction<double>, std::shared_ptr<SubmodularFunction<double>>>(m_fp64, "ExemplarClustering")
        .def(py::init<>(&constructFunction<double, double>), py::arg("ground_set"), py::arg("device") = "gpu");
}

#endif // EXEMCL_PYTHONBINDING_H
