#ifndef EXEMCL_DATATYPES_H
#define EXEMCL_DATATYPES_H

#include <Eigen/Eigen>
#include <memory>

namespace exemcl {
    template<typename HostDataType, Eigen::StorageOptions DefaultStorage = Eigen::RowMajor>
    using MatrixX = Eigen::Matrix<HostDataType, Eigen::Dynamic, Eigen::Dynamic, DefaultStorage>;

    template<typename HostDataType>
    using VectorX = Eigen::Matrix<HostDataType, Eigen::Dynamic, 1>;

    template<typename HostDataType>
    using VectorXRef = Eigen::Ref<VectorX<HostDataType>, 0, Eigen::InnerStride<>>;

    template<typename HostDataType>
    using ConstVectorXRef = Eigen::Ref<const VectorX<HostDataType>, 0, Eigen::InnerStride<>>;
}

#endif // EXEMCL_DATATYPES_H
