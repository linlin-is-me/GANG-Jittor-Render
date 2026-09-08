#include "var.h"
#include "simple_knn_v2_op.h"
#include <vector_types.h>
#include "simple_knn.h"

namespace jittor {

#ifndef JIT

SimpleKnnV2Op::SimpleKnnV2Op(Var* points) : points(points) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(nullptr, ns_float32);
}

void SimpleKnnV2Op::infer_shape() {
    output->set_shape(NanoVector(points->shape[0]));
}

void SimpleKnnV2Op::jit_prepare(JK& jk) {
    add_jit_define(jk, "Tx", points->dtype());
    add_jit_define(jk, "Ty", output->dtype());
}

#else

#ifdef JIT_cuda
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

void SimpleKnnV2Op::jit_run() {
    SimpleKNN::knn(
        static_cast<int>(points->shape[0]),
        reinterpret_cast<float3*>(points->ptr<Tx>()),
        output->ptr<Ty>());
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("SimpleKNN CUDA failure: ") + cudaGetErrorString(error));
    }
}

#else

void SimpleKnnV2Op::jit_run() {
    throw std::runtime_error("SimpleKNN is available only on CUDA");
}

#endif
#endif

} // namespace jittor
