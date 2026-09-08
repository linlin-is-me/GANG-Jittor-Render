#pragma once
#include "op.h"

namespace jittor {

struct SimpleKnnV2Op : Op {
    Var* points;
    Var* output;

    SimpleKnnV2Op(Var* points);
    const char* name() const override { return "simple_knn_v2"; }
    void infer_shape() override;
    DECLARE_jit_run;
};

} // namespace jittor
