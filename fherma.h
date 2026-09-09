// GENERATED from polymul/negacyclic@1.0.0. Do not edit — `--update` rewrites it.
//
// The types your answer is written against, derived from the signature: one
// field per value parameter, per argument, per result. A tensor is typed,
// because the signature already settled what its elements are.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fherma {

template <class T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;          // row-major

    int64_t count() const {
        int64_t total = 1;
        for (auto d : shape) total *= d;
        return total;
    }
};

struct Point {
    uint32_t N = 0;   // u32
    uint32_t W = 0;   // u32
    uint32_t L = 0;   // u32
    Tensor<uint32_t> q;   // tensor<L x u32>
};

struct Inputs {
    Tensor<uint32_t> a;   // tensor<N x L x u32>
    Tensor<uint32_t> b;   // tensor<N x L x u32>
};

struct Outputs {
    Tensor<uint32_t> c;   // tensor<N x L x u32>
};

}  // namespace fherma

// What you write, in solve.cpp.
void* fherma_init(const fherma::Point& p);
fherma::Outputs fherma_run(void* state, const fherma::Inputs& in);
void fherma_free(void* state);
