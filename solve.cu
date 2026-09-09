// A cuPQC BigInt answer to polymul/negacyclic@1.0.0.
//
// Negacyclic polynomial multiplication in Z_q[X]/(X^N + 1) with wide integer
// coefficients. This is a schoolbook O(N^2) baseline written for correctness,
// not speed: one CUDA thread per output coefficient, each accumulating the N
// products that land on it, with every wide-integer operation done by cuPQC's
// device BigInt (mul_mod / add_mod / sub_mod). A production answer would use an
// NTT; this one exists to show the interface and the arithmetic working end to
// end on the GPU.
//
// A coefficient is L little-endian u32 limbs, and the challenge's modulus q is
// wide too. cuPQC's BigInt width is a compile-time template, so the kernel is
// instantiated for a fixed set of limb counts and dispatched on the point's L
// at run time. The widths below cover the small test points (L = 4, 8) and the
// challenge point (L = 28, W = 868). All are Thread-execution (TPI = 1)
// configurations the shipped library instantiates.
#include "fherma.h"

#include <cupqc/bigint.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace cupqc;

namespace {

// c[k] = sum_{i<=k} a[i]*b[k-i]  -  sum_{i>k} a[i]*b[k-i+N]   (mod q)
//
// The first sum is the ordinary convolution terms whose degree stays below N;
// the second is the terms that wrapped past X^N, which in Z[X]/(X^N + 1) come
// back negated. Both operands of every sub_mod are already reduced mod q, so
// its precondition (this < q, other < q) holds throughout.
template <unsigned int NUM_LIMBS>
__global__ void negacyclic(const uint32_t* __restrict__ a,
                           const uint32_t* __restrict__ b,
                           const uint32_t* __restrict__ q_limbs,
                           uint32_t* __restrict__ c,
                           unsigned int N) {
    using BI = decltype(BitWidth<NUM_LIMBS * 32>() + SM<800>() + Thread());
    using bigint = typename BI::bigint;

    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    const bigint q(q_limbs, 0);
    bigint acc(static_cast<uint32_t>(0));

    for (unsigned int i = 0; i <= k; ++i) {
        const bigint ai(a, i);
        const bigint bj(b, k - i);
        acc = acc.add_mod(ai.mul_mod(bj, q), q);
    }
    for (unsigned int i = k + 1; i < N; ++i) {
        const bigint ai(a, i);
        const bigint bj(b, k - i + N);
        acc = acc.sub_mod(ai.mul_mod(bj, q), q);
    }

    acc.store(c, k);
}

struct State {
    unsigned int N = 0;
    unsigned int L = 0;
    uint32_t* d_a = nullptr;
    uint32_t* d_b = nullptr;
    uint32_t* d_c = nullptr;
    uint32_t* d_q = nullptr;
};

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

template <unsigned int NUM_LIMBS>
void launch(const State& s) {
    const unsigned int threads = 128;
    const unsigned int blocks = (s.N + threads - 1) / threads;
    negacyclic<NUM_LIMBS><<<blocks, threads>>>(s.d_a, s.d_b, s.d_q, s.d_c, s.N);
}

}  // namespace

// Setup, once per point and not measured: the point fixes N, L and q, so the
// device buffers and the modulus are prepared here rather than per case.
void* fherma_init(const fherma::Point& p) {
    if (!(p.L == 4 || p.L == 8 || p.L == 28)) {
        throw std::runtime_error("this build covers L in {4, 8, 28}; got L=" +
                                 std::to_string(p.L));
    }

    auto* s = new State{};
    s->N = p.N;
    s->L = p.L;

    const size_t coeffs = static_cast<size_t>(p.N) * p.L;
    check(cudaMalloc(&s->d_a, coeffs * sizeof(uint32_t)), "cudaMalloc a");
    check(cudaMalloc(&s->d_b, coeffs * sizeof(uint32_t)), "cudaMalloc b");
    check(cudaMalloc(&s->d_c, coeffs * sizeof(uint32_t)), "cudaMalloc c");
    check(cudaMalloc(&s->d_q, static_cast<size_t>(p.L) * sizeof(uint32_t)), "cudaMalloc q");
    check(cudaMemcpy(s->d_q, p.q.data.data(), p.L * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy q");
    return s;
}

// The measured operation: the two host-to-device copies and the device-to-host
// copy of the result are part of it, because timing begins with the operands in
// coefficient form and ends with the answer in coefficient form.
fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto* s = static_cast<State*>(state);
    const size_t coeffs = static_cast<size_t>(s->N) * s->L;

    check(cudaMemcpy(s->d_a, in.a.data.data(), coeffs * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy a");
    check(cudaMemcpy(s->d_b, in.b.data.data(), coeffs * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy b");

    switch (s->L) {
        case 4:  launch<4>(*s);  break;
        case 8:  launch<8>(*s);  break;
        case 28: launch<28>(*s); break;
    }
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "kernel");

    fherma::Outputs out;
    out.c.shape = {static_cast<int64_t>(s->N), static_cast<int64_t>(s->L)};
    out.c.data.resize(coeffs);
    check(cudaMemcpy(out.c.data.data(), s->d_c, coeffs * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost), "copy c");
    return out;
}

void fherma_free(void* state) {
    auto* s = static_cast<State*>(state);
    if (!s) return;
    cudaFree(s->d_a);
    cudaFree(s->d_b);
    cudaFree(s->d_c);
    cudaFree(s->d_q);
    delete s;
}
