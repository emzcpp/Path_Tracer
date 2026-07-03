#pragma once

// PCG32 — small, fast, statistically solid, and 16 bytes of state, so every
// pixel (or GPU thread) can own one. std::mt19937 is 2.5KB of state and not
// something you'd put in a Metal kernel; this is.
// Reference: O'Neill, "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Algorithms for Random Number Generation" (2014).

#include <cstdint>

#include "vec3.h"

// splitmix64 finalizer: a cheap, high-quality bit mixer. Used to decorrelate
// per-(pixel, pass) RNG seeds — PCG streams that share a raw seed start from
// correlated states, so hash the seed first. Branchless, GPU-portable.
inline std::uint64_t mix64(std::uint64_t v) {
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33;
    v *= 0xc4ceb9fe1a85ec53ULL;
    return v ^ (v >> 33);
}

struct RNG {
    std::uint64_t state = 0;
    std::uint64_t inc   = 1;

    explicit RNG(std::uint64_t seed, std::uint64_t sequence = 1) {
        inc = (sequence << 1u) | 1u;
        next_u32();
        state += seed;
        next_u32();
    }

    std::uint32_t next_u32() {
        const std::uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        const auto xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const auto rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
    }

    // Uniform in [0,1): top 24 bits scaled by 2^-24 (float has a 24-bit
    // mantissa, so every value is exactly representable and 1.0 is excluded).
    float next_float() {
        return static_cast<float>(next_u32() >> 8) * (1.0f / 16777216.0f);
    }
};

// Uniform direction on the unit sphere, sampled analytically (z uniform in
// [-1,1], azimuth uniform) — branchless, no rejection loop, GPU-friendly.
inline vec3 random_unit_vector(RNG& rng) {
    const float z = 1.0f - 2.0f * rng.next_float();
    const float phi = 6.28318530717958647692f * rng.next_float();
    const float r = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
    return {r * std::cos(phi), r * std::sin(phi), z};
}
