#pragma once

// 16-bit linear texture storage + bilinear sampling (CPU side).
//
// Textures are decoded to LINEAR at load time (sRGB sources through the
// pure pow-2.2 curve, matching encode_channel's pure-power display encode —
// a texel viewed head-on under unit light round-trips to its authored
// value through THIS pipeline). Storage is uint16 per channel: 8-bit linear
// would band in the darks, half has no native CPU type (a parity surface),
// float doubles bandwidth for nothing.
//
// The sampler below is duplicated in pathtrace.metal with identical text —
// deliberately NOT hardware texture filtering, whose fixed-point weights
// would break GPU/CPU parity by construction. Any change here must be
// mirrored there; --parity is the drift detector.

#include <cmath>
#include <cstdint>
#include <vector>

#include "vec3.h"

struct Texture16 {
    int w = 0, h = 0;
    std::vector<std::uint16_t> texels;   // RGBA interleaved, linear
    bool valid() const { return w > 0 && h > 0; }
};

// Convert stb's 8-bit RGB rows to linear ushort4.
inline Texture16 texture_from_rgb8(const unsigned char* rgb, int w, int h,
                                   bool srgb_to_linear) {
    Texture16 t;
    t.w = w;
    t.h = h;
    t.texels.resize(std::size_t(w) * h * 4);
    for (std::size_t i = 0, n = std::size_t(w) * h; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float f = rgb[i * 3 + c] * (1.0f / 255.0f);
            if (srgb_to_linear) f = std::pow(f, 2.2f);
            t.texels[i * 4 + c] =
                std::uint16_t(std::lrintf(f * 65535.0f));
        }
        t.texels[i * 4 + 3] = 65535;
    }
    return t;
}

inline color texel_fetch(const Texture16& t, int x, int y) {
    const std::uint16_t* p = &t.texels[(std::size_t(y) * t.w + x) * 4];
    const float k = 1.0f / 65535.0f;
    return color(p[0] * k, p[1] * k, p[2] * k);
}

// REPEAT wrap, texel centers at +0.5. The helmet's UVs live entirely in
// v ∈ [1, 2] — the wrap is mandatory, not defensive.
inline color sample_bilinear(const Texture16& t, float u, float v) {
    u = u - std::floor(u);
    v = v - std::floor(v);
    const float x = u * float(t.w) - 0.5f;
    const float y = v * float(t.h) - 0.5f;
    const float fx = std::floor(x), fy = std::floor(y);
    const float ax = x - fx, ay = y - fy;
    int x0 = int(fx), y0 = int(fy), x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 += t.w;         // only -1 is possible after wrap
    if (y0 < 0) y0 += t.h;
    if (x1 >= t.w) x1 -= t.w;
    if (y1 >= t.h) y1 -= t.h;
    const color c00 = texel_fetch(t, x0, y0), c10 = texel_fetch(t, x1, y0);
    const color c01 = texel_fetch(t, x0, y1), c11 = texel_fetch(t, x1, y1);
    // Literal weight expression — keep textually identical to the MSL copy.
    return (1.0f - ax) * (1.0f - ay) * c00 + ax * (1.0f - ay) * c10 +
           (1.0f - ax) * ay * c01 + ax * ay * c11;
}
