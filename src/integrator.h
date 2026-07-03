#pragma once

// The path tracer core — future Metal kernel. Iterative (a for-loop, not
// recursion) because that's the shape a GPU wants, and it makes the state
// explicit: `throughput` is the product of all albedos along the path so
// far, i.e. "what fraction of light found from here on will survive the
// trip back to the camera".

#include <limits>

#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "rng.h"

// The dome light: sky white at the horizon blending to blue overhead.
// Paths that escape the scene pick up this radiance. Emissive surfaces in
// the scene add their own light on top.
inline color sky_radiance(const Ray& r) {
    const float t = 0.5f * (normalize(r.dir).y + 1.0f);
    return lerp(color(1.0f, 1.0f, 1.0f), color(0.5f, 0.7f, 1.0f), t);
}

// ---- Session F: equirectangular HDRI environment ------------------------
// Lookup ONLY (no NEE / importance sampling — next session): missed rays
// sample the env map exactly where they sampled the gradient. All code
// below is mirrored line-for-line in pathtrace.metal; --parity is the
// drift detector, with special suspicion on atan2/acos ULPs.

struct EnvLookup {
    const float* texels = nullptr;   // RGBA float, linear radiance
    int w = 0, h = 0;
    float intensity = 1.0f;
    float yaw_norm = 0.0f;           // yaw / 2pi, added to u
};

inline color env_fetch(const float* tex, int W, int x, int y) {
    const float* p = tex + (std::size_t(y) * W + x) * 4;
    return color(p[0], p[1], p[2]);
}

// Same manual-bilinear structure as material textures, with equirect
// semantics: u WRAPS (azimuth seam), v CLAMPS (poles).
inline color sample_env_bilinear(const float* tex, int W, int H, float u,
                                 float v) {
    u = u - std::floor(u);
    v = std::fmin(std::fmax(v, 0.0f), 1.0f);
    const float x = u * float(W) - 0.5f;
    const float y = v * float(H) - 0.5f;
    const float fx = std::floor(x), fy = std::floor(y);
    const float ax = x - fx, ay = y - fy;
    int x0 = int(fx), x1 = x0 + 1;
    int y0 = int(fy), y1 = y0 + 1;
    if (x0 < 0) x0 += W;
    if (x1 >= W) x1 -= W;
    if (y0 < 0) y0 = 0;
    if (y1 >= H) y1 = H - 1;
    const color c00 = env_fetch(tex, W, x0, y0), c10 = env_fetch(tex, W, x1, y0);
    const color c01 = env_fetch(tex, W, x0, y1), c11 = env_fetch(tex, W, x1, y1);
    return (1.0f - ax) * (1.0f - ay) * c00 + ax * (1.0f - ay) * c10 +
           (1.0f - ax) * ay * c01 + ax * ay * c11;
}

// Direction -> lat-long UV -> radiance. Falls back to the gradient when no
// map is loaded (--no HDRI, legacy scenes, CPU-viewer default).
inline color miss_radiance(const EnvLookup& env, const Ray& r) {
    if (!env.texels) return sky_radiance(r);
    const vec3 d = normalize(r.dir);
    const float u = std::atan2(d.z, d.x) * 0.15915494309189533577f + 0.5f +
                    env.yaw_norm;
    const float v =
        std::acos(std::fmin(std::fmax(d.y, -1.0f), 1.0f)) *
        0.31830988618379067154f;
    return sample_env_bilinear(env.texels, env.w, env.h, u, v) *
           env.intensity;
}

// Scale an indirect contribution so its max component <= m (0 = off).
inline color clamp_contribution(const color& c, float m) {
    if (m <= 0.0f) return c;
    const float mx = std::fmax(c.x, std::fmax(c.y, c.z));
    return mx > m ? c * (m / mx) : c;
}

inline color trace(Ray ray, const Hittable& world, RNG& rng, int max_depth,
                   const EnvLookup& env, float clamp_indirect) {
    color radiance(0.0f);      // light collected so far
    color throughput(1.0f);    // fraction of it that survives back to the eye

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord rec;
        // t_min = 1e-3: a bounced ray starts ON a surface; float error can
        // put its origin a hair inside, and t_min=0 would let it re-hit the
        // same surface at t≈0 ("shadow acne" — dark speckles everywhere).
        if (!world.hit(ray, 1e-3f, std::numeric_limits<float>::infinity(), rec)) {
            const color c = throughput * miss_radiance(env, ray);
            return radiance +
                   (depth == 0 ? c : clamp_contribution(c, clamp_indirect));
        }

        // Collect whatever this surface emits (zero for non-lights), THEN
        // try to continue the path. Indirect pickups are firefly-clamped;
        // depth 0 (directly visible lights/background) never is.
        {
            const color c = throughput * rec.mat.emission;
            radiance +=
                depth == 0 ? c : clamp_contribution(c, clamp_indirect);
        }

        color attenuation;
        Ray scattered;
        if (!scatter(rec.mat, ray, rec, rng, attenuation, scattered)) {
            return radiance;                         // light hit, or absorbed
        }
        throughput *= attenuation;
        ray = scattered;

        // Russian roulette: after a few bounces, kill dim paths with
        // probability (1 - p) and divide survivors by p. The expected
        // contribution is unchanged — E[x] = p * (x/p) — so this stays
        // unbiased while skipping work on paths that barely matter.
        if (depth >= 3) {
            const float p = std::fmin(
                std::fmax(throughput.x, std::fmax(throughput.y, throughput.z)),
                0.95f);
            // p == 0 (all-black throughput) must terminate BEFORE the rng
            // test: next_float() can return exactly 0.0, and 0/0 would put
            // a NaN in the accumulator.
            if (p <= 0.0f) break;
            if (rng.next_float() > p) break;
            throughput /= p;
        }
    }
    return radiance;   // ran out of bounces; keep what was collected
}
