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
    // Session H: importance-sampling CDFs (null = NEE unavailable) and the
    // light-sampling toggle. Brute force remains the ground-truth mode.
    const float* row_cdf = nullptr;
    const float* cond_cdf = nullptr;
    bool nee = false;
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

// ---- Session H: env importance sampling (mirrored in pathtrace.metal) --
// The distribution is luminance x sin(theta) over the equirect image;
// image-space density converts to solid-angle pdf via
//   pdf_sa = p_uv / (2 pi^2 sin(theta)),
// with the SAME sin(theta) that weighted the CDF rows — carried
// consistently in both sample() and pdf() below.

// First index whose cdf value exceeds u. Identical loop on both backends.
inline int cdf_find(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (cdf[mid] > u) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Draw a direction from the env distribution; returns pdf w.r.t. solid
// angle (0 on degenerate rows/poles — caller skips those samples).
inline vec3 env_sample(const EnvLookup& env, float u1, float u2,
                       float& pdf_sa) {
    const int W = env.w, H = env.h;
    const int y = cdf_find(env.row_cdf, H, u1);
    const float* crow = env.cond_cdf + std::size_t(y) * W;
    const int x = cdf_find(crow, W, u2);
    const float row_lo = y > 0 ? env.row_cdf[y - 1] : 0.0f;
    const float row_w = env.row_cdf[y] - row_lo;
    const float col_lo = x > 0 ? crow[x - 1] : 0.0f;
    const float col_w = crow[x] - col_lo;
    // Continuous inversion inside the chosen texel (keeps stratification).
    const float fy = row_w > 0.0f ? (u1 - row_lo) / row_w : 0.5f;
    const float fx = col_w > 0.0f ? (u2 - col_lo) / col_w : 0.5f;
    const float u = (float(x) + fx) / float(W);
    const float v = (float(y) + fy) / float(H);
    // UV -> direction: exact inverse of the miss mapping, yaw included.
    const float phi = (u - 0.5f - env.yaw_norm) * 6.28318530717958648f;
    const float theta = v * 3.14159265358979f;
    const float st = std::sin(theta);
    const vec3 d(st * std::cos(phi), std::cos(theta), st * std::sin(phi));
    const float p_uv = row_w * col_w * float(W) * float(H);
    pdf_sa = st > 1e-6f
                 ? p_uv / (2.0f * 3.14159265358979f * 3.14159265358979f * st)
                 : 0.0f;
    return d;
}

// Solid-angle pdf the sampler above would assign to an arbitrary
// direction — the MIS counterpart. Same dir->UV text as miss_radiance.
inline float env_pdf(const EnvLookup& env, const vec3& dir) {
    const vec3 d = normalize(dir);
    float u = std::atan2(d.z, d.x) * 0.15915494309189533577f + 0.5f +
              env.yaw_norm;
    u = u - std::floor(u);
    const float v = std::acos(std::fmin(std::fmax(d.y, -1.0f), 1.0f)) *
                    0.31830988618379067154f;
    const int W = env.w, H = env.h;
    int x = int(u * float(W));
    int y = int(v * float(H));
    if (x >= W) x = W - 1;
    if (y >= H) y = H - 1;
    const float* crow = env.cond_cdf + std::size_t(y) * W;
    const float row_lo = y > 0 ? env.row_cdf[y - 1] : 0.0f;
    const float row_w = env.row_cdf[y] - row_lo;
    const float col_lo = x > 0 ? crow[x - 1] : 0.0f;
    const float col_w = crow[x] - col_lo;
    const float st = std::sqrt(std::fmax(0.0f, 1.0f - d.y * d.y));
    if (st <= 1e-6f) return 0.0f;
    return row_w * col_w * float(W) * float(H) /
           (2.0f * 3.14159265358979f * 3.14159265358979f * st);
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
    bool prev_nee = false;     // env NEE ran at the previous path vertex

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord rec;
        // t_min = 1e-3: a bounced ray starts ON a surface; float error can
        // put its origin a hair inside, and t_min=0 would let it re-hit the
        // same surface at t≈0 ("shadow acne" — dark speckles everywhere).
        if (!world.hit(ray, 1e-3f, std::numeric_limits<float>::infinity(), rec)) {
            // NEE double-count rule: a vertex that ran env NEE already
            // collected the environment's direct term, so its continuation
            // miss must not add it again. Camera rays and delta/glass
            // continuations (no NEE there) still see the env in full.
            if (!prev_nee) {
                const color c = throughput * miss_radiance(env, ray);
                radiance +=
                    (depth == 0 ? c : clamp_contribution(c, clamp_indirect));
            }
            return radiance;
        }

        // Collect whatever this surface emits (zero for non-lights), THEN
        // try to continue the path. Indirect pickups are firefly-clamped;
        // depth 0 (directly visible lights/background) never is.
        {
            const color c = throughput * rec.mat.emission;
            radiance +=
                depth == 0 ? c : clamp_contribution(c, clamp_indirect);
        }

        // ---- Session H: next-event estimation toward the environment.
        // Deliberately sample the env distribution (the sun), evaluate the
        // BSDF for that direction, and add the contribution if the shadow
        // ray escapes. Delta glass is skipped — BSDF sampling owns it.
        // Near-specular skip (glass is delta; a roughness-0.02 metal's
        // lobe is spiked enough that env-sampling it produces jackpot
        // noise while suppressing the clean BSDF mirror estimator) —
        // those vertices rely on BSDF sampling. MIS (Stage 3) weights,
        // rather than gates, this trade.
        const bool can_nee = env.nee && env.row_cdf != nullptr &&
                             rec.mat.transmission <= 0.5f &&
                             rec.mat.roughness >= 0.1f;
        if (can_nee) {
            const float u1 = rng.next_float();
            const float u2 = rng.next_float();
            float pdf_env = 0.0f;
            const vec3 ldir = env_sample(env, u1, u2, pdf_env);
            if (pdf_env > 1e-12f) {
                float pdf_b = 0.0f;
                const vec3 vdir = -normalize(ray.dir);
                const color f = eval_bsdf(rec.mat, rec, vdir, ldir, pdf_b);
                const float nl = dot(rec.normal, ldir);
                if (nl > 1e-6f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)) {
                    const Ray shadow(rec.p, ldir);
                    if (!world.occluded(
                            shadow, 1e-3f,
                            std::numeric_limits<float>::infinity())) {
                        const color c = throughput * f * nl *
                                        miss_radiance(env, shadow) /
                                        pdf_env;
                        radiance += clamp_contribution(c, clamp_indirect);
                    }
                }
            }
        }
        prev_nee = can_nee;

        color attenuation;
        Ray scattered;
        float scatter_pdf = 0.0f;
        bool scatter_delta = false;
        if (!scatter(rec.mat, ray, rec, rng, attenuation, scattered,
                     scatter_pdf, scatter_delta)) {
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
