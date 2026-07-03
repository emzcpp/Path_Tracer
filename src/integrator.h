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

inline color trace(Ray ray, const Hittable& world, RNG& rng, int max_depth) {
    color radiance(0.0f);      // light collected so far
    color throughput(1.0f);    // fraction of it that survives back to the eye

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord rec;
        // t_min = 1e-3: a bounced ray starts ON a surface; float error can
        // put its origin a hair inside, and t_min=0 would let it re-hit the
        // same surface at t≈0 ("shadow acne" — dark speckles everywhere).
        if (!world.hit(ray, 1e-3f, std::numeric_limits<float>::infinity(), rec)) {
            return radiance + throughput * sky_radiance(ray);
        }

        // Collect whatever this surface emits (zero for non-lights), THEN
        // try to continue the path.
        radiance += throughput * rec.mat.emission;

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
            if (rng.next_float() > p) break;
            throughput /= p;
        }
    }
    return radiance;   // ran out of bounces; keep what was collected
}
