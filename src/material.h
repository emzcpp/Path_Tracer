#pragma once

// Scattering — future-Metal-kernel code, mirrored line-for-line in
// pathtrace.metal. Pure functions over plain structs: no I/O, no
// allocation, no virtual calls. Any change here must land in BOTH copies;
// --parity is the drift detector.
//
// Session B: principled metallic-roughness GGX BSDF (the glTF-standard
// model). The integrator's contract is unchanged — scatter() returns the
// full Monte Carlo weight f·cosθ/pdf for the direction it sampled — so
// all D/G/F evaluation and pdf bookkeeping lives here.
//
// The model:
//   specular: Cook-Torrance, GGX/Trowbridge-Reitz D, height-correlated
//             Smith G2, Schlick F with F0 = mix(ior-derived, baseColor,
//             metallic). alpha = roughness^2 (perceptual remap).
//   diffuse:  (1 - metallic) · baseColor/π, coupled by (1 - F_dielectric)
//             per the glTF spec, so specular and diffuse never sum past 1.
//   sampling: VNDF (Heitz 2018) for specular — its weight is bounded by
//             construction (F·G2/G1 ≤ 1), the structural defense against
//             fireflies — cosine-weighted for diffuse; the two combine via
//             a one-sample balance heuristic: pick a lobe by luminance
//             probability, evaluate the FULL bsdf, divide by the MIXED
//             pdf. Unbiased, and the diffuse pdf term floors the mixture
//             wherever that lobe can be chosen.
//   glass:    transmission = 1 keeps the delta dielectric (Schlick
//             reflect/refract) — rough transmission is a later session.
//
// Known limitation (deliberate): single-scattering GGX loses some energy
// at high roughness (no multiple-scatter compensation), so rough metals
// read slightly dark. That is the textbook baseline, not a bug.

#include <cmath>

#include "hittable.h"
#include "ray.h"
#include "rng.h"

inline float luminance(const color& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Branchless orthonormal basis (Duff et al. 2017).
inline void build_onb(const vec3& n, vec3& t, vec3& b) {
    const float s = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float xy = n.x * n.y * a;
    t = vec3(1.0f + s * n.x * n.x * a, s * xy, -s * n.x);
    b = vec3(xy, s + n.y * n.y * a, -n.y);
}

inline float schlick_scalar(float f0, float cos_theta) {
    const float m = 1.0f - cos_theta;
    return f0 + (1.0f - f0) * m * m * m * m * m;
}

inline color schlick_color(const color& f0, float cos_theta) {
    const float m = 1.0f - cos_theta;
    const float w = m * m * m * m * m;
    return color(f0.x + (1.0f - f0.x) * w, f0.y + (1.0f - f0.y) * w,
                 f0.z + (1.0f - f0.z) * w);
}

// GGX normal distribution. nh = n·h, a2 = alpha^2.
inline float ggx_d(float nh, float a2) {
    // Cancellation-free form of nh^2*(a2-1)+1: at tiny a2 the naive form
    // rounds (a2-1) to -1 and collapses to 0 at nh=1 -> D=inf -> NaN.
    const float d = (1.0f - nh * nh) + nh * nh * a2;
    return a2 / (3.14159265358979f * d * d);
}

// Smith Lambda for GGX; c = cosθ (clamped > 0 by the caller).
inline float smith_lambda(float c, float a2) {
    const float c2 = c * c;
    return (-1.0f + std::sqrt(1.0f + a2 * (1.0f - c2) / c2)) * 0.5f;
}

// Sample the GGX distribution of VISIBLE normals (Heitz 2018).
// vt = view direction in tangent space (z = up), returns half-vector in
// tangent space. pdf(l) = G1(v)·D(h) / (4·n·v).
inline vec3 sample_vndf(const vec3& vt, float alpha, float u1, float u2) {
    const vec3 vh = normalize(vec3(alpha * vt.x, alpha * vt.y, vt.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const vec3 T1 = lensq > 0.0f
        ? vec3(-vh.y, vh.x, 0.0f) / std::sqrt(lensq)
        : vec3(1.0f, 0.0f, 0.0f);
    const vec3 T2 = cross(vh, T1);
    const float r = std::sqrt(u1);
    const float phi = 6.28318530717958647692f * u2;
    const float p1 = r * std::cos(phi);
    float p2 = r * std::sin(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * std::sqrt(std::fmax(0.0f, 1.0f - p1 * p1)) + s * p2;
    const vec3 nh = p1 * T1 + p2 * T2 +
                    std::sqrt(std::fmax(0.0f, 1.0f - p1 * p1 - p2 * p2)) * vh;
    return normalize(
        vec3(alpha * nh.x, alpha * nh.y, std::fmax(1e-6f, nh.z)));
}

// Scatter the incoming ray at the hit. Returns false if the path ends
// (absorbed / degenerate-grazing sample); the integrator has already
// collected any emission before calling this. `attenuation` is the full
// estimator weight f·cosθ/pdf for the sampled direction.
inline bool scatter(const Material& mat, const Ray& in, const HitRecord& rec,
                    RNG& rng, color& attenuation, Ray& scattered) {
    // ---- glass: delta dielectric, unchanged from the original model ----
    if (mat.transmission > 0.5f) {
        attenuation = color(1.0f, 1.0f, 1.0f);
        const float eta = rec.front_face ? 1.0f / mat.ior : mat.ior;
        const vec3 unit = normalize(in.dir);
        const float cos_theta = std::fmin(dot(-unit, rec.normal), 1.0f);
        const float sin_theta =
            std::sqrt(std::fmax(0.0f, 1.0f - cos_theta * cos_theta));
        // Short-circuit || matters: on TIR the rng draw is NOT consumed.
        vec3 dir;
        const float fr0 = (1.0f - eta) / (1.0f + eta);
        if (eta * sin_theta > 1.0f ||
            schlick_scalar(fr0 * fr0, cos_theta) > rng.next_float()) {
            dir = reflect(unit, rec.normal);
        } else {
            dir = refract(unit, rec.normal, eta);
        }
        scattered = Ray(rec.p, dir);
        return true;
    }

    // ---- GGX metallic-roughness surface ----
    const vec3 n = rec.normal;
    const vec3 v = -normalize(in.dir);
    const float nv = dot(n, v);
    if (nv <= 1e-4f) return false;   // grazing/below: absorb, never divide

    const float alpha =
        std::fmax(1e-4f, mat.roughness * mat.roughness);
    const float a2 = alpha * alpha;
    const float f0d_s = (mat.ior - 1.0f) / (mat.ior + 1.0f);
    const float f0_diel = f0d_s * f0d_s;   // ~0.04 at ior 1.5
    const color f0((1.0f - mat.metallic) * f0_diel +
                       mat.metallic * mat.base_color.x,
                   (1.0f - mat.metallic) * f0_diel +
                       mat.metallic * mat.base_color.y,
                   (1.0f - mat.metallic) * f0_diel +
                       mat.metallic * mat.base_color.z);
    const color diffuse_albedo = (1.0f - mat.metallic) * mat.base_color;

    // Lobe probability from relative importance at this view angle.
    const float spec_lum = luminance(schlick_color(f0, nv));
    const float diff_lum = luminance(diffuse_albedo);
    float p_spec = 1.0f;
    if (diff_lum > 0.0f) {
        p_spec = spec_lum / (spec_lum + diff_lum);
        p_spec = std::fmin(std::fmax(p_spec, 0.1f), 0.9f);
    }

    // Tangent frame; all sampling happens with z = normal.
    vec3 t, b;
    build_onb(n, t, b);
    const vec3 vt(dot(v, t), dot(v, b), nv);

    const float u_lobe = rng.next_float();
    const float u1 = rng.next_float();
    const float u2 = rng.next_float();

    vec3 lt;
    if (u_lobe < p_spec) {
        const vec3 ht = sample_vndf(vt, alpha, u1, u2);
        lt = reflect(-vt, ht);
    } else {
        const float rad = std::sqrt(u1);
        const float phi = 6.28318530717958647692f * u2;
        lt = vec3(rad * std::cos(phi), rad * std::sin(phi),
                  std::sqrt(std::fmax(0.0f, 1.0f - u1)));
    }
    const float nl = lt.z;
    if (nl <= 1e-6f) return false;   // sampled below the surface: absorb

    // Evaluate the FULL bsdf and the MIXED pdf at the sampled direction
    // (one-sample balance heuristic).
    const vec3 h = normalize(vt + lt);
    const float nh = std::fmax(0.0f, h.z);
    const float vh = std::fmax(1e-6f, dot(vt, h));

    const float D = ggx_d(nh, a2);
    const float Lv = smith_lambda(nv, a2);
    const float Ll = smith_lambda(nl, a2);
    const float G2 = 1.0f / (1.0f + Lv + Ll);
    const float G1v = 1.0f / (1.0f + Lv);

    const color F = schlick_color(f0, vh);
    const float F_diel = schlick_scalar(f0_diel, vh);

    const float spec_k = D * G2 / (4.0f * nv * nl);
    const color f_spec = spec_k * F;
    const color f_diff = (1.0f - F_diel) * (1.0f / 3.14159265358979f) *
                         diffuse_albedo;

    const float pdf_spec = G1v * D / (4.0f * nv);     // VNDF pdf of lt
    const float pdf_diff = nl * (1.0f / 3.14159265358979f);
    const float pdf = p_spec * pdf_spec + (1.0f - p_spec) * pdf_diff;
    if (pdf <= 1e-8f) return false;

    attenuation = (f_spec + f_diff) * (nl / pdf);
    scattered = Ray(rec.p, t * lt.x + b * lt.y + n * lt.z);
    return true;
}