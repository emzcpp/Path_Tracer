// The path-tracing kernel: a line-by-line port of the CPU reference
// (rng.h / sphere.h / material.h / integrator.h). Any change to the CPU
// files must be mirrored here — the --parity mode exists to catch drift.
//
// Parity contract with renderer.cpp:
//   - per-(pixel, pass) seed: pcg_init(mix64(px ^ (pass << 32)), px),
//     and pcg_init performs the SAME two warm-up next_u32() calls as the
//     CPU constructor
//   - RNG draw order per sample: u-jitter, v-jitter, then trace
//   - u/v computed as (x + r) * (1/w), matching the CPU's precomputed
//     reciprocal (a division would round differently)

#include <metal_stdlib>
using namespace metal;

#include "kernel_types.h"

inline float3 c3(pt_float3 v) { return float3(v.x, v.y, v.z); }

// Hand-written cross product (literal vec3.h formula): metal::cross may
// fma-contract differently in the last ulp, and the Möller-Trumbore
// determinant sits directly on branch decisions.
inline float3 cross_pt(float3 a, float3 b) {
    return float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

// ---------------------------------------------------------------- PCG32
// Port of rng.h. ulong math is emulated on the 32-bit GPU ALU (~3 muls per
// 64-bit mul) but RNG is <5% of kernel cost — exact parity is worth it.

struct PRNG {
    ulong state;
    ulong inc;
};

inline uint pcg_next_u32(thread PRNG& r) {
    const ulong old = r.state;
    r.state = old * 6364136223846793005UL + r.inc;
    const uint xorshifted = uint(((old >> 18) ^ old) >> 27);
    const uint rot = uint(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

inline PRNG pcg_init(ulong seed, ulong sequence) {
    PRNG r;
    r.state = 0UL;
    r.inc = (sequence << 1) | 1UL;
    pcg_next_u32(r);        // two warm-up draws, exactly like RNG's ctor
    r.state += seed;
    pcg_next_u32(r);
    return r;
}

inline float pcg_next_float(thread PRNG& r) {
    return float(pcg_next_u32(r) >> 8) * (1.0f / 16777216.0f);
}

inline ulong mix64(ulong v) {
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdUL;
    v ^= v >> 33;
    v *= 0xc4ceb9fe1a85ec53UL;
    return v ^ (v >> 33);
}

inline float3 random_unit_vector(thread PRNG& rng) {
    const float z = 1.0f - 2.0f * pcg_next_float(rng);
    const float phi = 6.28318530717958647692f * pcg_next_float(rng);
    const float r = sqrt(fmax(0.0f, 1.0f - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

// ------------------------------------------------------------- geometry

struct HitRec {
    float3 p;
    float3 normal;
    float t;
    bool front_face;
    uint sphere_idx;
};

inline bool hit_sphere(constant GPUSphere& s, float3 ro, float3 rd,
                       float t_min, float t_max, thread HitRec& rec) {
    const float3 oc = ro - c3(s.center);
    const float a = dot(rd, rd);
    const float half_b = dot(oc, rd);
    const float cc = dot(oc, oc) - s.radius * s.radius;

    const float discriminant = half_b * half_b - a * cc;
    if (discriminant < 0.0f) return false;
    const float sqrt_d = sqrt(discriminant);

    float root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max) return false;
    }

    rec.t = root;
    rec.p = ro + root * rd;
    const float3 outward = (rec.p - c3(s.center)) / s.radius;
    rec.front_face = dot(rd, outward) < 0.0f;
    rec.normal = rec.front_face ? outward : -outward;
    return true;
}

inline bool hit_scene(constant GPUSphere* spheres, uint n, float3 ro,
                      float3 rd, float t_min, float t_max,
                      thread HitRec& rec) {
    bool hit_anything = false;
    float closest = t_max;
    for (uint i = 0; i < n; ++i) {
        HitRec temp;
        if (hit_sphere(spheres[i], ro, rd, t_min, closest, temp)) {
            hit_anything = true;
            closest = temp.t;
            rec = temp;
            rec.sphere_idx = i;
        }
    }
    return hit_anything;
}

// ---------------------------------------------------------- mesh + BVH
// Line-for-line mirror of mesh.h (CPU). Any change must land in BOTH.

struct TriHit {
    float u;
    float v;
    uint tri;
};

// Möller–Trumbore. Absolute det epsilon is REQUIRED (det==0 -> inf/NaN
// leaks through range tests). No backface culling. Tie rule `t > closest`
// rejects — identical to the sphere path.
inline bool hit_tri(const GPUTriangle T, float3 ro, float3 rd, float t_min,
                    thread float& closest, thread TriHit& h) {
    const float3 pv = cross_pt(rd, c3(T.e2));
    const float det = dot(c3(T.e1), pv);
    if (fabs(det) < 1e-8f) return false;
    const float inv_det = 1.0f / det;
    const float3 tv = ro - c3(T.p0);
    const float u = dot(tv, pv) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const float3 qv = cross_pt(tv, c3(T.e1));
    const float v = dot(rd, qv) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = dot(c3(T.e2), qv) * inv_det;
    if (t < t_min || t > closest) return false;
    closest = t;
    h.u = u;
    h.v = v;
    return true;
}

inline bool hit_aabb(pt_float3 mn, pt_float3 mx, float3 ro, float3 inv,
                     float t_min, float t_max, thread float& tnear) {
    float a = (mn.x - ro.x) * inv.x, b = (mx.x - ro.x) * inv.x;
    float t0 = fmin(a, b), t1 = fmax(a, b);
    a = (mn.y - ro.y) * inv.y;
    b = (mx.y - ro.y) * inv.y;
    t0 = fmax(t0, fmin(a, b));
    t1 = fmin(t1, fmax(a, b));
    a = (mn.z - ro.z) * inv.z;
    b = (mx.z - ro.z) * inv.z;
    t0 = fmax(t0, fmin(a, b));
    t1 = fmin(t1, fmax(a, b));
    tnear = t0;
    return t0 <= t1 && t0 <= t_max && t1 >= t_min;
}

// Ordered traversal, near child first, deterministic tie -> left. Depth
// cap 30 at build makes stack[32] a guarantee.
inline bool bvh_hit(device const BVHNode* nodes,
                    device const GPUTriangle* tris, float3 ro, float3 rd,
                    float t_min, thread float& closest, thread TriHit& best) {
    const float3 inv = float3(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);
    float tn;
    if (!hit_aabb(nodes[0].mn, nodes[0].mx, ro, inv, t_min, closest, tn))
        return false;
    uint stack[32];
    uint sp = 0;
    uint idx = 0;
    bool found = false;
    for (;;) {
        const BVHNode n = nodes[idx];
        if (n.tri_count > 0) {   // leaf
            for (uint i = 0; i < n.tri_count; ++i) {
                if (hit_tri(tris[n.left_or_first + i], ro, rd, t_min, closest,
                            best)) {
                    found = true;
                    best.tri = n.left_or_first + i;
                }
            }
            if (sp == 0) break;
            idx = stack[--sp];
        } else {                 // internal
            uint l = n.left_or_first, r = l + 1;
            float tl, tr;
            const bool hl =
                hit_aabb(nodes[l].mn, nodes[l].mx, ro, inv, t_min, closest, tl);
            const bool hr =
                hit_aabb(nodes[r].mn, nodes[r].mx, ro, inv, t_min, closest, tr);
            if (hl && hr) {
                if (tr < tl) {
                    const uint tmp = l;
                    l = r;
                    r = tmp;
                }
                stack[sp++] = r;
                idx = l;
            } else if (hl) {
                idx = l;
            } else if (hr) {
                idx = r;
            } else {
                if (sp == 0) break;
                idx = stack[--sp];
            }
        }
    }
    return found;
}

// ----------------------------------------------------- texture sampling
// Mirror of texture.h. Manual bilinear on ushort4 linear texels — NOT
// hardware filtering, whose fixed-point weights would break parity.

inline float3 fetch_texel(device const ushort* tex, uint W, int x, int y) {
    device const ushort* p = tex + (ulong(uint(y)) * W + uint(x)) * 4;
    const float k = 1.0f / 65535.0f;
    return float3(p[0] * k, p[1] * k, p[2] * k);
}

inline float3 sample_bilinear(device const ushort* tex, uint W, uint H,
                              float u, float v) {
    u = u - floor(u);
    v = v - floor(v);
    const float x = u * float(W) - 0.5f;
    const float y = v * float(H) - 0.5f;
    const float fx = floor(x), fy = floor(y);
    const float ax = x - fx, ay = y - fy;
    int x0 = int(fx), y0 = int(fy), x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 += int(W);
    if (y0 < 0) y0 += int(H);
    if (x1 >= int(W)) x1 -= int(W);
    if (y1 >= int(H)) y1 -= int(H);
    const float3 c00 = fetch_texel(tex, W, x0, y0),
                 c10 = fetch_texel(tex, W, x1, y0);
    const float3 c01 = fetch_texel(tex, W, x0, y1),
                 c11 = fetch_texel(tex, W, x1, y1);
    return (1.0f - ax) * (1.0f - ay) * c00 + ax * (1.0f - ay) * c10 +
           (1.0f - ax) * ay * c01 + ax * ay * c11;
}

// ------------------------------------------------------------ materials
// Mirror of material.h (CPU). Session B: principled metallic-roughness
// GGX BSDF — VNDF specular sampling + cosine diffuse, one-sample balance
// heuristic. Any change must land in BOTH copies; --parity is the drift
// detector.

inline float luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Branchless orthonormal basis (Duff et al. 2017).
inline void build_onb(float3 n, thread float3& t, thread float3& b) {
    const float s = n.z >= 0.0f ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float xy = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * xy, -s * n.x);
    b = float3(xy, s + n.y * n.y * a, -n.y);
}

inline float schlick_scalar(float f0, float cos_theta) {
    const float m = 1.0f - cos_theta;
    return f0 + (1.0f - f0) * m * m * m * m * m;
}

inline float3 schlick_color(float3 f0, float cos_theta) {
    const float m = 1.0f - cos_theta;
    const float w = m * m * m * m * m;
    return float3(f0.x + (1.0f - f0.x) * w, f0.y + (1.0f - f0.y) * w,
                  f0.z + (1.0f - f0.z) * w);
}

inline float ggx_d(float nh, float a2) {
    // Cancellation-free form of nh^2*(a2-1)+1: at tiny a2 the naive form
    // rounds (a2-1) to -1 and collapses to 0 at nh=1 -> D=inf -> NaN.
    const float d = (1.0f - nh * nh) + nh * nh * a2;
    return a2 / (3.14159265358979f * d * d);
}

inline float smith_lambda(float c, float a2) {
    const float c2 = c * c;
    return (-1.0f + sqrt(1.0f + a2 * (1.0f - c2) / c2)) * 0.5f;
}

// GGX visible-normal sampling (Heitz 2018); vt = view in tangent space.
inline float3 sample_vndf(float3 vt, float alpha, float u1, float u2) {
    const float3 vh = normalize(float3(alpha * vt.x, alpha * vt.y, vt.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 T1 = lensq > 0.0f
        ? float3(-vh.y, vh.x, 0.0f) / sqrt(lensq)
        : float3(1.0f, 0.0f, 0.0f);
    const float3 T2 = cross_pt(vh, T1);
    const float r = sqrt(u1);
    const float phi = 6.28318530717958647692f * u2;
    const float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrt(fmax(0.0f, 1.0f - p1 * p1)) + s * p2;
    const float3 nh = p1 * T1 + p2 * T2 +
                      sqrt(fmax(0.0f, 1.0f - p1 * p1 - p2 * p2)) * vh;
    return normalize(
        float3(alpha * nh.x, alpha * nh.y, fmax(1e-6f, nh.z)));
}

// Manual port: Metal's builtin refract() returns zero on TIR — different
// semantics from vec3.h's refract, so we don't risk it.
inline float3 refract_pt(float3 uv, float3 n, float eta) {
    const float cos_theta = fmin(dot(-uv, n), 1.0f);
    const float3 r_perp = eta * (uv + cos_theta * n);
    const float3 r_parallel = -sqrt(fabs(1.0f - dot(r_perp, r_perp))) * n;
    return r_perp + r_parallel;
}

// Evaluated per-hit material: filled from sphere fields or from mesh
// texture samples — scatter() below serves both.
struct EvalMat {
    float3 base_color;
    float3 emission;
    float metallic;
    float roughness;
    float ior;
    float transmission;
};

inline EvalMat eval_sphere(constant GPUSphere& s) {
    EvalMat m;
    m.base_color = c3(s.base_color);
    m.emission = c3(s.emission);
    m.metallic = s.metallic;
    m.roughness = s.roughness;
    m.ior = s.ior;
    m.transmission = s.transmission;
    return m;
}

// Mirror of mesh.h's eval_mesh_material: glTF metallic-roughness textures
// feed the BSDF continuously — no threshold.
inline EvalMat eval_mesh_material(device const ushort* tex_base,
                                  device const ushort* tex_mr,
                                  device const ushort* tex_emis,
                                  constant MeshUniforms& MU,
                                  float u, float v) {
    const float3 base = sample_bilinear(tex_base, MU.base_w, MU.base_h, u, v);
    const float3 mr = MU.mr_w != 0u
                          ? sample_bilinear(tex_mr, MU.mr_w, MU.mr_h, u, v)
                          : float3(0.0f, 1.0f, 0.0f);
    const float3 emis =
        MU.emis_w != 0u ? sample_bilinear(tex_emis, MU.emis_w, MU.emis_h, u, v)
                        : float3(0.0f);
    EvalMat m;
    m.base_color = base;
    m.emission = emis * MU.emissive_scale;
    m.metallic = mr.z;    // glTF: B = metallic
    m.roughness = mr.y;   // glTF: G = roughness (perceptual)
    m.ior = 1.5f;
    m.transmission = 0.0f;
    return m;
}

// Full estimator weight f·cosθ/pdf for the sampled direction — mirror of
// material.h's scatter, line for line.
inline bool scatter(thread const EvalMat& mat, float3 in_dir, float3 normal,
                    bool front_face, thread PRNG& rng,
                    thread float3& attenuation, thread float3& out_dir) {
    // ---- glass: delta dielectric ----
    if (mat.transmission > 0.5f) {
        attenuation = float3(1.0f);
        const float eta = front_face ? 1.0f / mat.ior : mat.ior;
        const float3 unit = normalize(in_dir);
        const float cos_theta = fmin(dot(-unit, normal), 1.0f);
        const float sin_theta =
            sqrt(fmax(0.0f, 1.0f - cos_theta * cos_theta));
        // Short-circuit || matters: on TIR the rng draw is NOT consumed.
        float3 dir;
        const float fr0 = (1.0f - eta) / (1.0f + eta);
        if (eta * sin_theta > 1.0f ||
            schlick_scalar(fr0 * fr0, cos_theta) > pcg_next_float(rng)) {
            dir = reflect(unit, normal);
        } else {
            dir = refract_pt(unit, normal, eta);
        }
        out_dir = dir;
        return true;
    }

    // ---- GGX metallic-roughness surface ----
    const float3 n = normal;
    const float3 v = -normalize(in_dir);
    const float nv = dot(n, v);
    if (nv <= 1e-4f) return false;   // grazing/below: absorb, never divide

    const float alpha = fmax(1e-4f, mat.roughness * mat.roughness);
    const float a2 = alpha * alpha;
    const float f0d_s = (mat.ior - 1.0f) / (mat.ior + 1.0f);
    const float f0_diel = f0d_s * f0d_s;
    const float3 f0 = float3((1.0f - mat.metallic) * f0_diel +
                                 mat.metallic * mat.base_color.x,
                             (1.0f - mat.metallic) * f0_diel +
                                 mat.metallic * mat.base_color.y,
                             (1.0f - mat.metallic) * f0_diel +
                                 mat.metallic * mat.base_color.z);
    const float3 diffuse_albedo = (1.0f - mat.metallic) * mat.base_color;

    const float spec_lum = luminance(schlick_color(f0, nv));
    const float diff_lum = luminance(diffuse_albedo);
    float p_spec = 1.0f;
    if (diff_lum > 0.0f) {
        p_spec = spec_lum / (spec_lum + diff_lum);
        p_spec = fmin(fmax(p_spec, 0.1f), 0.9f);
    }

    float3 t, b;
    build_onb(n, t, b);
    const float3 vt = float3(dot(v, t), dot(v, b), nv);

    const float u_lobe = pcg_next_float(rng);
    const float u1 = pcg_next_float(rng);
    const float u2 = pcg_next_float(rng);

    float3 lt;
    if (u_lobe < p_spec) {
        const float3 ht = sample_vndf(vt, alpha, u1, u2);
        lt = reflect(-vt, ht);
    } else {
        const float rad = sqrt(u1);
        const float phi = 6.28318530717958647692f * u2;
        lt = float3(rad * cos(phi), rad * sin(phi),
                    sqrt(fmax(0.0f, 1.0f - u1)));
    }
    const float nl = lt.z;
    if (nl <= 1e-6f) return false;

    const float3 h = normalize(vt + lt);
    const float nh = fmax(0.0f, h.z);
    const float vh = fmax(1e-6f, dot(vt, h));

    const float D = ggx_d(nh, a2);
    const float Lv = smith_lambda(nv, a2);
    const float Ll = smith_lambda(nl, a2);
    const float G2 = 1.0f / (1.0f + Lv + Ll);
    const float G1v = 1.0f / (1.0f + Lv);

    const float3 F = schlick_color(f0, vh);
    const float F_diel = schlick_scalar(f0_diel, vh);

    const float spec_k = D * G2 / (4.0f * nv * nl);
    const float3 f_spec = spec_k * F;
    const float3 f_diff = (1.0f - F_diel) * (1.0f / 3.14159265358979f) *
                          diffuse_albedo;

    const float pdf_spec = G1v * D / (4.0f * nv);
    const float pdf_diff = nl * (1.0f / 3.14159265358979f);
    const float pdf = p_spec * pdf_spec + (1.0f - p_spec) * pdf_diff;
    if (pdf <= 1e-8f) return false;

    attenuation = (f_spec + f_diff) * (nl / pdf);
    out_dir = t * lt.x + b * lt.y + n * lt.z;
    return true;
}

// ----------------------------------------------------------- integrator

// Manual lerp matching vec3.h's (1-t)*a + t*b (metal::mix computes
// a + (b-a)*t — different rounding).
inline float3 lerp_pt(float3 a, float3 b, float t) {
    return (1.0f - t) * a + t * b;
}

inline float3 sky_radiance(float3 rd) {
    const float t = 0.5f * (normalize(rd).y + 1.0f);
    return lerp_pt(float3(1.0f), float3(0.5f, 0.7f, 1.0f), t);
}

// ---- Session F: equirect HDRI environment (mirror of integrator.h) ----

inline float3 env_fetch(device const float* tex, uint W, int x, int y) {
    device const float* p = tex + (ulong(uint(y)) * W + uint(x)) * 4;
    return float3(p[0], p[1], p[2]);
}

// Same manual-bilinear structure as material textures, with equirect
// semantics: u WRAPS (azimuth seam), v CLAMPS (poles).
inline float3 sample_env_bilinear(device const float* tex, uint W, uint H,
                                  float u, float v) {
    u = u - floor(u);
    v = fmin(fmax(v, 0.0f), 1.0f);
    const float x = u * float(W) - 0.5f;
    const float y = v * float(H) - 0.5f;
    const float fx = floor(x), fy = floor(y);
    const float ax = x - fx, ay = y - fy;
    int x0 = int(fx), x1 = x0 + 1;
    int y0 = int(fy), y1 = y0 + 1;
    if (x0 < 0) x0 += int(W);
    if (x1 >= int(W)) x1 -= int(W);
    if (y0 < 0) y0 = 0;
    if (y1 >= int(H)) y1 = int(H) - 1;
    const float3 c00 = env_fetch(tex, W, x0, y0), c10 = env_fetch(tex, W, x1, y0);
    const float3 c01 = env_fetch(tex, W, x0, y1), c11 = env_fetch(tex, W, x1, y1);
    return (1.0f - ax) * (1.0f - ay) * c00 + ax * (1.0f - ay) * c10 +
           (1.0f - ax) * ay * c01 + ax * ay * c11;
}

inline float3 miss_radiance(device const float* env_texels, uint env_w,
                            uint env_h, float intensity, float yaw_norm,
                            float3 rd) {
    if (env_w == 0u) return sky_radiance(rd);
    const float3 d = normalize(rd);
    const float u = atan2(d.z, d.x) * 0.15915494309189533577f + 0.5f +
                    yaw_norm;
    const float v =
        acos(fmin(fmax(d.y, -1.0f), 1.0f)) * 0.31830988618379067154f;
    return sample_env_bilinear(env_texels, env_w, env_h, u, v) * intensity;
}

// Scale an indirect contribution so its max component <= m (0 = off).
inline float3 clamp_contribution(float3 c, float m) {
    if (m <= 0.0f) return c;
    const float mx = fmax(c.x, fmax(c.y, c.z));
    return mx > m ? c * (m / mx) : c;
}

inline float3 trace(float3 ro, float3 rd, constant GPUSphere* spheres,
                    uint n, device const BVHNode* nodes,
                    device const GPUTriangle* tris,
                    device const ushort* tex_base,
                    device const ushort* tex_mr,
                    device const ushort* tex_emis,
                    device const ushort* tex_norm,
                    device const float* env_texels,
                    constant MeshUniforms& MU, constant PassUniforms& U,
                    thread PRNG& rng, uint max_depth) {
    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);

    for (uint depth = 0; depth < max_depth; ++depth) {
        // Closest hit: spheres first, then the mesh BVH seeded with the
        // sphere winner's t — mirrors the CPU's Scene ordering (mesh added
        // last), so tie-breaking is identical on both backends.
        HitRec rec;
        const bool hit_s = hit_scene(spheres, n, ro, rd, 1e-3f, INFINITY, rec);
        float closest = hit_s ? rec.t : INFINITY;
        TriHit th;
        bool hit_m = false;
        if (MU.has_mesh != 0u) {
            hit_m = bvh_hit(nodes, tris, ro, rd, 1e-3f, closest, th);
        }
        if (!hit_s && !hit_m) {
            const float3 c = throughput *
                             miss_radiance(env_texels, U.env_w, U.env_h,
                                           U.env_intensity, U.env_yaw_norm,
                                           rd);
            return radiance + (depth == 0u
                                   ? c
                                   : clamp_contribution(c, U.clamp_indirect));
        }

        EvalMat mat;
        float3 hp, hn;
        bool front;
        if (hit_m) {
            const GPUTriangle T = tris[th.tri];
            hp = ro + closest * rd;
            // Sidedness from the GEOMETRIC normal; the interpolated shading
            // normal only bends the scatter lobe. Mirrors mesh.h.
            const float3 ng = cross_pt(c3(T.e1), c3(T.e2));
            front = dot(rd, ng) < 0.0f;
            const float w = 1.0f - th.u - th.v;
            float3 ns = normalize(w * c3(T.n0) + th.u * c3(T.n1) +
                                  th.v * c3(T.n2));
            const float u = w * T.u0 + th.u * T.u1 + th.v * T.u2;
            const float v = w * T.v0 + th.u * T.v1 + th.v * T.v2;
            // Tangent-space normal mapping — mirror of mesh.h. glTF:
            // LINEAR texels, +Y up, [0,1] -> [-1,1]; bitangent takes the
            // .w handedness sign.
            if (MU.norm_w != 0u) {
                const float3 tn =
                    sample_bilinear(tex_norm, MU.norm_w, MU.norm_h, u, v);
                const float3 tN = float3(2.0f * tn.x - 1.0f,
                                         2.0f * tn.y - 1.0f,
                                         2.0f * tn.z - 1.0f);
                float3 tang = w * c3(T.t0) + th.u * c3(T.t1) +
                              th.v * c3(T.t2);
                tang = tang - ns * dot(ns, tang);
                const float tl = length(tang);
                if (tl > 1e-6f) {
                    tang = tang / tl;
                    const float3 bit = cross_pt(ns, tang) * T.w0;
                    ns = normalize(tN.x * tang + tN.y * bit + tN.z * ns);
                }
            }
            hn = front ? ns : -ns;
            mat = eval_mesh_material(tex_base, tex_mr, tex_emis, MU, u, v);
        } else {
            hp = rec.p;
            hn = rec.normal;
            front = rec.front_face;
            mat = eval_sphere(spheres[rec.sphere_idx]);
        }

        {
            const float3 c = throughput * mat.emission;
            radiance += depth == 0u
                            ? c
                            : clamp_contribution(c, U.clamp_indirect);
        }

        float3 attenuation, new_dir;
        if (!scatter(mat, rd, hn, front, rng, attenuation, new_dir)) {
            return radiance;
        }
        throughput *= attenuation;
        ro = hp;
        rd = new_dir;

        if (depth >= 3) {
            const float p = fmin(
                fmax(throughput.x, fmax(throughput.y, throughput.z)), 0.95f);
            if (pcg_next_float(rng) > p) break;
            throughput /= p;
        }
    }
    return radiance;
}

// -------------------------------------------------------------- kernels

// One thread per pixel, K passes looped in-kernel: one accum read-modify-
// write per K passes, and lanes whose paths die early start their next
// pass instead of idling — the sum of K path lengths has much lower
// variance than one, so SIMD groups re-converge.
kernel void accumulate(device float4* accum            [[buffer(0)]],
                       constant GPUSphere* spheres     [[buffer(1)]],
                       constant PassUniforms& U        [[buffer(2)]],
                       device const BVHNode* nodes     [[buffer(3)]],
                       device const GPUTriangle* tris  [[buffer(4)]],
                       device const ushort* tex_base   [[buffer(5)]],
                       device const ushort* tex_mr     [[buffer(6)]],
                       device const ushort* tex_emis   [[buffer(7)]],
                       constant MeshUniforms& MU       [[buffer(8)]],
                       device const ushort* tex_norm   [[buffer(9)]],
                       device const float* env_texels  [[buffer(10)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // The dispatch may cover a row slice [row_offset, row_offset+rows);
    // everything below keys off the FRAME pixel, so slicing is invisible
    // to the math (identical seeds and jitter per (pixel, pass)).
    const uint py = gid.y + U.row_offset;
    if (py >= U.height) return;
    const ulong px = ulong(py) * U.width + gid.x;
    const float inv_w = 1.0f / float(U.width);
    const float inv_h = 1.0f / float(U.height);

    float3 total = float3(0.0f);
    for (uint k = 0; k < U.pass_count; ++k) {
        const ulong pass = ulong(U.pass_base + k);
        PRNG rng = pcg_init(mix64(px ^ (pass << 32)), px);

        const float u = (float(gid.x) + pcg_next_float(rng)) * inv_w;
        const float v = 1.0f - (float(py) + pcg_next_float(rng)) * inv_h;
        const float3 ro = c3(U.cam.origin);
        const float3 rd = c3(U.cam.lower_left) + u * c3(U.cam.horizontal) +
                          v * c3(U.cam.vertical) - ro;

        total += trace(ro, rd, spheres, U.sphere_count, nodes, tris, tex_base,
                       tex_mr, tex_emis, tex_norm, env_texels, MU, U, rng,
                       U.max_depth);
    }
    accum[px] += float4(total, 0.0f);
}

// accum/passes -> gamma 1/2.2 -> drawable. Nearest-neighbor coordinate
// scaling handles the half-res preview (accum dims < drawable dims).
kernel void resolve(device const float4* accum         [[buffer(0)]],
                    constant ResolveUniforms& U        [[buffer(1)]],
                    texture2d<float, access::write> out [[texture(0)]],
                    uint2 gid [[thread_position_in_grid]]) {
    const uint sx = gid.x * U.accum_w / U.out_w;
    const uint sy = gid.y * U.accum_h / U.out_h;
    const float4 sum = accum[ulong(sy) * U.accum_w + sx];

    // Rows above the partial-pass cursor carry one extra sample; normalize
    // per source row so mid-pass frames display without banding.
    const uint n = U.pass_total + (sy < U.rows_plus1 ? 1u : 0u);
    const float inv_n = 1.0f / float(max(n, 1u));
    // Same gamma as image.h's encode_channel, so the screen matches PNGs.
    float3 c = pow(fmax(sum.xyz * inv_n, 0.0f), float3(1.0f / 2.2f));
    out.write(float4(saturate(c), 1.0f), gid);
}


// ================= Session E: raster navigation preview =================
// Preview-only pipeline for fast navigation. Deliberately independent of
// the tracing kernels above: it reads the SAME GPUTriangle/GPUSphere
// buffers the tracer consumes (no duplicated geometry) but never touches
// the accumulation buffer or the resolve path, so tracing correctness and
// parity are structurally unaffected.

struct RasterUniforms {          // mirrored in gpu_renderer.mm
    float4x4 vp;                 // GL-style clip; z remapped below
    float4 cam_pos;              // xyz = camera position (headlight)
    uint4 misc;                  // x: 1 = overlay tint
};

struct RasterOut {
    float4 pos [[position]];
    float3 world;
    float3 normal;
    float3 albedo;
};

// GL clip -> Metal NDC depth ([-w,w] -> [0,w]); x/y untouched so the
// preview registers with the traced image to sub-pixel.
inline float4 to_metal_clip(float4 clip) {
    clip.z = (clip.z + clip.w) * 0.5f;
    return clip;
}

// Meshes: fetch straight from the tracer's triangle records.
vertex RasterOut raster_mesh_vs(uint vid [[vertex_id]],
                                device const GPUTriangle* tris [[buffer(0)]],
                                constant RasterUniforms& U [[buffer(1)]]) {
    const uint t = vid / 3u;
    const uint c = vid % 3u;
    const GPUTriangle T = tris[t];
    float3 p = c3(T.p0);
    if (c == 1u) p += c3(T.e1);
    if (c == 2u) p += c3(T.e2);
    const float3 n = c == 0u ? c3(T.n0) : (c == 1u ? c3(T.n1) : c3(T.n2));

    RasterOut o;
    o.pos = to_metal_clip(U.vp * float4(p, 1.0f));
    o.world = p;
    o.normal = n;
    o.albedo = float3(0.72f);   // neutral read of form; nav aid, not beauty
    return o;
}

// Spheres: one unit-sphere proxy, instanced over the tracer's sphere
// buffer (center + radius + color per instance).
vertex RasterOut raster_sphere_vs(uint vid [[vertex_id]],
                                  uint iid [[instance_id]],
                                  device const float* unit [[buffer(0)]],
                                  constant RasterUniforms& U [[buffer(1)]],
                                  constant GPUSphere* spheres [[buffer(2)]]) {
    const float3 up = float3(unit[vid * 3u], unit[vid * 3u + 1u],
                             unit[vid * 3u + 2u]);
    constant GPUSphere& s = spheres[iid];
    const float3 p = c3(s.center) + s.radius * up;

    RasterOut o;
    o.pos = to_metal_clip(U.vp * float4(p, 1.0f));
    o.world = p;
    o.normal = up;
    // Emissive spheres read as bright so lights stay identifiable.
    const float3 e = c3(s.emission);
    o.albedo = c3(s.base_color) + min(float3(1.0f), e);
    return o;
}

fragment float4 raster_fs(RasterOut in [[stage_in]],
                          constant RasterUniforms& U [[buffer(1)]]) {
    if (U.misc.x == 1u) {
        return float4(1.0f, 0.55f, 0.1f, 1.0f);   // selection overlay tint
    }
    const float3 L = normalize(U.cam_pos.xyz - in.world);
    const float ndl = fmax(0.25f, fabs(dot(normalize(in.normal), L)));
    // Same display gamma as the traced path so brightness feels consistent.
    const float3 c = pow(in.albedo * ndl, float3(1.0f / 2.2f));
    return float4(saturate(c), 1.0f);
}
