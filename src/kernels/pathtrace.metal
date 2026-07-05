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
inline EvalMat eval_mesh_material(device const GPUMaterialArgs* mats,
                                  uint mat_id,
                                  constant MeshUniforms& MU, float u,
                                  float v) {
    device const GPUMaterialArgs& M = mats[mat_id];
    const float3 base = sample_bilinear(M.base, M.base_w, M.base_h, u, v);
    const float3 mr =
        M.mr_w != 0u ? sample_bilinear(M.mr, M.mr_w, M.mr_h, u, v)
                     : float3(0.0f, 1.0f, 0.0f);
    const float3 emis =
        M.emis_w != 0u ? sample_bilinear(M.emis, M.emis_w, M.emis_h, u, v)
                       : float3(0.0f);
    EvalMat m;
    m.base_color = base;
    m.emission = emis * MU.emissive_scale;
    m.metallic = mr.z;    // glTF: B = metallic
    m.roughness = mr.y;   // glTF: G = roughness (perceptual)
    m.ior = M.ior;
    m.transmission = M.transmission;
    return m;
}

// Full estimator weight f·cosθ/pdf for the sampled direction — mirror of
// material.h's scatter, line for line.
// Session H (NEE/MIS): evaluate the BSDF value and the MIXED pdf that
// scatter() below would assign to an ARBITRARY light direction l. Textual
// factoring of scatter()'s evaluation half — MUST stay in sync with it
// and with integrator.h. Delta glass evaluates to zero.
inline float3 eval_bsdf(thread const EvalMat& mat, float3 normal, float3 v,
                        float3 l, thread float& pdf) {
    pdf = 0.0f;
    if (mat.transmission > 0.5f) return float3(0.0f);

    const float3 n = normal;
    const float nv = dot(n, v);
    if (nv <= 1e-4f) return float3(0.0f);

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
    const float3 lt = float3(dot(l, t), dot(l, b), dot(l, n));
    const float nl = lt.z;
    if (nl <= 1e-6f) return float3(0.0f);

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
    pdf = p_spec * pdf_spec + (1.0f - p_spec) * pdf_diff;
    return f_spec + f_diff;
}

inline bool scatter(thread const EvalMat& mat, float3 in_dir, float3 normal,
                    bool front_face, thread PRNG& rng,
                    thread float3& attenuation, thread float3& out_dir,
                    thread float& out_pdf, thread bool& out_delta,
                    float glass_ior) {
    out_pdf = 0.0f;
    out_delta = false;
    // ---- glass: delta dielectric ----
    if (mat.transmission > 0.5f) {
        out_delta = true;
        attenuation = float3(1.0f);
        // glass_ior < 0 -> use mat.ior (RGB); else the n(lambda) from trace.
        const float ior_use = glass_ior > 0.0f ? glass_ior : mat.ior;
        const float eta = front_face ? 1.0f / ior_use : ior_use;
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
    out_pdf = pdf;
    return true;
}

// Session H shadow rays: any-hit occlusion, mirrors mesh.h/scene.h.
inline bool bvh_occluded(device const BVHNode* nodes,
                         device const GPUTriangle* tris, float3 ro, float3 rd,
                         float t_min, float t_max) {
    const float3 inv = float3(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);
    float tn;
    if (!hit_aabb(nodes[0].mn, nodes[0].mx, ro, inv, t_min, t_max, tn))
        return false;
    uint stack[32];
    int sp = 0;
    stack[sp++] = 0u;
    while (sp > 0) {
        const BVHNode node = nodes[stack[--sp]];
        if (node.tri_count > 0u) {
            for (uint i = 0; i < node.tri_count; ++i) {
                TriHit th;
                if (hit_tri(tris[node.left_or_first + i], ro, rd, t_min,
                            t_max, th))
                    return true;
            }
            continue;
        }
        const uint l = node.left_or_first, r = l + 1u;
        if (hit_aabb(nodes[l].mn, nodes[l].mx, ro, inv, t_min, t_max, tn))
            stack[sp++] = l;
        if (hit_aabb(nodes[r].mn, nodes[r].mx, ro, inv, t_min, t_max, tn))
            stack[sp++] = r;
    }
    return false;
}

inline bool occluded_scene(float3 ro, float3 rd, float t_max,
                           constant GPUSphere* spheres, uint n,
                           device const BVHNode* nodes,
                           device const GPUTriangle* tris,
                           constant MeshUniforms& MU) {
    for (uint i = 0; i < n; ++i) {
        HitRec tmp;
        if (hit_sphere(spheres[i], ro, rd, 1e-3f, t_max, tmp)) return true;
    }
    if (MU.has_mesh != 0u) {
        return bvh_occluded(nodes, tris, ro, rd, 1e-3f, t_max);
    }
    return false;
}

// ---- Session J: area-light NEE (mirror of integrator.h) ---------------

inline float light_dir_pdf(device const GPULight& L, float3 x, float3 dir,
                           float t_hit) {
    if (L.kind == 0u) {
        const float3 cx = c3(L.p0) - x;
        const float d2 = dot(cx, cx);
        const float d = sqrt(d2);
        if (d <= L.radius * 1.0001f) return 0.0f;
        const float sin2max = (L.radius * L.radius) / d2;
        const float cosmax = sqrt(fmax(0.0f, 1.0f - sin2max));
        const float one_minus = 1.0f - cosmax;
        if (one_minus < 1e-8f) return 0.0f;
        return 1.0f / (6.28318530717958648f * one_minus);
    }
    const float3 e1 = c3(L.e1);
    const float3 e2 = c3(L.e2);
    const float3 cr = cross_pt(e1, e2);
    const float two_area = length(cr);
    if (two_area < 1e-12f) return 0.0f;
    const float cos_l = fabs(dot(cr, dir)) / two_area;
    if (cos_l < 1e-6f) return 0.0f;
    return (t_hit * t_hit) / (cos_l * 0.5f * two_area);
}

inline float3 sample_tri_light(device const GPULight& L, float3 x, float u1,
                               float u2, thread float& pdf_sa,
                               thread float& t_light, thread float& bu,
                               thread float& bv) {
    pdf_sa = 0.0f;
    t_light = 0.0f;
    const float su = sqrt(u1);
    bu = 1.0f - su;
    bv = u2 * su;
    const float3 p0 = c3(L.p0);
    const float3 e1 = c3(L.e1);
    const float3 e2 = c3(L.e2);
    const float3 y = p0 + bu * e1 + bv * e2;
    const float3 d = y - x;
    const float r2 = dot(d, d);
    if (r2 < 1e-12f) return float3(0.0f, 0.0f, 1.0f);
    const float r = sqrt(r2);
    const float3 dir = d / r;
    const float3 cr = cross_pt(e1, e2);
    const float two_area = length(cr);
    if (two_area < 1e-12f) return float3(0.0f, 0.0f, 1.0f);
    const float cos_l = fabs(dot(cr, dir)) / two_area;
    if (cos_l < 1e-6f) return float3(0.0f, 0.0f, 1.0f);
    pdf_sa = r2 / (cos_l * 0.5f * two_area);
    t_light = r;
    return dir;
}

inline int light_pick(device const GPULight* lights, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (lights[mid].sel_cdf > u) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

inline float3 sample_sphere_light(device const GPULight& L, float3 x,
                                  float u1, float u2, thread float& pdf_sa,
                                  thread float& t_light) {
    pdf_sa = 0.0f;
    t_light = 0.0f;
    const float3 cx = c3(L.p0) - x;
    const float d2 = dot(cx, cx);
    const float d = sqrt(d2);
    if (d <= L.radius * 1.0001f) return float3(0.0f, 0.0f, 1.0f);
    const float sin2max = (L.radius * L.radius) / d2;
    const float cosmax = sqrt(fmax(0.0f, 1.0f - sin2max));
    const float one_minus = 1.0f - cosmax;
    if (one_minus < 1e-8f) return float3(0.0f, 0.0f, 1.0f);
    const float cost = 1.0f - u1 * one_minus;
    const float sint = sqrt(fmax(0.0f, 1.0f - cost * cost));
    const float phi = 6.28318530717958648f * u2;
    const float3 w = cx / d;
    float3 t, b;
    build_onb(w, t, b);
    const float3 dir = normalize(t * (cos(phi) * sint) +
                                 b * (sin(phi) * sint) + w * cost);
    pdf_sa = 1.0f / (6.28318530717958648f * one_minus);
    const float dc = dot(cx, dir);
    const float disc = L.radius * L.radius - (d2 - dc * dc);
    t_light = dc - sqrt(fmax(0.0f, disc));
    return dir;
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

// ---- Session H: env importance sampling (mirror of integrator.h) ----

// First index whose cdf value exceeds u. Identical loop on both backends.
inline int cdf_find(device const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (cdf[mid] > u) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

inline float3 env_sample(device const float* row_cdf,
                         device const float* cond_cdf, int W, int H,
                         float yaw_norm, float u1, float u2,
                         thread float& pdf_sa) {
    const int y = cdf_find(row_cdf, H, u1);
    device const float* crow = cond_cdf + ulong(y) * W;
    const int x = cdf_find(crow, W, u2);
    const float row_lo = y > 0 ? row_cdf[y - 1] : 0.0f;
    const float row_w = row_cdf[y] - row_lo;
    const float col_lo = x > 0 ? crow[x - 1] : 0.0f;
    const float col_w = crow[x] - col_lo;
    // Continuous inversion inside the chosen texel (keeps stratification).
    const float fy = row_w > 0.0f ? (u1 - row_lo) / row_w : 0.5f;
    const float fx = col_w > 0.0f ? (u2 - col_lo) / col_w : 0.5f;
    const float u = (float(x) + fx) / float(W);
    const float v = (float(y) + fy) / float(H);
    // UV -> direction: exact inverse of the miss mapping, yaw included.
    const float phi = (u - 0.5f - yaw_norm) * 6.28318530717958648f;
    const float theta = v * 3.14159265358979f;
    const float st = sin(theta);
    const float3 d = float3(st * cos(phi), cos(theta), st * sin(phi));
    const float p_uv = row_w * col_w * float(W) * float(H);
    pdf_sa = st > 1e-6f
                 ? p_uv / (2.0f * 3.14159265358979f * 3.14159265358979f * st)
                 : 0.0f;
    return d;
}

inline float env_pdf(device const float* row_cdf,
                     device const float* cond_cdf, int W, int H,
                     float yaw_norm, float3 dir) {
    const float3 d = normalize(dir);
    float u = atan2(d.z, d.x) * 0.15915494309189533577f + 0.5f + yaw_norm;
    u = u - floor(u);
    const float v = acos(fmin(fmax(d.y, -1.0f), 1.0f)) *
                    0.31830988618379067154f;
    int x = int(u * float(W));
    int y = int(v * float(H));
    if (x >= W) x = W - 1;
    if (y >= H) y = H - 1;
    device const float* crow = cond_cdf + ulong(y) * W;
    const float row_lo = y > 0 ? row_cdf[y - 1] : 0.0f;
    const float row_w = row_cdf[y] - row_lo;
    const float col_lo = x > 0 ? crow[x - 1] : 0.0f;
    const float col_w = crow[x] - col_lo;
    const float st = sqrt(fmax(0.0f, 1.0f - d.y * d.y));
    if (st <= 1e-6f) return 0.0f;
    return row_w * col_w * float(W) * float(H) /
           (2.0f * 3.14159265358979f * 3.14159265358979f * st);
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

// First-hit evaluation: closest hit across spheres + mesh, surface
// reconstruction (normal mapping included), material eval, and the light
// id of the hit emitter. Factored (Session K / ReSTIR Stage 0.5) so the
// monolithic kernel and the partitioned g_primary phase share ONE
// implementation — divergence between pipelines is impossible here.
struct HitInfo {
    float t;
    float3 hp, hn;
    EvalMat mat;
    bool front;
    int light_id;
};

inline bool scene_hit_eval(float3 ro, float3 rd, constant GPUSphere* spheres,
                           uint n, device const BVHNode* nodes,
                           device const GPUTriangle* tris,
                           device const GPUMaterialArgs* materials,
                           device const uint* tri_mat,
                           device const uint* tri_light,
                           constant MeshUniforms& MU, thread HitInfo& out) {
    HitRec rec;
    const bool hit_s = hit_scene(spheres, n, ro, rd, 1e-3f, INFINITY, rec);
    float closest = hit_s ? rec.t : INFINITY;
    TriHit th;
    bool hit_m = false;
    if (MU.has_mesh != 0u) {
        hit_m = bvh_hit(nodes, tris, ro, rd, 1e-3f, closest, th);
    }
    if (!hit_s && !hit_m) return false;

    out.t = closest;
    if (hit_m) {
        const GPUTriangle T = tris[th.tri];
        out.hp = ro + closest * rd;
        // Sidedness from the GEOMETRIC normal; the interpolated shading
        // normal only bends the scatter lobe. Mirrors mesh.h.
        const float3 ng = cross_pt(c3(T.e1), c3(T.e2));
        out.front = dot(rd, ng) < 0.0f;
        const float w = 1.0f - th.u - th.v;
        float3 ns = normalize(w * c3(T.n0) + th.u * c3(T.n1) +
                              th.v * c3(T.n2));
        const float u = w * T.u0 + th.u * T.u1 + th.v * T.u2;
        const float v = w * T.v0 + th.u * T.v1 + th.v * T.v2;
        const uint mid = tri_mat[th.tri];
        // Tangent-space normal mapping — mirror of mesh.h. glTF:
        // LINEAR texels, +Y up, [0,1] -> [-1,1]; bitangent takes the
        // .w handedness sign.
        device const GPUMaterialArgs& NM = materials[mid];
        if (NM.norm_w != 0u) {
            const float3 tn =
                sample_bilinear(NM.norm, NM.norm_w, NM.norm_h, u, v);
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
        out.hn = out.front ? ns : -ns;
        out.mat = eval_mesh_material(materials, mid, MU, u, v);
        // Glass refracts through the GEOMETRIC normal — mirror of mesh.h.
        if (out.mat.transmission > 0.5f) {
            out.hn = out.front ? normalize(ng) : -normalize(ng);
        }
        out.light_id = int(tri_light[th.tri]) - 1;
    } else {
        out.hp = rec.p;
        out.hn = rec.normal;
        out.front = rec.front_face;
        out.mat = eval_sphere(spheres[rec.sphere_idx]);
        out.light_id = int(spheres[rec.sphere_idx].pad[0]) - 1;
    }
    return true;
}

// Vertex direct lighting — mirror of integrator.h's sample_direct: ONE
// estimator shared by the monolithic kernel and the partitioned direct
// phase. Appends into `radiance` in the original order; same rng draws
// under the same conditions.
// ---- v1.2 spectral core — mirror of spectral.h (--parity is the drift
// detector). See that header for the derivation; constants are identical.
constant float SPEC_LMIN = 400.0f;
constant float SPEC_BAND = 300.0f;

inline float spec_sample_wavelength(float u) {
    return SPEC_LMIN + u * SPEC_BAND;
}
inline float wyman_g(float x, float mu, float s1, float s2) {
    const float t = (x - mu) * (x < mu ? 1.0f / s1 : 1.0f / s2);
    return exp(-0.5f * t * t);
}
inline float3 cie_xyz_fit(float l) {
    const float x = 1.056f * wyman_g(l, 599.8f, 37.9f, 31.0f) +
                    0.362f * wyman_g(l, 442.0f, 16.0f, 26.7f) -
                    0.065f * wyman_g(l, 501.1f, 20.4f, 26.2f);
    const float y = 0.821f * wyman_g(l, 568.8f, 46.9f, 40.5f) +
                    0.286f * wyman_g(l, 530.9f, 16.3f, 31.1f);
    const float z = 1.217f * wyman_g(l, 437.0f, 11.8f, 36.0f) +
                    0.681f * wyman_g(l, 459.0f, 26.0f, 13.8f);
    return float3(x, y, z);
}
inline float3 spec_response(float l) {
    const float3 c = cie_xyz_fit(l);
    const float R = 3.2406f * c.x - 1.5372f * c.y - 0.4986f * c.z;
    const float G = -0.9689f * c.x + 1.8758f * c.y + 0.0415f * c.z;
    const float B = 0.0557f * c.x - 0.2040f * c.y + 1.0570f * c.z;
    return float3(R * 2.33749215f, G * 2.95510157f, B * 3.10746416f);
}
inline float spec_reflectance(float3 rgb0, float l) {
    // Round-trip correction (C^-1) — mirror of spectral.h.
    const float3 rgb =
        float3(0.838717f * rgb0.x + 0.148845f * rgb0.y + 0.012780f * rgb0.z,
               -0.117719f * rgb0.x + 1.103062f * rgb0.y + 0.014987f * rgb0.z,
               -0.023000f * rgb0.x + 0.011697f * rgb0.y + 1.011686f * rgb0.z);
    const float3 c = cie_xyz_fit(l);
    const float R = fmax(0.0f, 3.2406f * c.x - 1.5372f * c.y - 0.4986f * c.z);
    const float G = fmax(0.0f, -0.9689f * c.x + 1.8758f * c.y + 0.0415f * c.z);
    const float B = fmax(0.0f, 0.0557f * c.x - 0.2040f * c.y + 1.0570f * c.z);
    const float sum = R + G + B;
    if (sum <= 1e-8f) return 0.0f;
    return (rgb.x * R + rgb.y * G + rgb.z * B) / sum;
}

// Stage 2 Cauchy IOR — mirror of spectral.h::ior_lambda.
inline float spec_ior_lambda(float base, float B, float lam_nm) {
    const float um = lam_nm * 0.001f;
    return base + B * (1.0f / (um * um) - 3.30578512f);
}

struct SpecCtx {
    bool on;
    float lam;
    float3 resp;
};
inline SpecCtx spec_begin(bool on, thread PRNG& rng) {
    SpecCtx s;
    s.on = on;
    s.lam = 0.0f;
    s.resp = float3(1.0f);
    if (on) {
        s.lam = spec_sample_wavelength(pcg_next_float(rng));
        s.resp = spec_response(s.lam);
    }
    return s;
}
inline float3 spec_dep(thread const SpecCtx& s, float3 src) {
    return s.on ? s.resp * spec_reflectance(src, s.lam) : src;
}
inline float3 spec_dep2(thread const SpecCtx& s, float3 a, float3 b) {
    if (!s.on) return a * b;
    return s.resp * (spec_reflectance(a, s.lam) * spec_reflectance(b, s.lam));
}
inline float3 spec_atten(thread const SpecCtx& s, float3 a) {
    if (!s.on) return a;
    return float3(spec_reflectance(a, s.lam));
}

// v1.3 homogeneous-medium transmittance — mirror of integrator.h::fog_tr.
inline float fog_tr(float sigma, float dist) {
    return sigma > 0.0f ? exp(-sigma * dist) : 1.0f;
}

inline void sample_direct(float3 hp, float3 hn, thread const EvalMat& mat,
                          float3 rd, thread PRNG& rng, float3 throughput,
                          thread float3& radiance,
                          constant GPUSphere* spheres, uint n,
                          device const BVHNode* nodes,
                          device const GPUTriangle* tris,
                          device const GPUMaterialArgs* materials,
                          device const float* env_texels,
                          device const float* env_row_cdf,
                          device const float* env_cond_cdf,
                          device const GPULight* lights,
                          constant MeshUniforms& MU,
                          constant PassUniforms& U,
                          thread bool& can_nee_out,
                          thread bool& can_nee_light_out,
                          thread const SpecCtx& sc, float fog_sigma) {
        // ---- Session H: next-event estimation toward the environment.
        // Deliberately sample the env distribution (the sun), evaluate the
        // BSDF for that direction, and add the contribution if the shadow
        // ray escapes. Delta glass is skipped — BSDF sampling owns it.
        // Delta glass skipped; near-specular handled by MIS weights —
        // mirror of integrator.h.
        const bool can_nee = U.env_nee != 0u && U.env_w != 0u &&
                             mat.transmission <= 0.5f;
        if (can_nee) {
            const float u1 = pcg_next_float(rng);
            const float u2 = pcg_next_float(rng);
            float pdf_env = 0.0f;
            const float3 ldir =
                env_sample(env_row_cdf, env_cond_cdf, int(U.env_w),
                           int(U.env_h), U.env_yaw_norm, u1, u2, pdf_env);
            if (pdf_env > 1e-12f) {
                float pdf_b = 0.0f;
                const float3 vdir = -normalize(rd);
                const float3 f = eval_bsdf(mat, hn, vdir, ldir, pdf_b);
                const float nl = dot(hn, ldir);
                if (nl > 1e-6f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)) {
                    if (!occluded_scene(hp, ldir, INFINITY, spheres, n,
                                        nodes, tris, MU)) {
                        // MIS weight vs the BSDF sampler (power heuristic).
                        const float w =
                            (pdf_env * pdf_env) /
                            (pdf_env * pdf_env + pdf_b * pdf_b + 1e-20f);
                        // Env at infinity: global fog extinguishes it
                        // (fog_tr with +inf -> 0); vacuum -> 1.
                        const float3 c =
                            throughput *
                            spec_dep2(sc, f,
                                      miss_radiance(env_texels, U.env_w,
                                                    U.env_h, U.env_intensity,
                                                    U.env_yaw_norm, ldir)) *
                            (nl * w / pdf_env) * fog_tr(fog_sigma, INFINITY);
                        radiance += clamp_contribution(c, U.clamp_indirect);
                    }
                }
            }
        }
        can_nee_out = can_nee;

        // ---- Session J: one area-light sample per vertex — mirror of
        // integrator.h. Same gating as env NEE; power-proportional pick.
        // Delta glass skipped; near-specular handled by the MIS weights
        // — mirror of integrator.h.
        const bool can_nee_light = U.env_nee != 0u && U.light_count > 0u &&
                                   mat.transmission <= 0.5f;
        if (can_nee_light) {
            const float us = pcg_next_float(rng);
            const float u1 = pcg_next_float(rng);
            const float u2 = pcg_next_float(rng);
            const int li = light_pick(lights, int(U.light_count), us);
            device const GPULight& L = lights[li];
            float pdf_sa = 0.0f, t_light = 0.0f;
            float bu = 0.0f, bv = 0.0f;
            const float3 ldir =
                L.kind == 0u
                    ? sample_sphere_light(L, hp, u1, u2, pdf_sa, t_light)
                    : sample_tri_light(L, hp, u1, u2, pdf_sa, t_light, bu,
                                       bv);
            if (pdf_sa > 1e-12f && L.sel_pdf > 0.0f) {
                float pdf_b = 0.0f;
                const float3 vdir = -normalize(rd);
                const float3 f = eval_bsdf(mat, hn, vdir, ldir, pdf_b);
                const float nl = dot(hn, ldir);
                if (nl > 1e-6f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)) {
                    if (!occluded_scene(hp, ldir, t_light * (1.0f - 1e-3f),
                                        spheres, n, nodes, tris, MU)) {
                        float3 Le = c3(L.emission);
                        if (L.kind == 1u) {
                            // Textured Le at the sampled point — same
                            // bilinear + emissive_scale as shading.
                            const float b0 = 1.0f - bu - bv;
                            const float tu =
                                b0 * L.u0 + bu * L.u1 + bv * L.u2;
                            const float tv =
                                b0 * L.v0 + bu * L.v1 + bv * L.v2;
                            device const GPUMaterialArgs& LM =
                                materials[L.mat_id];
                            Le = sample_bilinear(LM.emis, LM.emis_w,
                                                 LM.emis_h, tu, tv) *
                                 MU.emissive_scale;
                        }
                        // MIS weight vs the BSDF sampler.
                        const float pl = pdf_sa * L.sel_pdf;
                        const float w =
                            (pl * pl) / (pl * pl + pdf_b * pdf_b + 1e-20f);
                        // Area light at finite t_light: dim by fog along
                        // the shadow ray.
                        const float3 c = throughput * spec_dep2(sc, f, Le) *
                                         (nl * w / pl) *
                                         fog_tr(fog_sigma, t_light);
                        radiance += clamp_contribution(c, U.clamp_indirect);
                    }
                }
            }
        }
        can_nee_light_out = can_nee_light;
}

inline float3 trace(float3 ro, float3 rd, constant GPUSphere* spheres,
                    uint n, device const BVHNode* nodes,
                    device const GPUTriangle* tris,
                    device const GPUMaterialArgs* materials,
                    device const uint* tri_mat,
                    device const float* env_texels,
                    device const float* env_row_cdf,
                    device const float* env_cond_cdf,
                    device const GPULight* lights,
                    device const uint* tri_light,
                    constant MeshUniforms& MU, constant PassUniforms& U,
                    thread PRNG& rng, uint max_depth,
                    // Session K: partitioned pipeline injects the primary
                    // hit (g_primary already found it) and skips vertex-0
                    // direct lighting (the direct phase computed it with
                    // the same rng draws).
                    bool has_pre, thread const HitInfo& pre,
                    bool skip_v0_direct, bool spectral, float fog_sigma) {
    float3 radiance = float3(0.0f);
    float3 throughput = float3(1.0f);
    // v1.2 hero-wavelength: one lambda per path (one rng draw, here, so
    // both backends align) — identity context when spectral is off.
    const SpecCtx sc = spec_begin(spectral, rng);
    bool prev_nee = false;   // env NEE ran at the previous path vertex
    bool prev_nee_light = false;   // area-light NEE ran there too
    float prev_pdf = 0.0f;   // BSDF pdf of the ray we're now following

    for (uint depth = 0; depth < max_depth; ++depth) {
        // Closest hit: spheres first, then the mesh BVH seeded with the
        // sphere winner's t — mirrors the CPU's Scene ordering (mesh added
        // last), so tie-breaking is identical on both backends.
        HitInfo hi;
        bool hit_any;
        if (depth == 0u && has_pre) {
            hi = pre;
            hit_any = pre.t >= 0.0f;
        } else {
            hit_any = scene_hit_eval(ro, rd, spheres, n, nodes, tris,
                                     materials, tri_mat, tri_light, MU, hi);
        }
        // v1.3 Stage 1: Beer-Lambert transmittance over this segment —
        // mirror of integrator.h (miss -> +inf -> 0; guard avoids 0*inf).
        if (fog_sigma > 0.0f) {
            const float seg_len = hit_any ? hi.t : INFINITY;
            throughput *= exp(-fog_sigma * seg_len);
        }
        if (!hit_any) {
            // MIS (power heuristic): vertices that ran env NEE weight
            // their continuation's env hit by the BSDF-sampling share;
            // camera rays and delta/glass continuations (no NEE there)
            // see the env in full.
            float w = 1.0f;
            if (prev_nee) {
                const float pe =
                    env_pdf(env_row_cdf, env_cond_cdf, int(U.env_w),
                            int(U.env_h), U.env_yaw_norm, rd);
                w = (prev_pdf * prev_pdf) /
                    (prev_pdf * prev_pdf + pe * pe + 1e-20f);
            }
            const float3 c =
                throughput *
                spec_dep(sc, miss_radiance(env_texels, U.env_w, U.env_h,
                                           U.env_intensity, U.env_yaw_norm,
                                           rd)) *
                w;
            radiance +=
                (depth == 0u ? c : clamp_contribution(c, U.clamp_indirect));
            return radiance;
        }

        const EvalMat mat = hi.mat;
        const float3 hp = hi.hp;
        const float3 hn = hi.hn;
        const bool front = hi.front;
        const float closest = hi.t;

        {
            // MIS (power heuristic) — mirror of integrator.h: BSDF hits on
            // LISTED emitters weight against the area-NEE pdf for this
            // direction (x selection pdf).
            const int light_id = hi.light_id;
            float w = 1.0f;
            if (prev_nee_light && depth > 0u && light_id >= 0 &&
                prev_pdf > 0.0f) {
                device const GPULight& L = lights[light_id];
                const float pl =
                    light_dir_pdf(L, ro, normalize(rd), closest) * L.sel_pdf;
                w = (prev_pdf * prev_pdf) /
                    (prev_pdf * prev_pdf + pl * pl + 1e-20f);
            }
            const float3 c = throughput * spec_dep(sc, mat.emission) * w;
            radiance += depth == 0u
                            ? c
                            : clamp_contribution(c, U.clamp_indirect);
        }

        if (depth == 0u && skip_v0_direct) {
            // The direct phase already ran sample_direct with these draws;
            // reproduce only its condition flags for the MIS bookkeeping.
            prev_nee = U.env_nee != 0u && U.env_w != 0u &&
                       mat.transmission <= 0.5f;
            prev_nee_light = U.env_nee != 0u && U.light_count > 0u &&
                             mat.transmission <= 0.5f;
        } else {
            bool v_nee = false, v_nee_light = false;
            sample_direct(hp, hn, mat, rd, rng, throughput, radiance,
                          spheres, n, nodes, tris, materials, env_texels,
                          env_row_cdf, env_cond_cdf, lights, MU, U, v_nee,
                          v_nee_light, sc, fog_sigma);
            prev_nee = v_nee;
            prev_nee_light = v_nee_light;
        }

        float3 attenuation, new_dir;
        float scatter_pdf = 0.0f;
        bool scatter_delta = false;
        // Stage 2 dispersion: refract this path's wavelength at its own IOR.
        const float gior =
            sc.on ? spec_ior_lambda(mat.ior, U.dispersion_b, sc.lam) : -1.0f;
        if (!scatter(mat, rd, hn, front, rng, attenuation, new_dir,
                     scatter_pdf, scatter_delta, gior)) {
            return radiance;
        }
        throughput *= spec_atten(sc, attenuation);
        ro = hp;
        rd = new_dir;
        prev_pdf = scatter_delta ? 0.0f : scatter_pdf;
        if (scatter_delta) prev_nee = false;   // delta chains keep full env

        if (depth >= 3) {
            const float p = fmin(
                fmax(throughput.x, fmax(throughput.y, throughput.z)), 0.95f);
            // p == 0 (all-black throughput) must terminate BEFORE the rng
            // test: next_float() can return exactly 0.0, and 0/0 would put
            // a NaN in the accumulator.
            if (p <= 0.0f) break;
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
                       device const GPUMaterialArgs* materials [[buffer(5)]],
                       device const uint* tri_mat      [[buffer(6)]],
                       constant MeshUniforms& MU       [[buffer(8)]],
                       device const float* env_texels  [[buffer(10)]],
                       device const float* env_row_cdf [[buffer(11)]],
                       device const float* env_cond_cdf [[buffer(12)]],
                       device const GPULight* lights   [[buffer(13)]],
                       device const uint* tri_light    [[buffer(14)]],
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

        HitInfo no_pre;
        total += trace(ro, rd, spheres, U.sphere_count, nodes, tris,
                       materials, tri_mat, env_texels, env_row_cdf,
                       env_cond_cdf, lights, tri_light, MU, U, rng,
                       U.max_depth, false, no_pre, false,
                       U.spectral != 0u,
                       U.fog != 0u ? U.fog_sigma : 0.0f);
    }
    accum[px] += float4(total, 0.0f);
}

// ============ Session K: partitioned pipeline (ReSTIR Stage 0.5) ========
// Three phases reproducing the monolithic estimator exactly: g_primary
// (jitter + first hit -> G-buffer + rng state), direct_v0 (vertex-0
// direct lighting via the SAME sample_direct and the SAME draws), and
// indirect_v0 (the path continuation with the first hit injected and
// vertex-0 direct skipped). Phase order is enforced by command-queue
// ordering; the rng stream travels through the G-buffer.

// ---- Session K: ReSTIR direct lighting — mirror of integrator.h -------

inline float target_env_t(thread const EvalMat& mat, float3 hn, float3 hp,
                          float3 vdir, float3 dir,
                          device const float* env_texels,
                          device const float* env_row_cdf,
                          device const float* env_cond_cdf,
                          constant PassUniforms& U,
                          thread float3& contrib_out,
                          thread float& wmis_out) {
    contrib_out = float3(0.0f);
    wmis_out = 0.0f;
    float pb = 0.0f;
    const float3 f = eval_bsdf(mat, hn, vdir, dir, pb);
    const float nl = dot(hn, dir);
    if (nl <= 1e-6f || (f.x <= 0.0f && f.y <= 0.0f && f.z <= 0.0f))
        return 0.0f;
    contrib_out = f * nl *
                  miss_radiance(env_texels, U.env_w, U.env_h,
                                U.env_intensity, U.env_yaw_norm, dir);
    const float pl = env_pdf(env_row_cdf, env_cond_cdf, int(U.env_w),
                             int(U.env_h), U.env_yaw_norm, dir);
    wmis_out = (pl * pl) / (pl * pl + pb * pb + 1e-20f);
    return luminance(contrib_out * wmis_out);
}

inline float target_area_t(thread const EvalMat& mat, float3 hn, float3 hp,
                           float3 vdir, device const GPULight& L, float ax,
                           float ay, float az,
                           device const GPUMaterialArgs* materials,
                           constant MeshUniforms& MU,
                           thread float3& contrib_out,
                           thread float& wmis_out, thread float& G_out,
                           thread float3& dir_out, thread float& t_out) {
    contrib_out = float3(0.0f);
    wmis_out = 0.0f;
    G_out = 0.0f;
    t_out = 0.0f;
    dir_out = float3(0.0f, 0.0f, 1.0f);
    float3 y;
    float bu = 0.0f, bv = 0.0f;
    if (L.kind == 0u) {
        y = float3(ax, ay, az);
    } else {
        bu = ax;
        bv = ay;
        y = c3(L.p0) + bu * c3(L.e1) + bv * c3(L.e2);
    }
    const float3 d = y - hp;
    const float r2 = dot(d, d);
    if (r2 < 1e-12f) return 0.0f;
    const float r = sqrt(r2);
    dir_out = d / r;
    t_out = r;
    float pb = 0.0f;
    const float3 f = eval_bsdf(mat, hn, vdir, dir_out, pb);
    const float nl = dot(hn, dir_out);
    if (nl <= 1e-6f || (f.x <= 0.0f && f.y <= 0.0f && f.z <= 0.0f))
        return 0.0f;
    float3 Le = c3(L.emission);
    float3 n_y;
    if (L.kind == 0u) {
        n_y = normalize(y - c3(L.p0));
    } else {
        const float b0 = 1.0f - bu - bv;
        const float tu = b0 * L.u0 + bu * L.u1 + bv * L.u2;
        const float tv = b0 * L.v0 + bu * L.v1 + bv * L.v2;
        device const GPUMaterialArgs& LM = materials[L.mat_id];
        Le = sample_bilinear(LM.emis, LM.emis_w, LM.emis_h, tu, tv) *
             MU.emissive_scale;
        n_y = normalize(cross_pt(c3(L.e1), c3(L.e2)));
    }
    G_out = fabs(dot(n_y, dir_out)) / r2;
    const float pl = light_dir_pdf(L, hp, dir_out, r) * L.sel_pdf;
    contrib_out = f * nl * Le;
    wmis_out = (pl * pl) / (pl * pl + pb * pb + 1e-20f);
    return luminance(contrib_out * wmis_out) * G_out;
}

inline HitInfo hitinfo_from_gbuf(thread const GBufferPx& g) {
    HitInfo hi;
    hi.t = g.t;
    hi.hp = c3(g.pos);
    hi.hn = c3(g.normal);
    hi.front = (g.flags & 1u) != 0u;
    hi.mat.base_color = c3(g.base_color);
    hi.mat.emission = c3(g.emission);
    hi.mat.metallic = g.metallic;
    hi.mat.roughness = g.roughness;
    hi.mat.ior = g.ior;
    hi.mat.transmission = g.transmission;
    hi.light_id = int(g.light_id_p1) - 1;
    return hi;
}

// Phase D1 — mirror of integrator.h's restir_build: candidates + the
// Talbot balance-heuristic temporal combine (history = a neighbor in
// time, target evaluated on LAST frame's surface via gprev).
inline void restir_build(thread const HitInfo& hi, float3 rd,
                         thread PRNG& rng, int M, bool temporal, int mcap,
                         device const ReSTIRPixel& hist,
                         device const GBufferPx& gprev,
                         device ReSTIRPixel& cur,
                         constant GPUSphere* spheres, uint n,
                         device const GPUMaterialArgs* materials,
                         device const float* env_texels,
                         device const float* env_row_cdf,
                         device const float* env_cond_cdf,
                         device const GPULight* lights,
                         constant MeshUniforms& MU,
                         constant PassUniforms& U) {
    const bool can_env = U.env_nee != 0u && U.env_w != 0u &&
                         hi.mat.transmission <= 0.5f;
    const bool can_area = U.env_nee != 0u && U.light_count > 0u &&
                          hi.mat.transmission <= 0.5f;
    const float3 vdir = -normalize(rd);
    const float McapF = float(mcap) * float(M);
    const float3 pn = c3(gprev.normal);
    const bool hist_ok =
        temporal && gprev.t > 0.0f && hi.t > 0.0f &&
        fabs(gprev.t - hi.t) < 0.1f * hi.t &&
        (pn.x * hi.hn.x + pn.y * hi.hn.y + pn.z * hi.hn.z) > 0.9f;
    HitInfo ph;
    float3 pvdir = float3(0.0f, 0.0f, 1.0f);
    if (hist_ok) {
        const GBufferPx gp = gprev;   // device -> thread copy
        ph = hitinfo_from_gbuf(gp);
        pvdir = -normalize(c3(gp.rd));
    }

    // --- env slot ---
    if (can_env) {
        float wsum_c = 0.0f, win_that = 0.0f;
        float3 win_dir = float3(0.0f, 0.0f, 1.0f);
        for (int i = 0; i < M; ++i) {
            const float u1 = pcg_next_float(rng);
            const float u2 = pcg_next_float(rng);
            const float ur = pcg_next_float(rng);
            float pl = 0.0f;
            const float3 ldir =
                env_sample(env_row_cdf, env_cond_cdf, int(U.env_w),
                           int(U.env_h), U.env_yaw_norm, u1, u2, pl);
            float w_i = 0.0f, that = 0.0f;
            if (pl > 1e-12f) {
                float3 contrib;
                float w_mis;
                that = target_env_t(hi.mat, hi.hn, hi.hp, vdir, ldir,
                                    env_texels, env_row_cdf, env_cond_cdf,
                                    U, contrib, w_mis);
                if (that > 0.0f) w_i = that / pl;
            }
            wsum_c += w_i;
            if (w_i > 0.0f && ur < w_i / wsum_c) {
                win_dir = ldir;
                win_that = that;
            }
        }
        const float Wc = (wsum_c > 0.0f && win_that > 0.0f)
                             ? wsum_c / (float(M) * win_that)
                             : 0.0f;
        const float ur_h = pcg_next_float(rng);
        float out_that = win_that;
        float3 out_dir = win_dir;
        float out_wsum = 0.0f, out_M = float(M);
        const bool have_hist = hist_ok && hist.env_slot.M > 0.0f;
        if (!have_hist) {
            out_wsum = win_that * Wc;
        } else {
            const float Mh = fmin(hist.env_slot.M, McapF);
            out_M += Mh;
            float3 cc;
            float wm;
            float w_c = 0.0f;
            if (Wc > 0.0f && win_that > 0.0f) {
                const float p_prev_c =
                    target_env_t(ph.mat, ph.hn, ph.hp, pvdir, win_dir,
                                 env_texels, env_row_cdf, env_cond_cdf, U,
                                 cc, wm);
                const float denom = float(M) * win_that + Mh * p_prev_c;
                if (denom > 0.0f) {
                    w_c = (float(M) * win_that / denom) * win_that * Wc;
                }
            }
            float w_h = 0.0f, that_h_cur = 0.0f;
            const float3 hdir = float3(hist.env_slot.ax, hist.env_slot.ay,
                                       hist.env_slot.az);
            if (hist.env_slot.W > 0.0f) {
                that_h_cur =
                    target_env_t(hi.mat, hi.hn, hi.hp, vdir, hdir,
                                 env_texels, env_row_cdf, env_cond_cdf, U,
                                 cc, wm);
                if (that_h_cur > 0.0f) {
                    const float p_prev_h =
                        target_env_t(ph.mat, ph.hn, ph.hp, pvdir, hdir,
                                     env_texels, env_row_cdf, env_cond_cdf,
                                     U, cc, wm);
                    const float denom =
                        float(M) * that_h_cur + Mh * p_prev_h;
                    if (denom > 0.0f && p_prev_h > 0.0f) {
                        w_h = (Mh * p_prev_h / denom) * that_h_cur *
                              hist.env_slot.W;
                    }
                }
            }
            out_wsum = w_c + w_h;
            if (w_h > 0.0f && ur_h < w_h / out_wsum) {
                out_dir = hdir;
                out_that = that_h_cur;
            }
        }
        ReSTIRSlot es;
        es.ax = out_dir.x;
        es.ay = out_dir.y;
        es.az = out_dir.z;
        es.W = (out_wsum > 0.0f && out_that > 0.0f) ? out_wsum / out_that
                                                    : 0.0f;
        es.M = fmin(out_M, McapF);
        es.light_id_p1 = 0u;
        es.pad0 = 0u;
        es.pad1 = 0u;
        cur.env_slot = es;
    } else {
        ReSTIRSlot z;
        z.ax = 0.0f; z.ay = 0.0f; z.az = 1.0f;
        z.W = 0.0f; z.M = 0.0f; z.light_id_p1 = 0u; z.pad0 = 0u; z.pad1 = 0u;
        cur.env_slot = z;
    }

    // --- area slot ---
    if (can_area) {
        float wsum_c = 0.0f, win_thatA = 0.0f;
        float win_ax = 0.0f, win_ay = 0.0f, win_az = 0.0f;
        uint win_id_p1 = 0u;
        for (int i = 0; i < M; ++i) {
            const float us = pcg_next_float(rng);
            const float u1 = pcg_next_float(rng);
            const float u2 = pcg_next_float(rng);
            const float ur = pcg_next_float(rng);
            const int li = light_pick(lights, int(U.light_count), us);
            device const GPULight& L = lights[li];
            float pdf_sa = 0.0f, t_light = 0.0f;
            float bu = 0.0f, bv = 0.0f;
            const float3 ldir =
                L.kind == 0u
                    ? sample_sphere_light(L, hi.hp, u1, u2, pdf_sa, t_light)
                    : sample_tri_light(L, hi.hp, u1, u2, pdf_sa, t_light,
                                       bu, bv);
            float w_i = 0.0f, thatA = 0.0f;
            float sax = 0.0f, say = 0.0f, saz = 0.0f;
            if (pdf_sa > 1e-12f && L.sel_pdf > 0.0f && t_light > 1e-6f) {
                if (L.kind == 0u) {
                    const float3 yy = hi.hp + ldir * t_light;
                    sax = yy.x;
                    say = yy.y;
                    saz = yy.z;
                } else {
                    sax = bu;
                    say = bv;
                }
                float3 contrib;
                float w_mis, G, t_o;
                float3 d_o;
                thatA = target_area_t(hi.mat, hi.hn, hi.hp, vdir, L, sax,
                                      say, saz, materials, MU, contrib,
                                      w_mis, G, d_o, t_o);
                const float pl = pdf_sa * L.sel_pdf;
                if (thatA > 0.0f && G > 0.0f) w_i = thatA / (pl * G);
            }
            wsum_c += w_i;
            if (w_i > 0.0f && ur < w_i / wsum_c) {
                win_thatA = thatA;
                win_ax = sax;
                win_ay = say;
                win_az = saz;
                win_id_p1 = uint(li + 1);
            }
        }
        const float Wc = (wsum_c > 0.0f && win_thatA > 0.0f)
                             ? wsum_c / (float(M) * win_thatA)
                             : 0.0f;
        const float ur_h = pcg_next_float(rng);
        float out_that = win_thatA;
        float out_ax = win_ax, out_ay = win_ay, out_az = win_az;
        uint out_id = win_id_p1;
        float out_wsum = 0.0f, out_M = float(M);
        const bool have_hist = hist_ok && hist.area_slot.M > 0.0f;
        if (!have_hist) {
            out_wsum = win_thatA * Wc;
        } else {
            const float Mh = fmin(hist.area_slot.M, McapF);
            out_M += Mh;
            float3 cc;
            float wm, gg, tt;
            float3 dd;
            float w_c = 0.0f;
            if (Wc > 0.0f && win_thatA > 0.0f && win_id_p1 > 0u) {
                device const GPULight& Lc = lights[int(win_id_p1) - 1];
                const float p_prev_c = target_area_t(
                    ph.mat, ph.hn, ph.hp, pvdir, Lc, win_ax, win_ay,
                    win_az, materials, MU, cc, wm, gg, dd, tt);
                const float denom = float(M) * win_thatA + Mh * p_prev_c;
                if (denom > 0.0f) {
                    w_c = (float(M) * win_thatA / denom) * win_thatA * Wc;
                }
            }
            float w_h = 0.0f, that_h_cur = 0.0f;
            if (hist.area_slot.W > 0.0f && hist.area_slot.light_id_p1 > 0u &&
                hist.area_slot.light_id_p1 <= U.light_count) {
                device const GPULight& Lh =
                    lights[int(hist.area_slot.light_id_p1) - 1];
                that_h_cur = target_area_t(
                    hi.mat, hi.hn, hi.hp, vdir, Lh, hist.area_slot.ax,
                    hist.area_slot.ay, hist.area_slot.az, materials, MU,
                    cc, wm, gg, dd, tt);
                if (that_h_cur > 0.0f) {
                    const float p_prev_h = target_area_t(
                        ph.mat, ph.hn, ph.hp, pvdir, Lh, hist.area_slot.ax,
                        hist.area_slot.ay, hist.area_slot.az, materials,
                        MU, cc, wm, gg, dd, tt);
                    const float denom =
                        float(M) * that_h_cur + Mh * p_prev_h;
                    if (denom > 0.0f && p_prev_h > 0.0f) {
                        w_h = (Mh * p_prev_h / denom) * that_h_cur *
                              hist.area_slot.W;
                    }
                }
            }
            out_wsum = w_c + w_h;
            if (w_h > 0.0f && ur_h < w_h / out_wsum) {
                out_that = that_h_cur;
                out_ax = hist.area_slot.ax;
                out_ay = hist.area_slot.ay;
                out_az = hist.area_slot.az;
                out_id = hist.area_slot.light_id_p1;
            }
        }
        ReSTIRSlot as;
        as.ax = out_ax;
        as.ay = out_ay;
        as.az = out_az;
        as.W = (out_wsum > 0.0f && out_that > 0.0f) ? out_wsum / out_that
                                                    : 0.0f;
        as.M = fmin(out_M, McapF);
        as.light_id_p1 = out_id;
        as.pad0 = 0u;
        as.pad1 = 0u;
        cur.area_slot = as;
    } else {
        ReSTIRSlot z;
        z.ax = 0.0f; z.ay = 0.0f; z.az = 1.0f;
        z.W = 0.0f; z.M = 0.0f; z.light_id_p1 = 0u; z.pad0 = 0u; z.pad1 = 0u;
        cur.area_slot = z;
    }
}

inline PRNG pcg_restore(uint lo, uint hi, ulong px) {
    PRNG r;
    r.state = (ulong(hi) << 32) | ulong(lo);
    r.inc = (px << 1) | 1UL;
    return r;
}

kernel void g_primary(device GBufferPx* gbuf          [[buffer(0)]],
                      constant GPUSphere* spheres     [[buffer(1)]],
                      constant PassUniforms& U        [[buffer(2)]],
                      device const BVHNode* nodes     [[buffer(3)]],
                      device const GPUTriangle* tris  [[buffer(4)]],
                      device const GPUMaterialArgs* materials [[buffer(5)]],
                      device const uint* tri_mat      [[buffer(6)]],
                      device const uint* tri_light    [[buffer(7)]],
                      constant MeshUniforms& MU       [[buffer(8)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint py = gid.y + U.row_offset;
    if (py >= U.height) return;
    const ulong px = ulong(py) * U.width + gid.x;
    const float inv_w = 1.0f / float(U.width);
    const float inv_h = 1.0f / float(U.height);
    const ulong pass = ulong(U.pass_base);
    PRNG rng = pcg_init(mix64(px ^ (pass << 32)), px);
    const float u = (float(gid.x) + pcg_next_float(rng)) * inv_w;
    const float v = 1.0f - (float(py) + pcg_next_float(rng)) * inv_h;
    const float3 ro = c3(U.cam.origin);
    const float3 rd = c3(U.cam.lower_left) + u * c3(U.cam.horizontal) +
                      v * c3(U.cam.vertical) - ro;

    GBufferPx g;
    HitInfo hi;
    if (scene_hit_eval(ro, rd, spheres, U.sphere_count, nodes, tris,
                       materials, tri_mat, tri_light, MU, hi)) {
        g.pos = pt_float3{hi.hp.x, hi.hp.y, hi.hp.z};
        g.t = hi.t;
        g.normal = pt_float3{hi.hn.x, hi.hn.y, hi.hn.z};
        g.flags = hi.front ? 1u : 0u;
        g.base_color = pt_float3{hi.mat.base_color.x, hi.mat.base_color.y,
                                 hi.mat.base_color.z};
        g.metallic = hi.mat.metallic;
        g.emission = pt_float3{hi.mat.emission.x, hi.mat.emission.y,
                               hi.mat.emission.z};
        g.ior = hi.mat.ior;
        g.roughness = hi.mat.roughness;
        g.transmission = hi.mat.transmission;
        g.light_id_p1 = uint(hi.light_id + 1);
    } else {
        g.t = -1.0f;
        g.pos = pt_float3{0.0f, 0.0f, 0.0f};
        g.normal = pt_float3{0.0f, 1.0f, 0.0f};
        g.flags = 0u;
        g.base_color = pt_float3{0.0f, 0.0f, 0.0f};
        g.metallic = 0.0f;
        g.emission = pt_float3{0.0f, 0.0f, 0.0f};
        g.ior = 1.5f;
        g.roughness = 1.0f;
        g.transmission = 0.0f;
        g.light_id_p1 = 0u;
    }
    g.rd = pt_float3{rd.x, rd.y, rd.z};
    g.rng_lo = uint(rng.state & 0xffffffffUL);
    g.rng_hi = uint(rng.state >> 32);
    gbuf[px] = g;
}

kernel void direct_v0(device float4* accum            [[buffer(0)]],
                      constant GPUSphere* spheres     [[buffer(1)]],
                      constant PassUniforms& U        [[buffer(2)]],
                      device const BVHNode* nodes     [[buffer(3)]],
                      device const GPUTriangle* tris  [[buffer(4)]],
                      device const GPUMaterialArgs* materials [[buffer(5)]],
                      device GBufferPx* gbuf          [[buffer(6)]],
                      constant MeshUniforms& MU       [[buffer(8)]],
                      device const float* env_texels  [[buffer(10)]],
                      device const float* env_row_cdf [[buffer(11)]],
                      device const float* env_cond_cdf [[buffer(12)]],
                      device const GPULight* lights   [[buffer(13)]],
                      device const ReSTIRPixel* resv  [[buffer(14)]],
                      device ReSTIRPixel* resv_cur    [[buffer(15)]],
                      device const GBufferPx* gbuf_prev [[buffer(16)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint py = gid.y + U.row_offset;
    if (py >= U.height) return;
    const ulong px = ulong(py) * U.width + gid.x;
    GBufferPx g = gbuf[px];
    if (g.t < 0.0f) return;
    const HitInfo hi = hitinfo_from_gbuf(g);
    PRNG rng = pcg_restore(g.rng_lo, g.rng_hi, px);

    restir_build(hi, c3(g.rd), rng, int(U.restir_m),
                 U.restir_temporal != 0u, int(U.restir_mcap), resv[px],
                 gbuf_prev[px], resv_cur[px], spheres, U.sphere_count,
                 materials, env_texels, env_row_cdf, env_cond_cdf, lights,
                 MU, U);

    gbuf[px].rng_lo = uint(rng.state & 0xffffffffUL);
    gbuf[px].rng_hi = uint(rng.state >> 32);
}

// Phase D2 — mirror of integrator.h's restir_spatial_shade: Talbot
// balance-heuristic spatial combine (unbiased for any support overlap),
// the slot's single shadow ray, shading, and pre-spatial history store.
kernel void spatial_v0(device float4* accum            [[buffer(0)]],
                       constant GPUSphere* spheres     [[buffer(1)]],
                       constant PassUniforms& U        [[buffer(2)]],
                       device const BVHNode* nodes     [[buffer(3)]],
                       device const GPUTriangle* tris  [[buffer(4)]],
                       device const GPUMaterialArgs* materials [[buffer(5)]],
                       device GBufferPx* gbuf          [[buffer(6)]],
                       device const ReSTIRPixel* resv_cur [[buffer(7)]],
                       constant MeshUniforms& MU       [[buffer(8)]],
                       device ReSTIRPixel* resv        [[buffer(9)]],
                       device const float* env_texels  [[buffer(10)]],
                       device const float* env_row_cdf [[buffer(11)]],
                       device const float* env_cond_cdf [[buffer(12)]],
                       device const GPULight* lights   [[buffer(13)]],
                       uint2 gid [[thread_position_in_grid]]) {
    const uint py = gid.y + U.row_offset;
    if (py >= U.height) return;
    const ulong px = ulong(py) * U.width + gid.x;
    GBufferPx g = gbuf[px];
    if (g.t < 0.0f) return;
    const HitInfo hi = hitinfo_from_gbuf(g);
    PRNG rng = pcg_restore(g.rng_lo, g.rng_hi, px);
    const float3 vdir = -normalize(c3(g.rd));
    const bool can_env = U.env_nee != 0u && U.env_w != 0u &&
                         hi.mat.transmission <= 0.5f;
    const bool can_area = U.env_nee != 0u && U.light_count > 0u &&
                          hi.mat.transmission <= 0.5f;
    device const ReSTIRPixel& own = resv_cur[px];
    float3 radiance = float3(0.0f);

    int nxs[PT_RESTIR_NEIGHBORS], nys[PT_RESTIR_NEIGHBORS];
    bool ok[PT_RESTIR_NEIGHBORS];
    const int K = U.restir_spatial != 0u
                      ? (int(U.restir_k) < PT_RESTIR_NEIGHBORS
                             ? int(U.restir_k)
                             : PT_RESTIR_NEIGHBORS)
                      : 0;
    for (int k = 0; k < K; ++k) {
        const float u1 = pcg_next_float(rng);
        const float u2 = pcg_next_float(rng);
        int nx = int(gid.x) +
                 int((u1 * 2.0f - 1.0f) * float(U.restir_radius));
        int ny = int(py) + int((u2 * 2.0f - 1.0f) * float(U.restir_radius));
        nx = nx < 0 ? 0 : (nx >= int(U.width) ? int(U.width) - 1 : nx);
        ny = ny < 0 ? 0 : (ny >= int(U.height) ? int(U.height) - 1 : ny);
        nxs[k] = nx;
        nys[k] = ny;
        ok[k] = false;
        if (nx == int(gid.x) && ny == int(py)) continue;
        const GBufferPx gn = gbuf[ulong(ny) * U.width + nx];
        if (gn.t < 0.0f || gn.transmission > 0.5f) continue;
        if (fabs(gn.t - hi.t) >= 0.1f * hi.t) continue;
        if (dot(c3(gn.normal), hi.hn) <= 0.9f) continue;
        ok[k] = true;
    }

    // Participant surfaces cached as HitInfo (own = index -1).
    HitInfo nsurf[PT_RESTIR_NEIGHBORS];
    float3 nvdir[PT_RESTIR_NEIGHBORS];
    for (int k = 0; k < K; ++k) {
        if (!ok[k]) continue;
        const GBufferPx gn = gbuf[ulong(nys[k]) * U.width + nxs[k]];
        nsurf[k] = hitinfo_from_gbuf(gn);
        nvdir[k] = -normalize(c3(gn.rd));
    }

    // --- env slot ---
    if (can_env) {
        // Mirror of integrator.h: denominators count every M>0 input;
        // selection needs W>0.
        ReSTIRSlot pslot[1 + PT_RESTIR_NEIGHBORS];
        int psurf[1 + PT_RESTIR_NEIGHBORS];
        int np = 0;
        int dsurf[1 + PT_RESTIR_NEIGHBORS];
        float dM[1 + PT_RESTIR_NEIGHBORS];
        int nd = 0;
        if (own.env_slot.M > 0.0f) {
            dsurf[nd] = -1;
            dM[nd] = own.env_slot.M;
            ++nd;
            if (own.env_slot.W > 0.0f) {
                pslot[np] = own.env_slot;
                psurf[np] = -1;
                ++np;
            }
        }
        int pidx[PT_RESTIR_NEIGHBORS];
        for (int k = 0; k < K; ++k) {
            pidx[k] = -1;
            if (!ok[k]) continue;
            device const ReSTIRPixel& nb =
                resv_cur[ulong(nys[k]) * U.width + nxs[k]];
            if (nb.env_slot.M <= 0.0f) continue;
            dsurf[nd] = k;
            dM[nd] = nb.env_slot.M;
            ++nd;
            if (nb.env_slot.W <= 0.0f) continue;
            pslot[np] = nb.env_slot;
            psurf[np] = k;
            pidx[k] = np;
            ++np;
        }
        float3 win_dir = float3(0.0f, 0.0f, 1.0f);
        float3 win_c = float3(0.0f);
        float win_mis = 0.0f, win_that = 0.0f, wsum = 0.0f;
        // Own seeds without a draw; neighbors take one lottery draw per
        // SLOT k (drawn regardless of participation) — CPU order.
        for (int i = 0; i < np && psurf[i] < 0; ++i) {
            const float ur = 0.0f;
            const float3 sdir =
                float3(pslot[i].ax, pslot[i].ay, pslot[i].az);
            float3 c;
            float wm;
            const float that_own =
                target_env_t(hi.mat, hi.hn, hi.hp, vdir, sdir, env_texels,
                             env_row_cdf, env_cond_cdf, U, c, wm);
            float w_i = 0.0f;
            if (that_own > 0.0f) {
                float denom = 0.0f, self = 0.0f;
                for (int j = 0; j < nd; ++j) {
                    float3 cc;
                    float wmm;
                    float p;
                    if (dsurf[j] < 0) {
                        p = target_env_t(hi.mat, hi.hn, hi.hp, vdir, sdir,
                                         env_texels, env_row_cdf,
                                         env_cond_cdf, U, cc, wmm);
                    } else {
                        const int kk = dsurf[j];
                        p = target_env_t(nsurf[kk].mat, nsurf[kk].hn,
                                         nsurf[kk].hp, nvdir[kk], sdir,
                                         env_texels, env_row_cdf,
                                         env_cond_cdf, U, cc, wmm);
                    }
                    denom += dM[j] * p;
                    if (dsurf[j] == psurf[i]) self = p;
                }
                if (denom > 0.0f && self > 0.0f) {
                    w_i = (pslot[i].M * self / denom) * that_own *
                          pslot[i].W;
                }
            }
            (void)ur;
            wsum += w_i;
            if (w_i > 0.0f) {
                win_dir = sdir;
                win_c = c;
                win_mis = wm;
                win_that = that_own;
            }
        }
        for (int k = 0; k < K; ++k) {
            const float ur = pcg_next_float(rng);
            const int i = pidx[k];
            if (i < 0) continue;
            const float3 sdir =
                float3(pslot[i].ax, pslot[i].ay, pslot[i].az);
            float3 c;
            float wm;
            const float that_own =
                target_env_t(hi.mat, hi.hn, hi.hp, vdir, sdir, env_texels,
                             env_row_cdf, env_cond_cdf, U, c, wm);
            float w_i = 0.0f;
            if (that_own > 0.0f) {
                float denom = 0.0f, self = 0.0f;
                for (int j = 0; j < nd; ++j) {
                    float3 cc;
                    float wmm;
                    float p;
                    if (dsurf[j] < 0) {
                        p = target_env_t(hi.mat, hi.hn, hi.hp, vdir, sdir,
                                         env_texels, env_row_cdf,
                                         env_cond_cdf, U, cc, wmm);
                    } else {
                        const int kk = dsurf[j];
                        p = target_env_t(nsurf[kk].mat, nsurf[kk].hn,
                                         nsurf[kk].hp, nvdir[kk], sdir,
                                         env_texels, env_row_cdf,
                                         env_cond_cdf, U, cc, wmm);
                    }
                    denom += dM[j] * p;
                    if (dsurf[j] == psurf[i]) self = p;
                }
                if (denom > 0.0f && self > 0.0f) {
                    w_i = (pslot[i].M * self / denom) * that_own *
                          pslot[i].W;
                }
            }
            wsum += w_i;
            if (w_i > 0.0f && ur < w_i / wsum) {
                win_dir = sdir;
                win_c = c;
                win_mis = wm;
                win_that = that_own;
            }
        }
        const float Wnew =
            (wsum > 0.0f && win_that > 0.0f) ? wsum / win_that : 0.0f;
        resv[px].env_slot = own.env_slot;   // pre-spatial history
        if (Wnew > 0.0f) {
            if (!occluded_scene(hi.hp, win_dir, INFINITY, spheres,
                                U.sphere_count, nodes, tris, MU)) {
                const float3 c = win_c * win_mis * Wnew;
                radiance += clamp_contribution(c, U.clamp_indirect);
            }
        }
    } else {
        for (int k = 0; k < K; ++k) pcg_next_float(rng);
        ReSTIRSlot z;
        z.ax = 0.0f; z.ay = 0.0f; z.az = 1.0f;
        z.W = 0.0f; z.M = 0.0f; z.light_id_p1 = 0u; z.pad0 = 0u; z.pad1 = 0u;
        resv[px].env_slot = z;
    }

    // --- area slot ---
    if (can_area) {
        ReSTIRSlot pslot[1 + PT_RESTIR_NEIGHBORS];
        int psurf[1 + PT_RESTIR_NEIGHBORS];
        int np = 0;
        int dsurf[1 + PT_RESTIR_NEIGHBORS];
        float dM[1 + PT_RESTIR_NEIGHBORS];
        int nd = 0;
        if (own.area_slot.M > 0.0f) {
            dsurf[nd] = -1;
            dM[nd] = own.area_slot.M;
            ++nd;
            if (own.area_slot.W > 0.0f &&
                own.area_slot.light_id_p1 > 0u &&
                own.area_slot.light_id_p1 <= U.light_count) {
                pslot[np] = own.area_slot;
                psurf[np] = -1;
                ++np;
            }
        }
        int pidx[PT_RESTIR_NEIGHBORS];
        for (int k = 0; k < K; ++k) {
            pidx[k] = -1;
            if (!ok[k]) continue;
            device const ReSTIRPixel& nb =
                resv_cur[ulong(nys[k]) * U.width + nxs[k]];
            if (nb.area_slot.M <= 0.0f) continue;
            dsurf[nd] = k;
            dM[nd] = nb.area_slot.M;
            ++nd;
            if (nb.area_slot.W <= 0.0f ||
                nb.area_slot.light_id_p1 == 0u ||
                nb.area_slot.light_id_p1 > U.light_count)
                continue;
            pslot[np] = nb.area_slot;
            psurf[np] = k;
            pidx[k] = np;
            ++np;
        }
        float3 win_c = float3(0.0f);
        float win_mis = 0.0f, win_G = 0.0f, win_t = 0.0f, win_that = 0.0f;
        float3 win_dir = float3(0.0f, 0.0f, 1.0f);
        float wsum = 0.0f;
        bool have_win = false;
        for (int i = 0; i < np && psurf[i] < 0; ++i) {
            const float ur = 0.0f;
            device const GPULight& L =
                lights[int(pslot[i].light_id_p1) - 1];
            float3 c;
            float wm, G, t;
            float3 d;
            const float that_own =
                target_area_t(hi.mat, hi.hn, hi.hp, vdir, L, pslot[i].ax,
                              pslot[i].ay, pslot[i].az, materials, MU, c,
                              wm, G, d, t);
            float w_i = 0.0f;
            if (that_own > 0.0f) {
                float denom = 0.0f, self = 0.0f;
                for (int j = 0; j < nd; ++j) {
                    float3 cc;
                    float wmm, gg, tt;
                    float3 dd;
                    float p;
                    if (dsurf[j] < 0) {
                        p = target_area_t(hi.mat, hi.hn, hi.hp, vdir, L,
                                          pslot[i].ax, pslot[i].ay,
                                          pslot[i].az, materials, MU, cc,
                                          wmm, gg, dd, tt);
                    } else {
                        const int kk = dsurf[j];
                        p = target_area_t(nsurf[kk].mat, nsurf[kk].hn,
                                          nsurf[kk].hp, nvdir[kk], L,
                                          pslot[i].ax, pslot[i].ay,
                                          pslot[i].az, materials, MU, cc,
                                          wmm, gg, dd, tt);
                    }
                    denom += dM[j] * p;
                    if (dsurf[j] == psurf[i]) self = p;
                }
                if (denom > 0.0f && self > 0.0f) {
                    w_i = (pslot[i].M * self / denom) * that_own *
                          pslot[i].W;
                }
            }
            (void)ur;
            wsum += w_i;
            if (w_i > 0.0f) {
                win_c = c;
                win_mis = wm;
                win_G = G;
                win_dir = d;
                win_t = t;
                win_that = that_own;
                have_win = true;
            }
        }
        for (int k = 0; k < K; ++k) {
            const float ur = pcg_next_float(rng);
            const int i = pidx[k];
            if (i < 0) continue;
            device const GPULight& L =
                lights[int(pslot[i].light_id_p1) - 1];
            float3 c;
            float wm, G, t;
            float3 d;
            const float that_own =
                target_area_t(hi.mat, hi.hn, hi.hp, vdir, L, pslot[i].ax,
                              pslot[i].ay, pslot[i].az, materials, MU, c,
                              wm, G, d, t);
            float w_i = 0.0f;
            if (that_own > 0.0f) {
                float denom = 0.0f, self = 0.0f;
                for (int j = 0; j < nd; ++j) {
                    float3 cc;
                    float wmm, gg, tt;
                    float3 dd;
                    float p;
                    if (dsurf[j] < 0) {
                        p = target_area_t(hi.mat, hi.hn, hi.hp, vdir, L,
                                          pslot[i].ax, pslot[i].ay,
                                          pslot[i].az, materials, MU, cc,
                                          wmm, gg, dd, tt);
                    } else {
                        const int kk = dsurf[j];
                        p = target_area_t(nsurf[kk].mat, nsurf[kk].hn,
                                          nsurf[kk].hp, nvdir[kk], L,
                                          pslot[i].ax, pslot[i].ay,
                                          pslot[i].az, materials, MU, cc,
                                          wmm, gg, dd, tt);
                    }
                    denom += dM[j] * p;
                    if (dsurf[j] == psurf[i]) self = p;
                }
                if (denom > 0.0f && self > 0.0f) {
                    w_i = (pslot[i].M * self / denom) * that_own *
                          pslot[i].W;
                }
            }
            wsum += w_i;
            if (w_i > 0.0f && ur < w_i / wsum) {
                win_c = c;
                win_mis = wm;
                win_G = G;
                win_dir = d;
                win_t = t;
                win_that = that_own;
                have_win = true;
            }
        }
        const float Wnew = (wsum > 0.0f && win_that > 0.0f && have_win)
                               ? wsum / win_that
                               : 0.0f;
        resv[px].area_slot = own.area_slot;   // pre-spatial history
        if (Wnew > 0.0f) {
            if (!occluded_scene(hi.hp, win_dir, win_t * (1.0f - 1e-3f),
                                spheres, U.sphere_count, nodes, tris, MU)) {
                const float3 c = win_c * win_mis * (win_G * Wnew);
                radiance += clamp_contribution(c, U.clamp_indirect);
            }
        }
    } else {
        for (int k = 0; k < K; ++k) pcg_next_float(rng);
        ReSTIRSlot z;
        z.ax = 0.0f; z.ay = 0.0f; z.az = 1.0f;
        z.W = 0.0f; z.M = 0.0f; z.light_id_p1 = 0u; z.pad0 = 0u; z.pad1 = 0u;
        resv[px].area_slot = z;
    }

    resv[px].prev_normal = pt_float3{hi.hn.x, hi.hn.y, hi.hn.z};
    resv[px].prev_t = hi.t;
    accum[px] += float4(radiance, 0.0f);

    gbuf[px].rng_lo = uint(rng.state & 0xffffffffUL);
    gbuf[px].rng_hi = uint(rng.state >> 32);
}

kernel void indirect_v0(device float4* accum           [[buffer(0)]],
                        constant GPUSphere* spheres    [[buffer(1)]],
                        constant PassUniforms& U       [[buffer(2)]],
                        device const BVHNode* nodes    [[buffer(3)]],
                        device const GPUTriangle* tris [[buffer(4)]],
                        device const GPUMaterialArgs* materials [[buffer(5)]],
                        device const GBufferPx* gbuf   [[buffer(6)]],
                        device const uint* tri_mat     [[buffer(7)]],
                        constant MeshUniforms& MU      [[buffer(8)]],
                        device const uint* tri_light   [[buffer(9)]],
                        device const float* env_texels [[buffer(10)]],
                        device const float* env_row_cdf [[buffer(11)]],
                        device const float* env_cond_cdf [[buffer(12)]],
                        device const GPULight* lights  [[buffer(13)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint py = gid.y + U.row_offset;
    if (py >= U.height) return;
    const ulong px = ulong(py) * U.width + gid.x;
    const GBufferPx g = gbuf[px];
    PRNG rng = pcg_restore(g.rng_lo, g.rng_hi, px);
    const float3 rd = c3(g.rd);

    if (g.t < 0.0f) {
        // Primary miss: full env, exactly the monolithic depth-0 branch.
        const float3 c = miss_radiance(env_texels, U.env_w, U.env_h,
                                       U.env_intensity, U.env_yaw_norm, rd);
        accum[px] += float4(c, 0.0f);
        return;
    }
    const HitInfo pre = hitinfo_from_gbuf(g);
    const float3 c =
        trace(c3(g.pos), rd, spheres, U.sphere_count, nodes, tris, materials,
              tri_mat, env_texels, env_row_cdf, env_cond_cdf, lights,
              tri_light, MU, U, rng, U.max_depth, true, pre, true, false,
              0.0f);   // ReSTIR indirect: fog deferred (monolithic path only)
    accum[px] += float4(c, 0.0f);
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

// ---------------------------------------------------------------------
// v1.1 DISPLAY-ONLY DENOISER — everything below runs strictly between the
// resolve copy and the drawable. It reads the accumulator and G-buffer,
// writes only scratch + the drawable, and is never encoded by the parity
// / offline / FINAL-export paths. It is OUTSIDE the parity surface.
// ---------------------------------------------------------------------

// accum/N + albedo demodulation -> illum scratch (accum resolution).
// w channel: 1 = surface (demodulated), 0 = primary miss (passthrough).
kernel void dn_resolve(device const float4* accum       [[buffer(0)]],
                       constant DenoiseUniforms& U      [[buffer(1)]],
                       device const GBufferPx* gbuf     [[buffer(2)]],
                       device float4* illum             [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= U.accum_w || gid.y >= U.accum_h) return;
    const ulong px = ulong(gid.y) * U.accum_w + gid.x;
    const uint n = U.pass_total + (gid.y < U.rows_plus1 ? 1u : 0u);
    const float3 c = accum[px].xyz / float(max(n, 1u));
    const GBufferPx g = gbuf[px];
    if (g.t < 0.0f) {
        illum[px] = float4(c, 0.0f);
        return;
    }
    const float3 alb = fmax(c3(g.base_color), float3(1e-3f));
    illum[px] = float4(c / alb, 1.0f);
}

// One a-trous iteration: 5x5 B3-spline taps dilated by U.step, weighted
// by normal / relative-depth / illumination-luminance edge stops.
kernel void dn_atrous(device const float4* src          [[buffer(0)]],
                      constant DenoiseUniforms& U       [[buffer(1)]],
                      device const GBufferPx* gbuf      [[buffer(2)]],
                      device float4* dst                [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= U.accum_w || gid.y >= U.accum_h) return;
    const ulong px = ulong(gid.y) * U.accum_w + gid.x;
    const float4 center = src[px];
    if (center.w < 0.5f) {   // sky: never filtered, never a filter input
        dst[px] = center;
        return;
    }
    const GBufferPx gc = gbuf[px];
    const float3 cn = c3(gc.normal);
    const float cl =
        0.2126f * center.x + 0.7152f * center.y + 0.0722f * center.z;
    const float hk[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    float3 sum = float3(0.0f);
    float wsum = 0.0f;
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int tx = int(gid.x) + dx * int(U.step);
            const int ty = int(gid.y) + dy * int(U.step);
            if (tx < 0 || ty < 0 || tx >= int(U.accum_w) ||
                ty >= int(U.accum_h))
                continue;
            const ulong tp = ulong(ty) * U.accum_w + tx;
            const float4 t = src[tp];
            if (t.w < 0.5f) continue;
            const GBufferPx gt = gbuf[tp];
            const float w_n =
                pow(fmax(dot(cn, c3(gt.normal)), 0.0f), U.sigma_n);
            const float w_z = exp(-fabs(gt.t - gc.t) /
                                  (U.sigma_z * fmax(gc.t, 1e-3f)));
            const float tl =
                0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
            const float w_l = exp(-fabs(tl - cl) / (U.sigma_l + 1e-6f));
            const float w = hk[dx + 2] * hk[dy + 2] * w_n * w_z * w_l;
            sum += t.xyz * w;
            wsum += w;
        }
    }
    dst[px] = wsum > 0.0f ? float4(sum / wsum, 1.0f) : center;
}

// Remodulate, fade toward the raw accumulation, gamma, wipe, AOVs.
// The raw side of the wipe recomputes accum/N with the EXACT resolve
// expression so "denoise off" and "left of the wipe" are the true image.
kernel void dn_finish(device const float4* accum        [[buffer(0)]],
                      constant DenoiseUniforms& U       [[buffer(1)]],
                      device const GBufferPx* gbuf      [[buffer(2)]],
                      device const float4* illum        [[buffer(3)]],
                      texture2d<float, access::write> out [[texture(0)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= U.out_w || gid.y >= U.out_h) return;
    const uint sx = gid.x * U.accum_w / U.out_w;
    const uint sy = gid.y * U.accum_h / U.out_h;
    const ulong px = ulong(sy) * U.accum_w + sx;
    const uint n = U.pass_total + (sy < U.rows_plus1 ? 1u : 0u);
    const float inv_n = 1.0f / float(max(n, 1u));
    const float3 raw = accum[px].xyz * inv_n;
    const GBufferPx g = gbuf[px];
    float3 c;
    if (U.aov == 1u) {
        c = g.t < 0.0f ? float3(0.0f) : c3(g.normal) * 0.5f + 0.5f;
    } else if (U.aov == 2u) {
        c = float3(g.t < 0.0f ? 0.0f : g.t / (1.0f + g.t));
    } else if (U.aov == 3u) {
        c = g.t < 0.0f ? float3(0.0f) : c3(g.base_color);
    } else if (U.aov == 4u) {
        c = illum[px].xyz;
    } else if (gid.x < U.wipe_x) {
        c = raw;
    } else {
        const float4 il = illum[px];
        const float3 dn =
            il.w < 0.5f
                ? il.xyz
                : il.xyz * fmax(c3(g.base_color), float3(1e-3f));
        c = mix(raw, dn, U.alpha);
    }
    c = pow(fmax(c, 0.0f), float3(1.0f / 2.2f));
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
