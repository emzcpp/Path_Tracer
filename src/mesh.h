#pragma once

// CPU-side triangle mesh: BVH traversal + Möller–Trumbore + deferred
// textured-material evaluation. The traversal and intersection code here is
// duplicated line-for-line in pathtrace.metal — any change must be mirrored
// there; --parity is the drift detector.
//
// Note the deliberate structure: the BVH walk stores only (t, u, v, tri);
// textures are sampled exactly ONCE, after the winning triangle is final,
// and the result is written into the existing Material POD — so hittable.h,
// material.h, and integrator.h need zero changes.

#include <cmath>
#include <cstdint>
#include <memory>

#include "gltf_loader.h"
#include "hittable.h"
#include "texture.h"

inline vec3 mesh_c3(const pt_float3& v) { return {v.x, v.y, v.z}; }

struct TriHit {
    float u = 0.0f, v = 0.0f;
    std::uint32_t tri = 0;
};

// Möller–Trumbore. The absolute det epsilon is REQUIRED: det == 0 would
// give inf/NaN u/v/t that leak through the range tests. No backface
// culling — refracted/interior rays legitimately see backfaces. Tie rule
// `t > closest` rejects (ties replace), identical to sphere.h.
inline bool hit_tri(const GPUTriangle& T, const vec3& ro, const vec3& rd,
                    float t_min, float& closest, TriHit& h) {
    const vec3 pv = cross(rd, mesh_c3(T.e2));
    const float det = dot(mesh_c3(T.e1), pv);
    if (std::fabs(det) < 1e-8f) return false;
    const float inv_det = 1.0f / det;
    const vec3 tv = ro - mesh_c3(T.p0);
    const float u = dot(tv, pv) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const vec3 qv = cross(tv, mesh_c3(T.e1));
    const float v = dot(rd, qv) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = dot(mesh_c3(T.e2), qv) * inv_det;
    if (t < t_min || t > closest) return false;
    closest = t;
    h.u = u;
    h.v = v;
    return true;
}

// Slab test. fmin/fmax resolve the 0*inf NaN case identically in std:: and
// metal:: (IEEE minNum/maxNum semantics).
inline bool hit_aabb(const pt_float3& mn, const pt_float3& mx, const vec3& ro,
                     const vec3& inv, float t_min, float t_max,
                     float& tnear) {
    float a = (mn.x - ro.x) * inv.x, b = (mx.x - ro.x) * inv.x;
    float t0 = std::fmin(a, b), t1 = std::fmax(a, b);
    a = (mn.y - ro.y) * inv.y;
    b = (mx.y - ro.y) * inv.y;
    t0 = std::fmax(t0, std::fmin(a, b));
    t1 = std::fmin(t1, std::fmax(a, b));
    a = (mn.z - ro.z) * inv.z;
    b = (mx.z - ro.z) * inv.z;
    t0 = std::fmax(t0, std::fmin(a, b));
    t1 = std::fmin(t1, std::fmax(a, b));
    tnear = t0;
    return t0 <= t1 && t0 <= t_max && t1 >= t_min;
}

// Ordered BVH traversal: parent tests both children, near child first
// (deterministic tie -> left), plain uint stack. Depth cap 30 at build
// makes stack[32] a guarantee.
inline bool bvh_hit(const BVHNode* nodes, const GPUTriangle* tris,
                    const vec3& ro, const vec3& rd, float t_min,
                    float& closest, TriHit& best) {
    const vec3 inv(1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.z);
    float tn;
    if (!hit_aabb(nodes[0].mn, nodes[0].mx, ro, inv, t_min, closest, tn))
        return false;
    std::uint32_t stack[32];
    std::uint32_t sp = 0;
    std::uint32_t idx = 0;
    bool found = false;
    for (;;) {
        const BVHNode n = nodes[idx];
        if (n.tri_count > 0) {   // leaf
            for (std::uint32_t i = 0; i < n.tri_count; ++i) {
                if (hit_tri(tris[n.left_or_first + i], ro, rd, t_min, closest,
                            best)) {
                    found = true;
                    best.tri = n.left_or_first + i;
                }
            }
            if (sp == 0) break;
            idx = stack[--sp];
        } else {                 // internal
            std::uint32_t l = n.left_or_first, r = l + 1;
            float tl, tr;
            const bool hl =
                hit_aabb(nodes[l].mn, nodes[l].mx, ro, inv, t_min, closest, tl);
            const bool hr =
                hit_aabb(nodes[r].mn, nodes[r].mx, ro, inv, t_min, closest, tr);
            if (hl && hr) {
                if (tr < tl) {
                    const std::uint32_t tmp = l;
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

// Textured material -> the Material POD. With the GGX BSDF the glTF
// metallic-roughness textures feed the model CONTINUOUSLY — no threshold.
// Emission rides on the scattering surface (the integrator collects
// rec.mat.emission unconditionally before scatter).
inline Material eval_mesh_material(const MeshData& m, float u, float v) {
    const color base = sample_bilinear(m.base, u, v);
    const color mr = m.mr.valid() ? sample_bilinear(m.mr, u, v)
                                  : color(0.0f, 1.0f, 0.0f);
    const color emis = m.emissive.valid() ? sample_bilinear(m.emissive, u, v)
                                          : color(0.0f, 0.0f, 0.0f);
    Material out;
    out.base_color = base;
    out.emission = emis * m.emissive_scale;
    out.metallic = mr.z;    // glTF: B = metallic
    out.roughness = mr.y;   // glTF: G = roughness (perceptual)
    out.ior = 1.5f;
    out.transmission = 0.0f;
    return out;
}

class Mesh : public Hittable {
public:
    explicit Mesh(std::shared_ptr<const MeshData> data)
        : data_(std::move(data)) {}

    bool hit(const Ray& r, float t_min, float t_max,
             HitRecord& rec) const override {
        float closest = t_max;
        TriHit best;
        if (!bvh_hit(data_->nodes.data(), data_->tris.data(), r.origin, r.dir,
                     t_min, closest, best))
            return false;

        const GPUTriangle& T = data_->tris[best.tri];
        rec.t = closest;
        rec.p = r.at(closest);

        // Sidedness from the GEOMETRIC normal (drives dielectric eta and
        // metal absorption — robust); the interpolated shading normal only
        // bends the scatter lobe.
        const vec3 ng = cross(mesh_c3(T.e1), mesh_c3(T.e2));
        const bool front = dot(r.dir, ng) < 0.0f;
        const float w = 1.0f - best.u - best.v;
        vec3 ns = normalize(w * mesh_c3(T.n0) + best.u * mesh_c3(T.n1) +
                            best.v * mesh_c3(T.n2));

        const float u = w * T.u0 + best.u * T.u1 + best.v * T.u2;
        const float v = w * T.v0 + best.u * T.v1 + best.v * T.v2;

        // Tangent-space normal mapping (Session D). glTF convention:
        // LINEAR texels, +Y-up (OpenGL); [0,1] -> [-1,1]. The tangent is
        // interpolated, Gram-Schmidt'd against the shading normal, and the
        // bitangent takes the glTF .w handedness sign. Mirrored in
        // pathtrace.metal.
        if (data_->normal.valid()) {
            const color tn = sample_bilinear(data_->normal, u, v);
            const vec3 tN(2.0f * tn.x - 1.0f, 2.0f * tn.y - 1.0f,
                          2.0f * tn.z - 1.0f);
            vec3 tang = w * mesh_c3(T.t0) + best.u * mesh_c3(T.t1) +
                        best.v * mesh_c3(T.t2);
            tang = tang - ns * dot(ns, tang);
            const float tl = tang.length();
            if (tl > 1e-6f) {
                tang = tang / tl;
                const vec3 bit = cross(ns, tang) * T.w0;
                ns = normalize(tN.x * tang + tN.y * bit + tN.z * ns);
            }
        }
        rec.normal = front ? ns : -ns;
        rec.front_face = front;
        rec.mat = eval_mesh_material(*data_, u, v);
        return true;
    }

private:
    std::shared_ptr<const MeshData> data_;
};
