#pragma once

// Procedurally-generated meshes for the built-in demos (no external asset
// needed). Used by --prism-mesh: a triangular glass prism whose flat faces
// give strong wavelength-dependent deviation, so spectral mode splits a
// bright source into a visible spectrum. Reuses the standard MeshData +
// build_bvh path, so a generated mesh is indistinguishable from a loaded
// glTF one to the renderer.

#include <memory>
#include <vector>

#include "bvh.h"
#include "gltf_loader.h"
#include "kernel_types.h"
#include "texture.h"
#include "vec3.h"

namespace mesh_gen {

inline Texture16 white_1x1() {
    Texture16 t;
    t.w = 1;
    t.h = 1;
    t.texels = {65535, 65535, 65535, 65535};
    return t;
}

inline Texture16 solid_1x1(float r, float g, float b) {
    const auto q = [](float x) {
        return std::uint16_t(std::lrint(std::fmax(0.0f, std::fmin(1.0f, x)) *
                                        65535.0f));
    };
    Texture16 t;
    t.w = 1;
    t.h = 1;
    t.texels = {q(r), q(g), q(b), 65535};
    return t;
}

inline void add_tri(std::vector<GPUTriangle>& v, const vec3& p0,
                    const vec3& p1, const vec3& p2) {
    const vec3 e1 = p1 - p0, e2 = p2 - p0;
    const vec3 nn = normalize(cross(e1, e2));   // FLAT per-face normal
    GPUTriangle t{};
    t.p0 = {p0.x, p0.y, p0.z};
    t.e1 = {e1.x, e1.y, e1.z};
    t.e2 = {e2.x, e2.y, e2.z};
    t.n0 = {nn.x, nn.y, nn.z};
    t.n1 = {nn.x, nn.y, nn.z};
    t.n2 = {nn.x, nn.y, nn.z};
    // UVs unused (glass has no textures); tangents unused (no normal map).
    t.t0 = {1, 0, 0};
    t.w0 = 1;
    t.t1 = {1, 0, 0};
    t.w1 = 1;
    t.t2 = {1, 0, 0};
    t.w2 = 1;
    v.push_back(t);
}

inline void add_quad(std::vector<GPUTriangle>& v, const vec3& a, const vec3& b,
                     const vec3& c, const vec3& d) {
    add_tri(v, a, b, c);
    add_tri(v, a, c, d);
}

inline std::shared_ptr<MeshData> finish_opaque(std::vector<GPUTriangle> tris) {
    auto m = std::make_shared<MeshData>();
    m->tri_mat.assign(tris.size(), 0u);
    build_bvh(tris, m->nodes, &m->tri_mat);
    m->tris = std::move(tris);
    MeshMaterial mat;
    mat.base = solid_1x1(0.45f, 0.46f, 0.5f);   // mid-gray walls
    mat.transmission = 0.0f;
    m->materials = {mat};
    m->tri_light.assign(m->tris.size(), 0u);
    m->light_tri_count = 0;
    m->emissive_scale = 1.0f;
    return m;
}

// An axis-aligned room (box with INWARD-facing walls) — the camera sits
// inside and sees the corridor. hx/hy/hz are half-extents about center c.
inline std::shared_ptr<MeshData> room(const point3& c, float hx, float hy,
                                      float hz) {
    std::vector<GPUTriangle> v;
    const vec3 p[8] = {{c.x - hx, c.y - hy, c.z - hz},
                       {c.x + hx, c.y - hy, c.z - hz},
                       {c.x + hx, c.y + hy, c.z - hz},
                       {c.x - hx, c.y + hy, c.z - hz},
                       {c.x - hx, c.y - hy, c.z + hz},
                       {c.x + hx, c.y - hy, c.z + hz},
                       {c.x + hx, c.y + hy, c.z + hz},
                       {c.x - hx, c.y + hy, c.z + hz}};
    // Winding chosen so each face's normal points INWARD.
    add_quad(v, p[5], p[4], p[7], p[6]);  // +z wall (normal -z, inward)
    add_quad(v, p[0], p[1], p[2], p[3]);  // -z wall (normal +z)
    add_quad(v, p[1], p[5], p[6], p[2]);  // +x wall (normal -x)
    add_quad(v, p[4], p[0], p[3], p[7]);  // -x wall (normal +x)
    add_quad(v, p[3], p[2], p[6], p[7]);  // +y ceiling (normal -y)
    add_quad(v, p[4], p[5], p[1], p[0]);  // -y floor (normal +y)
    return finish_opaque(std::move(v));
}

inline std::shared_ptr<MeshData> finish_glass(std::vector<GPUTriangle> tris,
                                              float ior) {
    auto m = std::make_shared<MeshData>();
    m->tri_mat.assign(tris.size(), 0u);
    build_bvh(tris, m->nodes, &m->tri_mat);   // reorders tris + tri_mat
    m->tris = std::move(tris);
    MeshMaterial glass;
    glass.base = white_1x1();
    glass.transmission = 1.0f;
    glass.ior = ior;
    m->materials = {glass};
    m->tri_light.assign(m->tris.size(), 0u);
    m->light_tri_count = 0;
    m->emissive_scale = 1.0f;
    return m;
}

// Triangular prism, axis along Z, upward-pointing triangular cross-section
// of side `s`; half-depth `d` along Z.
inline std::shared_ptr<MeshData> glass_prism(const point3& c, float s, float d,
                                             float ior) {
    std::vector<GPUTriangle> v;
    const float H = s * 0.8660254f;   // equilateral height
    const vec3 a0(c.x - s * 0.5f, c.y - H * 0.5f, c.z - d);
    const vec3 b0(c.x + s * 0.5f, c.y - H * 0.5f, c.z - d);
    const vec3 t0(c.x, c.y + H * 0.5f, c.z - d);
    const vec3 a1(c.x - s * 0.5f, c.y - H * 0.5f, c.z + d);
    const vec3 b1(c.x + s * 0.5f, c.y - H * 0.5f, c.z + d);
    const vec3 t1(c.x, c.y + H * 0.5f, c.z + d);
    add_tri(v, a0, t0, b0);       // back cap (normal -Z)
    add_tri(v, a1, b1, t1);       // front cap (normal +Z)
    add_quad(v, a0, b0, b1, a1);  // bottom face
    add_quad(v, b0, t0, t1, b1);  // right slope
    add_quad(v, t0, a0, a1, t1);  // left slope
    return finish_glass(std::move(v), ior);
}

}  // namespace mesh_gen
