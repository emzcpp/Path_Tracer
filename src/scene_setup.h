#pragma once

// The demo scene, shared by offline renders and the live viewer: three hero
// spheres (matte, glass, metal) over a field of random small spheres, plus
// a glowing lamp. Generated with a fixed RNG seed so every run — and every
// backend — sees the identical world.
//
// The scene is described once as plain data (SceneDesc); the CPU Scene and
// the flat GPU sphere buffer are both derived from it, so the two backends
// can never drift apart.

#include <cstdio>
#include <memory>
#include <vector>

#include "gltf_loader.h"
#include "integrator.h"
#include "kernel_types.h"
#include "mesh.h"
#include "rng.h"
#include "scene.h"
#include "sphere.h"

struct SphereData {
    point3 center;
    float radius;
    Material mat;
    std::string name;   // editable label; empty = UI shows a generated one
};

struct SceneDesc {
    std::vector<SphereData> spheres;
    std::shared_ptr<const MeshData> mesh;   // null = sphere-only scene
    std::string mesh_source_path;           // where the mesh came from (save/load)
    std::string mesh_name;                  // editable label for the mesh

    // Session F: HDRI environment (null = gradient dome fallback).
    std::shared_ptr<const EnvMap> env;
    std::string env_source_path;
    float env_intensity = 1.0f;
    float env_yaw_deg = 0.0f;

    // v1.3 recursive portals (empty = none, so no-portal scenes are
    // byte-identical). Built in pairs by make_portal_pair.
    std::vector<GPUPortal> portals;
};

// Build one portal rectangle from a frame (center + world half-edges u,v),
// with the rigid transform to a partner portal precomputed:
//   R = R_B * Flip180(up) * R_A^T,   t = cB - R*cA
// The standard portal convention: a ray entering the FRONT of A exits the
// FRONT of B (Flip180 about the up axis turns it around). This is what
// makes FACING portals recurse (exit-front loops A->B->A). R_A^T maps
// world -> A-local; R_B maps A-local -> world in B's frame.
inline GPUPortal make_portal(const point3& c, const vec3& u, const vec3& v,
                             const point3& pc, const vec3& pu, const vec3& pv) {
    const auto basis = [](const vec3& uu, const vec3& vv, vec3& r, vec3& up,
                          vec3& n) {
        r = normalize(uu);
        up = normalize(vv);
        n = normalize(cross(uu, vv));
    };
    vec3 rA, upA, nA, rB, upB, nB;
    basis(u, v, rA, upA, nA);
    basis(pu, pv, rB, upB, nB);
    // R_A columns = (rA, upA, nA); R_A^T rows = those. Flip180 about up:
    // diag(-1, 1, -1) in local space. Combined rotation R = R_B * F * R_A^T.
    // Row i of R = R_B * F * (col i of R_A). Compute R via basis images.
    const auto Rt = [&](const vec3& p) {   // R_A^T * p  (world -> A local)
        return vec3(dot(rA, p), dot(upA, p), dot(nA, p));
    };
    const auto FB = [&](const vec3& l) {   // Flip180(up) then R_B
        const vec3 f(-l.x, l.y, -l.z);
        return rB * f.x + upB * f.y + nB * f.z;
    };
    const auto R = [&](const vec3& p) { return FB(Rt(p)); };
    GPUPortal P{};
    P.center = {c.x, c.y, c.z};
    P.u = {u.x, u.y, u.z};
    P.v = {v.x, v.y, v.z};
    P.normal = {nA.x, nA.y, nA.z};
    const vec3 t = pc - R(vec3(c.x, c.y, c.z));
    // Store R's ROWS: r0 = R applied to basis vectors' x-components. Easiest:
    // R's rows are (R(e_x).comp? ) — instead store rows as R(e_i) transposed.
    const vec3 rx = R(vec3(1, 0, 0)), ry = R(vec3(0, 1, 0)), rz = R(vec3(0, 0, 1));
    // rx/ry/rz are the COLUMNS of R; its rows are (rx.x,ry.x,rz.x), etc.
    P.r0 = {rx.x, ry.x, rz.x};
    P.tx = t.x;
    P.r1 = {rx.y, ry.y, rz.y};
    P.ty = t.y;
    P.r2 = {rx.z, ry.z, rz.z};
    P.tz = t.z;
    return P;
}

// A↔B pair: two rectangles, each transporting to the other.
inline void make_portal_pair(std::vector<GPUPortal>& out, const point3& ca,
                             const vec3& ua, const vec3& va, const point3& cb,
                             const vec3& ub, const vec3& vb) {
    out.push_back(make_portal(ca, ua, va, cb, ub, vb));   // A -> B
    out.push_back(make_portal(cb, ub, vb, ca, ua, va));   // B -> A
}

inline EnvLookup env_lookup(const SceneDesc& desc, bool nee = true) {
    EnvLookup e;
    e.nee = nee;
    if (desc.env && desc.env->valid()) {
        e.texels = desc.env->texels.data();
        e.w = desc.env->w;
        e.h = desc.env->h;
        e.intensity = desc.env_intensity;
        e.yaw_norm = desc.env_yaw_deg * (1.0f / 360.0f);
        if (!desc.env->row_cdf.empty()) {
            e.row_cdf = desc.env->row_cdf.data();
            e.cond_cdf = desc.env->cond_cdf.data();
        }
    }
    return e;
}

// Session J: enumerate scene emitters for area-light NEE. Emissive
// spheres now; emissive mesh triangles join in Stage 2. Selection is
// power-proportional (luminance x surface area), normalized into a CDF
// carried inside the entries. Rebuilt (cheap, derived data) whenever the
// spheres or the mesh change — the same discipline as tri_mat.
// `live_*` (viewer): the re-baked leaf-order arrays after a gizmo edit;
// defaults use the MeshData's own (load-time) arrays. Triangle entries
// ALWAYS precede sphere entries and follow tri_light's ordinal order.
inline std::vector<GPULight> build_light_list(
    const SceneDesc& desc, const std::vector<GPUTriangle>* live_tris = nullptr,
    const std::vector<std::uint32_t>* live_tri_mat = nullptr,
    const std::vector<std::uint32_t>* live_tri_light = nullptr) {
    std::vector<GPULight> out;
    std::vector<double> power;
    if (desc.mesh && desc.mesh->light_tri_count > 0) {
        const std::vector<GPUTriangle>& tris =
            live_tris ? *live_tris : desc.mesh->tris;
        const std::vector<std::uint32_t>& tri_mat =
            live_tri_mat ? *live_tri_mat : desc.mesh->tri_mat;
        const std::vector<std::uint32_t>& tri_light =
            live_tri_light ? *live_tri_light : desc.mesh->tri_light;
        for (std::size_t i = 0; i < tris.size(); ++i) {
            if (tri_light[i] == 0u) continue;
            const GPUTriangle& T = tris[i];
            GPULight L{};
            L.kind = 1;
            L.p0 = T.p0;
            L.e1 = T.e1;
            L.e2 = T.e2;
            L.u0 = T.u0; L.v0 = T.v0;
            L.u1 = T.u1; L.v1 = T.v1;
            L.u2 = T.u2; L.v2 = T.v2;
            L.mat_id = tri_mat[i];
            const float probe =
                tri_emissive_probe(desc.mesh->materials[tri_mat[i]], T) *
                desc.mesh->emissive_scale;
            L.emission = {probe, probe, probe};   // selection only; Le textured
            out.push_back(L);
            const vec3 cr = cross(vec3(T.e1.x, T.e1.y, T.e1.z),
                                  vec3(T.e2.x, T.e2.y, T.e2.z));
            power.push_back(double(probe) * 0.5 * double(cr.length()) +
                            1e-12);
        }
    }
    for (const SphereData& s : desc.spheres) {
        const color& e = s.mat.emission;
        if (e.x <= 0.0f && e.y <= 0.0f && e.z <= 0.0f) continue;
        GPULight L{};
        L.kind = 0;
        L.p0 = {s.center.x, s.center.y, s.center.z};
        L.radius = s.radius;
        L.emission = {e.x, e.y, e.z};
        out.push_back(L);
        const double lum = 0.2126 * e.x + 0.7152 * e.y + 0.0722 * e.z;
        power.push_back(lum * double(s.radius) * double(s.radius));
    }
    double total = 0.0;
    for (double p : power) total += p;
    if (total <= 0.0) {
        out.clear();
        return out;
    }
    double run = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        run += power[i];
        out[i].sel_cdf = float(run / total);
        out[i].sel_pdf = float(power[i] / total);
    }
    out.back().sel_cdf = 1.0f;
    return out;
}

inline SceneDesc build_scene_desc(
    std::shared_ptr<const MeshData> mesh = nullptr) {
    SceneDesc desc;
    desc.mesh = std::move(mesh);
    RNG rng(197001ULL);
    const auto add = [&desc](const point3& c, float r, const Material& m) {
        desc.spheres.push_back({c, r, m});
    };

    // Ground: one huge sphere, so its top is locally flat at y=0.
    add(point3(0.0f, -1000.0f, 0.0f), 1000.0f,
        Material::lambertian(color(0.58f, 0.57f, 0.55f)));

    // Heroes. Sphere-only scene: matte red / clear glass / metal, left to
    // right. With a mesh, the model takes center stage: glass moves to the
    // left slot, red is dropped. The clearance positions below stay the
    // same either way, so the random field is identical in both scenes.
    desc.spheres.back().name = "Ground";
    const point3 hero[3] = {{-3.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {3.0f, 1.0f, 0.0f}};
    if (desc.mesh) {
        add(hero[0], 1.0f, Material::dielectric(1.5f));
        desc.spheres.back().name = "Glass Hero";
        desc.mesh_name = "Damaged Helmet";
    } else {
        add(hero[0], 1.0f, Material::lambertian(color(0.70f, 0.13f, 0.10f)));
        desc.spheres.back().name = "Matte Hero";
        add(hero[1], 1.0f, Material::dielectric(1.5f));
        desc.spheres.back().name = "Glass Hero";
    }
    add(hero[2], 1.0f, Material::metal(color(0.88f, 0.86f, 0.82f), 0.02f));
    desc.spheres.back().name = "Metal Hero";

    // A warm lamp among the field — the only non-sky light source.
    const point3 lamp_pos(2.0f, 0.38f, 3.1f);
    add(lamp_pos, 0.38f, Material::emissive(color(7.0f, 4.2f, 1.8f)));
    desc.spheres.back().name = "Lamp";

    // Field of small spheres on a jittered grid.
    for (int a = -5; a <= 5; ++a) {
        for (int b = -4; b <= 3; ++b) {
            const float radius = 0.14f + 0.08f * rng.next_float();
            const point3 c(a + 0.65f * rng.next_float(), radius,
                           b + 0.65f * rng.next_float());

            // Keep clear of the heroes and the lamp.
            bool blocked = false;
            for (const point3& h : hero) {
                const float dx = c.x - h.x, dz = c.z - h.z;
                if (dx * dx + dz * dz < 1.45f * 1.45f) blocked = true;
            }
            {
                const float dx = c.x - lamp_pos.x, dz = c.z - lamp_pos.z;
                if (dx * dx + dz * dz < 0.95f * 0.95f) blocked = true;
            }
            if (blocked) continue;

            const float pick = rng.next_float();
            Material mat;
            if (pick < 0.62f) {
                // Squaring biases toward saturated darks, so the palette
                // reads as colorful rather than pastel.
                const color albedo(rng.next_float() * rng.next_float(),
                                   rng.next_float() * rng.next_float(),
                                   rng.next_float() * rng.next_float());
                mat = Material::lambertian(albedo);
            } else if (pick < 0.85f) {
                const color albedo(0.5f + 0.5f * rng.next_float(),
                                   0.5f + 0.5f * rng.next_float(),
                                   0.5f + 0.5f * rng.next_float());
                const float fuzz = 0.35f * rng.next_float() * rng.next_float();
                mat = Material::metal(albedo, fuzz);
            } else {
                mat = Material::dielectric(1.5f);
            }
            add(c, radius, mat);
        }
    }
    return desc;
}

inline Scene make_scene(const SceneDesc& desc) {
    Scene scene;
    int next_light =
        desc.mesh ? int(desc.mesh->light_tri_count) : 0;   // tris first
    for (const SphereData& s : desc.spheres) {
        const color& e = s.mat.emission;
        const bool lit = e.x > 0.0f || e.y > 0.0f || e.z > 0.0f;
        // Ids follow build_light_list's enumeration order exactly.
        scene.add(std::make_unique<Sphere>(s.center, s.radius, s.mat,
                                           lit ? next_light : -1));
        if (lit) ++next_light;
    }
    // Mesh LAST: the sphere loop's running-closest then prunes the BVH.
    // The GPU kernel mirrors this order (spheres, then bvh_hit seeded with
    // the sphere winner's t).
    if (desc.mesh) scene.add(std::make_unique<Mesh>(desc.mesh));
    return scene;
}

inline std::vector<GPUSphere> flatten_scene(const SceneDesc& desc) {
    std::vector<GPUSphere> out;
    out.reserve(desc.spheres.size());
    int next_light =
        desc.mesh ? int(desc.mesh->light_tri_count) : 0;   // tris first
    for (const SphereData& s : desc.spheres) {
        GPUSphere g{};
        g.center = {s.center.x, s.center.y, s.center.z};
        g.radius = s.radius;
        g.base_color = {s.mat.base_color.x, s.mat.base_color.y,
                        s.mat.base_color.z};
        g.metallic = s.mat.metallic;
        g.emission = {s.mat.emission.x, s.mat.emission.y, s.mat.emission.z};
        g.ior = s.mat.ior;
        g.roughness = s.mat.roughness;
        g.transmission = s.mat.transmission;
        // Session J: pad[0] = light-list index + 1 (0 = not an emitter);
        // same enumeration order as build_light_list / make_scene.
        const color& e = s.mat.emission;
        if (e.x > 0.0f || e.y > 0.0f || e.z > 0.0f) {
            g.pad[0] = pt_uint(next_light + 1);
            ++next_light;
        }
        out.push_back(g);
    }
    return out;
}

// Session B validation scene: the standard GGX check — roughness sweeps
// left→right (0→1), metallic sweeps front→back (0→1), one red base color
// so metal tinting is obvious, a lamp for a readable specular highlight.
inline SceneDesc build_grid_desc() {
    SceneDesc desc;
    desc.spheres.push_back({point3(0.0f, -1000.0f, 0.0f), 1000.0f,
                            Material::lambertian(color(0.55f, 0.55f, 0.55f)),
                            "Ground"});
    const int N = 6;
    for (int im = 0; im < N; ++im) {
        for (int ir = 0; ir < N; ++ir) {
            SphereData s;
            s.radius = 0.42f;
            s.center = point3((float(ir) - 0.5f * (N - 1)) * 1.05f, 0.42f,
                              (float(im) - 0.5f * (N - 1)) * 1.05f);
            s.mat.base_color = color(0.80f, 0.30f, 0.25f);
            s.mat.metallic = float(im) / float(N - 1);
            s.mat.roughness = float(ir) / float(N - 1);
            char name[48];
            std::snprintf(name, sizeof name, "r=%.1f m=%.1f",
                          s.mat.roughness, s.mat.metallic);
            s.name = name;
            desc.spheres.push_back(s);
        }
    }
    SphereData lamp;
    lamp.center = point3(5.0f, 4.0f, 5.0f);
    lamp.radius = 1.0f;
    lamp.mat = Material::emissive(color(14.0f, 13.0f, 12.0f));
    lamp.name = "Lamp";
    desc.spheres.push_back(lamp);
    return desc;
}
