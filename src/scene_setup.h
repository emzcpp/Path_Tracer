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
};

inline EnvLookup env_lookup(const SceneDesc& desc) {
    EnvLookup e;
    if (desc.env && desc.env->valid()) {
        e.texels = desc.env->texels.data();
        e.w = desc.env->w;
        e.h = desc.env->h;
        e.intensity = desc.env_intensity;
        e.yaw_norm = desc.env_yaw_deg * (1.0f / 360.0f);
    }
    return e;
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
    for (const SphereData& s : desc.spheres) {
        scene.add(std::make_unique<Sphere>(s.center, s.radius, s.mat));
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
