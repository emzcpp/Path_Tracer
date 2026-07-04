#pragma once

#include "ray.h"
#include "vec3.h"

// Plain-data material, stored BY VALUE in the hit record — no pointers, no
// virtual dispatch. Principled metallic-roughness parameterization (the
// glTF-standard model): one GGX surface covers diffuse through mirror;
// transmission = 1 selects the delta dielectric (glass) lobe instead.
struct Material {
    color base_color {0.8f, 0.8f, 0.8f};
    color emission {0.0f, 0.0f, 0.0f};
    float metallic = 0.0f;
    float roughness = 1.0f;      // perceptual; alpha = roughness^2 inside
    float ior = 1.5f;            // drives dielectric F0 and the glass lobe
    float transmission = 0.0f;   // 1 = glass (delta dielectric for now)

    // Factories keep the old call-site names.
    static Material lambertian(const color& a) {
        Material m; m.base_color = a; return m;
    }
    static Material metal(const color& a, float roughness) {
        Material m; m.base_color = a; m.metallic = 1.0f;
        m.roughness = roughness; return m;
    }
    static Material dielectric(float ior) {
        Material m; m.base_color = color(1.0f, 1.0f, 1.0f);
        m.transmission = 1.0f; m.ior = ior; return m;
    }
    static Material emissive(const color& e) {
        Material m; m.base_color = color(0.0f, 0.0f, 0.0f);
        m.emission = e; return m;
    }
};

struct HitRecord {
    // Session J: index into the area-light list when the hit surface IS a
    // listed emitter (-1 otherwise). Drives NEE double-count suppression
    // and the Stage-3 MIS weight.
    int light_id = -1;
    point3   p;            // hit point in world space
    vec3     normal;       // always opposes the incoming ray (see below)
    float    t = 0.0f;     // ray parameter at the hit
    bool     front_face = false;
    Material mat;

    // Store the normal against the ray so shading never has to re-derive
    // which side of the surface it's on; front_face remembers the original
    // orientation (matters later for glass, harmless for diffuse).
    void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = dot(r.dir, outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

// Traversal interface. This is the ONLY virtual dispatch in the renderer,
// and it lives outside the shading math on purpose: the Metal port replaces
// it with a loop over a flat sphere buffer, leaving integrator/material
// code untouched.
class Hittable {
public:
    virtual ~Hittable() = default;
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;

    // Session H (NEE shadow rays): any-hit occlusion. Default falls back
    // to closest-hit; Mesh overrides with an early-out BVH traversal.
    virtual bool occluded(const Ray& r, float t_min, float t_max) const {
        HitRecord tmp;
        return hit(r, t_min, t_max, tmp);
    }
};
