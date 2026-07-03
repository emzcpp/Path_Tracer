#pragma once

#include <cmath>

#include "hittable.h"

class Sphere : public Hittable {
public:
    Sphere(const point3& center, float radius, const Material& mat)
        : center_(center), radius_(radius), mat_(mat) {}

    // Ray–sphere: solve |o + t*d - c|^2 = r^2, a quadratic in t.
    // With oc = o - c and half_b = oc·d, the discriminant simplifies to
    // half_b^2 - a*c (the usual /2 and *4 factors cancel — fewer ops, same roots).
    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        const vec3 oc = r.origin - center_;
        const float a = r.dir.length_squared();
        const float half_b = dot(oc, r.dir);
        const float c = oc.length_squared() - radius_ * radius_;

        const float discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0f) return false;
        const float sqrt_d = std::sqrt(discriminant);

        // Nearest root inside [t_min, t_max]; fall back to the far root so
        // rays starting inside the sphere still hit its back wall.
        float root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > t_max) return false;
        }

        rec.t = root;
        rec.p = r.at(root);
        rec.set_face_normal(r, (rec.p - center_) / radius_);
        rec.mat = mat_;
        return true;
    }

private:
    point3   center_;
    float    radius_;
    Material mat_;
};
