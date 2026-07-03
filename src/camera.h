#pragma once

// Pinhole camera. Precomputes the image-plane basis once; per-ray work is
// two multiply-adds, so this is trivially portable to a GPU constant buffer.

#include <cmath>

#include "ray.h"
#include "vec3.h"

class Camera {
public:
    Camera() = default;   // degenerate; overwritten before first use

    Camera(const point3& look_from, const point3& look_at, const vec3& vup,
           float vfov_deg, float aspect) {
        // Vertical fov defines the image plane height at unit focal distance.
        const float theta = vfov_deg * 3.14159265358979323846f / 180.0f;
        const float half_h = std::tan(theta / 2.0f);
        const float half_w = aspect * half_h;

        // Orthonormal basis: w points *backwards* (from target to camera),
        // u is camera-right, v is camera-up.
        const vec3 w = normalize(look_from - look_at);
        const vec3 u = normalize(cross(vup, w));
        const vec3 v = cross(w, u);

        origin_     = look_from;
        horizontal_ = 2.0f * half_w * u;
        vertical_   = 2.0f * half_h * v;
        lower_left_ = origin_ - half_w * u - half_h * v - w;
    }

    // s,t in [0,1): s scans left→right, t scans bottom→top.
    Ray get_ray(float s, float t) const {
        return Ray(origin_, lower_left_ + s * horizontal_ + t * vertical_ - origin_);
    }

    // Basis accessors so the GPU camera struct is filled from the exact
    // same derivation instead of re-deriving the trig.
    const point3& origin() const     { return origin_; }
    const point3& lower_left() const { return lower_left_; }
    const vec3& horizontal() const   { return horizontal_; }
    const vec3& vertical() const     { return vertical_; }

private:
    point3 origin_;
    point3 lower_left_;
    vec3   horizontal_;
    vec3   vertical_;
};
