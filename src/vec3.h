#pragma once

// Core 3-vector. float (not double) on purpose: Metal compute kernels are
// float32, and keeping the CPU reference in float means the eventual GPU
// port produces bit-comparable images instead of subtly different ones.

#include <cmath>

struct vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    constexpr vec3() = default;
    constexpr vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    constexpr explicit vec3(float s) : x(s), y(s), z(s) {}

    constexpr vec3 operator-() const { return {-x, -y, -z}; }

    constexpr vec3& operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    constexpr vec3& operator-=(const vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    constexpr vec3& operator*=(const vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    constexpr vec3& operator*=(float s)       { x *= s;   y *= s;   z *= s;   return *this; }
    constexpr vec3& operator/=(float s)       { return *this *= (1.0f / s); }

    float length() const { return std::sqrt(length_squared()); }
    constexpr float length_squared() const { return x * x + y * y + z * z; }

    // True when all components are ~0; used to reject degenerate scatter
    // directions (normal + random unit vector can nearly cancel).
    bool near_zero() const {
        constexpr float eps = 1e-6f;
        return std::fabs(x) < eps && std::fabs(y) < eps && std::fabs(z) < eps;
    }
};

// Semantic aliases: same layout, different intent.
using point3 = vec3;
using color  = vec3;

constexpr vec3 operator+(vec3 a, const vec3& b) { return a += b; }
constexpr vec3 operator-(vec3 a, const vec3& b) { return a -= b; }
constexpr vec3 operator*(vec3 a, const vec3& b) { return a *= b; }   // component-wise (color filtering)
constexpr vec3 operator*(vec3 a, float s)       { return a *= s; }
constexpr vec3 operator*(float s, vec3 a)       { return a *= s; }
constexpr vec3 operator/(vec3 a, float s)       { return a /= s; }

constexpr float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr vec3 cross(const vec3& a, const vec3& b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline vec3 normalize(const vec3& v) { return v / v.length(); }

constexpr vec3 lerp(const vec3& a, const vec3& b, float t) {
    return (1.0f - t) * a + t * b;
}

// Mirror reflection about a unit normal: v bounces off the plane, keeping
// its tangential component and flipping the normal one.
constexpr vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

// Snell's law refraction. uv must be unit length; eta = n_in / n_out.
// Decomposes the refracted ray into components perpendicular and parallel
// to the normal: the perpendicular part scales by eta (Snell), the parallel
// part is whatever keeps the result unit length.
inline vec3 refract(const vec3& uv, const vec3& n, float eta) {
    const float cos_theta = std::fmin(dot(-uv, n), 1.0f);
    const vec3 r_perp = eta * (uv + cos_theta * n);
    const vec3 r_parallel =
        -std::sqrt(std::fabs(1.0f - r_perp.length_squared())) * n;
    return r_perp + r_parallel;
}
