#pragma once

#include "vec3.h"

struct Ray {
    point3 origin;
    vec3   dir;   // not necessarily normalized; intersection code handles that

    constexpr Ray() = default;
    constexpr Ray(const point3& o, const vec3& d) : origin(o), dir(d) {}

    constexpr point3 at(float t) const { return origin + t * dir; }
};
