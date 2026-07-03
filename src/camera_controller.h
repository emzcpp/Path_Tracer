#pragma once

// Fly-style camera for the interactive viewer: position + yaw/pitch, driven
// by key/mouse input on the UI thread. A value snapshot is handed to the
// render thread, which turns it into the existing Camera via make_camera().
// Pure C++ — no windowing types leak in here.

#include <algorithm>
#include <cmath>

#include "camera.h"
#include "vec3.h"

struct InputState {
    bool fwd = false, back = false, left = false, right = false;
    bool up = false, down = false;
    bool boost = false;   // shift

    bool any_movement() const { return fwd || back || left || right || up || down; }
};

struct FlyCamera {
    point3 pos;
    float yaw   = 0.0f;        // radians around +Y; 0 looks down -Z
    float pitch = 0.0f;        // radians; clamped to ±89° so 'up' never flips
    float move_speed = 2.0f;   // world units per second (scroll to adjust)

    static FlyCamera from_look_at(const point3& p, const point3& target) {
        const vec3 d = normalize(target - p);
        FlyCamera c;
        c.pos = p;
        c.pitch = std::asin(std::clamp(d.y, -1.0f, 1.0f));
        c.yaw = std::atan2(-d.x, -d.z);
        return c;
    }

    vec3 forward() const {
        const float cp = std::cos(pitch);
        return {-std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp};
    }

    // Horizon-parallel right vector (world-up cross), so strafing never
    // gains altitude even while looking up or down.
    vec3 right() const {
        return normalize(cross(forward(), vec3(0.0f, 1.0f, 0.0f)));
    }

    void apply_move(const InputState& in, float dt) {
        vec3 dir(0.0f);
        if (in.fwd)   dir += forward();
        if (in.back)  dir -= forward();
        if (in.right) dir += right();
        if (in.left)  dir -= right();
        if (in.up)    dir += vec3(0.0f, 1.0f, 0.0f);
        if (in.down)  dir -= vec3(0.0f, 1.0f, 0.0f);
        if (dir.length_squared() < 1e-12f) return;
        pos += normalize(dir) * (move_speed * (in.boost ? 4.0f : 1.0f) * dt);
    }

    void apply_look(float dx, float dy) {
        constexpr float sens = 0.0035f;   // radians per point of drag
        yaw -= dx * sens;
        pitch = std::clamp(pitch - dy * sens, -1.5533f, 1.5533f);   // ±89°
    }

    Camera make_camera(float vfov_deg, float aspect) const {
        return Camera(pos, pos + forward(), vec3(0.0f, 1.0f, 0.0f), vfov_deg,
                      aspect);
    }
};
