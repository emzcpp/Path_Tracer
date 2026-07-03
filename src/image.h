#pragma once

// Framebuffer + file output. Pixels are stored as LINEAR radiance (floats).
// Gamma correction happens exactly once, at write time — never mix linear
// math with gamma-encoded values.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "vec3.h"

// Why gamma: the framebuffer holds physically linear radiance, but displays
// expect sRGB-encoded values — they apply roughly a 2.2 power curve on the
// way to the screen. Writing linear values directly would make the image
// look far too dark in the midtones. So we encode with the inverse: ^(1/2.2).
// Shared by file output and the live viewer so saved PNGs match the screen.
inline unsigned char encode_channel(float linear) {
    const float v = std::pow(std::max(linear, 0.0f), 1.0f / 2.2f);
    return static_cast<unsigned char>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

class Image {
public:
    Image(int width, int height)
        : width_(width), height_(height), pixels_(size_t(width) * height) {}

    int width() const  { return width_; }
    int height() const { return height_; }

    color& at(int x, int y)             { return pixels_[size_t(y) * width_ + x]; }
    const color& at(int x, int y) const { return pixels_[size_t(y) * width_ + x]; }

    // Both writers apply gamma 1/2.2 and clamp to [0,255].
    bool write_ppm(const std::string& path) const;
    bool write_png(const std::string& path) const;

private:
    std::vector<unsigned char> to_srgb_bytes() const;

    int width_;
    int height_;
    std::vector<color> pixels_;
};
