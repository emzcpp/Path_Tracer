#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

// Third-party header; macOS deprecation warnings in it aren't ours to fix.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma clang diagnostic pop

std::vector<unsigned char> Image::to_srgb_bytes() const {
    std::vector<unsigned char> bytes;
    bytes.reserve(pixels_.size() * 3);
    for (const color& c : pixels_) {
        bytes.push_back(encode_channel(c.x));
        bytes.push_back(encode_channel(c.y));
        bytes.push_back(encode_channel(c.z));
    }
    return bytes;
}

bool Image::write_ppm(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << width_ << ' ' << height_ << "\n255\n";
    const auto bytes = to_srgb_bytes();
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return bool(out);
}

bool Image::write_png(const std::string& path) const {
    const auto bytes = to_srgb_bytes();
    return stbi_write_png(path.c_str(), width_, height_, 3, bytes.data(),
                          width_ * 3) != 0;
}
