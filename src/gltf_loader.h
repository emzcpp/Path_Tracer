#pragma once

// glTF loader (Session D: built on cgltf). Walks the full node hierarchy,
// composes world transforms, concatenates all triangle primitives, reads
// or computes per-vertex tangents, and decodes the PBR texture set
// (baseColor + metallicRoughness + emissive + normal; JPEG or PNG,
// embedded or file-relative). Geometry is baked to world space at load.
//
// Color space (the #1 glTF bug, handled here once): baseColor and
// emissive are sRGB — decoded to linear at load via the pipeline's pure
// pow-2.2; metallicRoughness and normal maps are LINEAR data and are
// stored untouched.
//
// Scope note: materials bind as ONE texture set (multi-material meshes
// take the first textured primitive's material; proper per-primitive sets
// need Metal argument buffers — a later session).

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "bvh.h"
#include "kernel_types.h"
#include "texture.h"

struct MeshPlacement {
    float target_height = 2.2f;   // world-Y extent after transform
    // Spin about +Y (after the node transform). The helmet's face points
    // -Z in node space; camera sits at atan2(7.4, 5.8) ~ 52 deg, +180 to
    // face it.
    float yaw_deg = 232.0f;
};

// Session I: one glTF material's texture set. Every channel keeps the
// established color-space rules (base/emissive sRGB-decoded at load via
// pure pow-2.2; mr and normal LINEAR, untouched) and factor baking.
struct MeshMaterial {
    Texture16 base;       // linear (sRGB-decoded), factors baked in
    Texture16 mr;         // linear: G = roughness, B = metallic
    Texture16 emissive;   // linear (sRGB-decoded), emissiveFactor baked in
    Texture16 normal;     // LINEAR tangent-space normal map (never sRGB)
};

struct MeshData {
    std::vector<GPUTriangle> tris;   // world space, BVH leaf order
    std::vector<BVHNode> nodes;
    // Session I (multi-material): flat material table + a per-triangle
    // index into it, kept in BVH LEAF ORDER alongside tris (build_bvh
    // permutes both together — including on every gizmo re-bake).
    std::vector<MeshMaterial> materials;
    std::vector<std::uint32_t> tri_mat;
    float emissive_scale = 1.0f;   // user knob on top of the baked factor

    struct Info {           // --mesh-info diagnostics
        std::size_t vert_count = 0;
        std::size_t index_count = 0;
        std::size_t material_count = 0;
        std::size_t texture_bytes = 0;   // decoded ushort4 payload
        float uv_min[2]{}, uv_max[2]{};
        float pre_min[3]{}, pre_max[3]{};     // mesh space
        float post_min[3]{}, post_max[3]{};   // world space, after placement
        BvhStats bvh;
    } info;
};

// nullptr + `error` filled on any parse/decode failure.
std::shared_ptr<const MeshData> load_glb(const std::string& path,
                                         const MeshPlacement& placement,
                                         std::string& error);

// Session F: equirectangular HDRI environment map. Radiance .hdr data is
// ALREADY linear — stored untouched (the inverse of the baseColor rule:
// that was sRGB and needed decoding; this must not be).
struct EnvMap {
    int w = 0, h = 0;
    std::vector<float> texels;   // RGBA float, linear radiance

    // Session H: importance-sampling distribution over the equirect image,
    // weighted by luminance x sin(theta) (rows near the poles subtend less
    // solid angle). row_cdf: marginal over rows (h entries, last == 1).
    // cond_cdf: per-row normalized column CDFs (h*w, row-major). A small
    // support floor keeps every texel sampleable (bilinear shading can
    // read radiance from texels whose own luminance is zero).
    std::vector<float> row_cdf;
    std::vector<float> cond_cdf;

    bool valid() const { return w > 0 && h > 0; }
};

std::shared_ptr<const EnvMap> load_hdr(const std::string& path,
                                       std::string& error);
