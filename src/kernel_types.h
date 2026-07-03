#ifndef PT_KERNEL_TYPES_H
#define PT_KERNEL_TYPES_H

// Types shared between C++ host code and the MSL kernel. This header is
// compiled by BOTH languages: the CMake embed step inlines it into the
// shader source, so keep it types-only (MSL functions would need address-
// space qualifiers) and avoid anything with layout ambiguity:
//   - explicit-scalar pt_float3, never MSL float3 (16-byte alignment)
//   - plain enum constants, not enum class (no underlying-type doubt)
//   - 64-bit literals elsewhere use UL, not ULL (MSL has no long long)

#ifndef __METAL_VERSION__
#include <cstdint>
typedef std::uint32_t pt_uint;
#else
typedef uint pt_uint;
#endif

struct pt_float3 {
    float x, y, z;
};

// 64 bytes = one cacheline, four float4 loads. Metallic-roughness
// parameterization (Session B): one GGX surface, transmission = 1 selects
// the delta glass lobe.
struct GPUSphere {
    pt_float3 center;     float radius;
    pt_float3 base_color; float metallic;
    pt_float3 emission;   float ior;
    float roughness;      float transmission;
    pt_uint pad[2];
};

struct GPUCamera {
    pt_float3 origin;
    pt_float3 lower_left;
    pt_float3 horizontal;
    pt_float3 vertical;
};

struct PassUniforms {
    GPUCamera cam;
    pt_uint width, height;
    pt_uint pass_base;      // first pass index in this dispatch (RNG seed)
    pt_uint pass_count;     // K passes looped inside the kernel
    pt_uint sphere_count;
    pt_uint max_depth;
    // Session F: equirect HDRI environment. env_w == 0 -> gradient dome.
    pt_uint env_w, env_h;
    float env_intensity;
    float env_yaw_norm;     // yaw as a u offset (yaw / 2pi)
    float clamp_indirect;   // firefly clamp; 0 = off
    // Session G (responsiveness): dispatches may cover a row slice of the
    // frame so no single command buffer runs long. Seeds depend only on
    // (pixel, pass), so the accumulated result is bit-identical no matter
    // how work is sliced.
    pt_uint row_offset;
    // Session H: 1 = env NEE+MIS, 0 = brute-force ground truth.
    pt_uint env_nee;
    pt_uint pad_h0, pad_h1, pad_h2;
};

struct ResolveUniforms {
    pt_uint accum_w, accum_h;   // accumulation dims (may be preview-sized)
    pt_uint out_w, out_h;       // drawable dims (always full res)
    pt_uint pass_total;         // completed full passes
    pt_uint rows_plus1;         // rows [0, rows_plus1) carry one extra pass
};

// ---- Triangle mesh (glTF import) ----

// 32 bytes = two float4 loads.
struct BVHNode {
    pt_float3 mn;  pt_uint left_or_first;  // internal: left child (right = +1);
    pt_float3 mx;  pt_uint tri_count;      // 0 = internal; >0 = leaf, tris
};                                          //   [left_or_first, +tri_count)

// 144 bytes = nine 16-byte rows. UVs/handedness ride the pad lanes; BVH
// traversal touches only rows 0-2 (p0/e1/e2); normals, UVs, and tangents
// are fetched once, for the single winning triangle. Rows 6-8 (Session D)
// carry per-vertex tangents + the glTF .w handedness sign.
struct GPUTriangle {
    pt_float3 p0;  float u0;
    pt_float3 e1;  float v0;   // e1 = p1 - p0 (world space, baked at load)
    pt_float3 e2;  float u1;   // e2 = p2 - p0
    pt_float3 n0;  float v1;
    pt_float3 n1;  float u2;
    pt_float3 n2;  float v2;
    pt_float3 t0;  float w0;
    pt_float3 t1;  float w1;
    pt_float3 t2;  float w2;
};

// Separate from PassUniforms so the sphere path's layout stays untouched.
// Session I: per-set texture dims moved into GPUMaterialArgs.
struct MeshUniforms {
    pt_uint has_mesh, tri_count, node_count, mat_count;
    float   emissive_scale;
    pt_uint pad0, pad1, pad2;
};

// Session I: one material's texture set, as an entry in the bindless
// argument-buffer table. The MSL side holds device pointers; the C++ side
// writes the buffers' gpuAddress values (Metal 3) into the same 8-byte
// slots. Addressing changes, FILTERING does not: the kernel dereferences
// these exactly like the old single-set bindings, so the hand-matched
// bilinear/color-space code is untouched. A dim of 0 marks an absent map.
struct GPUMaterialArgs {
#ifdef __METAL_VERSION__
    device const ushort* base;
    device const ushort* mr;
    device const ushort* emis;
    device const ushort* norm;
#else
    unsigned long long base, mr, emis, norm;   // MTLBuffer.gpuAddress
#endif
    pt_uint base_w, base_h, mr_w, mr_h;
    pt_uint emis_w, emis_h, norm_w, norm_h;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(pt_float3) == 12, "pt_float3 must be 12 bytes");
static_assert(sizeof(GPUSphere) == 64, "GPUSphere must be one cacheline");
static_assert(sizeof(GPUCamera) == 48, "GPUCamera layout drifted");
static_assert(sizeof(PassUniforms) == 112, "PassUniforms layout drifted");
static_assert(sizeof(ResolveUniforms) == 24, "ResolveUniforms layout drifted");
static_assert(sizeof(BVHNode) == 32, "BVHNode must be two float4 loads");
static_assert(sizeof(GPUTriangle) == 144, "GPUTriangle must be nine 16B rows");
static_assert(sizeof(MeshUniforms) == 32, "MeshUniforms layout drifted");
static_assert(sizeof(GPUMaterialArgs) == 64,
              "GPUMaterialArgs must be four 8B pointers + eight uints");
#endif

#endif // PT_KERNEL_TYPES_H
