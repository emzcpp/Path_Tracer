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
    // Session J: number of area lights (emissive spheres/triangles) in the
    // light list; 0 = none (area NEE off).
    pt_uint light_count;
    // Session K: RIS candidate count per light slot (partitioned pipeline)
    // and the reuse toggles (Stage 2 temporal, Stage 3 spatial).
    pt_uint restir_m;
    pt_uint restir_temporal;
    pt_uint restir_spatial;
    // Consolidation session: the remaining ReSTIR knobs, made runtime.
    pt_uint restir_k;        // spatial neighbors (<= PT_RESTIR_NEIGHBORS)
    pt_uint restir_radius;   // spatial radius, pixels
    pt_uint restir_mcap;     // temporal M-cap multiplier (x M)
    // v1.2 spectral rendering (hero-wavelength). spectral == 0 -> RGB path,
    // byte-identical to pre-feature. dispersion_b feeds the Cauchy IOR.
    pt_uint spectral;
    float dispersion_b;
    pt_uint pad_s0, pad_s1;
};

// v1.1 denoiser (display-only post-process; OUTSIDE the parity surface —
// nothing here reads back into accumulation).
struct DenoiseUniforms {
    pt_uint accum_w, accum_h, out_w, out_h;   // same mapping as resolve
    pt_uint pass_total, rows_plus1;           // accum normalization
    pt_uint step;      // a-trous dilation for this iteration
    pt_uint aov;       // 0 off | 1 normal | 2 depth | 3 albedo | 4 illum
    float sigma_n, sigma_z, sigma_l;          // edge-stopping params
    float alpha;       // fade: 0 = raw accumulation, 1 = fully denoised
    pt_uint wipe_x;    // drawable-x split; pixels left of it show RAW
    pt_uint pad0, pad1, pad2;
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

// Session J: one scene emitter for area-light NEE. Shared POD, built on
// host, bit-identical arrays on both backends (the BVH discipline).
// kind 0 = sphere (p0 = center, radius; e1/e2/uv unused).
// kind 1 = triangle (p0/e1/e2 copied from the LEAF-ORDER GPUTriangle at
// list-build time — rebuilt on every BVH re-bake; uv lanes + mat_id let
// the estimator evaluate the actual textured Le at the sampled point).
// sel_cdf is the cumulative power-proportional selection distribution
// (binary-searched); sel_pdf this light's own selection probability.
struct GPULight {
    pt_float3 p0;       float radius;
    pt_float3 e1;       float u0;
    pt_float3 e2;       float v0;
    pt_float3 emission; float sel_cdf;   // sphere Le / tri MEAN Le (selection)
    float u1, v1, u2, v2;
    pt_uint kind;       pt_uint mat_id;
    float sel_pdf;      pt_uint pad0;
};

// Session K Stage 2: per-pixel persistent reservoirs (temporal reuse).
// One slot per light strategy. The sample representation is chosen so a
// sample is a FIXED object independent of the (jittered) receiving
// surface: env slot stores the direction (solid-angle measure); the area
// slot stores the light id plus either the world-space point (spheres —
// cone samples aren't portable) or the barycentrics (triangles), with
// its RIS bookkeeping in the AREA measure so re-evaluating the target at
// a new surface applies the correct cos/r^2 Jacobian. W and M follow the
// standard reservoir form; M == 0 marks an empty/invalidated slot.
// Stage 3 spatial-reuse shape, shared by both backends.
enum { PT_RESTIR_NEIGHBORS = 3, PT_RESTIR_RADIUS = 16 };

struct ReSTIRSlot {
    float ax, ay, az;     // env: direction | sphere: point | tri: (bu,bv,-)
    float W;
    float M;
    pt_uint light_id_p1;  // 0 for the env slot
    pt_uint pad0, pad1;
};

struct ReSTIRPixel {
    ReSTIRSlot env_slot;
    ReSTIRSlot area_slot;
    pt_float3 prev_normal;   // similarity gate for history validity
    float prev_t;
    pt_uint pad[4];
};

// Session K (ReSTIR Stage 0.5): per-pixel primary-hit record. Written by
// the g_primary phase, consumed by the direct phase (vertex-0 lighting)
// and the indirect continuation. rng_lo/hi carry the PCG stream state
// across phases so every draw lands exactly where the monolithic kernel
// puts it (the stream increment derives from the pixel id). t < 0 marks
// a primary miss.
struct GBufferPx {
    pt_float3 pos;        float t;
    pt_float3 normal;     pt_uint flags;      // bit0: front_face
    pt_float3 rd;         float roughness;    // primary ray direction
    pt_float3 base_color; float metallic;
    pt_float3 emission;   float ior;
    float transmission;   pt_uint rng_lo, rng_hi;
    pt_uint light_id_p1;                      // hit emitter id + 1 (0 = none)
};

#ifndef __METAL_VERSION__
static_assert(sizeof(pt_float3) == 12, "pt_float3 must be 12 bytes");
static_assert(sizeof(GPUSphere) == 64, "GPUSphere must be one cacheline");
static_assert(sizeof(GPUCamera) == 48, "GPUCamera layout drifted");
static_assert(sizeof(PassUniforms) == 144, "PassUniforms layout drifted");
static_assert(sizeof(ResolveUniforms) == 24, "ResolveUniforms layout drifted");
static_assert(sizeof(DenoiseUniforms) == 64, "DenoiseUniforms layout drifted");
static_assert(sizeof(BVHNode) == 32, "BVHNode must be two float4 loads");
static_assert(sizeof(GPUTriangle) == 144, "GPUTriangle must be nine 16B rows");
static_assert(sizeof(MeshUniforms) == 32, "MeshUniforms layout drifted");
static_assert(sizeof(GPULight) == 96, "GPULight must be six 16B rows");
static_assert(sizeof(GBufferPx) == 96, "GBufferPx must be six 16B rows");
static_assert(sizeof(ReSTIRSlot) == 32, "ReSTIRSlot layout drifted");
static_assert(sizeof(ReSTIRPixel) == 96, "ReSTIRPixel layout drifted");
static_assert(sizeof(GPUMaterialArgs) == 64,
              "GPUMaterialArgs must be four 8B pointers + eight uints");
#endif

#endif // PT_KERNEL_TYPES_H
