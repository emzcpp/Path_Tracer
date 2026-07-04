#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_renderer.h"
#include "image.h"

#include "pathtrace_msl.h"   // generated: pt_msl_source[]

namespace {

// Watchdog guard: macOS kills command buffers after a few seconds of GPU
// time. At ~1-4 ms per full-res pass, 64 passes per command buffer leaves
// two orders of magnitude of headroom.
constexpr int kMaxPassesPerCommandBuffer = 64;

id<MTLLibrary> compile_library(id<MTLDevice> device, std::string& error) {
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Safe math for CPU parity: fast math reassociates and assumes no NaN,
    // which would make the GPU image diverge from the reference for no
    // performance we care about.
#if defined(MAC_OS_VERSION_15_0)
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
        opts.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else
#endif
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }

    NSError* err = nil;
    NSString* src = [NSString
        stringWithUTF8String:reinterpret_cast<const char*>(pt_msl_source)];
    id<MTLLibrary> lib = [device newLibraryWithSource:src
                                              options:opts
                                                error:&err];
    if (!lib) {
        error = "MSL compile failed: ";
        error += err ? err.localizedDescription.UTF8String : "(no error info)";
    }
    return lib;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device,
                                          id<MTLLibrary> lib, NSString* name,
                                          std::string& error) {
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) {
        error = std::string("kernel not found: ") + name.UTF8String;
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        error = std::string("pipeline failed for ") + name.UTF8String + ": " +
                (err ? err.localizedDescription.UTF8String : "(no error info)");
    }
    return pso;
}

// C++ mirror of the MSL RasterUniforms (float4x4 = 16 floats col-major).
struct RasterUniformsGPU {
    float vp[16];
    float cam_pos[4];
    std::uint32_t misc[4];
};
static_assert(sizeof(RasterUniformsGPU) == 96, "raster uniforms drifted");

// c = a * b, column-major 4x4.
void mat_mul4(const float* a, const float* b, float* c) {
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float v = 0.0f;
            for (int k = 0; k < 4; ++k) v += a[k * 4 + row] * b[col * 4 + k];
            c[col * 4 + row] = v;
        }
    }
}

// Unit UV-sphere triangle list (positions only; the position IS the
// normal on a unit sphere). Proxy geometry for the nav preview.
std::vector<float> make_unit_sphere(int slices, int stacks) {
    std::vector<float> v;
    const float PI = 3.14159265358979f;
    auto pt = [&](int sl, int st) {
        const float phi = 2.0f * PI * float(sl) / float(slices);
        const float theta = PI * float(st) / float(stacks);
        v.push_back(std::sin(theta) * std::cos(phi));
        v.push_back(std::cos(theta));
        v.push_back(std::sin(theta) * std::sin(phi));
    };
    for (int st = 0; st < stacks; ++st) {
        for (int sl = 0; sl < slices; ++sl) {
            pt(sl, st);     pt(sl + 1, st);     pt(sl, st + 1);
            pt(sl + 1, st); pt(sl + 1, st + 1); pt(sl, st + 1);
        }
    }
    return v;
}

} // namespace

struct GpuRenderer::Impl {
    RenderSettings settings;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> accumulate_pso;
    id<MTLComputePipelineState> resolve_pso;
    id<MTLBuffer> accum;     // full-res float4, StorageModeShared
    id<MTLBuffer> spheres;   // bound as constant GPUSphere*
    // Mesh data (placeholder-sized when no mesh — Metal requires every
    // declared kernel argument to be bound even if never dereferenced).
    id<MTLBuffer> bvh_nodes;
    id<MTLBuffer> tris;
    id<MTLBuffer> tri_mat;   // per-triangle material id, leaf order
    id<MTLBuffer> tri_light; // per-triangle light ordinal + 1 (0 = none)
    // Session I bindless: one raw ushort buffer per texture, referenced
    // from the GPUMaterialArgs table by gpuAddress. Kept in this vector so
    // (a) ARC retains them and (b) encode marks them resident
    // (useResource) — indirectly-addressed resources aren't tracked
    // automatically.
    std::vector<id<MTLBuffer>> mat_textures;
    id<MTLBuffer> mat_table;
    MeshUniforms mesh_u{};   // has_mesh = 0 by default
    // Session J: area-light list.
    id<MTLBuffer> lights;
    pt_uint light_count = 0;
    // Session F: HDRI environment (env_w == 0 -> gradient fallback).
    id<MTLBuffer> env_texels;
    id<MTLBuffer> env_row_cdf;
    id<MTLBuffer> env_cond_cdf;
    pt_uint env_w = 0, env_h = 0;
    float env_intensity = 1.0f, env_yaw_norm = 0.0f;
    // BVH register pressure can lower maxTotalThreadsPerThreadgroup below
    // a hardcoded 16x16=256, which would FAIL dispatch at runtime — pick
    // the threadgroup shape per pipeline at creation instead.
    MTLSize tg_accum, tg_resolve;

    // Session E: raster nav preview. Preview-only — never touches accum.
    id<MTLRenderPipelineState> raster_mesh_pso;
    id<MTLRenderPipelineState> raster_sphere_pso;
    id<MTLDepthStencilState> depth_less;
    id<MTLDepthStencilState> depth_always;   // selection overlay
    id<MTLTexture> raster_depth;
    id<MTLBuffer> sphere_proxy;
    int sphere_proxy_verts = 0;
    id<MTLCommandBuffer> last_cb;
    pt_uint sphere_count = 0;
    int w = 0, h = 0;        // current accumulation dims
    int passes = 0;          // completed full passes scheduled since reset
    int cursor = 0;          // next row of the in-progress pass (0 = none)
    bool needs_clear = true;
    // Timing of the last completed accumulate batch (written from Metal's
    // completion handler thread, read by the main-thread tick).
    std::atomic<float> last_gpu_ms{0.0f};
    std::atomic<unsigned long long> last_px_passes{0};

    void encode_clear_if_needed(id<MTLCommandBuffer> cb) {
        if (!needs_clear) return;
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        // fillBuffer in-command-buffer: GPU-timeline ordering makes the
        // clear race-free against in-flight older passes.
        [blit fillBuffer:accum
                   range:NSMakeRange(0, size_t(w) * h * sizeof(float) * 4)
                   value:0];
        [blit endEncoding];
        needs_clear = false;
        passes = 0;
        cursor = 0;
    }

    // Full frame (row_count == h): K passes in-kernel — the headless
    // parity/offline path and cheap interactive frames. Slice (row_count <
    // h): a single pass over [row_start, row_start+row_count), so heavy
    // frames are spread across many short command buffers.
    void encode_accumulate(id<MTLCommandBuffer> cb, const GPUCamera& cam,
                           int count, int row_start = 0, int row_count = 0) {
        const bool slice = row_count > 0 && row_count < h;
        PassUniforms u{};
        u.cam = cam;
        u.width = pt_uint(w);
        u.height = pt_uint(h);
        u.pass_base = pt_uint(passes);
        u.pass_count = pt_uint(slice ? 1 : count);
        u.row_offset = pt_uint(slice ? row_start : 0);
        u.sphere_count = sphere_count;
        u.max_depth = pt_uint(settings.max_depth);
        u.env_w = env_w;
        u.env_h = env_h;
        u.env_intensity = env_intensity;
        u.env_yaw_norm = env_yaw_norm;
        u.clamp_indirect = settings.clamp_indirect;
        u.env_nee = pt_uint(settings.env_nee != 0 ? 1 : 0);
        u.light_count = light_count;

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:accumulate_pso];
        [enc setBuffer:accum offset:0 atIndex:0];
        [enc setBuffer:spheres offset:0 atIndex:1];
        [enc setBytes:&u length:sizeof u atIndex:2];
        [enc setBuffer:bvh_nodes offset:0 atIndex:3];
        [enc setBuffer:tris offset:0 atIndex:4];
        [enc setBuffer:mat_table offset:0 atIndex:5];
        [enc setBuffer:tri_mat offset:0 atIndex:6];
        [enc setBytes:&mesh_u length:sizeof mesh_u atIndex:8];
        // Bindless residency: buffers reached via gpuAddress are invisible
        // to Metal's automatic hazard/residency tracking.
        for (id<MTLBuffer> t : mat_textures) {
            [enc useResource:t usage:MTLResourceUsageRead];
        }
        [enc setBuffer:env_texels offset:0 atIndex:10];
        [enc setBuffer:env_row_cdf offset:0 atIndex:11];
        [enc setBuffer:env_cond_cdf offset:0 atIndex:12];
        [enc setBuffer:lights offset:0 atIndex:13];
        [enc setBuffer:tri_light offset:0 atIndex:14];
        [enc dispatchThreads:MTLSizeMake(w, slice ? row_count : h, 1)
            threadsPerThreadgroup:tg_accum];
        [enc endEncoding];
        if (slice) {
            cursor = row_start + row_count;
            if (cursor >= h) {
                passes += 1;
                cursor = 0;
            }
        } else {
            passes += count;
        }
    }

    void ensure_raster_depth(id<MTLTexture> target) {
        if (raster_depth && raster_depth.width == target.width &&
            raster_depth.height == target.height)
            return;
        MTLTextureDescriptor* d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                         width:target.width
                                        height:target.height
                                     mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget;
        d.storageMode = MTLStorageModePrivate;
        raster_depth = [device newTextureWithDescriptor:d];
    }

    // kind: 0 = whole scene, 1 = one sphere (index), 2 = mesh only.
    void draw_raster(id<MTLRenderCommandEncoder> enc,
                     const RasterUniformsGPU& u, bool wireframe, int kind,
                     int index) {
        [enc setTriangleFillMode:wireframe ? MTLTriangleFillModeLines
                                           : MTLTriangleFillModeFill];
        [enc setCullMode:MTLCullModeNone];
        if ((kind == 0 || kind == 2) && mesh_u.has_mesh) {
            [enc setRenderPipelineState:raster_mesh_pso];
            [enc setVertexBuffer:tris offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof u atIndex:1];
            [enc setFragmentBytes:&u length:sizeof u atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                     vertexStart:0
                     vertexCount:NSUInteger(mesh_u.tri_count) * 3];
        }
        if ((kind == 0 || kind == 1) && sphere_count > 0) {
            [enc setRenderPipelineState:raster_sphere_pso];
            [enc setVertexBuffer:sphere_proxy offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof u atIndex:1];
            [enc setVertexBuffer:spheres offset:0 atIndex:2];
            [enc setFragmentBytes:&u length:sizeof u atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                     vertexStart:0
                     vertexCount:NSUInteger(sphere_proxy_verts)
                   instanceCount:kind == 1 ? 1 : sphere_count
                    baseInstance:kind == 1 ? NSUInteger(index) : 0];
        }
    }

    id<MTLBuffer> upload(const void* data, size_t len) {
        // Metal validates that every declared argument is bound, so bind a
        // small placeholder when there is no mesh (never dereferenced —
        // the kernel guards on has_mesh).
        if (!data || len == 0)
            return [device newBufferWithLength:64
                                       options:MTLResourceStorageModeShared];
        return [device newBufferWithBytes:data
                                   length:len
                                  options:MTLResourceStorageModeShared];
    }

    // Full mesh (re)upload: geometry, textures, uniforms. nullptr removes.
    void upload_mesh(const MeshData* mesh) {
        mesh_u = MeshUniforms{};   // has_mesh = 0
        if (!mesh) {
            bvh_nodes = upload(nullptr, 0);
            tris = upload(nullptr, 0);
            tri_mat = upload(nullptr, 0);
            tri_light = upload(nullptr, 0);
            mat_textures.clear();
            mat_table = upload(nullptr, 0);
            return;
        }
        bvh_nodes = upload(mesh->nodes.data(),
                           mesh->nodes.size() * sizeof(BVHNode));
        tris = upload(mesh->tris.data(),
                      mesh->tris.size() * sizeof(GPUTriangle));
        tri_mat = upload(mesh->tri_mat.data(),
                         mesh->tri_mat.size() * sizeof(std::uint32_t));
        tri_light = upload(mesh->tri_light.data(),
                           mesh->tri_light.size() * sizeof(std::uint32_t));

        // Bindless material table: raw ushort buffers per channel, their
        // Metal-3 gpuAddress written into GPUMaterialArgs entries. New
        // buffers + new table on every (re)upload — the buffer-swap
        // discipline: in-flight command buffers retain the old ones.
        mat_textures.clear();
        std::vector<GPUMaterialArgs> table;
        table.reserve(mesh->materials.size());
        const auto tex_addr = [&](const Texture16& t) -> unsigned long long {
            id<MTLBuffer> buf =
                upload(t.texels.empty() ? nullptr : t.texels.data(),
                       t.texels.size() * sizeof(std::uint16_t));
            mat_textures.push_back(buf);
            return buf.gpuAddress;
        };
        for (const MeshMaterial& m : mesh->materials) {
            GPUMaterialArgs e{};
            e.base = tex_addr(m.base);
            e.mr = tex_addr(m.mr);
            e.emis = tex_addr(m.emissive);
            e.norm = tex_addr(m.normal);
            e.base_w = pt_uint(m.base.w);
            e.base_h = pt_uint(m.base.h);
            e.mr_w = pt_uint(m.mr.w);
            e.mr_h = pt_uint(m.mr.h);
            e.emis_w = pt_uint(m.emissive.w);
            e.emis_h = pt_uint(m.emissive.h);
            e.norm_w = pt_uint(m.normal.w);
            e.norm_h = pt_uint(m.normal.h);
            table.push_back(e);
        }
        mat_table =
            upload(table.data(), table.size() * sizeof(GPUMaterialArgs));

        mesh_u.has_mesh = 1;
        mesh_u.tri_count = pt_uint(mesh->tris.size());
        mesh_u.node_count = pt_uint(mesh->nodes.size());
        mesh_u.mat_count = pt_uint(mesh->materials.size());
        mesh_u.emissive_scale = mesh->emissive_scale;
    }

    void encode_resolve(id<MTLCommandBuffer> cb, id<MTLTexture> target) {
        ResolveUniforms u{};
        u.accum_w = pt_uint(w);
        u.accum_h = pt_uint(h);
        u.out_w = pt_uint(target.width);
        u.out_h = pt_uint(target.height);
        u.pass_total = pt_uint(passes);
        u.rows_plus1 = pt_uint(cursor);

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:resolve_pso];
        [enc setBuffer:accum offset:0 atIndex:0];
        [enc setBytes:&u length:sizeof u atIndex:1];
        [enc setTexture:target atIndex:0];
        [enc dispatchThreads:MTLSizeMake(target.width, target.height, 1)
            threadsPerThreadgroup:tg_resolve];
        [enc endEncoding];
    }
};

GpuRenderer::GpuRenderer() : impl_(new Impl) {}
GpuRenderer::~GpuRenderer() { wait_idle(); }

std::unique_ptr<GpuRenderer> GpuRenderer::create(
    const RenderSettings& settings, const std::vector<GPUSphere>& spheres,
    const MeshData* mesh, std::string& error) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        error = "no Metal device";
        return nullptr;
    }
    id<MTLLibrary> lib = compile_library(device, error);
    if (!lib) return nullptr;

    auto r = std::unique_ptr<GpuRenderer>(new GpuRenderer());
    Impl& im = *r->impl_;
    im.settings = settings;
    im.device = device;
    im.queue = [device newCommandQueue];
    im.accumulate_pso = make_pipeline(device, lib, @"accumulate", error);
    im.resolve_pso = make_pipeline(device, lib, @"resolve", error);
    if (!im.accumulate_pso || !im.resolve_pso) return nullptr;

    // Raster nav-preview pipelines (color = drawable format, depth32).
    const auto make_raster_pso = [&](NSString* vs_name,
                                     std::string& e) -> id<MTLRenderPipelineState> {
        id<MTLFunction> vs = [lib newFunctionWithName:vs_name];
        id<MTLFunction> fs = [lib newFunctionWithName:@"raster_fs"];
        if (!vs || !fs) {
            e = "raster shader missing";
            return nil;
        }
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = vs;
        d.fragmentFunction = fs;
        d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        NSError* nserr = nil;
        id<MTLRenderPipelineState> pso =
            [device newRenderPipelineStateWithDescriptor:d error:&nserr];
        if (!pso)
            e = std::string("raster pipeline: ") +
                (nserr ? nserr.localizedDescription.UTF8String : "?");
        return pso;
    };

    const auto pick_tg = [](id<MTLComputePipelineState> pso) {
        return pso.maxTotalThreadsPerThreadgroup >= 256
                   ? MTLSizeMake(16, 16, 1)
                   : MTLSizeMake(8, 8, 1);
    };
    im.tg_accum = pick_tg(im.accumulate_pso);
    im.tg_resolve = pick_tg(im.resolve_pso);

    im.raster_mesh_pso = make_raster_pso(@"raster_mesh_vs", error);
    im.raster_sphere_pso = make_raster_pso(@"raster_sphere_vs", error);
    if (!im.raster_mesh_pso || !im.raster_sphere_pso) return nullptr;
    {
        MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = MTLCompareFunctionLess;
        dd.depthWriteEnabled = YES;
        im.depth_less = [device newDepthStencilStateWithDescriptor:dd];
        dd.depthCompareFunction = MTLCompareFunctionAlways;
        dd.depthWriteEnabled = NO;
        im.depth_always = [device newDepthStencilStateWithDescriptor:dd];
    }
    {
        const std::vector<float> proxy = make_unit_sphere(24, 16);
        im.sphere_proxy =
            [device newBufferWithBytes:proxy.data()
                                length:proxy.size() * sizeof(float)
                               options:MTLResourceStorageModeShared];
        im.sphere_proxy_verts = int(proxy.size() / 3);
    }

    im.accum = [device
        newBufferWithLength:size_t(settings.width) * settings.height *
                            sizeof(float) * 4
                    options:MTLResourceStorageModeShared];
    // Guarded like update_spheres: --only-model scenes have no spheres,
    // and a zero-length newBufferWithBytes returns NIL (unbound buffer(1)).
    im.spheres = im.upload(spheres.empty() ? nullptr : spheres.data(),
                           spheres.size() * sizeof(GPUSphere));
    im.sphere_count = pt_uint(spheres.size());

    im.upload_mesh(mesh);
    im.lights = im.upload(nullptr, 0);       // none until set_lights
    im.env_texels = im.upload(nullptr, 0);   // gradient until set_env
    im.env_row_cdf = im.upload(nullptr, 0);
    im.env_cond_cdf = im.upload(nullptr, 0);
    im.w = settings.width;
    im.h = settings.height;
    return r;
}

void GpuRenderer::set_lights(const std::vector<GPULight>& lights) {
    impl_->lights = impl_->upload(
        lights.empty() ? nullptr : lights.data(),
        lights.size() * sizeof(GPULight));
    impl_->light_count = pt_uint(lights.size());
}

void GpuRenderer::set_env(const EnvMap* env) {
    if (env && env->valid()) {
        impl_->env_texels =
            impl_->upload(env->texels.data(),
                          env->texels.size() * sizeof(float));
        impl_->env_row_cdf =
            impl_->upload(env->row_cdf.data(),
                          env->row_cdf.size() * sizeof(float));
        impl_->env_cond_cdf =
            impl_->upload(env->cond_cdf.data(),
                          env->cond_cdf.size() * sizeof(float));
        impl_->env_w = pt_uint(env->w);
        impl_->env_h = pt_uint(env->h);
    } else {
        impl_->env_texels = impl_->upload(nullptr, 0);
        impl_->env_row_cdf = impl_->upload(nullptr, 0);
        impl_->env_cond_cdf = impl_->upload(nullptr, 0);
        impl_->env_w = 0;
        impl_->env_h = 0;
    }
}

void GpuRenderer::set_env_params(float intensity, float yaw_norm) {
    impl_->env_intensity = intensity;
    impl_->env_yaw_norm = yaw_norm;
}

void GpuRenderer::set_mesh(const MeshData* mesh) {
    impl_->upload_mesh(mesh);
}

void GpuRenderer::reset(int w, int h) {
    // Resolution scales above 1x need more accumulation space than the
    // base allocation. Grow-only buffer swap: in-flight command buffers
    // retain the old one, so no stall and no torn frames.
    const size_t needed = size_t(w) * h * sizeof(float) * 4;
    if (needed > impl_->accum.length) {
        impl_->accum =
            [impl_->device newBufferWithLength:needed
                                       options:MTLResourceStorageModeShared];
    }
    impl_->w = w;
    impl_->h = h;
    impl_->needs_clear = true;
    impl_->cursor = 0;
}

int GpuRenderer::width() const  { return impl_->w; }
int GpuRenderer::height() const { return impl_->h; }
int GpuRenderer::passes() const { return impl_->passes; }
bool GpuRenderer::reset_pending() const { return impl_->needs_clear; }

void GpuRenderer::render_passes_blocking(const GPUCamera& cam, int count) {
    while (count > 0) {
        const int chunk = std::min(count, kMaxPassesPerCommandBuffer);
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        impl_->encode_clear_if_needed(cb);
        impl_->encode_accumulate(cb, cam, chunk);
        [cb commit];
        impl_->last_cb = cb;
        count -= chunk;
    }
    wait_idle();
}

const float* GpuRenderer::accum_data() const {
    return static_cast<const float*>(impl_->accum.contents);
}

void GpuRenderer::wait_idle() const {
    if (impl_->last_cb) [impl_->last_cb waitUntilCompleted];
}

bool GpuRenderer::save_png(const std::string& path) const {
    wait_idle();
    const float* a = accum_data();
    const float inv_n = 1.0f / float(std::max(1, impl_->passes));
    Image img(impl_->w, impl_->h);
    for (int y = 0; y < impl_->h; ++y) {
        for (int x = 0; x < impl_->w; ++x) {
            const size_t i = (size_t(y) * impl_->w + x) * 4;
            img.at(x, y) = color(a[i], a[i + 1], a[i + 2]) * inv_n;
        }
    }
    return img.write_png(path);
}

namespace {
RasterUniformsGPU raster_uniforms(const GpuRenderer::RasterParams& p,
                                  bool overlay_tint) {
    RasterUniformsGPU u{};
    mat_mul4(p.proj, p.view, u.vp);
    u.cam_pos[0] = p.cam_pos[0];
    u.cam_pos[1] = p.cam_pos[1];
    u.cam_pos[2] = p.cam_pos[2];
    u.cam_pos[3] = 1.0f;
    u.misc[0] = overlay_tint ? 1u : 0u;
    return u;
}
} // namespace

void GpuRenderer::encode_raster_frame(const RasterParams& params, void* layer,
                                      std::function<void()> on_complete,
                                      const UiEncoder& ui) {
    CAMetalLayer* metal_layer = (__bridge CAMetalLayer*)layer;
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    @autoreleasepool {
        id<CAMetalDrawable> drawable = [metal_layer nextDrawable];
        if (drawable) {
            impl_->ensure_raster_depth(drawable.texture);
            MTLRenderPassDescriptor* rpd =
                [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = drawable.texture;
            rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[0].clearColor =
                MTLClearColorMake(0.16, 0.17, 0.20, 1.0);
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            rpd.depthAttachment.texture = impl_->raster_depth;
            rpd.depthAttachment.loadAction = MTLLoadActionClear;
            rpd.depthAttachment.clearDepth = 1.0;
            rpd.depthAttachment.storeAction = MTLStoreActionDontCare;

            id<MTLRenderCommandEncoder> enc =
                [cb renderCommandEncoderWithDescriptor:rpd];
            [enc setDepthStencilState:impl_->depth_less];
            const RasterUniformsGPU u = raster_uniforms(params, false);
            impl_->draw_raster(enc, u, params.wireframe, 0, 0);
            // Selection stays visible while navigating too.
            if (params.overlay_kind != 0) {
                [enc setDepthStencilState:impl_->depth_always];
                const RasterUniformsGPU ou = raster_uniforms(params, true);
                impl_->draw_raster(enc, ou, true, params.overlay_kind,
                                   params.overlay_index);
            }
            [enc endEncoding];

            if (ui) ui((__bridge void*)cb, (__bridge void*)drawable.texture);
            [cb presentDrawable:drawable];
        }
    }
    if (on_complete) {
        [cb addCompletedHandler:^(id<MTLCommandBuffer>) { on_complete(); }];
    }
    [cb commit];
    impl_->last_cb = cb;
}

void GpuRenderer::encode_frame(const GPUCamera& cam, const TraceWork& work,
                               void* layer,
                               std::function<void()> on_complete,
                               const UiEncoder& ui,
                               const RasterParams* overlay) {
    CAMetalLayer* metal_layer = (__bridge CAMetalLayer*)layer;
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    impl_->encode_clear_if_needed(cb);
    unsigned long long px_passes = 0;
    if (work.passes > 0) {
        impl_->encode_accumulate(cb, cam, work.passes, work.row_start,
                                 work.row_count);
        const bool slice = work.row_count > 0 && work.row_count < impl_->h;
        px_passes = (unsigned long long)(impl_->w) *
                    (slice ? work.row_count : impl_->h) *
                    (slice ? 1 : work.passes);
    }

    // Drawable acquired LAST (after all CPU-side work) and nil-checked:
    // when the pool is exhausted nextDrawable can block, so the pacing
    // semaphore upstream keeps in-flight batches below the drawable count.
    @autoreleasepool {
        id<CAMetalDrawable> drawable = [metal_layer nextDrawable];
        if (drawable) {
            impl_->encode_resolve(cb, drawable.texture);
            if (overlay && overlay->overlay_kind != 0) {
                impl_->ensure_raster_depth(drawable.texture);
                MTLRenderPassDescriptor* rpd =
                    [MTLRenderPassDescriptor renderPassDescriptor];
                rpd.colorAttachments[0].texture = drawable.texture;
                rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
                rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
                rpd.depthAttachment.texture = impl_->raster_depth;
                rpd.depthAttachment.loadAction = MTLLoadActionClear;
                rpd.depthAttachment.clearDepth = 1.0;
                rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
                id<MTLRenderCommandEncoder> enc =
                    [cb renderCommandEncoderWithDescriptor:rpd];
                [enc setDepthStencilState:impl_->depth_always];
                const RasterUniformsGPU ou = raster_uniforms(*overlay, true);
                impl_->draw_raster(enc, ou, true, overlay->overlay_kind,
                                   overlay->overlay_index);
                [enc endEncoding];
            }
            if (ui) ui((__bridge void*)cb, (__bridge void*)drawable.texture);
            [cb presentDrawable:drawable];
        }
    }
    // Timing lands from Metal's completion thread; the tick reads the
    // atomics to steer the next frame's budget.
    Impl* impl = impl_.get();
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (px_passes > 0) {
            const double ms = (done.GPUEndTime - done.GPUStartTime) * 1000.0;
            impl->last_gpu_ms.store(float(ms), std::memory_order_relaxed);
            impl->last_px_passes.store(px_passes, std::memory_order_relaxed);
        }
        if (on_complete) on_complete();
    }];
    [cb commit];
    impl_->last_cb = cb;
}

int GpuRenderer::partial_row() const { return impl_->cursor; }
float GpuRenderer::last_batch_gpu_ms() const {
    return impl_->last_gpu_ms.load(std::memory_order_relaxed);
}
unsigned long long GpuRenderer::last_batch_px_passes() const {
    return impl_->last_px_passes.load(std::memory_order_relaxed);
}

void GpuRenderer::save_png_async(const std::string& path,
                                 std::function<void(bool ok)> done) {
    // Snapshot the ingredients now; the caller pauses accumulate encodes
    // until `done`, so the accum contents are stable once this CB retires.
    id<MTLCommandBuffer> cb = impl_->last_cb;
    Impl* impl = impl_.get();
    const int w = impl_->w, h = impl_->h;
    const int passes = impl_->passes, cursor = impl_->cursor;
    dispatch_async(
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            if (cb) [cb waitUntilCompleted];
            const float* a = static_cast<const float*>(impl->accum.contents);
            Image img(w, h);
            for (int y = 0; y < h; ++y) {
                const float inv_n =
                    1.0f / float(std::max(1, passes + (y < cursor ? 1 : 0)));
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t(y) * w + x) * 4;
                    img.at(x, y) = color(a[i], a[i + 1], a[i + 2]) * inv_n;
                }
            }
            const bool ok = img.write_png(path);
            dispatch_async(dispatch_get_main_queue(), ^{ done(ok); });
        });
}

void GpuRenderer::set_max_depth(int depth) {
    impl_->settings.max_depth = depth;
}

void GpuRenderer::set_clamp_indirect(float clamp) {
    impl_->settings.clamp_indirect = clamp;
}

void GpuRenderer::set_env_nee(bool on) {
    impl_->settings.env_nee = on ? 1 : 0;
}

void GpuRenderer::update_spheres(const std::vector<GPUSphere>& spheres) {
    // upload() guards the empty case (--only-model scenes): a zero-length
    // newBufferWithBytes returns NIL, which would leave the kernel's
    // declared buffer(1) unbound.
    impl_->spheres = impl_->upload(
        spheres.empty() ? nullptr : spheres.data(),
        spheres.size() * sizeof(GPUSphere));
    impl_->sphere_count = pt_uint(spheres.size());
}

void GpuRenderer::update_mesh_geometry(const std::vector<GPUTriangle>& tris,
                                       const std::vector<BVHNode>& nodes,
                                       const std::vector<pt_uint>& tri_mat,
                                       const std::vector<pt_uint>& tri_light) {
    impl_->tri_mat = impl_->upload(tri_mat.data(),
                                   tri_mat.size() * sizeof(pt_uint));
    impl_->tri_light = impl_->upload(tri_light.data(),
                                     tri_light.size() * sizeof(pt_uint));
    impl_->tris =
        [impl_->device newBufferWithBytes:tris.data()
                                   length:tris.size() * sizeof(GPUTriangle)
                                  options:MTLResourceStorageModeShared];
    impl_->bvh_nodes =
        [impl_->device newBufferWithBytes:nodes.data()
                                   length:nodes.size() * sizeof(BVHNode)
                                  options:MTLResourceStorageModeShared];
    impl_->mesh_u.tri_count = pt_uint(tris.size());
    impl_->mesh_u.node_count = pt_uint(nodes.size());
}

void* GpuRenderer::metal_device() const {
    return (__bridge void*)impl_->device;
}

bool gpu_check() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::fprintf(stderr, "gpu-check: no Metal device\n");
        return false;
    }
    std::string error;
    id<MTLLibrary> lib = compile_library(device, error);
    if (!lib) {
        std::fprintf(stderr, "gpu-check: %s\n", error.c_str());
        return false;
    }
    id<MTLComputePipelineState> acc =
        make_pipeline(device, lib, @"accumulate", error);
    id<MTLComputePipelineState> res =
        make_pipeline(device, lib, @"resolve", error);
    if (!acc || !res) {
        std::fprintf(stderr, "gpu-check: %s\n", error.c_str());
        return false;
    }
    std::printf("device:                        %s\n", device.name.UTF8String);
    std::printf("threadExecutionWidth:          %lu\n",
                (unsigned long)acc.threadExecutionWidth);
    std::printf("maxTotalThreadsPerThreadgroup: %lu (accumulate)\n",
                (unsigned long)acc.maxTotalThreadsPerThreadgroup);
    std::printf("threadgroup:                   %s (accumulate)\n",
                acc.maxTotalThreadsPerThreadgroup >= 256 ? "16x16" : "8x8");
    if (acc.maxTotalThreadsPerThreadgroup < 64) {
        std::fprintf(stderr, "gpu-check: kernel too register-heavy for 8x8\n");
        return false;
    }
    std::printf("kernels compiled OK (safe math)\n");
    return true;
}
