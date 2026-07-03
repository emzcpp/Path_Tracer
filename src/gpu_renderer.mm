#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <cstdio>

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
    id<MTLBuffer> tex_base;
    id<MTLBuffer> tex_mr;
    id<MTLBuffer> tex_emis;
    id<MTLBuffer> tex_norm;
    MeshUniforms mesh_u{};   // has_mesh = 0 by default
    // BVH register pressure can lower maxTotalThreadsPerThreadgroup below
    // a hardcoded 16x16=256, which would FAIL dispatch at runtime — pick
    // the threadgroup shape per pipeline at creation instead.
    MTLSize tg_accum, tg_resolve;
    id<MTLCommandBuffer> last_cb;
    pt_uint sphere_count = 0;
    int w = 0, h = 0;        // current accumulation dims
    int passes = 0;          // scheduled since reset
    bool needs_clear = true;

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
    }

    void encode_accumulate(id<MTLCommandBuffer> cb, const GPUCamera& cam,
                           int count) {
        PassUniforms u{};
        u.cam = cam;
        u.width = pt_uint(w);
        u.height = pt_uint(h);
        u.pass_base = pt_uint(passes);
        u.pass_count = pt_uint(count);
        u.sphere_count = sphere_count;
        u.max_depth = pt_uint(settings.max_depth);

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:accumulate_pso];
        [enc setBuffer:accum offset:0 atIndex:0];
        [enc setBuffer:spheres offset:0 atIndex:1];
        [enc setBytes:&u length:sizeof u atIndex:2];
        [enc setBuffer:bvh_nodes offset:0 atIndex:3];
        [enc setBuffer:tris offset:0 atIndex:4];
        [enc setBuffer:tex_base offset:0 atIndex:5];
        [enc setBuffer:tex_mr offset:0 atIndex:6];
        [enc setBuffer:tex_emis offset:0 atIndex:7];
        [enc setBytes:&mesh_u length:sizeof mesh_u atIndex:8];
        [enc setBuffer:tex_norm offset:0 atIndex:9];
        [enc dispatchThreads:MTLSizeMake(w, h, 1)
            threadsPerThreadgroup:tg_accum];
        [enc endEncoding];
        passes += count;
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
            tex_base = upload(nullptr, 0);
            tex_mr = upload(nullptr, 0);
            tex_emis = upload(nullptr, 0);
            tex_norm = upload(nullptr, 0);
            return;
        }
        bvh_nodes = upload(mesh->nodes.data(),
                           mesh->nodes.size() * sizeof(BVHNode));
        tris = upload(mesh->tris.data(),
                      mesh->tris.size() * sizeof(GPUTriangle));
        tex_base = upload(mesh->base.texels.data(),
                          mesh->base.texels.size() * sizeof(std::uint16_t));
        tex_mr = upload(mesh->mr.texels.data(),
                        mesh->mr.texels.size() * sizeof(std::uint16_t));
        tex_emis = upload(mesh->emissive.texels.data(),
                          mesh->emissive.texels.size() * sizeof(std::uint16_t));
        tex_norm = upload(mesh->normal.texels.data(),
                          mesh->normal.texels.size() * sizeof(std::uint16_t));
        mesh_u.has_mesh = 1;
        mesh_u.tri_count = pt_uint(mesh->tris.size());
        mesh_u.node_count = pt_uint(mesh->nodes.size());
        mesh_u.base_w = pt_uint(mesh->base.w);
        mesh_u.base_h = pt_uint(mesh->base.h);
        mesh_u.mr_w = pt_uint(mesh->mr.w);
        mesh_u.mr_h = pt_uint(mesh->mr.h);
        mesh_u.emis_w = pt_uint(mesh->emissive.w);
        mesh_u.emis_h = pt_uint(mesh->emissive.h);
        mesh_u.norm_w = pt_uint(mesh->normal.w);
        mesh_u.norm_h = pt_uint(mesh->normal.h);
        mesh_u.emissive_scale = mesh->emissive_scale;
    }

    void encode_resolve(id<MTLCommandBuffer> cb, id<MTLTexture> target) {
        ResolveUniforms u{};
        u.accum_w = pt_uint(w);
        u.accum_h = pt_uint(h);
        u.out_w = pt_uint(target.width);
        u.out_h = pt_uint(target.height);
        u.pass_total = pt_uint(passes);

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

    const auto pick_tg = [](id<MTLComputePipelineState> pso) {
        return pso.maxTotalThreadsPerThreadgroup >= 256
                   ? MTLSizeMake(16, 16, 1)
                   : MTLSizeMake(8, 8, 1);
    };
    im.tg_accum = pick_tg(im.accumulate_pso);
    im.tg_resolve = pick_tg(im.resolve_pso);

    im.accum = [device
        newBufferWithLength:size_t(settings.width) * settings.height *
                            sizeof(float) * 4
                    options:MTLResourceStorageModeShared];
    im.spheres = [device newBufferWithBytes:spheres.data()
                                     length:spheres.size() * sizeof(GPUSphere)
                                    options:MTLResourceStorageModeShared];
    im.sphere_count = pt_uint(spheres.size());

    im.upload_mesh(mesh);
    im.w = settings.width;
    im.h = settings.height;
    return r;
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

void GpuRenderer::encode_frame(const GPUCamera& cam, int count, void* layer,
                               std::function<void()> on_complete,
                               const UiEncoder& ui) {
    CAMetalLayer* metal_layer = (__bridge CAMetalLayer*)layer;
    id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
    impl_->encode_clear_if_needed(cb);
    if (count > 0) impl_->encode_accumulate(cb, cam, count);

    // Drawable acquired LAST (after all CPU-side work) and nil-checked:
    // when the pool is exhausted nextDrawable can block, so the pacing
    // semaphore upstream keeps in-flight batches below the drawable count.
    @autoreleasepool {
        id<CAMetalDrawable> drawable = [metal_layer nextDrawable];
        if (drawable) {
            impl_->encode_resolve(cb, drawable.texture);
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

void GpuRenderer::set_max_depth(int depth) {
    impl_->settings.max_depth = depth;
}

void GpuRenderer::update_spheres(const std::vector<GPUSphere>& spheres) {
    impl_->spheres =
        [impl_->device newBufferWithBytes:spheres.data()
                                   length:spheres.size() * sizeof(GPUSphere)
                                  options:MTLResourceStorageModeShared];
    impl_->sphere_count = pt_uint(spheres.size());
}

void GpuRenderer::update_mesh_geometry(const std::vector<GPUTriangle>& tris,
                                       const std::vector<BVHNode>& nodes) {
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
