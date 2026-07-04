#pragma once

// GPU progressive renderer (Metal compute). This header is C++-clean —
// all Objective-C lives in gpu_renderer.mm behind the pimpl — so pure C++
// translation units (main.cpp, the parity harness) can drive the GPU.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "camera.h"
#include "gltf_loader.h"
#include "kernel_types.h"
#include "settings.h"

// Fill the GPU camera from the CPU camera's own basis — same derivation,
// no re-derived trig.
inline GPUCamera to_gpu_camera(const Camera& cam) {
    const auto p3 = [](const vec3& v) { return pt_float3{v.x, v.y, v.z}; };
    return GPUCamera{p3(cam.origin()), p3(cam.lower_left()),
                     p3(cam.horizontal()), p3(cam.vertical())};
}

class GpuRenderer {
public:
    // nullptr + `error` filled if there's no Metal device or the embedded
    // kernel fails to compile. `mesh` may be null (sphere-only scene).
    static std::unique_ptr<GpuRenderer> create(
        const RenderSettings& settings, const std::vector<GPUSphere>& spheres,
        const MeshData* mesh, std::string& error);
    ~GpuRenderer();

    // Switch accumulation dims (full <-> preview) and clear. The buffer is
    // allocated once at full res; smaller dims use a prefix of it.
    void reset(int w, int h);

    int width() const;
    int height() const;
    int passes() const;   // passes scheduled since the last reset
    // True between reset() and the next encoded batch. While pending,
    // passes() still reports the PRE-reset count — callers gating on
    // passes-reached-target must also check this.
    bool reset_pending() const;

    // Headless path (parity/offline): render `count` passes, block until
    // the GPU finishes. Chunked internally to stay under the GPU watchdog.
    void render_passes_blocking(const GPUCamera& cam, int count);

    // Linear accumulation sums, width*height float4s (RGBX). Valid after a
    // blocking render or wait_idle().
    const float* accum_data() const;

    void wait_idle() const;
    bool save_png(const std::string& path) const;

    // Optional UI overlay pass: called synchronously after resolve, before
    // present, with the command buffer and drawable texture (bridged
    // id<MTLCommandBuffer> / id<MTLTexture>).
    using UiEncoder = std::function<void(void* command_buffer,
                                         void* target_texture)>;

    // Session G: one tick's worth of trace work, bounded by the caller so
    // no single command buffer monopolizes the GPU. passes == 0 encodes a
    // present-only frame. row_count == 0 means the full frame; a nonzero
    // row_count dispatches a K=1 slice of rows [row_start, row_start +
    // row_count). Slicing never changes the accumulated image — seeds are
    // per (pixel, pass) — only the schedule.
    struct TraceWork {
        int passes = 0;
        int row_start = 0;
        int row_count = 0;
    };

    // Row where the in-progress pass stops (0 = at a pass boundary).
    int partial_row() const;

    // GPU time and size of the most recently COMPLETED accumulate batch
    // (from MTLCommandBuffer GPUStart/EndTime). Zero until one completes.
    // The tick's budget controller derives ns/(pixel*pass) from these.
    float last_batch_gpu_ms() const;
    unsigned long long last_batch_px_passes() const;

    // Async PNG export: waits for the GPU and encodes on a background
    // queue; `done` fires on the MAIN queue. The caller must pause new
    // accumulate work until then (present-only frames are fine) so the
    // snapshot is race-free. Never blocks the calling thread.
    void save_png_async(const std::string& path,
                        std::function<void(bool ok)> done);

    // Session E: raster navigation preview + selection overlay.
    struct RasterParams {
        float view[16];          // the SAME matrices ImGuizmo uses —
        float proj[16];          // verified vs get_ray to ~1e-6
        float cam_pos[3];
        bool wireframe = false;
        // Selection overlay: 0 = none, 1 = sphere (overlay_index), 2 = mesh.
        int overlay_kind = 0;
        int overlay_index = 0;
    };

    // Viewer path: encode [clear if pending +] `count` passes + resolve
    // into the CAMetalLayer's next drawable + present. `count` may be 0:
    // a present-only frame (resolve + UI) that keeps the overlay live
    // while accumulation is parked. Non-blocking; on_complete fires on an
    // arbitrary thread when the GPU finishes. `layer` is a CAMetalLayer*
    // passed as void* to keep this header ObjC-free. When `overlay` is
    // non-null, the selected object's wireframe is drawn over the traced
    // image (registration guaranteed: same camera matrices).
    void encode_frame(const GPUCamera& cam, const TraceWork& work,
                      void* layer, std::function<void()> on_complete,
                      const UiEncoder& ui = {},
                      const RasterParams* overlay = nullptr);

    // Fast-nav: a rasterized preview frame INSTEAD of tracing — used while
    // the camera moves so navigation stays at full framerate. Never touches
    // the accumulation buffer; the tracer reconverges on settle via the
    // existing generation/reset machinery.
    void encode_raster_frame(const RasterParams& params, void* layer,
                             std::function<void()> on_complete,
                             const UiEncoder& ui = {});

    // Live-tunable from the UI (the renderer holds its own settings copy).
    void set_max_depth(int depth);
    void set_clamp_indirect(float clamp);

    // Phase 4 scene edits. Both swap in NEW buffers rather than mutating in
    // place: Metal retains the old buffers for in-flight command buffers,
    // so there is no GPU stall and no torn geometry — the next encoded
    // batch simply binds the new data.
    void update_spheres(const std::vector<GPUSphere>& spheres);
    void update_mesh_geometry(const std::vector<GPUTriangle>& tris,
                              const std::vector<BVHNode>& nodes,
                              const std::vector<pt_uint>& tri_mat,
                              const std::vector<pt_uint>& tri_light);

    // Phase 5 (scene load): swap the whole mesh — geometry AND textures —
    // or remove it (nullptr). Same buffer-swap semantics as above.
    void set_mesh(const MeshData* mesh);

    // Session F: HDRI environment. set_env swaps the texel buffer (nullptr
    // reverts to the gradient dome); set_env_params updates intensity/yaw
    // only (cheap, per-edit).
    // Session J: swap the area-light list (emissive spheres/triangles).
    // Same buffer-swap discipline; count rides in PassUniforms.
    void set_lights(const std::vector<GPULight>& lights);

    void set_env(const EnvMap* env);
    void set_env_params(float intensity, float yaw_norm);
    void set_env_nee(bool on);   // Session H: NEE+MIS vs brute force

    // Bridged id<MTLDevice>, for CAMetalLayer.device.
    void* metal_device() const;

private:
    GpuRenderer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --gpu-check: create the device, compile the embedded kernels, print
// device capabilities. Returns false on any failure.
bool gpu_check();
