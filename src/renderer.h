#pragma once

// Progressive renderer: owns the accumulation buffer and a persistent worker
// pool. One render_pass() adds 1 sample per pixel to the accumulator, and
// each worker also tone-maps its own rows into an RGBA staging buffer —
// resolving 1.5M pixels in a single thread afterwards would cost more than
// the trace pass itself.
//
// Progressive refinement is just Monte Carlo bookkeeping: pass k stores
// sum-of-k-samples in `accum`, and the displayed image is accum/k. Rendering
// N passes of 1 spp is mathematically identical to one N-spp render, which
// is why offline mode and the live viewer share this exact code path.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "camera.h"
#include "integrator.h"
#include "scene.h"
#include "settings.h"
#include "vec3.h"

class ProgressiveRenderer {
public:
    // total_threads includes the calling thread: render_pass() works rows
    // itself, so the pool holds total_threads - 1 workers.
    ProgressiveRenderer(const Scene& scene, const RenderSettings& settings,
                        unsigned total_threads, const EnvLookup& env = {},
                        const LightsLookup& lights = {});
    ~ProgressiveRenderer();

    ProgressiveRenderer(const ProgressiveRenderer&) = delete;
    ProgressiveRenderer& operator=(const ProgressiveRenderer&) = delete;

    // Clear accumulation and (re)size buffers — switches between full and
    // preview resolution.
    void reset(int w, int h);

    int width() const  { return w_; }
    int height() const { return h_; }
    int passes() const { return pass_count_; }

    // Render one 1-spp pass with this camera; blocks until the pass is done.
    void render_pass(const Camera& camera);

    // Resolved 8-bit frame of the latest completed pass (w*h*4, RGBX order).
    const std::vector<std::uint8_t>& rgba() const { return rgba_; }

    // Linear radiance sums (divide by passes() for the mean). Used by the
    // GPU parity harness.
    const std::vector<color>& accum() const { return accum_; }

    bool save_png(const std::string& path) const;
    bool save_ppm(const std::string& path) const;

    // Offline mode: spp passes at settings resolution, writes out.ppm/out.png.
    bool render_offline(int spp);

private:
    void worker_loop();
    void work_rows();
    void render_row(int y);

    void render_pass_partitioned();
    const Scene& scene_;
    RenderSettings settings_;
    EnvLookup env_;
    LightsLookup lights_;
    std::vector<GBufferPx> gbuf_;   // Session K: per-pixel primary hits
    std::vector<ReSTIRPixel> resv_;     // persistent (temporal history)
    std::vector<ReSTIRPixel> resv_cur_; // this frame, post-temporal

    int w_ = 0, h_ = 0;
    int pass_count_ = 0;      // completed passes
    int current_pass_ = 0;    // index of the in-flight pass (seeds the RNG)
    Camera camera_;           // camera for the in-flight pass

    std::vector<color> accum_;         // linear radiance sums
    std::vector<std::uint8_t> rgba_;   // tone-mapped resolve of latest pass

    // Worker pool: threads park on cv_start_ between passes; job_id_ tells
    // them a new pass began. Workers steal rows via the atomic counter, then
    // report their row counts; the pass is done when rows_done_ == job_rows_.
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::uint64_t job_id_ = 0;
    bool quit_ = false;
    std::atomic<int> next_row_{0};
    int rows_done_ = 0;
    int job_rows_ = 0;
};
