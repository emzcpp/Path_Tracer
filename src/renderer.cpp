#include "renderer.h"

#include <chrono>
#include <cstdio>

#include "image.h"
#include "integrator.h"
#include "rng.h"

ProgressiveRenderer::ProgressiveRenderer(const Scene& scene,
                                         const RenderSettings& settings,
                                         unsigned total_threads,
                                         const EnvLookup& env,
                                         const LightsLookup& lights)
    : scene_(scene), settings_(settings), env_(env), lights_(lights) {
    reset(settings.width, settings.height);
    const unsigned n_workers = total_threads > 1 ? total_threads - 1 : 0;
    workers_.reserve(n_workers);
    for (unsigned i = 0; i < n_workers; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ProgressiveRenderer::~ProgressiveRenderer() {
    {
        std::lock_guard lk(m_);
        quit_ = true;
    }
    cv_start_.notify_all();
    for (auto& w : workers_) w.join();
}

void ProgressiveRenderer::reset(int w, int h) {
    w_ = w;
    h_ = h;
    pass_count_ = 0;
    accum_.assign(std::size_t(w) * h, color(0.0f));
    rgba_.assign(std::size_t(w) * h * 4, 0);
}

void ProgressiveRenderer::render_pass(const Camera& camera) {
    current_pass_ = pass_count_;
    camera_ = camera;
    {
        std::lock_guard lk(m_);
        next_row_ = 0;
        rows_done_ = 0;
        job_rows_ = h_;
        ++job_id_;
    }
    cv_start_.notify_all();
    work_rows();   // the calling thread pulls rows too
    {
        std::unique_lock lk(m_);
        cv_done_.wait(lk, [this] { return rows_done_ == job_rows_; });
    }
    ++pass_count_;
}

void ProgressiveRenderer::worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock lk(m_);
            cv_start_.wait(lk, [&] { return quit_ || job_id_ != seen; });
            if (quit_) return;
            seen = job_id_;
        }
        work_rows();
    }
}

void ProgressiveRenderer::work_rows() {
    int done = 0;
    for (int y = next_row_.fetch_add(1); y < h_; y = next_row_.fetch_add(1)) {
        render_row(y);
        ++done;
    }
    // A worker's nonzero contribution always belongs to the current pass:
    // the orchestrator can't start the next one until every row is counted.
    std::lock_guard lk(m_);
    rows_done_ += done;
    if (rows_done_ == job_rows_) cv_done_.notify_one();
}

void ProgressiveRenderer::render_row(int y) {
    const float inv_w = 1.0f / float(w_);
    const float inv_h = 1.0f / float(h_);
    const float inv_n = 1.0f / float(current_pass_ + 1);
    color* accum_row = &accum_[std::size_t(y) * w_];
    std::uint8_t* out = &rgba_[std::size_t(y) * w_ * 4];

    for (int x = 0; x < w_; ++x) {
        // Seed with hash(pixel, pass) and a per-pixel PCG stream: samples
        // must differ across passes (or accumulation would just repeat the
        // same estimate) yet stay independent of thread scheduling, so the
        // image is deterministic at any thread count.
        const std::uint64_t px = std::uint64_t(y) * w_ + x;
        RNG rng(mix64(px ^ (std::uint64_t(current_pass_) << 32)), px);

        const float u = (x + rng.next_float()) * inv_w;
        const float v = 1.0f - (y + rng.next_float()) * inv_h;
        accum_row[x] += trace(camera_.get_ray(u, v), scene_, rng,
                              settings_.max_depth, env_, lights_,
                              settings_.clamp_indirect);

        const color c = accum_row[x] * inv_n;
        out[x * 4 + 0] = encode_channel(c.x);
        out[x * 4 + 1] = encode_channel(c.y);
        out[x * 4 + 2] = encode_channel(c.z);
        out[x * 4 + 3] = 255;
    }
}

bool ProgressiveRenderer::save_png(const std::string& path) const {
    const float inv_n = 1.0f / float(std::max(1, pass_count_));
    Image img(w_, h_);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
            img.at(x, y) = accum_[std::size_t(y) * w_ + x] * inv_n;
    return img.write_png(path);
}

bool ProgressiveRenderer::save_ppm(const std::string& path) const {
    const float inv_n = 1.0f / float(std::max(1, pass_count_));
    Image img(w_, h_);
    for (int y = 0; y < h_; ++y)
        for (int x = 0; x < w_; ++x)
            img.at(x, y) = accum_[std::size_t(y) * w_ + x] * inv_n;
    return img.write_ppm(path);
}

bool ProgressiveRenderer::render_offline(int spp) {
    reset(settings_.width, settings_.height);
    const Camera camera(settings_.cam_pos, settings_.cam_look_at,
                        settings_.cam_up, settings_.vfov_deg,
                        settings_.aspect());

    const auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < spp; ++s) {
        render_pass(camera);
        if ((s + 1) % 16 == 0 || s + 1 == spp) {
            std::fprintf(stderr, "\r%d/%d passes", s + 1, spp);
            std::fflush(stderr);
        }
    }
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "\rdone in %.2fs        \n", elapsed);

    return save_ppm("out.ppm") && save_png("out.png");
}
