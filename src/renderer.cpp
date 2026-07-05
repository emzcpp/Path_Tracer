#include "renderer.h"

#include <atomic>
#include <chrono>
#include <thread>
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
    // Session K: reservoirs reset with the accumulation — this IS the
    // temporal-reuse ghosting guarantee (central reset -> fresh history).
    if (settings_.restir != 0) {
        resv_.assign(std::size_t(w) * h, ReSTIRPixel{});
    }
}

void ProgressiveRenderer::render_pass(const Camera& camera) {
    current_pass_ = pass_count_;
    camera_ = camera;
    if (settings_.restir != 0) {
        render_pass_partitioned();
        ++pass_count_;
        return;
    }
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
                              settings_.clamp_indirect, nullptr, false,
                              settings_.spectral != 0,
                              settings_.dispersion_b);

        const color c = accum_row[x] * inv_n;
        out[x * 4 + 0] = encode_channel(c.x);
        out[x * 4 + 1] = encode_channel(c.y);
        out[x * 4 + 2] = encode_channel(c.z);
        out[x * 4 + 3] = 255;
    }
}

// Session K (ReSTIR Stage 0.5): the partitioned pass — three sequential
// row-parallel phases with barriers, mirroring the GPU pipeline. Phase A
// finds primary hits into the G-buffer (jitter draws), phase D adds
// vertex-0 direct lighting via the SAME sample_direct and the SAME rng
// stream (state travels through the G-buffer), phase I continues the
// path with the first hit injected and vertex-0 direct skipped.
void ProgressiveRenderer::render_pass_partitioned() {
    if (gbuf_.size() != std::size_t(w_) * h_) {
        gbuf_.assign(std::size_t(w_) * h_, GBufferPx{});
    }
    if (gbuf_prev_.size() != std::size_t(w_) * h_) {
        gbuf_prev_.assign(std::size_t(w_) * h_, GBufferPx{});
    }
    if (resv_.size() != std::size_t(w_) * h_) {
        resv_.assign(std::size_t(w_) * h_, ReSTIRPixel{});
    }
    if (resv_cur_.size() != std::size_t(w_) * h_) {
        resv_cur_.assign(std::size_t(w_) * h_, ReSTIRPixel{});
    }
    const auto par_rows = [&](auto&& fn) {
        const unsigned T =
            std::max(1u, std::thread::hardware_concurrency());
        std::atomic<int> next{0};
        std::vector<std::thread> ts;
        ts.reserve(T);
        for (unsigned i = 0; i < T; ++i) {
            ts.emplace_back([&] {
                for (int y = next.fetch_add(1); y < h_;
                     y = next.fetch_add(1)) {
                    fn(y);
                }
            });
        }
        for (auto& t : ts) t.join();
    };
    const float inv_w = 1.0f / float(w_);
    const float inv_h = 1.0f / float(h_);
    const float inv_n = 1.0f / float(current_pass_ + 1);

    // Phase A: primary hits.
    par_rows([&](int y) {
        for (int x = 0; x < w_; ++x) {
            const std::uint64_t px = std::uint64_t(y) * w_ + x;
            RNG rng(mix64(px ^ (std::uint64_t(current_pass_) << 32)), px);
            const float u = (x + rng.next_float()) * inv_w;
            const float v = 1.0f - (y + rng.next_float()) * inv_h;
            const Ray r = camera_.get_ray(u, v);
            GBufferPx& g = gbuf_[px];
            HitRecord rec;
            if (scene_.hit(r, 1e-3f,
                           std::numeric_limits<float>::infinity(), rec)) {
                g.pos = {rec.p.x, rec.p.y, rec.p.z};
                g.t = rec.t;
                g.normal = {rec.normal.x, rec.normal.y, rec.normal.z};
                g.flags = rec.front_face ? 1u : 0u;
                g.base_color = {rec.mat.base_color.x, rec.mat.base_color.y,
                                rec.mat.base_color.z};
                g.metallic = rec.mat.metallic;
                g.emission = {rec.mat.emission.x, rec.mat.emission.y,
                              rec.mat.emission.z};
                g.ior = rec.mat.ior;
                g.roughness = rec.mat.roughness;
                g.transmission = rec.mat.transmission;
                g.light_id_p1 = pt_uint(rec.light_id + 1);
            } else {
                g.t = -1.0f;
            }
            g.rd = {r.dir.x, r.dir.y, r.dir.z};
            g.rng_lo = pt_uint(rng.state & 0xffffffffULL);
            g.rng_hi = pt_uint(rng.state >> 32);
        }
    });

    // Phase D1: candidates + temporal merge -> per-frame reservoirs.
    // Shared helper builds the surface record from the G-buffer.
    const auto rec_from_gbuf = [](const GBufferPx& g, HitRecord& rec) {
        rec.p = point3(g.pos.x, g.pos.y, g.pos.z);
        rec.normal = vec3(g.normal.x, g.normal.y, g.normal.z);
        rec.front_face = (g.flags & 1u) != 0u;
        rec.mat.base_color =
            color(g.base_color.x, g.base_color.y, g.base_color.z);
        rec.mat.emission = color(g.emission.x, g.emission.y, g.emission.z);
        rec.mat.metallic = g.metallic;
        rec.mat.roughness = g.roughness;
        rec.mat.ior = g.ior;
        rec.mat.transmission = g.transmission;
        rec.t = g.t;
    };
    par_rows([&](int y) {
        for (int x = 0; x < w_; ++x) {
            const std::uint64_t px = std::uint64_t(y) * w_ + x;
            GBufferPx& g = gbuf_[px];
            if (g.t < 0.0f) continue;
            HitRecord rec;
            rec_from_gbuf(g, rec);
            RNG rng(0, px);
            rng.state =
                (std::uint64_t(g.rng_hi) << 32) | std::uint64_t(g.rng_lo);
            const Ray pray(rec.p, vec3(g.rd.x, g.rd.y, g.rd.z));
            restir_build(rec, pray, rng, env_, lights_, settings_.restir_m,
                         settings_.restir_temporal != 0,
                         settings_.restir_mcap, resv_[px], gbuf_prev_[px],
                         resv_cur_[px]);
            g.rng_lo = pt_uint(rng.state & 0xffffffffULL);
            g.rng_hi = pt_uint(rng.state >> 32);
        }
    });

    // Phase D2: spatial merge (unbiased 1/Z) + shadow ray + shading, and
    // the persistent store that feeds next frame's temporal reuse.
    par_rows([&](int y) {
        color* accum_row = &accum_[std::size_t(y) * w_];
        for (int x = 0; x < w_; ++x) {
            const std::uint64_t px = std::uint64_t(y) * w_ + x;
            GBufferPx& g = gbuf_[px];
            if (g.t < 0.0f) continue;
            HitRecord rec;
            rec_from_gbuf(g, rec);
            RNG rng(0, px);
            rng.state =
                (std::uint64_t(g.rng_hi) << 32) | std::uint64_t(g.rng_lo);
            const Ray pray(rec.p, vec3(g.rd.x, g.rd.y, g.rd.z));
            color rad(0.0f);
            restir_spatial_shade(rec, pray, scene_, rng, env_, lights_,
                                 settings_.restir_m,
                                 settings_.restir_spatial != 0,
                                 settings_.restir_k, settings_.restir_radius,
                                 gbuf_.data(), resv_cur_.data(), w_, h_, x,
                                 y, resv_[px], settings_.clamp_indirect,
                                 rad);
            accum_row[x] += rad;
            g.rng_lo = pt_uint(rng.state & 0xffffffffULL);
            g.rng_hi = pt_uint(rng.state >> 32);
        }
    });

    // Phase I: indirect continuation (+ display encode).
    par_rows([&](int y) {
        color* accum_row = &accum_[std::size_t(y) * w_];
        std::uint8_t* out = &rgba_[std::size_t(y) * w_ * 4];
        for (int x = 0; x < w_; ++x) {
            const std::uint64_t px = std::uint64_t(y) * w_ + x;
            const GBufferPx& g = gbuf_[px];
            RNG rng(0, px);
            rng.state =
                (std::uint64_t(g.rng_hi) << 32) | std::uint64_t(g.rng_lo);
            const Ray pray(point3(g.pos.x, g.pos.y, g.pos.z),
                           vec3(g.rd.x, g.rd.y, g.rd.z));
            if (g.t < 0.0f) {
                // Primary miss: the monolithic depth-0 env branch.
                accum_row[x] += miss_radiance(env_, pray);
            } else {
                HitRecord pre;
                pre.p = pray.origin;
                pre.t = g.t;
                pre.normal = vec3(g.normal.x, g.normal.y, g.normal.z);
                pre.front_face = (g.flags & 1u) != 0u;
                pre.mat.base_color =
                    color(g.base_color.x, g.base_color.y, g.base_color.z);
                pre.mat.emission =
                    color(g.emission.x, g.emission.y, g.emission.z);
                pre.mat.metallic = g.metallic;
                pre.mat.roughness = g.roughness;
                pre.mat.ior = g.ior;
                pre.mat.transmission = g.transmission;
                pre.light_id = int(g.light_id_p1) - 1;
                accum_row[x] +=
                    trace(pray, scene_, rng, settings_.max_depth, env_,
                          lights_, settings_.clamp_indirect, &pre, true);
            }
            const color c = accum_row[x] * inv_n;
            out[x * 4 + 0] = encode_channel(c.x);
            out[x * 4 + 1] = encode_channel(c.y);
            out[x * 4 + 2] = encode_channel(c.z);
            out[x * 4 + 3] = 255;
        }
    });

    // Ping-pong: this frame's surfaces become next frame's temporal
    // balance-weight reference.
    std::swap(gbuf_, gbuf_prev_);
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
