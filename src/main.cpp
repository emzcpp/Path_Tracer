// Mode dispatcher.
//   (default)          live interactive viewer, GPU-accelerated (macOS)
//   --cpu              viewer with the CPU reference backend
//   --offline [spp]    CPU render to out.ppm/out.png (bare spp also works)
//   --parity [spp]     render CPU and GPU with identical seeds and diff them
//   --gpu-check        compile kernels, print device capabilities

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#include "gltf_loader.h"
#include "image.h"
#include "renderer.h"
#include "scene_setup.h"
#include "settings.h"
#ifdef PT_HAVE_VIEWER
#include "viewer.h"
#endif
#ifdef PT_HAVE_METAL
#include "gpu_renderer.h"
#endif

#ifdef PT_HAVE_METAL
namespace {

struct DiffStats {
    double mean_byte = 0, max_byte = 0;      // after gamma encode, 0..255
    double pct_within_2 = 0, pct_exact = 0;  // per-pixel, all channels
    double mean_linear = 0, max_linear = 0;
};

// Compare two accumulation buffers (linear sums; inv_* are the 1/pass
// normalizers). Tone-mapped stats use the same encode_channel as file
// output, so thresholds are in real display LSBs.
DiffStats compare_accums(const std::vector<color>& a, float inv_a,
                         const std::vector<color>& b, float inv_b) {
    DiffStats s;
    size_t within2 = 0, exact = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float la[3] = {a[i].x * inv_a, a[i].y * inv_a, a[i].z * inv_a};
        const float lb[3] = {b[i].x * inv_b, b[i].y * inv_b, b[i].z * inv_b};
        int worst = 0;
        for (int c = 0; c < 3; ++c) {
            const double lin = std::fabs(double(la[c]) - double(lb[c]));
            s.mean_linear += lin;
            s.max_linear = std::max(s.max_linear, lin);
            const int byte_diff =
                std::abs(int(encode_channel(la[c])) - int(encode_channel(lb[c])));
            s.mean_byte += byte_diff;
            s.max_byte = std::max(s.max_byte, double(byte_diff));
            worst = std::max(worst, byte_diff);
        }
        if (worst <= 2) ++within2;
        if (worst == 0) ++exact;
    }
    const double n = double(a.size());
    s.mean_byte /= n * 3.0;
    s.mean_linear /= n * 3.0;
    s.pct_within_2 = 100.0 * double(within2) / n;
    s.pct_exact = 100.0 * double(exact) / n;
    return s;
}

void write_accum_png(const std::vector<color>& accum, float inv, int w, int h,
                     const char* path) {
    Image img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img.at(x, y) = accum[size_t(y) * w + x] * inv;
    img.write_png(path);
}

int run_parity(const RenderSettings& settings, int spp,
               const SceneDesc& desc) {
    using Clock = std::chrono::steady_clock;
    const int w = settings.width, h = settings.height;
    const size_t n = size_t(w) * h;
    std::printf("parity: %dx%d @ %d spp, identical per-(pixel,pass) seeds%s\n",
                w, h, spp, desc.mesh ? ", mesh present" : "");

    const Camera camera(settings.cam_pos, settings.cam_look_at,
                        settings.cam_up, settings.vfov_deg, settings.aspect());

    // --- GPU: passes 0..spp-1 ---
    std::string err;
    auto gpu = GpuRenderer::create(settings, flatten_scene(desc),
                                   desc.mesh.get(), err);
    if (!gpu) {
        std::fprintf(stderr, "parity: %s\n", err.c_str());
        return 1;
    }
    gpu->set_env(desc.env.get());
    gpu->set_env_params(desc.env_intensity, desc.env_yaw_deg / 360.0f);
    auto t0 = Clock::now();
    gpu->render_passes_blocking(to_gpu_camera(camera), spp);
    const double gpu_s = std::chrono::duration<double>(Clock::now() - t0).count();

    std::vector<color> gpu_accum(n);
    const float* ga = gpu->accum_data();
    for (size_t i = 0; i < n; ++i)
        gpu_accum[i] = color(ga[i * 4], ga[i * 4 + 1], ga[i * 4 + 2]);

    // --- CPU: passes 0..spp-1, then spp..2spp-1 for the noise floor ---
    const Scene scene = make_scene(desc);
    ProgressiveRenderer cpu(scene, settings,
                            std::max(1u, std::thread::hardware_concurrency()),
                            env_lookup(desc, settings.env_nee != 0));
    t0 = Clock::now();
    for (int s = 0; s < spp; ++s) cpu.render_pass(camera);
    const double cpu_s = std::chrono::duration<double>(Clock::now() - t0).count();
    const std::vector<color> cpu_a = cpu.accum();   // snapshot: sums 0..spp-1

    for (int s = 0; s < spp; ++s) cpu.render_pass(camera);
    std::vector<color> cpu_b(n);                    // sums spp..2spp-1
    for (size_t i = 0; i < n; ++i) cpu_b[i] = cpu.accum()[i] - cpu_a[i];

    // --- stats ---
    const float inv = 1.0f / float(spp);
    const DiffStats gd = compare_accums(gpu_accum, inv, cpu_a, inv);
    const DiffStats fl = compare_accums(cpu_a, inv, cpu_b, inv);

    std::printf("\n  timing: GPU %.2fs (%.0f passes/s) | CPU %.2fs (%.0f passes/s)\n",
                gpu_s, spp / gpu_s, cpu_s, spp / cpu_s);
    std::printf("\n  %-34s %14s %18s\n", "", "GPU vs CPU", "CPU noise floor");
    std::printf("  %-34s %14s %18s\n", "", "(same seeds)", "(independent spp)");
    std::printf("  %-34s %14.4f %18.4f\n", "mean |diff| tone-mapped (LSB)",
                gd.mean_byte, fl.mean_byte);
    std::printf("  %-34s %14.1f %18.1f\n", "max  |diff| tone-mapped (LSB)",
                gd.max_byte, fl.max_byte);
    std::printf("  %-34s %14.2f %18.2f\n", "%% pixels within 2 LSB",
                gd.pct_within_2, fl.pct_within_2);
    std::printf("  %-34s %14.2f %18.2f\n", "%% pixels exactly equal",
                gd.pct_exact, fl.pct_exact);
    std::printf("  %-34s %14.2e %18.2e\n", "mean |diff| linear", gd.mean_linear,
                fl.mean_linear);
    std::printf("  %-34s %14.2e %18.2e\n", "max  |diff| linear", gd.max_linear,
                fl.max_linear);

    // --- inspectable artifacts ---
    write_accum_png(cpu_a, inv, w, h, "parity_cpu.png");
    write_accum_png(gpu_accum, inv, w, h, "parity_gpu.png");
    std::vector<color> diff(n);
    for (size_t i = 0; i < n; ++i) {
        const color d = gpu_accum[i] * inv - cpu_a[i] * inv;
        diff[i] = color(std::fabs(d.x), std::fabs(d.y), std::fabs(d.z)) * 32.0f;
    }
    write_accum_png(diff, 1.0f, w, h, "parity_diff.png");
    std::printf("\n  wrote parity_cpu.png / parity_gpu.png / parity_diff.png (|diff|*32)\n");

    // Gate on the bulk, never the max: a ULP that flips a discrete branch
    // (discriminant, Schlick-vs-rng, RR) replaces that whole path — rare
    // divergent samples are expected and harmless. The absolute rates are
    // scene/framing dependent, so the load-bearing gate is RELATIVE: the
    // same-seed error must sit well below the Monte Carlo noise floor.
    const bool pass = gd.mean_byte <= 0.5 &&
                      gd.mean_byte * 5.0 <= fl.mean_byte &&
                      gd.pct_within_2 >= 98.0;
    std::printf("\n  %s (gates: mean <= 0.5 LSB, mean <= floor/5, "
                ">= 98%% within 2 LSB)\n",
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

} // namespace
#endif // PT_HAVE_METAL

namespace {

// Default model lookup: cwd, then cwd's parent (covers running from build/).
std::string resolve_model_path(const std::string& explicit_path,
                               bool no_model) {
    if (no_model) return {};
    if (!explicit_path.empty()) return explicit_path;
    for (const char* c : {"assets/DamagedHelmet.glb",
                          "../assets/DamagedHelmet.glb",
                          "Test_Models/Damaged Helmet.glb",
                          "../Test_Models/Damaged Helmet.glb"}) {
        if (std::filesystem::exists(c)) return c;
    }
    return {};
}

int run_mesh_info(const std::string& path) {
    if (path.empty()) {
        std::fprintf(stderr, "mesh-info: no model found (use --model <path>)\n");
        return 1;
    }
    std::string err;
    const auto mesh = load_glb(path, MeshPlacement{}, err);
    if (!mesh) {
        std::fprintf(stderr, "mesh-info: %s\n", err.c_str());
        return 1;
    }
    const auto& i = mesh->info;
    std::printf("model:      %s\n", path.c_str());
    std::printf("triangles:  %zu   vertices: %zu   indices: %zu\n",
                mesh->tris.size(), i.vert_count, i.index_count);
    std::printf("uv range:   u [%.4f, %.4f]  v [%.4f, %.4f]\n", i.uv_min[0],
                i.uv_max[0], i.uv_min[1], i.uv_max[1]);
    std::printf("materials:  %zu (%.1f MB decoded ushort4 textures)\n",
                i.material_count,
                double(i.texture_bytes) / (1024.0 * 1024.0));
    std::printf("textures:   base %dx%d  mr %dx%d  emissive %dx%d (mat 0)\n",
                mesh->materials[0].base.w, mesh->materials[0].base.h,
                mesh->materials[0].mr.w, mesh->materials[0].mr.h,
                mesh->materials[0].emissive.w, mesh->materials[0].emissive.h);
    std::printf("mesh bbox:  (%.3f %.3f %.3f) .. (%.3f %.3f %.3f)\n",
                i.pre_min[0], i.pre_min[1], i.pre_min[2], i.pre_max[0],
                i.pre_max[1], i.pre_max[2]);
    std::printf("world bbox: (%.3f %.3f %.3f) .. (%.3f %.3f %.3f)\n",
                i.post_min[0], i.post_min[1], i.post_min[2], i.post_max[0],
                i.post_max[1], i.post_max[2]);
    std::printf("bvh:        %d nodes, max depth %d, %d leaves, "
                "%.2f tris/leaf\n",
                i.bvh.node_count, i.bvh.max_depth, i.bvh.leaf_count,
                i.bvh.mean_leaf_tris);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    RenderSettings settings;
    enum class Mode { Viewer, CpuViewer, Offline, Parity, MeshInfo };
    Mode mode = Mode::Viewer;
    int spp_override = 0;
    std::string model_path;
    std::string env_path;
    bool no_model = false;
    bool grid = false;
    bool nan_check = false;
    bool brute = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--offline") == 0) {
            mode = Mode::Offline;
        } else if (std::strcmp(argv[i], "--cpu") == 0) {
            mode = Mode::CpuViewer;
        } else if (std::strcmp(argv[i], "--parity") == 0) {
            mode = Mode::Parity;
        } else if (std::strcmp(argv[i], "--mesh-info") == 0) {
            mode = Mode::MeshInfo;
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--no-model") == 0) {
            no_model = true;
        } else if (std::strcmp(argv[i], "--env") == 0 && i + 1 < argc) {
            env_path = argv[++i];
        } else if (std::strcmp(argv[i], "--nan-check") == 0) {
            nan_check = true;
        } else if (std::strcmp(argv[i], "--brute") == 0) {
            brute = true;
        } else if (std::strcmp(argv[i], "--grid") == 0) {
            grid = true;   // GGX validation: roughness x metallic array
        } else if (std::strcmp(argv[i], "--gpu-check") == 0) {
#ifdef PT_HAVE_METAL
            return gpu_check() ? 0 : 1;
#else
            std::fprintf(stderr, "built without Metal support\n");
            return 1;
#endif
        } else if (int spp = std::atoi(argv[i]); spp > 0) {
            spp_override = spp;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--offline|--parity|--cpu|--gpu-check|"
                         "--mesh-info] [--model <path>|--no-model] [spp]\n",
                         argv[0]);
            return 1;
        }
    }
    // Back-compat: a bare spp count means an offline render.
    if (mode == Mode::Viewer && spp_override > 0) mode = Mode::Offline;
    if (spp_override > 0) settings.samples_per_pixel = spp_override;

    const std::string resolved_model =
        resolve_model_path(model_path, no_model || grid);
    if (mode == Mode::MeshInfo) return run_mesh_info(resolved_model);

    // Build the scene description ONCE; every mode consumes it.
    std::shared_ptr<const MeshData> mesh;
    if (!resolved_model.empty()) {
        std::string err;
        mesh = load_glb(resolved_model, MeshPlacement{}, err);
        if (mesh) {
            std::printf("model: %s (%zu tris, bvh depth %d)\n",
                        resolved_model.c_str(), mesh->tris.size(),
                        mesh->info.bvh.max_depth);
        } else {
            std::fprintf(stderr, "model load failed (%s) — sphere scene\n",
                         err.c_str());
        }
    } else if (!no_model && !grid) {
        std::fprintf(stderr,
                     "no model found (looked for Test_Models/Damaged Helmet"
                     ".glb) — sphere scene; use --model <path>\n");
    }
    SceneDesc desc =
        grid ? build_grid_desc() : build_scene_desc(std::move(mesh));
    if (desc.mesh) desc.mesh_source_path = resolved_model;
    if (brute) {
        settings.env_nee = 0;
        std::printf("lighting: brute force (ground-truth mode)\n");
    }
    if (!env_path.empty()) {
        std::string err;
        auto env = load_hdr(env_path, err);
        if (env) {
            desc.env = env;
            desc.env_source_path = env_path;
            std::printf("environment: %s (%dx%d)\n", env_path.c_str(),
                        env->w, env->h);
        } else {
            std::fprintf(stderr, "%s\n", err.c_str());
        }
    }
    if (grid) {
        // Elevated view so all 36 spheres and both axes read clearly.
        settings.cam_pos = point3(0.0f, 7.5f, 8.5f);
        settings.cam_look_at = point3(0.0f, 0.0f, 0.2f);
        std::printf("grid: roughness 0->1 left to right, "
                    "metallic 0->1 front to back\n");
    }

    // Stage-0 diagnostics: render both backends and count non-finite
    // values that reached the accumulators. Any nonzero count is a bug by
    // definition (NaN/Inf never averages out), not Monte Carlo noise.
    if (nan_check) {
        const int spp = spp_override > 0 ? spp_override : 64;
        const size_t n = size_t(settings.width) * settings.height;
        const auto scan = [](const float* p, size_t floats) {
            size_t bad = 0;
            for (size_t i = 0; i < floats; ++i)
                if (!std::isfinite(p[i])) ++bad;
            return bad;
        };
        std::printf("nan-check: %dx%d @ %d spp, clamp_indirect=%.1f%s\n",
                    settings.width, settings.height, spp,
                    settings.clamp_indirect,
                    desc.env ? ", HDRI env" : ", gradient env");

        const Scene scene = make_scene(desc);
        ProgressiveRenderer cpu(scene, settings,
                                std::max(1u, std::thread::hardware_concurrency()),
                                env_lookup(desc, settings.env_nee != 0));
        const Camera camera(settings.cam_pos, settings.cam_look_at,
                            settings.cam_up, settings.vfov_deg,
                            settings.aspect());
        for (int i = 0; i < spp; ++i) cpu.render_pass(camera);
        const size_t cpu_bad =
            scan(reinterpret_cast<const float*>(cpu.accum().data()), n * 3);
        std::printf("  CPU accumulator: %zu non-finite of %zu floats\n",
                    cpu_bad, n * 3);

        size_t gpu_bad = 0;
        bool gpu_ran = false;
#ifdef PT_HAVE_METAL
        std::string err;
        if (auto gpu = GpuRenderer::create(settings, flatten_scene(desc),
                                           desc.mesh.get(), err)) {
            gpu->set_env(desc.env.get());
            gpu->set_env_params(desc.env_intensity, desc.env_yaw_deg / 360.0f);
            gpu->render_passes_blocking(to_gpu_camera(camera), spp);
            gpu_bad = scan(gpu->accum_data(), n * 4);
            gpu_ran = true;
            std::printf("  GPU accumulator: %zu non-finite of %zu floats\n",
                        gpu_bad, n * 4);
        } else {
            std::fprintf(stderr, "  GPU unavailable: %s\n", err.c_str());
        }
#endif
        const bool pass = cpu_bad == 0 && (!gpu_ran || gpu_bad == 0);
        std::printf("  %s\n", pass ? "PASS (zero non-finite values)"
                                    : "FAIL — NaN/Inf reached an accumulator");
        return pass ? 0 : 1;
    }

    if (mode == Mode::Parity) {
#ifdef PT_HAVE_METAL
        return run_parity(settings, spp_override > 0 ? spp_override : 64, desc);
#else
        std::fprintf(stderr, "built without Metal support\n");
        return 1;
#endif
    }

#ifdef PT_HAVE_VIEWER
    if (mode == Mode::Viewer || mode == Mode::CpuViewer) {
        return run_viewer(settings, /*use_gpu=*/mode == Mode::Viewer, desc);
    }
#else
    if (mode != Mode::Offline) {
        std::fprintf(stderr, "viewer unavailable on this platform; rendering offline\n");
    }
#endif

    const Scene scene = make_scene(desc);
    ProgressiveRenderer renderer(scene, settings,
                                 std::max(1u, std::thread::hardware_concurrency()),
                                 env_lookup(desc, settings.env_nee != 0));

    std::printf("rendering %dx%d @ %d spp, max depth %d\n", settings.width,
                settings.height, settings.samples_per_pixel, settings.max_depth);
    if (!renderer.render_offline(settings.samples_per_pixel)) {
        std::printf("FAILED to write output\n");
        return 1;
    }
    std::printf("wrote out.ppm and out.png\n");
    return 0;
}
