#pragma once

// Every render knob lives here. Tweak, rebuild, rerun.

#include "vec3.h"

struct RenderSettings {
    // Image
    int width  = 960;
    int height = 540;

    // Sampling
    int samples_per_pixel = 256;  // jittered samples averaged per pixel
    int max_depth         = 16;   // hard bounce cutoff (russian roulette
                                  // usually terminates paths well before this)

    // Camera
    point3 cam_pos     {7.4f, 1.9f, 5.8f};
    point3 cam_look_at {0.0f, 1.0f, 0.0f};
    vec3   cam_up      {0.0f, 1.0f, 0.0f};
    float  vfov_deg    = 35.0f;

    // Interactive viewer
    int   preview_divisor  = 2;      // resolution divisor while the camera moves
    int   max_accum_passes = 4096;   // stop refining after this many spp
    float settle_ms        = 150.0f; // camera still this long → back to full res
    int   final_target_spp = 2048;   // FINAL mode (F key): converge to this,
                                     // then auto-export a PNG
    // Firefly control (Cycles-style "clamp indirect"): cap the per-sample
    // contribution of indirect bounces. Direct hits (visible sun/lights/
    // background) are never clamped. Clamping is BIASED, so the default is
    // 0 = OFF: FINAL, --offline, and --parity render ground truth unless
    // explicitly overridden. The viewer applies its own preview-only clamp
    // in interactive mode (see ViewerCore) — HDRI suns otherwise leave
    // white speckle in the preview that outlives any sample count.
    float clamp_indirect = 0.0f;

    // GPU backend: passes encoded per 60 Hz tick (per command buffer).
    // These are CAPS; the budget controller below decides the actual work.
    int gpu_passes_per_tick_preview = 4;
    int gpu_passes_per_tick         = 8;

    // Session G: per-command-buffer GPU-time budget. No trace dispatch is
    // allowed to run much past this — heavy frames are sliced into row
    // ranges across ticks instead — so the compositor and the UI always
    // get the GPU within one budget window. FINAL trades some of that
    // headroom for convergence speed but stays preemptible/cancellable.
    float gpu_budget_ms       = 10.0f;
    float gpu_budget_ms_final = 24.0f;

    // Session H: environment light sampling. 1 = NEE+MIS (default),
    // 0 = brute force — kept as the ground-truth reference: both must
    // converge to the same image, NEE just gets there far cleaner.
    int env_nee = 1;

    // Session K: 1 = partitioned direct/indirect pipeline (ReSTIR
    // scaffolding; Stage 0.5 reproduces plain NEE+MIS through it),
    // 0 = the monolithic integrator.
    int restir = 0;
    // RIS candidates per light slot (env / area) at the primary vertex.
    int restir_m = 8;
    // Stage 2: temporal reservoir reuse. DEFAULT OFF pending the
    // prev-surface balance correction: the Mtot-style temporal combine
    // biases +5.7% on scenes with emitters very close to surfaces (the
    // target mismatch under sub-pixel jitter is large at close range;
    // benign scenes measure ~0.0%). The fix needs last frame's G-buffer
    // so history can carry proper balance-heuristic weights — scoped for
    // a follow-up session. Toggle lives in the panel for A/B.
    int restir_temporal = 0;
    // Stage 3: spatial reservoir reuse (K random neighbors per frame)
    // combined with Talbot balance-heuristic MIS weights — unbiased for
    // any sampler-support overlap (the earlier 1/Z counting variant
    // measured +1.3% bias from support asymmetry and was replaced).
    int restir_spatial = 1;
    int restir_k = 3;          // spatial neighbors (max 3)
    int restir_radius = 16;    // spatial radius, pixels
    int restir_mcap = 20;      // temporal history cap, x candidate count
    // Reservoir + G-buffer memory guard: above this budget the viewer
    // clamps resolution scale instead of silently allocating (~304 B/px).
    int restir_mem_budget_mb = 1500;

    // Execution
    int thread_count = 0;         // 0 = use all hardware threads

    float aspect() const { return float(width) / float(height); }
};
