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
    // Stage 2: temporal reservoir reuse — ON. The temporal merge is the
    // same Talbot balance-heuristic combine as spatial, with history's
    // target evaluated on LAST frame's surface (G-buffer ping-pong):
    // the old Mtot-style combine biased +5.7% on normal-mapped scenes
    // with empty-reservoir regions; the balance combine measures +0.1%
    // alongside RIS +0.06% and spatial +0.09% on the same gate.
    int restir_temporal = 1;
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
    // v1.1: display-only edge-aware denoiser (A-trous, G-buffer guided).
    // NEVER touches the accumulator/parity/FINAL paths; interactive Metal
    // viewer only. Fades out as spp approaches denoise_fade_spp so the
    // converged image is the true accumulated one (the pass is skipped
    // entirely at alpha == 0).
    int denoise = 0;
    int denoise_iters = 3;       // a-trous iterations (steps 1,2,4,..)
    int denoise_fade_spp = 96;
    // v1.2 spectral rendering: each path carries ONE wavelength; RGB is
    // reconstructed at the sensor. Changes tracing, so it resets
    // accumulation (NOT display-only). dispersion_b feeds the Cauchy IOR
    // n(lambda)=A+B/lambda^2 in Stage 2 (0 = no dispersion, matches Stage 1).
    int spectral = 0;
    float dispersion_b = 0.0f;
    // v1.3 participating medium (homogeneous global fog). Changes tracing,
    // so it resets accumulation. fog == 0 -> vacuum, byte-identical.
    // Stage 1 uses only density (Beer-Lambert transmittance); g + color
    // drive the Stage 2 in-scattering phase function / scatter albedo.
    int fog = 0;
    float fog_density = 0.10f;   // sigma_t (extinction per world unit)
    float fog_g = 0.0f;          // Henyey-Greenstein anisotropy [-1,1]
    color fog_color = color(1.0f, 1.0f, 1.0f);   // scattering albedo (RGB)

    // Execution
    int thread_count = 0;         // 0 = use all hardware threads

    float aspect() const { return float(width) / float(height); }
};
