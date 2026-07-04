#pragma once

// The path tracer core — future Metal kernel. Iterative (a for-loop, not
// recursion) because that's the shape a GPU wants, and it makes the state
// explicit: `throughput` is the product of all albedos along the path so
// far, i.e. "what fraction of light found from here on will survive the
// trip back to the camera".

#include <limits>

#include "hittable.h"
#include "gltf_loader.h"
#include "kernel_types.h"
#include "material.h"
#include "texture.h"
#include "ray.h"
#include "rng.h"

// The dome light: sky white at the horizon blending to blue overhead.
// Paths that escape the scene pick up this radiance. Emissive surfaces in
// the scene add their own light on top.
inline color sky_radiance(const Ray& r) {
    const float t = 0.5f * (normalize(r.dir).y + 1.0f);
    return lerp(color(1.0f, 1.0f, 1.0f), color(0.5f, 0.7f, 1.0f), t);
}

// ---- Session F: equirectangular HDRI environment ------------------------
// Lookup ONLY (no NEE / importance sampling — next session): missed rays
// sample the env map exactly where they sampled the gradient. All code
// below is mirrored line-for-line in pathtrace.metal; --parity is the
// drift detector, with special suspicion on atan2/acos ULPs.

struct EnvLookup {
    const float* texels = nullptr;   // RGBA float, linear radiance
    int w = 0, h = 0;
    float intensity = 1.0f;
    float yaw_norm = 0.0f;           // yaw / 2pi, added to u
    // Session H: importance-sampling CDFs (null = NEE unavailable) and the
    // light-sampling toggle. Brute force remains the ground-truth mode.
    const float* row_cdf = nullptr;
    const float* cond_cdf = nullptr;
    bool nee = false;
};

inline color env_fetch(const float* tex, int W, int x, int y) {
    const float* p = tex + (std::size_t(y) * W + x) * 4;
    return color(p[0], p[1], p[2]);
}

// Same manual-bilinear structure as material textures, with equirect
// semantics: u WRAPS (azimuth seam), v CLAMPS (poles).
inline color sample_env_bilinear(const float* tex, int W, int H, float u,
                                 float v) {
    u = u - std::floor(u);
    v = std::fmin(std::fmax(v, 0.0f), 1.0f);
    const float x = u * float(W) - 0.5f;
    const float y = v * float(H) - 0.5f;
    const float fx = std::floor(x), fy = std::floor(y);
    const float ax = x - fx, ay = y - fy;
    int x0 = int(fx), x1 = x0 + 1;
    int y0 = int(fy), y1 = y0 + 1;
    if (x0 < 0) x0 += W;
    if (x1 >= W) x1 -= W;
    if (y0 < 0) y0 = 0;
    if (y1 >= H) y1 = H - 1;
    const color c00 = env_fetch(tex, W, x0, y0), c10 = env_fetch(tex, W, x1, y0);
    const color c01 = env_fetch(tex, W, x0, y1), c11 = env_fetch(tex, W, x1, y1);
    return (1.0f - ax) * (1.0f - ay) * c00 + ax * (1.0f - ay) * c10 +
           (1.0f - ax) * ay * c01 + ax * ay * c11;
}

// ---- Session H: env importance sampling (mirrored in pathtrace.metal) --
// The distribution is luminance x sin(theta) over the equirect image;
// image-space density converts to solid-angle pdf via
//   pdf_sa = p_uv / (2 pi^2 sin(theta)),
// with the SAME sin(theta) that weighted the CDF rows — carried
// consistently in both sample() and pdf() below.

// First index whose cdf value exceeds u. Identical loop on both backends.
inline int cdf_find(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (cdf[mid] > u) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Draw a direction from the env distribution; returns pdf w.r.t. solid
// angle (0 on degenerate rows/poles — caller skips those samples).
inline vec3 env_sample(const EnvLookup& env, float u1, float u2,
                       float& pdf_sa) {
    const int W = env.w, H = env.h;
    const int y = cdf_find(env.row_cdf, H, u1);
    const float* crow = env.cond_cdf + std::size_t(y) * W;
    const int x = cdf_find(crow, W, u2);
    const float row_lo = y > 0 ? env.row_cdf[y - 1] : 0.0f;
    const float row_w = env.row_cdf[y] - row_lo;
    const float col_lo = x > 0 ? crow[x - 1] : 0.0f;
    const float col_w = crow[x] - col_lo;
    // Continuous inversion inside the chosen texel (keeps stratification).
    const float fy = row_w > 0.0f ? (u1 - row_lo) / row_w : 0.5f;
    const float fx = col_w > 0.0f ? (u2 - col_lo) / col_w : 0.5f;
    const float u = (float(x) + fx) / float(W);
    const float v = (float(y) + fy) / float(H);
    // UV -> direction: exact inverse of the miss mapping, yaw included.
    const float phi = (u - 0.5f - env.yaw_norm) * 6.28318530717958648f;
    const float theta = v * 3.14159265358979f;
    const float st = std::sin(theta);
    const vec3 d(st * std::cos(phi), std::cos(theta), st * std::sin(phi));
    const float p_uv = row_w * col_w * float(W) * float(H);
    pdf_sa = st > 1e-6f
                 ? p_uv / (2.0f * 3.14159265358979f * 3.14159265358979f * st)
                 : 0.0f;
    return d;
}

// Solid-angle pdf the sampler above would assign to an arbitrary
// direction — the MIS counterpart. Same dir->UV text as miss_radiance.
inline float env_pdf(const EnvLookup& env, const vec3& dir) {
    const vec3 d = normalize(dir);
    float u = std::atan2(d.z, d.x) * 0.15915494309189533577f + 0.5f +
              env.yaw_norm;
    u = u - std::floor(u);
    const float v = std::acos(std::fmin(std::fmax(d.y, -1.0f), 1.0f)) *
                    0.31830988618379067154f;
    const int W = env.w, H = env.h;
    int x = int(u * float(W));
    int y = int(v * float(H));
    if (x >= W) x = W - 1;
    if (y >= H) y = H - 1;
    const float* crow = env.cond_cdf + std::size_t(y) * W;
    const float row_lo = y > 0 ? env.row_cdf[y - 1] : 0.0f;
    const float row_w = env.row_cdf[y] - row_lo;
    const float col_lo = x > 0 ? crow[x - 1] : 0.0f;
    const float col_w = crow[x] - col_lo;
    const float st = std::sqrt(std::fmax(0.0f, 1.0f - d.y * d.y));
    if (st <= 1e-6f) return 0.0f;
    return row_w * col_w * float(W) * float(H) /
           (2.0f * 3.14159265358979f * 3.14159265358979f * st);
}

// ---- Session J: area-light NEE (mirrored in pathtrace.metal) -----------
// Scene emitters (emissive spheres now, mesh triangles in Stage 2) get the
// same treatment the env got in Session H: a deliberate shadow-rayed
// sample per vertex instead of waiting for a BSDF ray to stumble onto
// them. ONE light is picked per vertex, power-proportionally (1/sel_pdf
// in the estimator).

struct LightsLookup {
    const GPULight* lights = nullptr;
    int count = 0;
    // For triangle lights: the mesh whose materials hold the emissive
    // textures — Le is evaluated at the sampled point (textured emitters
    // must converge to brute force exactly; a constant Le would not).
    const MeshData* mesh = nullptr;
};

// Binary search of the strided per-entry selection CDF: first light whose
// sel_cdf exceeds u. Identical loop on both backends.
inline int light_pick(const GPULight* lights, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (lights[mid].sel_cdf > u) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

// Uniform solid-angle sampling of the cone a sphere light subtends from x
// (better behaved than area sampling: every sampled direction hits the
// light). Returns the direction; pdf_sa and the analytic distance to the
// sphere along it. pdf_sa = 0 flags degenerate cases (x inside/on the
// sphere, or a cone too small for float) — caller skips.
inline vec3 sample_sphere_light(const GPULight& L, const vec3& x, float u1,
                                float u2, float& pdf_sa, float& t_light) {
    pdf_sa = 0.0f;
    t_light = 0.0f;
    const vec3 cx = vec3(L.p0.x, L.p0.y, L.p0.z) - x;
    const float d2 = dot(cx, cx);
    const float d = std::sqrt(d2);
    if (d <= L.radius * 1.0001f) return vec3(0.0f, 0.0f, 1.0f);
    const float sin2max = (L.radius * L.radius) / d2;
    const float cosmax = std::sqrt(std::fmax(0.0f, 1.0f - sin2max));
    const float one_minus = 1.0f - cosmax;
    if (one_minus < 1e-8f) return vec3(0.0f, 0.0f, 1.0f);
    const float cost = 1.0f - u1 * one_minus;
    const float sint = std::sqrt(std::fmax(0.0f, 1.0f - cost * cost));
    const float phi = 6.28318530717958648f * u2;
    const vec3 w = cx / d;
    vec3 t, b;
    build_onb(w, t, b);
    const vec3 dir = normalize(t * (std::cos(phi) * sint) +
                               b * (std::sin(phi) * sint) + w * cost);
    pdf_sa = 1.0f / (6.28318530717958648f * one_minus);
    const float dc = dot(cx, dir);
    const float disc = L.radius * L.radius - (d2 - dc * dc);
    t_light = dc - std::sqrt(std::fmax(0.0f, disc));
    return dir;
}

// Solid-angle pdf the area-light sampler would assign to the direction
// x -> (hit at distance t on light L) — the MIS counterpart for BSDF rays
// that land on an emitter. Mirrors sample_sphere_light / sample_tri_light
// exactly; 0 = the sampler could not produce this direction (BSDF keeps
// full weight, consistent with the sampler contributing nothing there).
inline float light_dir_pdf(const GPULight& L, const vec3& x, const vec3& dir,
                           float t_hit) {
    if (L.kind == 0u) {
        const vec3 cx = vec3(L.p0.x, L.p0.y, L.p0.z) - x;
        const float d2 = dot(cx, cx);
        const float d = std::sqrt(d2);
        if (d <= L.radius * 1.0001f) return 0.0f;
        const float sin2max = (L.radius * L.radius) / d2;
        const float cosmax = std::sqrt(std::fmax(0.0f, 1.0f - sin2max));
        const float one_minus = 1.0f - cosmax;
        if (one_minus < 1e-8f) return 0.0f;
        return 1.0f / (6.28318530717958648f * one_minus);
    }
    const vec3 e1(L.e1.x, L.e1.y, L.e1.z);
    const vec3 e2(L.e2.x, L.e2.y, L.e2.z);
    const vec3 cr = cross(e1, e2);
    const float two_area = cr.length();
    if (two_area < 1e-12f) return 0.0f;
    const float cos_l = std::fabs(dot(cr, dir)) / two_area;
    if (cos_l < 1e-6f) return 0.0f;
    return (t_hit * t_hit) / (cos_l * 0.5f * two_area);
}

// Uniform-area triangle sampling, converted to a solid-angle pdf via the
// r^2 / cos(theta_light) Jacobian (THE classic mesh-light bug when wrong).
// Two-sided: the integrator collects emission on either face, so |cos|.
// bu/bv are the sampled barycentrics over (e1, e2) for the Le lookup.
inline vec3 sample_tri_light(const GPULight& L, const vec3& x, float u1,
                             float u2, float& pdf_sa, float& t_light,
                             float& bu, float& bv) {
    pdf_sa = 0.0f;
    t_light = 0.0f;
    const float su = std::sqrt(u1);
    bu = 1.0f - su;
    bv = u2 * su;
    const vec3 p0(L.p0.x, L.p0.y, L.p0.z);
    const vec3 e1(L.e1.x, L.e1.y, L.e1.z);
    const vec3 e2(L.e2.x, L.e2.y, L.e2.z);
    const vec3 y = p0 + bu * e1 + bv * e2;
    const vec3 d = y - x;
    const float r2 = dot(d, d);
    if (r2 < 1e-12f) return vec3(0.0f, 0.0f, 1.0f);
    const float r = std::sqrt(r2);
    const vec3 dir = d / r;
    const vec3 cr = cross(e1, e2);
    const float two_area = cr.length();
    if (two_area < 1e-12f) return vec3(0.0f, 0.0f, 1.0f);
    const float cos_l = std::fabs(dot(cr, dir)) / two_area;
    if (cos_l < 1e-6f) return vec3(0.0f, 0.0f, 1.0f);
    pdf_sa = r2 / (cos_l * 0.5f * two_area);
    t_light = r;
    return dir;
}

// Direction -> lat-long UV -> radiance. Falls back to the gradient when no
// map is loaded (--no HDRI, legacy scenes, CPU-viewer default).
inline color miss_radiance(const EnvLookup& env, const Ray& r) {
    if (!env.texels) return sky_radiance(r);
    const vec3 d = normalize(r.dir);
    const float u = std::atan2(d.z, d.x) * 0.15915494309189533577f + 0.5f +
                    env.yaw_norm;
    const float v =
        std::acos(std::fmin(std::fmax(d.y, -1.0f), 1.0f)) *
        0.31830988618379067154f;
    return sample_env_bilinear(env.texels, env.w, env.h, u, v) *
           env.intensity;
}

// Scale an indirect contribution so its max component <= m (0 = off).
inline color clamp_contribution(const color& c, float m) {
    if (m <= 0.0f) return c;
    const float mx = std::fmax(c.x, std::fmax(c.y, c.z));
    return mx > m ? c * (m / mx) : c;
}

// Vertex direct lighting — the env-NEE + area-NEE blocks, factored so
// the monolithic trace and the partitioned direct phase (ReSTIR Stage
// 0.5+) share ONE estimator: divergence between the two pipelines is
// structurally impossible at this layer. APPENDS into `radiance` in the
// original order (bit-preserving for the monolithic caller); consumes
// the same rng draws under the same conditions. The can_nee_* outputs
// feed the caller's MIS bookkeeping for its continuation ray.
inline void sample_direct(const HitRecord& rec, const Ray& ray,
                          const Hittable& world, RNG& rng,
                          const EnvLookup& env, const LightsLookup& lights,
                          float clamp_indirect, const color& throughput,
                          color& radiance, bool& can_nee_out,
                          bool& can_nee_light_out) {
        // ---- Session H: next-event estimation toward the environment.
        // Deliberately sample the env distribution (the sun), evaluate the
        // BSDF for that direction, and add the contribution if the shadow
        // ray escapes. Delta glass is skipped — BSDF sampling owns it.
        // Delta glass is skipped (no finite pdf); everything else —
        // including near-specular metals — is handled by the MIS weights:
        // where the BSDF lobe is sharp, pdf_bsdf dominates and the env-
        // sample weight goes to zero instead of spiking.
        const bool can_nee = env.nee && env.row_cdf != nullptr &&
                             rec.mat.transmission <= 0.5f;
        if (can_nee) {
            const float u1 = rng.next_float();
            const float u2 = rng.next_float();
            float pdf_env = 0.0f;
            const vec3 ldir = env_sample(env, u1, u2, pdf_env);
            if (pdf_env > 1e-12f) {
                float pdf_b = 0.0f;
                const vec3 vdir = -normalize(ray.dir);
                const color f = eval_bsdf(rec.mat, rec, vdir, ldir, pdf_b);
                const float nl = dot(rec.normal, ldir);
                if (nl > 1e-6f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)) {
                    const Ray shadow(rec.p, ldir);
                    if (!world.occluded(
                            shadow, 1e-3f,
                            std::numeric_limits<float>::infinity())) {
                        // MIS weight vs the BSDF sampler (power heuristic).
                        const float w =
                            (pdf_env * pdf_env) /
                            (pdf_env * pdf_env + pdf_b * pdf_b + 1e-20f);
                        const color c = throughput * f * nl *
                                        miss_radiance(env, shadow) *
                                        (w / pdf_env);
                        radiance += clamp_contribution(c, clamp_indirect);
                    }
                }
            }
        }
        can_nee_out = can_nee;

        // ---- Session J: one area-light sample per vertex. Same gating as
        // env NEE (delta glass excluded); selection is power-proportional.
        // Delta glass skipped; near-specular handled by the MIS weights
        // (sharp lobes hand themselves to BSDF sampling smoothly).
        const bool can_nee_light = env.nee && lights.count > 0 &&
                                   rec.mat.transmission <= 0.5f;
        if (can_nee_light) {
            const float us = rng.next_float();
            const float u1 = rng.next_float();
            const float u2 = rng.next_float();
            const int li = light_pick(lights.lights, lights.count, us);
            const GPULight& L = lights.lights[li];
            float pdf_sa = 0.0f, t_light = 0.0f;
            float bu = 0.0f, bv = 0.0f;
            const vec3 ldir =
                L.kind == 0u
                    ? sample_sphere_light(L, rec.p, u1, u2, pdf_sa, t_light)
                    : sample_tri_light(L, rec.p, u1, u2, pdf_sa, t_light,
                                       bu, bv);
            if (pdf_sa > 1e-12f && L.sel_pdf > 0.0f) {
                float pdf_b = 0.0f;
                const vec3 vdir = -normalize(ray.dir);
                const color f = eval_bsdf(rec.mat, rec, vdir, ldir, pdf_b);
                const float nl = dot(rec.normal, ldir);
                if (nl > 1e-6f && (f.x > 0.0f || f.y > 0.0f || f.z > 0.0f)) {
                    const Ray shadow(rec.p, ldir);
                    if (!world.occluded(shadow, 1e-3f,
                                        t_light * (1.0f - 1e-3f))) {
                        color Le(L.emission.x, L.emission.y, L.emission.z);
                        if (L.kind == 1u) {
                            // Textured Le at the sampled point (same
                            // bilinear + emissive_scale as shading).
                            const float b0 = 1.0f - bu - bv;
                            const float tu =
                                b0 * L.u0 + bu * L.u1 + bv * L.u2;
                            const float tv =
                                b0 * L.v0 + bu * L.v1 + bv * L.v2;
                            const MeshMaterial& mm =
                                lights.mesh->materials[L.mat_id];
                            Le = sample_bilinear(mm.emissive, tu, tv) *
                                 lights.mesh->emissive_scale;
                        }
                        // MIS weight vs the BSDF sampler.
                        const float pl = pdf_sa * L.sel_pdf;
                        const float w =
                            (pl * pl) / (pl * pl + pdf_b * pdf_b + 1e-20f);
                        const color c = throughput * f * nl * Le *
                                        (w / pl);
                        radiance += clamp_contribution(c, clamp_indirect);
                    }
                }
            }
        }
        can_nee_light_out = can_nee_light;
}

// ---- Session K: ReSTIR direct lighting (partitioned pipeline) ----------
// Stage 1: per-slot RIS over M candidates. Stage 2: temporal reservoirs.
// Stage 3: spatial reuse with the paper's UNBIASED 1/Z variant — the
// merged winner's target is re-evaluated on every contributing surface,
// and Z counts only reservoirs whose surface could have produced it.
// Mirrored in pathtrace.metal.

// Target evaluation for a fixed env-slot sample (a direction), on an
// arbitrary surface. Returns that (target value); fills the contribution
// pieces the shader needs when this surface is the receiver.
inline float target_env(const HitRecord& rec, const vec3& vdir,
                        const vec3& dir, const EnvLookup& env,
                        color& contrib_out, float& wmis_out) {
    contrib_out = color(0.0f);
    wmis_out = 0.0f;
    float pb = 0.0f;
    const color f = eval_bsdf(rec.mat, rec, vdir, dir, pb);
    const float nl = dot(rec.normal, dir);
    if (nl <= 1e-6f || (f.x <= 0.0f && f.y <= 0.0f && f.z <= 0.0f))
        return 0.0f;
    contrib_out = f * nl * miss_radiance(env, Ray(rec.p, dir));
    const float pl = env_pdf(env, dir);
    wmis_out = (pl * pl) / (pl * pl + pb * pb + 1e-20f);
    return luminance(contrib_out * wmis_out);
}

// Target evaluation for a fixed area-slot sample (a light point encoded
// as sphere world-point or triangle barycentrics), on an arbitrary
// surface. AREA measure: returns thatA = that * G.
inline float target_area(const HitRecord& rec, const vec3& vdir,
                         const GPULight& L, float ax, float ay, float az,
                         const LightsLookup& lights, color& contrib_out,
                         float& wmis_out, float& G_out, vec3& dir_out,
                         float& t_out) {
    contrib_out = color(0.0f);
    wmis_out = 0.0f;
    G_out = 0.0f;
    t_out = 0.0f;
    vec3 y;
    float bu = 0.0f, bv = 0.0f;
    if (L.kind == 0u) {
        y = vec3(ax, ay, az);
    } else {
        bu = ax;
        bv = ay;
        y = vec3(L.p0.x, L.p0.y, L.p0.z) +
            bu * vec3(L.e1.x, L.e1.y, L.e1.z) +
            bv * vec3(L.e2.x, L.e2.y, L.e2.z);
    }
    const vec3 d = y - rec.p;
    const float r2 = dot(d, d);
    if (r2 < 1e-12f) return 0.0f;
    const float r = std::sqrt(r2);
    dir_out = d / r;
    t_out = r;
    float pb = 0.0f;
    const color f = eval_bsdf(rec.mat, rec, vdir, dir_out, pb);
    const float nl = dot(rec.normal, dir_out);
    if (nl <= 1e-6f || (f.x <= 0.0f && f.y <= 0.0f && f.z <= 0.0f))
        return 0.0f;
    color Le(L.emission.x, L.emission.y, L.emission.z);
    vec3 n_y;
    if (L.kind == 0u) {
        n_y = normalize(y - vec3(L.p0.x, L.p0.y, L.p0.z));
    } else {
        const float b0 = 1.0f - bu - bv;
        const float tu = b0 * L.u0 + bu * L.u1 + bv * L.u2;
        const float tv = b0 * L.v0 + bu * L.v1 + bv * L.v2;
        const MeshMaterial& mm = lights.mesh->materials[L.mat_id];
        Le = sample_bilinear(mm.emissive, tu, tv) *
             lights.mesh->emissive_scale;
        n_y = normalize(cross(vec3(L.e1.x, L.e1.y, L.e1.z),
                              vec3(L.e2.x, L.e2.y, L.e2.z)));
    }
    G_out = std::fabs(dot(n_y, dir_out)) / r2;
    const float pl = light_dir_pdf(L, rec.p, dir_out, r) * L.sel_pdf;
    contrib_out = f * nl * Le;
    wmis_out = (pl * pl) / (pl * pl + pb * pb + 1e-20f);
    return luminance(contrib_out * wmis_out) * G_out;
}

// Phase D1: fresh candidates + temporal merge -> the per-frame reservoir
// (resv_cur). No shadow rays here; shading happens after spatial reuse.
inline void restir_build(const HitRecord& rec, const Ray& ray, RNG& rng,
                         const EnvLookup& env, const LightsLookup& lights,
                         int M, bool temporal, const ReSTIRPixel& hist,
                         ReSTIRPixel& cur) {
    const bool can_env = env.nee && env.row_cdf != nullptr &&
                         rec.mat.transmission <= 0.5f;
    const bool can_area = env.nee && lights.count > 0 &&
                          rec.mat.transmission <= 0.5f;
    const vec3 vdir = -normalize(ray.dir);
    const float McapF = 20.0f * float(M);
    const bool hist_ok =
        temporal && hist.prev_t > 0.0f && rec.t > 0.0f &&
        std::fabs(hist.prev_t - rec.t) < 0.1f * rec.t &&
        (hist.prev_normal.x * rec.normal.x +
         hist.prev_normal.y * rec.normal.y +
         hist.prev_normal.z * rec.normal.z) > 0.9f;

    // --- env slot ---
    if (can_env) {
        float wsum = 0.0f, win_that = 0.0f;
        vec3 win_dir(0.0f, 0.0f, 1.0f);
        for (int i = 0; i < M; ++i) {
            const float u1 = rng.next_float();
            const float u2 = rng.next_float();
            const float ur = rng.next_float();
            float pl = 0.0f;
            const vec3 ldir = env_sample(env, u1, u2, pl);
            float w_i = 0.0f, that = 0.0f;
            if (pl > 1e-12f) {
                color contrib;
                float w_mis;
                that = target_env(rec, vdir, ldir, env, contrib, w_mis);
                if (that > 0.0f) w_i = that / pl;
            }
            wsum += w_i;
            if (w_i > 0.0f && ur < w_i / wsum) {
                win_dir = ldir;
                win_that = that;
            }
        }
        float Mtot = float(M);
        const float ur_h = rng.next_float();
        if (hist_ok && hist.env_slot.M > 0.0f && hist.env_slot.W > 0.0f) {
            const vec3 hdir(hist.env_slot.ax, hist.env_slot.ay,
                            hist.env_slot.az);
            color contrib;
            float w_mis;
            const float that =
                target_env(rec, vdir, hdir, env, contrib, w_mis);
            const float Mh = std::fmin(hist.env_slot.M, McapF);
            const float wh = that * hist.env_slot.W * Mh;
            wsum += wh;
            Mtot += Mh;
            if (wh > 0.0f && ur_h < wh / wsum) {
                win_dir = hdir;
                win_that = that;
            }
        }
        cur.env_slot.ax = win_dir.x;
        cur.env_slot.ay = win_dir.y;
        cur.env_slot.az = win_dir.z;
        cur.env_slot.W = (wsum > 0.0f && win_that > 0.0f)
                             ? wsum / (Mtot * win_that)
                             : 0.0f;
        cur.env_slot.M = std::fmin(Mtot, McapF);
        cur.env_slot.light_id_p1 = 0u;
    } else {
        cur.env_slot = ReSTIRSlot{};
    }

    // --- area slot ---
    if (can_area) {
        float wsum = 0.0f, win_thatA = 0.0f;
        float win_ax = 0.0f, win_ay = 0.0f, win_az = 0.0f;
        pt_uint win_id_p1 = 0u;
        for (int i = 0; i < M; ++i) {
            const float us = rng.next_float();
            const float u1 = rng.next_float();
            const float u2 = rng.next_float();
            const float ur = rng.next_float();
            const int li = light_pick(lights.lights, lights.count, us);
            const GPULight& L = lights.lights[li];
            float pdf_sa = 0.0f, t_light = 0.0f;
            float bu = 0.0f, bv = 0.0f;
            const vec3 ldir =
                L.kind == 0u
                    ? sample_sphere_light(L, rec.p, u1, u2, pdf_sa, t_light)
                    : sample_tri_light(L, rec.p, u1, u2, pdf_sa, t_light,
                                       bu, bv);
            float w_i = 0.0f, thatA = 0.0f;
            float sax = 0.0f, say = 0.0f, saz = 0.0f;
            if (pdf_sa > 1e-12f && L.sel_pdf > 0.0f && t_light > 1e-6f) {
                if (L.kind == 0u) {
                    const vec3 yy = rec.p + ldir * t_light;
                    sax = yy.x;
                    say = yy.y;
                    saz = yy.z;
                } else {
                    sax = bu;
                    say = bv;
                }
                color contrib;
                float w_mis, G, t_o;
                vec3 d_o;
                thatA = target_area(rec, vdir, L, sax, say, saz, lights,
                                    contrib, w_mis, G, d_o, t_o);
                const float pl = pdf_sa * L.sel_pdf;
                if (thatA > 0.0f && G > 0.0f) w_i = thatA / (pl * G);
            }
            wsum += w_i;
            if (w_i > 0.0f && ur < w_i / wsum) {
                win_thatA = thatA;
                win_ax = sax;
                win_ay = say;
                win_az = saz;
                win_id_p1 = pt_uint(li + 1);
            }
        }
        float Mtot = float(M);
        const float ur_h = rng.next_float();
        if (hist_ok && hist.area_slot.M > 0.0f && hist.area_slot.W > 0.0f &&
            hist.area_slot.light_id_p1 > 0u &&
            int(hist.area_slot.light_id_p1) <= lights.count) {
            const GPULight& L =
                lights.lights[int(hist.area_slot.light_id_p1) - 1];
            color contrib;
            float w_mis, G, t_o;
            vec3 d_o;
            const float thatA =
                target_area(rec, vdir, L, hist.area_slot.ax,
                            hist.area_slot.ay, hist.area_slot.az, lights,
                            contrib, w_mis, G, d_o, t_o);
            const float Mh = std::fmin(hist.area_slot.M, McapF);
            const float wh = thatA * hist.area_slot.W * Mh;
            wsum += wh;
            Mtot += Mh;
            if (wh > 0.0f && ur_h < wh / wsum) {
                win_thatA = thatA;
                win_ax = hist.area_slot.ax;
                win_ay = hist.area_slot.ay;
                win_az = hist.area_slot.az;
                win_id_p1 = hist.area_slot.light_id_p1;
            }
        }
        cur.area_slot.ax = win_ax;
        cur.area_slot.ay = win_ay;
        cur.area_slot.az = win_az;
        cur.area_slot.W = (wsum > 0.0f && win_thatA > 0.0f)
                              ? wsum / (Mtot * win_thatA)
                              : 0.0f;
        cur.area_slot.M = std::fmin(Mtot, McapF);
        cur.area_slot.light_id_p1 = win_id_p1;
    } else {
        cur.area_slot = ReSTIRSlot{};
    }
}

// Phase D2: spatial merge with Talbot balance-heuristic MIS weights —
// unbiased for ANY sampler-support overlap (no Z counting, no support
// assumptions): each input i contributes
//   w_i = [M_i p^_i(s_i) / sum_j M_j p^_j(s_i)] * p^_own(s_i) * W_i,
// and the shading weight is wsum / p^_own(winner). History persists the
// PRE-spatial reservoir (temporal chains stay same-surface). Mirrored in
// pathtrace.metal.
inline void restir_spatial_shade(
    const HitRecord& rec, const Ray& ray, const Hittable& world, RNG& rng,
    const EnvLookup& env, const LightsLookup& lights, int M, bool spatial,
    const GBufferPx* gbuf, const ReSTIRPixel* cur_all, int w, int h, int x,
    int y, ReSTIRPixel& persist, float clamp_indirect, color& radiance) {
    const bool can_env = env.nee && env.row_cdf != nullptr &&
                         rec.mat.transmission <= 0.5f;
    const bool can_area = env.nee && lights.count > 0 &&
                          rec.mat.transmission <= 0.5f;
    const vec3 vdir = -normalize(ray.dir);
    const std::size_t px = std::size_t(y) * w + x;
    const ReSTIRPixel& own = cur_all[px];

    // Neighbor set: chosen once, shared by both slots; 2 offset draws per
    // neighbor, always consumed.
    int nxs[PT_RESTIR_NEIGHBORS], nys[PT_RESTIR_NEIGHBORS];
    bool ok[PT_RESTIR_NEIGHBORS];
    const int K = spatial ? PT_RESTIR_NEIGHBORS : 0;
    for (int k = 0; k < K; ++k) {
        const float u1 = rng.next_float();
        const float u2 = rng.next_float();
        int nx = x + int((u1 * 2.0f - 1.0f) * float(PT_RESTIR_RADIUS));
        int ny = y + int((u2 * 2.0f - 1.0f) * float(PT_RESTIR_RADIUS));
        nx = nx < 0 ? 0 : (nx >= w ? w - 1 : nx);
        ny = ny < 0 ? 0 : (ny >= h ? h - 1 : ny);
        nxs[k] = nx;
        nys[k] = ny;
        ok[k] = false;
        if (nx == x && ny == y) continue;
        const GBufferPx& gn = gbuf[std::size_t(ny) * w + nx];
        if (gn.t < 0.0f || gn.transmission > 0.5f) continue;
        if (std::fabs(gn.t - rec.t) >= 0.1f * rec.t) continue;
        const float ndot = gn.normal.x * rec.normal.x +
                           gn.normal.y * rec.normal.y +
                           gn.normal.z * rec.normal.z;
        if (ndot <= 0.9f) continue;
        ok[k] = true;
    }
    const auto neighbor_rec = [&](int k, HitRecord& nr, vec3& nvdir) {
        const GBufferPx& gn = gbuf[std::size_t(nys[k]) * w + nxs[k]];
        nr.p = point3(gn.pos.x, gn.pos.y, gn.pos.z);
        nr.normal = vec3(gn.normal.x, gn.normal.y, gn.normal.z);
        nr.front_face = (gn.flags & 1u) != 0u;
        nr.t = gn.t;
        nr.mat.base_color =
            color(gn.base_color.x, gn.base_color.y, gn.base_color.z);
        nr.mat.emission = color(gn.emission.x, gn.emission.y, gn.emission.z);
        nr.mat.metallic = gn.metallic;
        nr.mat.roughness = gn.roughness;
        nr.mat.ior = gn.ior;
        nr.mat.transmission = gn.transmission;
        nvdir = -normalize(vec3(gn.rd.x, gn.rd.y, gn.rd.z));
    };

    // --- env slot ---
    if (can_env) {
        const ReSTIRSlot* pslot[1 + PT_RESTIR_NEIGHBORS];
        int psurf[1 + PT_RESTIR_NEIGHBORS];
        int np = 0;
        if (own.env_slot.M > 0.0f && own.env_slot.W > 0.0f) {
            pslot[np] = &own.env_slot;
            psurf[np] = -1;
            ++np;
        }
        int pidx[PT_RESTIR_NEIGHBORS];
        for (int k = 0; k < K; ++k) {
            pidx[k] = -1;
            if (!ok[k]) continue;
            const ReSTIRPixel& nb = cur_all[std::size_t(nys[k]) * w + nxs[k]];
            if (nb.env_slot.M <= 0.0f || nb.env_slot.W <= 0.0f) continue;
            pslot[np] = &nb.env_slot;
            psurf[np] = k;
            pidx[k] = np;
            ++np;
        }
        const auto phat_env = [&](int surf_id, const vec3& dir, color& c,
                                  float& wm) {
            if (surf_id < 0) return target_env(rec, vdir, dir, env, c, wm);
            HitRecord nr;
            vec3 nvdir;
            neighbor_rec(surf_id, nr, nvdir);
            return target_env(nr, nvdir, dir, env, c, wm);
        };
        // w_i for participant i (balance m-weight folded in).
        const auto weight_env = [&](int i, vec3& sdir_o, color& c_o,
                                    float& wm_o, float& that_o) {
            const vec3 sdir(pslot[i]->ax, pslot[i]->ay, pslot[i]->az);
            color c;
            float wm;
            const float that_own = phat_env(-1, sdir, c, wm);
            if (that_own <= 0.0f) return 0.0f;
            float denom = 0.0f, self = 0.0f;
            for (int j = 0; j < np; ++j) {
                color cc;
                float wmm;
                const float p = phat_env(psurf[j], sdir, cc, wmm);
                denom += pslot[j]->M * p;
                if (j == i) self = p;
            }
            if (denom <= 0.0f || self <= 0.0f) return 0.0f;
            sdir_o = sdir;
            c_o = c;
            wm_o = wm;
            that_o = that_own;
            return (pslot[i]->M * self / denom) * that_own * pslot[i]->W;
        };
        vec3 win_dir(0.0f, 0.0f, 1.0f);
        color win_c(0.0f);
        float win_mis = 0.0f, win_that = 0.0f, wsum = 0.0f;
        if (np > 0 && psurf[0] == -1) {   // own seeds without a draw
            wsum = weight_env(0, win_dir, win_c, win_mis, win_that);
        }
        for (int k = 0; k < K; ++k) {
            const float ur = rng.next_float();
            const int i = pidx[k];
            if (i < 0) continue;
            vec3 sd;
            color c;
            float wm, th;
            const float w_i = weight_env(i, sd, c, wm, th);
            wsum += w_i;
            if (w_i > 0.0f && ur < w_i / wsum) {
                win_dir = sd;
                win_c = c;
                win_mis = wm;
                win_that = th;
            }
        }
        const float Wnew =
            (wsum > 0.0f && win_that > 0.0f) ? wsum / win_that : 0.0f;
        persist.env_slot = own.env_slot;   // pre-spatial history
        if (Wnew > 0.0f) {
            if (!world.occluded(Ray(rec.p, win_dir), 1e-3f,
                                std::numeric_limits<float>::infinity())) {
                const color c = win_c * win_mis * Wnew;
                radiance += clamp_contribution(c, clamp_indirect);
            }
        }
    } else {
        for (int k = 0; k < K; ++k) rng.next_float();
        persist.env_slot = ReSTIRSlot{};
    }

    // --- area slot ---
    if (can_area) {
        const ReSTIRSlot* pslot[1 + PT_RESTIR_NEIGHBORS];
        int psurf[1 + PT_RESTIR_NEIGHBORS];
        int np = 0;
        const auto slot_valid = [&](const ReSTIRSlot& sl) {
            return sl.M > 0.0f && sl.W > 0.0f && sl.light_id_p1 > 0u &&
                   int(sl.light_id_p1) <= lights.count;
        };
        if (slot_valid(own.area_slot)) {
            pslot[np] = &own.area_slot;
            psurf[np] = -1;
            ++np;
        }
        int pidx[PT_RESTIR_NEIGHBORS];
        for (int k = 0; k < K; ++k) {
            pidx[k] = -1;
            if (!ok[k]) continue;
            const ReSTIRPixel& nb = cur_all[std::size_t(nys[k]) * w + nxs[k]];
            if (!slot_valid(nb.area_slot)) continue;
            pslot[np] = &nb.area_slot;
            psurf[np] = k;
            pidx[k] = np;
            ++np;
        }
        const auto phat_area = [&](int surf_id, const ReSTIRSlot& sl,
                                   color& c, float& wm, float& G, vec3& d,
                                   float& t) {
            const GPULight& L = lights.lights[int(sl.light_id_p1) - 1];
            if (surf_id < 0)
                return target_area(rec, vdir, L, sl.ax, sl.ay, sl.az,
                                   lights, c, wm, G, d, t);
            HitRecord nr;
            vec3 nvdir;
            neighbor_rec(surf_id, nr, nvdir);
            return target_area(nr, nvdir, L, sl.ax, sl.ay, sl.az, lights, c,
                               wm, G, d, t);
        };
        const auto weight_area = [&](int i, color& c_o, float& wm_o,
                                     float& G_o, vec3& d_o, float& t_o,
                                     float& that_o) {
            color c;
            float wm, G, t;
            vec3 d;
            const float that_own =
                phat_area(-1, *pslot[i], c, wm, G, d, t);
            if (that_own <= 0.0f) return 0.0f;
            float denom = 0.0f, self = 0.0f;
            for (int j = 0; j < np; ++j) {
                color cc;
                float wmm, gg, tt;
                vec3 dd;
                const float p =
                    phat_area(psurf[j], *pslot[i], cc, wmm, gg, dd, tt);
                denom += pslot[j]->M * p;
                if (j == i) self = p;
            }
            if (denom <= 0.0f || self <= 0.0f) return 0.0f;
            c_o = c;
            wm_o = wm;
            G_o = G;
            d_o = d;
            t_o = t;
            that_o = that_own;
            return (pslot[i]->M * self / denom) * that_own * pslot[i]->W;
        };
        int win_i = -1;
        color win_c(0.0f);
        float win_mis = 0.0f, win_G = 0.0f, win_t = 0.0f, win_that = 0.0f;
        vec3 win_dir(0.0f, 0.0f, 1.0f);
        float wsum = 0.0f;
        if (np > 0 && psurf[0] == -1) {
            wsum = weight_area(0, win_c, win_mis, win_G, win_dir, win_t,
                               win_that);
            if (wsum > 0.0f) win_i = 0;
        }
        for (int k = 0; k < K; ++k) {
            const float ur = rng.next_float();
            const int i = pidx[k];
            if (i < 0) continue;
            color c;
            float wm, G, t, th;
            vec3 d;
            const float w_i = weight_area(i, c, wm, G, d, t, th);
            wsum += w_i;
            if (w_i > 0.0f && ur < w_i / wsum) {
                win_i = i;
                win_c = c;
                win_mis = wm;
                win_G = G;
                win_dir = d;
                win_t = t;
                win_that = th;
            }
        }
        const float Wnew =
            (wsum > 0.0f && win_that > 0.0f && win_i >= 0) ? wsum / win_that
                                                           : 0.0f;
        persist.area_slot = own.area_slot;   // pre-spatial history
        if (Wnew > 0.0f) {
            if (!world.occluded(Ray(rec.p, win_dir), 1e-3f,
                                win_t * (1.0f - 1e-3f))) {
                const color c = win_c * win_mis * (win_G * Wnew);
                radiance += clamp_contribution(c, clamp_indirect);
            }
        }
    } else {
        for (int k = 0; k < K; ++k) rng.next_float();
        persist.area_slot = ReSTIRSlot{};
    }

    persist.prev_normal = {rec.normal.x, rec.normal.y, rec.normal.z};
    persist.prev_t = rec.t;
}

inline color trace(Ray ray, const Hittable& world, RNG& rng, int max_depth,
                   const EnvLookup& env, const LightsLookup& lights,
                   float clamp_indirect,
                   // Session K: partitioned pipeline injects the primary
                   // hit and skips vertex-0 direct (the direct phase
                   // computed it with the same rng draws).
                   const HitRecord* pre = nullptr,
                   bool skip_v0_direct = false) {
    color radiance(0.0f);      // light collected so far
    color throughput(1.0f);    // fraction of it that survives back to the eye
    bool prev_nee = false;     // env NEE ran at the previous path vertex
    bool prev_nee_light = false;   // area-light NEE ran there too
    float prev_pdf = 0.0f;     // BSDF pdf of the ray we're now following

    for (int depth = 0; depth < max_depth; ++depth) {
        HitRecord rec;
        // t_min = 1e-3: a bounced ray starts ON a surface; float error can
        // put its origin a hair inside, and t_min=0 would let it re-hit the
        // same surface at t≈0 ("shadow acne" — dark speckles everywhere).
        bool hit_any;
        if (depth == 0 && pre) {
            rec = *pre;
            hit_any = true;
        } else {
            hit_any = world.hit(ray, 1e-3f,
                                std::numeric_limits<float>::infinity(), rec);
        }
        if (!hit_any) {
            // MIS (power heuristic): vertices that ran env NEE weight
            // their continuation's env hit by the BSDF-sampling share;
            // camera rays and delta/glass continuations (no NEE there)
            // see the env in full.
            float w = 1.0f;
            if (prev_nee) {
                const float pe = env_pdf(env, ray.dir);
                w = (prev_pdf * prev_pdf) /
                    (prev_pdf * prev_pdf + pe * pe + 1e-20f);
            }
            const color c = throughput * miss_radiance(env, ray) * w;
            radiance +=
                (depth == 0 ? c : clamp_contribution(c, clamp_indirect));
            return radiance;
        }

        // Collect whatever this surface emits (zero for non-lights), THEN
        // try to continue the path. Indirect pickups are firefly-clamped;
        // depth 0 (directly visible lights/background) never is.
        {
            // MIS (power heuristic): a BSDF ray that lands on a LISTED
            // emitter weights its emission against the pdf area-light NEE
            // would have assigned this direction (x selection pdf). Camera
            // rays, delta chains, and unlisted emitters keep full weight.
            float w = 1.0f;
            if (prev_nee_light && depth > 0 && rec.light_id >= 0 &&
                prev_pdf > 0.0f) {
                const GPULight& L = lights.lights[rec.light_id];
                const float pl = light_dir_pdf(L, ray.origin,
                                               normalize(ray.dir), rec.t) *
                                 L.sel_pdf;
                w = (prev_pdf * prev_pdf) /
                    (prev_pdf * prev_pdf + pl * pl + 1e-20f);
            }
            const color c = throughput * rec.mat.emission * w;
            radiance +=
                depth == 0 ? c : clamp_contribution(c, clamp_indirect);
        }

        if (depth == 0 && skip_v0_direct) {
            // The direct phase already ran sample_direct with these draws;
            // reproduce only its condition flags for the MIS bookkeeping.
            prev_nee = env.nee && env.row_cdf != nullptr &&
                       rec.mat.transmission <= 0.5f;
            prev_nee_light = env.nee && lights.count > 0 &&
                             rec.mat.transmission <= 0.5f;
        } else {
            bool v_nee = false, v_nee_light = false;
            sample_direct(rec, ray, world, rng, env, lights, clamp_indirect,
                          throughput, radiance, v_nee, v_nee_light);
            prev_nee = v_nee;
            prev_nee_light = v_nee_light;
        }

        color attenuation;
        Ray scattered;
        float scatter_pdf = 0.0f;
        bool scatter_delta = false;
        if (!scatter(rec.mat, ray, rec, rng, attenuation, scattered,
                     scatter_pdf, scatter_delta)) {
            return radiance;                         // light hit, or absorbed
        }
        throughput *= attenuation;
        ray = scattered;
        prev_pdf = scatter_delta ? 0.0f : scatter_pdf;
        if (scatter_delta) prev_nee = false;   // delta chains keep full env

        // Russian roulette: after a few bounces, kill dim paths with
        // probability (1 - p) and divide survivors by p. The expected
        // contribution is unchanged — E[x] = p * (x/p) — so this stays
        // unbiased while skipping work on paths that barely matter.
        if (depth >= 3) {
            const float p = std::fmin(
                std::fmax(throughput.x, std::fmax(throughput.y, throughput.z)),
                0.95f);
            // p == 0 (all-black throughput) must terminate BEFORE the rng
            // test: next_float() can return exactly 0.0, and 0/0 would put
            // a NaN in the accumulator.
            if (p <= 0.0f) break;
            if (rng.next_float() > p) break;
            throughput /= p;
        }
    }
    return radiance;   // ran out of bounces; keep what was collected
}
