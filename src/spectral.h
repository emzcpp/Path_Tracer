#pragma once

// Spectral rendering core (v1.2, hero-wavelength). A path may carry ONE
// wavelength lambda instead of an RGB triple; the sensor reconstructs RGB
// by accumulating many single-wavelength samples through the CIE color-
// matching functions. Mirrored line-for-line in pathtrace.metal (the
// --parity harness is the drift detector).
//
// The accumulator stays RGB: a spectral sample deposits
//   scalarThroughput * S(lambda) * response(lambda)
// where response() is the (1,1,1)-normalized CMF->XYZ->linear-sRGB map, so
// resolve / denoiser / clamp / PNG export are all untouched. A FLAT unit
// spectrum reconstructs to exactly (1,1,1) — that normalization is the
// white->white guarantee (Stage 1's checkpoint).
//
// Stage 1: IOR is still wavelength-INDEPENDENT (no dispersion yet); this
// file only establishes wavelength sampling + the color reconstruction.

#include "rng.h"
#include "vec3.h"

namespace spectral {

// Visible band for hero-wavelength sampling (uniform). Kept narrow enough
// that near-zero-response tails don't add variance; wide enough for a
// convincing prism. Both backends must agree on these bounds.
constexpr float LMIN = 400.0f;
constexpr float LMAX = 700.0f;
constexpr float BAND = LMAX - LMIN;

inline float sample_wavelength(float u) { return LMIN + u * BAND; }

// Wyman, Sloan, Shirley 2013 — multi-lobe Gaussian fit to the CIE 1931
// color-matching functions (no lookup tables; trivially hand-portable).
inline float wyman_g(float x, float mu, float s1, float s2) {
    const float t = (x - mu) * (x < mu ? 1.0f / s1 : 1.0f / s2);
    return std::exp(-0.5f * t * t);
}

inline vec3 cie_xyz(float l) {
    const float x = 1.056f * wyman_g(l, 599.8f, 37.9f, 31.0f) +
                    0.362f * wyman_g(l, 442.0f, 16.0f, 26.7f) -
                    0.065f * wyman_g(l, 501.1f, 20.4f, 26.2f);
    const float y = 0.821f * wyman_g(l, 568.8f, 46.9f, 40.5f) +
                    0.286f * wyman_g(l, 530.9f, 16.3f, 31.1f);
    const float z = 1.217f * wyman_g(l, 437.0f, 11.8f, 36.0f) +
                    0.681f * wyman_g(l, 459.0f, 26.0f, 13.8f);
    return vec3(x, y, z);
}

// (1,1,1)-normalized RGB response for depositing one uniform-lambda sample.
// The per-channel K constants = BAND / integral(M*cmf)_c over [LMIN,LMAX],
// so a flat spectrum S==1 integrates back to (1,1,1) EXACTLY (computed
// offline from this same fit at 0.25 nm steps). Individual samples may be
// negative in a channel — pure wavelengths fall outside the sRGB triangle;
// that is physical and averages out (clamped only at display).
inline vec3 lambda_response(float l) {
    const vec3 c = cie_xyz(l);
    const float R = 3.2406f * c.x - 1.5372f * c.y - 0.4986f * c.z;
    const float G = -0.9689f * c.x + 1.8758f * c.y + 0.0415f * c.z;
    const float B = 0.0557f * c.x - 0.2040f * c.y + 1.0570f * c.z;
    return vec3(R * 2.33749215f, G * 2.95510157f, B * 3.10746416f);
}

// RGB -> spectral value at lambda via a partition-of-unity basis built from
// the positive parts of the sRGB response: rho_R+rho_G+rho_B == 1, so white
// (1,1,1) upsamples to a FLAT spectrum (== 1) and a saturated primary
// concentrates where its response peaks (red -> long wavelengths). The
// result is a convex combination of the RGB components, so reflectances
// stay energy-conserving in [0, max(rgb)].
inline float rgb_to_reflectance(const vec3& rgb0, float l) {
    // Round-trip correction (C^-1, computed offline from this fit): the
    // uncorrected upsample->reconstruct map has off-diagonal channel bleed
    // (~+17% on saturated red) though it preserves neutral. This 3x3
    // pre-correction makes SINGLE-bounce RGB<->spectral near-exact, so
    // colored surfaces keep their hue AND value, not just neutral. Row
    // sums == 1, so white stays white.
    const vec3 rgb(0.838717f * rgb0.x + 0.148845f * rgb0.y +
                       0.012780f * rgb0.z,
                   -0.117719f * rgb0.x + 1.103062f * rgb0.y +
                       0.014987f * rgb0.z,
                   -0.023000f * rgb0.x + 0.011697f * rgb0.y +
                       1.011686f * rgb0.z);
    const vec3 c = cie_xyz(l);
    const float R = std::fmax(0.0f,
                              3.2406f * c.x - 1.5372f * c.y - 0.4986f * c.z);
    const float G = std::fmax(0.0f,
                              -0.9689f * c.x + 1.8758f * c.y + 0.0415f * c.z);
    const float B = std::fmax(0.0f,
                              0.0557f * c.x - 0.2040f * c.y + 1.0570f * c.z);
    const float sum = R + G + B;
    if (sum <= 1e-8f) return 0.0f;
    return (rgb.x * R + rgb.y * G + rgb.z * B) / sum;
}

}  // namespace spectral

// Per-path spectral context, threaded through trace()/sample_direct() so the
// deposit sites branch with a single object rather than three booleans.
// on == false makes every helper the identity (RGB pipeline untouched, and
// spec_begin draws NO rng — so --parity with spectral off is byte-exact).
struct SpecCtx {
    bool on;
    float lam;
    color resp;
};

inline SpecCtx spec_begin(bool on, RNG& rng) {
    SpecCtx s;
    s.on = on;
    s.lam = 0.0f;
    s.resp = color(1.0f, 1.0f, 1.0f);
    if (on) {
        s.lam = spectral::sample_wavelength(rng.next_float());
        s.resp = spectral::lambda_response(s.lam);
    }
    return s;
}

// Colorize an emissive/env RGB source for deposit: resp * S(lambda).
inline color spec_dep(const SpecCtx& s, const color& src) {
    return s.on ? s.resp * spectral::rgb_to_reflectance(src, s.lam) : src;
}

// Two spectra multiplied then colorized once (BSDF value * light radiance in
// the NEE deposits); RGB path is the plain elementwise product.
inline color spec_dep2(const SpecCtx& s, const color& a, const color& b) {
    if (!s.on) return color(a.x * b.x, a.y * b.y, a.z * b.z);
    const float p = spectral::rgb_to_reflectance(a, s.lam) *
                    spectral::rgb_to_reflectance(b, s.lam);
    return s.resp * p;
}

// Throughput multiplier for a BSDF attenuation: a scalar (broadcast to keep
// throughput equal-channel, so Russian roulette's max() stays correct).
inline color spec_atten(const SpecCtx& s, const color& a) {
    if (!s.on) return a;
    const float r = spectral::rgb_to_reflectance(a, s.lam);
    return color(r, r, r);
}
