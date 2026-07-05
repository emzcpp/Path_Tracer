# The Parity & Unbiasedness Methodology

This renderer maintains two complete, independent implementations of the same
integrator — a C++ CPU reference (`src/integrator.h`) and a Metal compute
kernel (`src/kernels/pathtrace.metal`) — and treats their agreement as a
continuously verified invariant rather than a hope. This document describes
the machinery that makes that claim testable, and the bias-gating discipline
built on top of it.

## 1. Determinism is the foundation

Every random draw in the renderer comes from PCG32, seeded per `(pixel, pass)`
through splitmix64. No RNG state crosses pixels or passes implicitly; the
stream increment derives from the pixel id, and when the pipeline was split
into phases (ReSTIR), the stream state rides through the G-buffer so every
draw lands exactly where the monolithic kernel would have put it.

Consequences, all load-bearing:

- Images are **bit-identical at any thread count** (CPU) and **any work
  slicing/schedule** (GPU command-buffer slicing, phase slicing).
- A CPU pixel and a GPU pixel consume the *same* random sequence, so their
  results may be compared sample-for-sample.
- Any scheduling feature (GPU-time budgeting, row slicing, ReSTIR phase
  slicing) can be gated on `memcmp` — and is.

## 2. The parity harness (`--parity [spp]`)

Both backends render the same scene with the same seeds. The harness then
reports, over tone-mapped 8-bit output:

- mean |diff| in least-significant-bits (LSB),
- % of pixels bit-exact and % within 2 LSB,
- worst pixel.

**The control:** two *independent CPU runs with different seed bases* are
diffed the same way. That is the Monte Carlo noise floor — what "the same
renderer, sampled differently" looks like. GPU-vs-CPU differences are judged
*relative to that floor*, which removes hand-waving about how much
disagreement is acceptable: the answer is "hundreds of times less than
statistically invisible."

**Gates:** mean ≤ 0.5 LSB, ≥ 98% of pixels within 2 LSB, and mean ≤ floor/5 —
**or**, when an estimator drives the floor itself to near zero (low-variance
regimes), an absolute ULP-regime gate applies instead: mean ≤ 0.02 LSB with
≥ 99.9% within 2 LSB. The two-regime structure exists because a better
estimator shrinks the floor, which would otherwise make the *relative* gate
stricter for no physical reason.

Typical numbers on this codebase: ~96–99% of pixels bit-exact; mean |diff|
around 0.02 LSB against noise floors of 3–8 LSB.

### Hand-matching disciplines that make this possible

- **No hardware samplers.** Texture filtering (bilinear, wrap, sRGB decode)
  is hand-written identically in both languages; hardware filtering units
  round differently and would break bit-exactness immediately.
- Cross products, lerps, and CDF binary searches are written out textually
  rather than delegated to intrinsics where fused-multiply-add contraction
  could flip a branch.
- **Draw order is part of the contract.** Both backends consume RNG draws in
  the same order down to degenerate branches: rejected candidates still burn
  their lottery draws so the stream shape is identical regardless of data.

## 3. Statefulness: adapting parity for ReSTIR

Reservoir reuse makes frame N depend on frames < N, so tiny ULP divergences
can *amplify* across frames instead of averaging out. The parity model was
extended rather than weakened:

- **N=8 anchor:** the standard gate runs at 8 frames of temporal/spatial
  history.
- **64-frame amplification probe:** the same gate at 64 frames confirms that
  cross-frame divergence stays bounded (measured: it does — 0.023 → 0.023
  LSB class results).
- **Bias gates become the primary correctness instrument** for stateful
  estimators (below), with parity guarding implementation agreement.

## 4. Unbiasedness: brute force is the referee

`--brute` renders with no NEE, no MIS, no ReSTIR — pure BSDF-sampled path
tracing. It is slow and noisy and *obviously correct*, which is the point.
Every variance-reduction technique in the renderer is gated on converging to
the same image as brute force **in the mean**:

- **Mean-image and block-mean gates**, never pixel diffs (different
  estimators legitimately produce different noise). Target band: within
  ~±0.1% of the brute mean, block means within a few percent at test spp.
- Gates run on scenes chosen to be *hostile* to the feature under test: a
  36-fixture many-light scene for ReSTIR, occlusion-heavy interiors,
  close-range emitters, normal-mapped surfaces.

### Case studies — bias bugs this discipline caught

All of these produced plausible images. None survived measurement.

| Bias | Mechanism | Fix |
|---|---|---|
| **+96%** | Spatial ReSTIR stored its 1/Z-corrected weight back into temporal history, compounding by M/Z per frame wherever neighbors couldn't see the winner | Shade with the corrected weight; persist the pre-spatial reservoir |
| **+10%** | Reservoirs that audited M samples but found nothing (W = 0) were excluded from the spatial balance denominators, inflating other inputs' MIS weights in occlusion-heavy regions | Denominators count every M > 0 input; selection still requires W > 0 |
| **+5.7%** | Temporal reuse: the same M-counting defect, plus the Mtot-style combine's target mismatch under sub-pixel jitter — largest on normal-mapped surfaces | Temporal history merged through the same Talbot balance-heuristic combine as spatial, its target evaluated on last frame's actual surface (G-buffer ping-pong) |
| **+1.3%** | The 1/Z "unbiased" spatial combine assumes target support equals sampler support — false for one-sided sphere-cone sampling against two-sided targets | Replace Z-counting with the Talbot balance heuristic (unbiased for arbitrary support overlap) |

The diagnostic pattern was identical every time: **bisect the estimator**
(RIS-only → +temporal → +spatial) on a cheap scene until the biased stage is
isolated, form a mechanism hypothesis, design a discriminating probe (does
the bias scale with M-cap? with light distance? does it appear on smooth
spheres or only textured meshes?), and only then write the fix. Twice, the
"obvious" mechanism was falsified by the probe before the real one was found.

## 5. Refactor and scheduling gates

- Pure refactors (kernel partitioning, factoring shared functions) are gated
  on **byte-identical output** (checksums), not on parity — stronger, and
  catches "harmless" reorderings.
- Work scheduling (row slicing, ReSTIR's phase-sliced frames) is gated on
  `memcmp` equality between sliced and unsliced runs.
- State reset (reservoirs, accumulation) is gated on a **ghosting test**:
  after a scene edit, the next N frames must be bit-identical to a freshly
  constructed renderer.
- `--nan-check` scans both accumulators for non-finite values (gate: zero).

## 6. Running the gates

```bash
./build/pathtracer --parity 64                     # CPU vs GPU + noise-floor control
./build/pathtracer --parity 8 --restir             # stateful anchor gate
./build/pathtracer --brute                         # ground truth for any scene
./build/pathtracer --nan-check 16 --restir         # non-finite scan
./build/pathtracer --grid                          # BSDF energy validation
```

The claim this buys: one renderer, three estimators (brute force, NEE+MIS,
ReSTIR DI), two hand-written backends — all provably converging to the same
image, with every gate re-runnable by anyone who clones the repo.
