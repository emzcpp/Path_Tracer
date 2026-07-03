# PathTracer

Path tracer with full global illumination — a principled GGX
metallic-roughness BSDF (glTF-style: baseColor/metallic/roughness, VNDF
importance sampling, Smith height-correlated masking) plus delta glass and
emissive lights — and a live interactive viewer running on **Metal
compute**, with the CPU implementation kept as a
bit-for-bit-seeded reference. C++20, no dependencies beyond the vendored
`stb_image_write.h` (the Metal kernel is embedded in the binary and
compiled at runtime, so not even Xcode is required — just Command Line
Tools).

## Build & run

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ./pathtracer            # interactive viewer, GPU backend
./pathtracer --cpu                  # interactive viewer, CPU reference backend
./pathtracer --offline              # CPU render to out.ppm/out.png @ 256 spp
./pathtracer 1024                   # offline render @ 1024 spp
./pathtracer --parity               # verify GPU kernel against CPU reference
./pathtracer --gpu-check            # compile kernels, print device info
./pathtracer --mesh-info            # model loader / BVH diagnostics
open out.png
```

All render settings — resolution, samples per pixel, max bounce depth,
camera, preview quality, GPU batch sizes — live in `src/settings.h`.

## Model import (glTF binary)

The scene auto-loads `assets/DamagedHelmet.glb` (or the legacy
`Test_Models/` path) when present; `--model <path>` loads any glTF/GLB,
`--no-model` forces the sphere-only scene. Parsing is via the vendored
single-header `cgltf` (node-hierarchy transforms composed to world space,
all triangle primitives, TANGENT accessors or UV-derived tangents), with
JPEG/PNG decode via `stb_image.h`. Materials map straight onto the GGX
metallic-roughness BSDF, including tangent-space normal mapping; baseColor
and emissive textures are sRGB-decoded to linear at load, while
metallicRoughness and normal maps stay linear.

What the import pipeline does: bakes the node transform + placement into
world-space triangles at load, decodes textures to linear ushort4 (sRGB via
pure pow-2.2, matching the display encode), builds a midpoint-split BVH
(depth-capped so the traversal stack is a guarantee), and maps glTF
metallic/roughness onto the existing material model (metallic > 0.5 →
metal with fuzz = roughness²; emissive texture rides on the scattering
surface). Both backends consume the identical host-built arrays: BVH
traversal, Möller–Trumbore, and the bilinear texture sampler are written
twice (C++ and MSL) with textually identical logic — `--parity` is the
drift detector, and the helmet scene passes with the same-seed error ~19×
below the Monte Carlo noise floor.

## GPU/CPU parity

Both backends render the same deterministic scene with the same
per-(pixel, pass) PCG32 seeds, so `--parity` can diff them meaningfully:
same-seed GPU-vs-CPU error sits ~60× below the Monte Carlo noise floor
(mean ~0.05 display LSB, >96% of pixels exactly equal at 64 spp). Rare
larger diffs are expected: a float ULP that flips a discrete branch
(discriminant, Fresnel-vs-rng, roulette) replaces that whole path — the
gate is on the bulk statistics, never the max. `parity_diff.png` makes any
systematic bug visible at a glance.

## Viewer controls

| Input | Action |
|---|---|
| `W A S D` | move forward / left / back / right |
| `Q` / `E` | move down / up |
| left-drag | look around |
| `Shift` | 4× speed boost |
| scroll | adjust move speed |
| `P` | save the current image as a timestamped PNG |
| `R` | FINAL mode: lock camera, converge to target spp, auto-export PNG |
| `F` | frame the selected object |
| `U` | show/hide the tool panel (ImGui) |
| click | select object under cursor (sky deselects) |
| `Tab` / `Shift+Tab` | cycle selection through the outliner |
| `1` / `2` / `3` | gizmo mode: move / rotate / scale the selection |
| `Backspace` | delete the selected object |
| `V` | fast-nav: cycle Off / Solid / Wireframe raster preview |
| `?` / `F1` | keyboard-shortcut overlay |
| `Esc` / `Cmd-Q` | close overlay / quit |

The tool panel exposes live render settings (reconverging through the
central reset hook), the selection readout, and scene Save/Load (JSON;
geometry stored by reference — sphere parameters and a relative mesh path,
never vertex data).

The environment is an equirectangular HDRI (`--env <path.hdr>` or the
panel's Environment section: load/clear, intensity, yaw — all persisted in
scene files as a relative path). Radiance data stays linear (never
sRGB-decoded), missed rays sample it with the same hand-written bilinear
as material textures (u wraps, v clamps), and with no HDRI loaded the
original gradient dome remains. Environment importance sampling / NEE is
deliberately not implemented yet — bright suns converge slowly for now.

Fast-nav (V) rasterizes the scene (flat-lit solid or wireframe) at full
framerate while the camera moves — same camera matrices as the tracer, so
it registers exactly — and hands back to path tracing on settle. The
selected object always shows a wireframe overlay, in both modes.

While the camera moves the renderer drops to a half-resolution preview;
once you stop for ~150 ms it switches back to full resolution and
progressively accumulates samples — the title bar shows the live sample
count and pass rate, and rendering parks once it reaches 4096 spp.

## Layout

| File | Role |
|---|---|
| `src/settings.h` | every render knob |
| `src/vec3.h`, `ray.h`, `rng.h` | core math + PCG32 RNG (GPU-portable) |
| `src/camera.h` | look-from/look-at/fov → primary rays |
| `src/camera_controller.h` | fly camera (pos + yaw/pitch) for the viewer |
| `src/hittable.h`, `sphere.h`, `scene.h` | geometry + traversal |
| `src/scene_setup.h` | the demo scene |
| `src/material.h` | Lambertian/metal/glass/emissive scatter — **future Metal kernel** |
| `src/integrator.h` | iterative bounce loop — **future Metal kernel** |
| `src/renderer.h/.cpp` | CPU progressive accumulation + worker pool |
| `src/gltf_loader.h/.cpp` | GLB/JSON/accessor parsing, texture decode, transform bake |
| `src/bvh.h/.cpp` | midpoint-split BVH build (host-only, shared by both backends) |
| `src/mesh.h`, `texture.h` | CPU BVH traversal, Möller–Trumbore, bilinear sampling |
| `src/kernel_types.h` | structs shared between C++ and MSL |
| `src/kernels/pathtrace.metal` | the GPU kernel — line-by-line port of the CPU reference |
| `src/gpu_renderer.h/.mm` | Metal device/pipelines/buffers, batching, readback |
| `src/image.h/.cpp` | linear framebuffer → gamma-corrected PPM/PNG |
| `src/viewer.h`, `viewer_macos.mm` | AppKit window, input, GPU present / CPU blit |
| `src/main.cpp` | mode dispatch: viewer / offline / parity / gpu-check |
