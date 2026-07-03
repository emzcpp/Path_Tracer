// Native macOS viewer: an AppKit shell around ProgressiveRenderer.
//
// Thread layout:
//   main thread          — Cocoa event loop; owns FlyCamera + input state;
//                          a 60 Hz timer applies movement, publishes camera
//                          snapshots, and blits the latest frame to the layer
//   render thread        — owns the ProgressiveRenderer; loops 1-spp passes,
//                          accumulating while the camera is still, dropping
//                          to a half-res preview while it moves
//   worker pool          — inside ProgressiveRenderer (renderer.cpp)
//
// The two threads meet at exactly two mutexes: camera_mutex (snapshot +
// generation counter, main → render) and frame_mutex (resolved RGBA bytes,
// render → main). Everything else is thread-private or atomic.

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/QuartzCore.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "camera_controller.h"
#include "gpu_renderer.h"
#include "renderer.h"
#include "scene_io.h"
#include "scene_setup.h"
#include "settings.h"
#include "viewer.h"

#include "imgui.h"
#include "imgui_impl_metal.h"
#include "imgui_impl_osx.h"

#include "ImGuizmo.h"

namespace {

using Clock = std::chrono::steady_clock;

// Phase 3: the single central selection state. Plain C++, no UI types —
// Phase 4's gizmos consume this. Sphere identity is the (stable) index
// into SceneDesc::spheres; the mesh is one object.
struct Selection {
    enum class Kind { None, Sphere, Mesh } kind = Kind::None;
    int index = -1;         // sphere index when kind == Sphere
    float distance = 0.0f;  // world-space hit distance at pick time
};

// Session C part 3: everything the editors can mutate, as one snapshot.
// The editable state is a few KB, so whole-state undo beats per-field
// commands: every operation (material edit, gizmo drag, duplicate/delete,
// rename, scene load) is "restore the snapshot from before it began".
struct SceneState {
    std::vector<SphereData> spheres;
    std::shared_ptr<const MeshData> mesh;   // presence + identity
    std::string mesh_source_path;
    std::string mesh_name;
    float mesh_model[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 1};
    // Session F: environment.
    std::shared_ptr<const EnvMap> env;
    std::string env_source_path;
    float env_intensity = 1.0f;
    float env_yaw_deg = 0.0f;
};

struct ViewerCore {
    RenderSettings settings;
    Scene scene;                       // CPU backend only
    SceneDesc desc;                    // retained for CPU-side picking
    Selection selection;               // main-thread only

    // Undo/redo (main-thread only). idle_stash rolls forward every frame
    // nothing is being edited, so edit-begin always has the TRUE pre-edit
    // state even though ImGui widgets mutate values before activation is
    // observable. A whole drag coalesces into one entry: begin on
    // activation, commit when everything goes idle again.
    std::vector<SceneState> undo_stack;
    std::vector<SceneState> redo_stack;
    SceneState idle_stash;
    SceneState pending_before;
    bool edit_active = false;
    bool use_gpu = false;
    std::unique_ptr<GpuRenderer> gpu;  // GPU backend only

    // Phase 4: gizmo + editable geometry (all main-thread only).
    int gizmo_op = 0;                  // 0 translate, 1 rotate, 2 scale
    bool gizmo_local = false;          // local vs world manipulation space
    bool snap_enabled = false;
    float snap_translate = 0.5f, snap_rotate_deg = 15.0f, snap_scale = 0.1f;

    // Session E: fast-nav raster preview. 0 = off, 1 = solid, 2 = wire.
    // Auto-handoff by design: raster only while the camera moves, traced
    // when still (the settle machinery already reconverges on handback).
    int fastnav = 0;

    // Phase 5.2 viewport/render QoL (main-thread only).
    float interactive_scale = 1.0f;    // interactive-mode resolution scale
    bool paused = false;               // freeze accumulation (UI stays live)
    bool show_help = false;            // shortcut cheat-sheet overlay
    SceneDesc initial_desc;            // startup state, for per-object reset
    FlyCamera initial_fly;             // startup camera, for camera reset
    float initial_vfov = 35.0f;
    // The mesh's world geometry is re-baked from an object-space baseline
    // (the load-time bake, treated as "model = identity") whenever the
    // gizmo edits mesh_model. Pick and GPU read the same re-baked arrays.
    std::vector<GPUTriangle> mesh_object_tris;
    std::vector<GPUTriangle> mesh_tris;
    std::vector<BVHNode> mesh_nodes;
    float mesh_model[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 1};
    bool mesh_apply_pending = false;   // throttled re-bake during drags
    Clock::time_point last_mesh_apply{};

    // Camera channel: main thread → render thread.
    std::mutex camera_mutex;
    std::condition_variable camera_cv;
    FlyCamera camera_snapshot;
    std::uint64_t generation = 1;
    Clock::time_point last_change{};   // epoch ⇒ "long since settled" at startup

    // Frame channel: render thread → main thread.
    std::mutex frame_mutex;
    std::vector<std::uint8_t> published;
    int pub_w = 0, pub_h = 0, pub_passes = 0;
    bool fresh = false;

    std::atomic<bool> quit{false};
    std::atomic<bool> save_requested{false};

    // Render mode: FINAL locks the camera and converges to
    // settings.final_target_spp, then auto-exports a PNG once.
    std::atomic<bool> final_mode{false};
    std::atomic<bool> final_saved{false};

    std::thread render_thread;

    // Main-thread-only state (never touched by the render thread).
    FlyCamera fly;
    InputState input;
    float mouse_dx = 0.0f, mouse_dy = 0.0f;
    Clock::time_point last_tick = Clock::now();
    bool show_ui = true;   // U toggles the overlay panel (GPU backend)

    // THE central accumulation-reset hook. Anything that invalidates the
    // image (camera movement, mode switches, future scene edits) calls
    // this and nothing else; the render side reacts to the generation bump
    // by clearing the accumulator. Main thread only.
    void mark_scene_dirty(bool camera_moved = false) {
        const auto now = Clock::now();
        if (use_gpu) {
            // No render thread in GPU mode — main-thread fields, no lock.
            ++generation;
            if (camera_moved) last_change = now;
        } else {
            {
                std::lock_guard lk(camera_mutex);
                camera_snapshot = fly;
                ++generation;
                if (camera_moved) last_change = now;
            }
            camera_cv.notify_one();
        }
    }
};

std::string timestamped_png_name() {
    char name[64];
    std::time_t t = std::time(nullptr);
    std::tm tm;
    localtime_r(&t, &tm);
    std::strftime(name, sizeof name, "render_%Y%m%d_%H%M%S.png", &tm);
    return name;
}

// Shared by the F key and the UI checkbox — one entry point for the mode.
void set_final_mode(ViewerCore& core, bool entering) {
    core.final_mode = entering;
    // Re-arm the export latch on ENTRY only. Re-arming on exit would open
    // a window where a stale converged accumulation could spuriously
    // export on the next entry tick.
    if (entering) core.final_saved = false;
    if (entering) {
        // Fresh full-res convergence run (final_mode bypasses the preview
        // hysteresis, so this resets straight to full res).
        core.mark_scene_dirty();
        std::printf("FINAL mode: camera locked, converging to %d spp — "
                    "auto-saves PNG at target, P saves early, R unlocks\n",
                    core.settings.final_target_spp);
    } else {
        std::printf("interactive mode\n");
    }
}

// True when ImGui exists (GPU backend) and wants the event.
bool ui_wants_keyboard() {
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureKeyboard;
}
bool ui_wants_mouse() {
    return ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse;
}
// Hovering or dragging the gizmo — camera and picking must stand down.
bool gizmo_busy() {
    return ImGui::GetCurrentContext() &&
           (ImGuizmo::IsOver() || ImGuizmo::IsUsing());
}

// Human label derived from the metallic-roughness parameters.
const char* mat_type_name(const Material& m) {
    if (m.transmission > 0.5f) return "glass";
    if (m.emission.x + m.emission.y + m.emission.z > 0.0f) return "emissive";
    if (m.metallic > 0.5f) return "metal";
    return m.roughness > 0.7f ? "diffuse" : "glossy";
}

// Single-ray pick — CPU-side, click-only, so it never touches render
// performance. Reuses the renderer's own intersectors: Sphere::hit for
// spheres and mesh.h's bvh_hit over the LIVE mesh arrays (the same ones
// the GPU renders from, including any gizmo edits), with the renderer's
// tie semantics (mesh tested last, ties replace).
Selection pick_scene(const ViewerCore& core, const Ray& ray) {
    Selection sel;
    float closest = std::numeric_limits<float>::infinity();
    for (int i = 0; i < int(core.desc.spheres.size()); ++i) {
        const SphereData& sd = core.desc.spheres[i];
        const Sphere s(sd.center, sd.radius, sd.mat);
        HitRecord rec;
        if (s.hit(ray, 1e-3f, closest, rec)) {
            closest = rec.t;
            sel.kind = Selection::Kind::Sphere;
            sel.index = i;
        }
    }
    if (!core.mesh_tris.empty()) {
        TriHit th;
        if (bvh_hit(core.mesh_nodes.data(), core.mesh_tris.data(), ray.origin,
                    ray.dir, 1e-3f, closest, th)) {
            sel.kind = Selection::Kind::Mesh;
            sel.index = -1;
        }
    }
    if (sel.kind != Selection::Kind::None) {
        sel.distance = closest * ray.dir.length();   // t is in |dir| units
    }
    return sel;
}

// ---- Phase 4: matrices + transform application -------------------------
// Column-major GL-style float[16] (the ImGuizmo convention). Verified to
// project points to the same NDC as Camera::get_ray to ~1e-6.

void mat_look_at(const vec3& eye, const vec3& at, const vec3& up, float* m) {
    const vec3 f = normalize(at - eye);
    const vec3 s = normalize(cross(f, up));
    const vec3 u = cross(s, f);
    m[0] = s.x;  m[4] = s.y;  m[8] = s.z;   m[12] = -dot(s, eye);
    m[1] = u.x;  m[5] = u.y;  m[9] = u.z;   m[13] = -dot(u, eye);
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z; m[14] = dot(f, eye);
    m[3] = 0;    m[7] = 0;    m[11] = 0;    m[15] = 1;
}

void mat_perspective(float vfov_deg, float aspect, float zn, float zf,
                     float* m) {
    const float f = 1.0f / std::tan(vfov_deg * 3.14159265358979f / 360.0f);
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zf + zn) / (zn - zf);
    m[11] = -1.0f;
    m[14] = 2.0f * zf * zn / (zn - zf);
}

vec3 mat_apply_point(const float* m, const vec3& p) {
    return {m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
            m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
}

// Upper-3x3 inverse transpose: the correct normal transform under
// non-uniform scale (ImGuizmo's scale handles are per-axis).
void mat_normal_transform(const float* m, float out[9]) {
    const float a = m[0], b = m[4], c = m[8];
    const float d = m[1], e = m[5], f = m[9];
    const float g = m[2], h = m[6], i = m[10];
    const float det = a * (e * i - f * h) - b * (d * i - f * g) +
                      c * (d * h - e * g);
    const float k = det != 0.0f ? 1.0f / det : 0.0f;
    // inverse (adjugate/det), stored TRANSPOSED => inverse-transpose
    out[0] = (e * i - f * h) * k; out[1] = (c * h - b * i) * k; out[2] = (b * f - c * e) * k;
    out[3] = (f * g - d * i) * k; out[4] = (a * i - c * g) * k; out[5] = (c * d - a * f) * k;
    out[6] = (d * h - e * g) * k; out[7] = (b * g - a * h) * k; out[8] = (a * e - b * d) * k;
}

// Re-bake the mesh: world tris = mesh_model x object-space baseline, then
// rebuild the BVH (the existing builder — milliseconds at 15k tris) and
// hand the new arrays to the GPU. Pick reads the same arrays, so selection
// stays consistent with what renders.
void apply_mesh_transform(ViewerCore& core) {
    if (core.mesh_object_tris.empty()) return;
    const float* M = core.mesh_model;
    float nt[9];
    mat_normal_transform(M, nt);
    const auto xn = [&nt](const pt_float3& n) {
        const vec3 v(nt[0] * n.x + nt[3] * n.y + nt[6] * n.z,
                     nt[1] * n.x + nt[4] * n.y + nt[7] * n.z,
                     nt[2] * n.x + nt[5] * n.y + nt[8] * n.z);
        const vec3 u = normalize(v);
        return pt_float3{u.x, u.y, u.z};
    };
    const auto p3 = [](const vec3& v) { return pt_float3{v.x, v.y, v.z}; };

    core.mesh_tris = core.mesh_object_tris;
    for (GPUTriangle& T : core.mesh_tris) {
        const vec3 o0 = mesh_c3(T.p0);
        const vec3 p0 = mat_apply_point(M, o0);
        const vec3 p1 = mat_apply_point(M, o0 + mesh_c3(T.e1));
        const vec3 p2 = mat_apply_point(M, o0 + mesh_c3(T.e2));
        T.p0 = p3(p0);
        T.e1 = p3(p1 - p0);
        T.e2 = p3(p2 - p0);
        T.n0 = xn(T.n0);
        T.n1 = xn(T.n1);
        T.n2 = xn(T.n2);
    }
    build_bvh(core.mesh_tris, core.mesh_nodes);
    if (core.gpu) core.gpu->update_mesh_geometry(core.mesh_tris, core.mesh_nodes);
    core.mark_scene_dirty();
    core.mesh_apply_pending = false;
    core.last_mesh_apply = Clock::now();
}

// Sphere edits are trivial: the tracer intersects the analytic sphere list
// directly (no BVH involved), so an edit is a 4.6KB buffer swap.
void apply_sphere_edit(ViewerCore& core) {
    if (core.gpu) core.gpu->update_spheres(flatten_scene(core.desc));
    core.mark_scene_dirty();
}

// ---- Phase 5.2: discrete scene operations --------------------------------
// Each is a self-contained mutation ending in the central reset — the unit
// a future undo system would capture. No other reset paths exist.

std::string display_name(const ViewerCore& core, const Selection& sel) {
    if (sel.kind == Selection::Kind::Mesh) {
        return core.desc.mesh_name.empty() ? "Mesh" : core.desc.mesh_name;
    }
    if (sel.kind == Selection::Kind::Sphere && sel.index >= 0 &&
        sel.index < int(core.desc.spheres.size())) {
        const SphereData& sd = core.desc.spheres[sel.index];
        if (!sd.name.empty()) return sd.name;
        char buf[48];
        std::snprintf(buf, sizeof buf, "Sphere %d · %s", sel.index,
                      mat_type_name(sd.mat));
        return buf;
    }
    return "None";
}

void remove_mesh(ViewerCore& core) {
    core.desc.mesh.reset();
    core.desc.mesh_source_path.clear();
    core.desc.mesh_name.clear();
    core.mesh_object_tris.clear();
    core.mesh_tris.clear();
    core.mesh_nodes.clear();
    if (core.gpu) core.gpu->set_mesh(nullptr);
    core.mark_scene_dirty();
}

// Duplicate: spheres only — the renderer supports a single mesh instance.
bool can_duplicate(const Selection& sel) {
    return sel.kind == Selection::Kind::Sphere;
}

void duplicate_selected(ViewerCore& core) {
    if (core.selection.kind != Selection::Kind::Sphere) return;
    SphereData copy = core.desc.spheres[core.selection.index];
    const float off = std::fmax(0.15f, copy.radius * 0.6f);
    copy.center = copy.center + vec3(off, 0.0f, off);
    if (!copy.name.empty()) copy.name += " copy";
    core.desc.spheres.push_back(copy);
    core.selection = Selection{Selection::Kind::Sphere,
                               int(core.desc.spheres.size()) - 1, 0.0f};
    apply_sphere_edit(core);
}

void delete_selected(ViewerCore& core) {
    if (core.selection.kind == Selection::Kind::Sphere) {
        core.desc.spheres.erase(core.desc.spheres.begin() +
                                core.selection.index);
        core.selection = Selection{};
        apply_sphere_edit(core);
    } else if (core.selection.kind == Selection::Kind::Mesh) {
        core.selection = Selection{};
        remove_mesh(core);
    }
}

// Tab / Shift+Tab: cycle spheres 0..N-1, then the mesh, then wrap.
void cycle_selection(ViewerCore& core, bool backward) {
    const int n_spheres = int(core.desc.spheres.size());
    const int total = n_spheres + (core.desc.mesh ? 1 : 0);
    if (total == 0) return;
    int slot;   // 0..total-1; mesh occupies the last slot
    if (core.selection.kind == Selection::Kind::Sphere) {
        slot = core.selection.index;
    } else if (core.selection.kind == Selection::Kind::Mesh) {
        slot = n_spheres;
    } else {
        slot = backward ? 0 : total - 1;   // so the step lands on an end
    }
    slot = (slot + (backward ? total - 1 : 1)) % total;
    core.selection = slot < n_spheres
        ? Selection{Selection::Kind::Sphere, slot, 0.0f}
        : Selection{Selection::Kind::Mesh, -1, 0.0f};
}

// F: dolly the camera along its current view direction so the selected
// object fills a comfortable fraction of the frame.
void frame_selected(ViewerCore& core) {
    if (core.selection.kind == Selection::Kind::None) return;
    vec3 center;
    float size = 1.0f;
    if (core.selection.kind == Selection::Kind::Sphere) {
        const SphereData& sd = core.desc.spheres[core.selection.index];
        center = sd.center;
        size = sd.radius;
    } else {
        if (core.mesh_nodes.empty()) return;
        const BVHNode& root = core.mesh_nodes[0];   // live world bounds
        center = 0.5f * (mesh_c3(root.mn) + mesh_c3(root.mx));
        size = 0.5f * (mesh_c3(root.mx) - mesh_c3(root.mn)).length();
    }
    const float half_fov =
        core.settings.vfov_deg * 3.14159265358979f / 360.0f;
    const float dist = std::fmax(0.5f, 2.2f * size / std::tan(half_fov));
    core.fly.pos = center - core.fly.forward() * dist;
    core.mark_scene_dirty(/*camera_moved=*/true);
}

// ---- Session C part 3: undo/redo ----------------------------------------

SceneState capture_state(const ViewerCore& core) {
    SceneState s;
    s.spheres = core.desc.spheres;
    s.mesh = core.desc.mesh;
    s.mesh_source_path = core.desc.mesh_source_path;
    s.mesh_name = core.desc.mesh_name;
    std::memcpy(s.mesh_model, core.mesh_model, sizeof s.mesh_model);
    s.env = core.desc.env;
    s.env_source_path = core.desc.env_source_path;
    s.env_intensity = core.desc.env_intensity;
    s.env_yaw_deg = core.desc.env_yaw_deg;
    return s;
}

bool states_equal(const SceneState& a, const SceneState& b) {
    if (a.mesh != b.mesh || a.mesh_name != b.mesh_name) return false;
    if (a.env != b.env || a.env_intensity != b.env_intensity ||
        a.env_yaw_deg != b.env_yaw_deg)
        return false;
    if (std::memcmp(a.mesh_model, b.mesh_model, sizeof a.mesh_model) != 0)
        return false;
    if (a.spheres.size() != b.spheres.size()) return false;
    for (std::size_t i = 0; i < a.spheres.size(); ++i) {
        const SphereData& x = a.spheres[i];
        const SphereData& y = b.spheres[i];
        if (x.name != y.name) return false;
        if (std::memcmp(&x.center, &y.center, sizeof(point3)) != 0 ||
            x.radius != y.radius ||
            std::memcmp(&x.mat, &y.mat, sizeof(Material)) != 0) {
            return false;
        }
    }
    return true;
}

// Push the desc's env state to the GPU (buffer swap only on identity
// change; params always) — the single env-apply path.
void apply_env(ViewerCore& core, bool env_changed) {
    if (!core.gpu) return;
    if (env_changed) core.gpu->set_env(core.desc.env.get());
    core.gpu->set_env_params(core.desc.env_intensity,
                             core.desc.env_yaw_deg / 360.0f);
    core.mark_scene_dirty();
}

// Restores through the SAME apply paths as interactive edits, ending in
// the one central reset.
void restore_state(ViewerCore& core, const SceneState& s) {
    core.desc.spheres = s.spheres;
    apply_sphere_edit(core);

    if (s.mesh != core.desc.mesh) {
        core.desc.mesh = s.mesh;
        core.desc.mesh_source_path = s.mesh_source_path;
        if (s.mesh) {
            core.mesh_object_tris = s.mesh->tris;
            if (core.gpu) core.gpu->set_mesh(s.mesh.get());
        } else {
            core.mesh_object_tris.clear();
            core.mesh_tris.clear();
            core.mesh_nodes.clear();
            if (core.gpu) core.gpu->set_mesh(nullptr);
        }
    }
    core.desc.mesh_name = s.mesh_name;
    std::memcpy(core.mesh_model, s.mesh_model, sizeof core.mesh_model);
    if (core.desc.mesh) apply_mesh_transform(core);

    const bool env_changed = s.env != core.desc.env;
    core.desc.env = s.env;
    core.desc.env_source_path = s.env_source_path;
    core.desc.env_intensity = s.env_intensity;
    core.desc.env_yaw_deg = s.env_yaw_deg;
    apply_env(core, env_changed);

    if (core.selection.kind == Selection::Kind::Sphere &&
        core.selection.index >= int(core.desc.spheres.size())) {
        core.selection = Selection{};
    }
    if (core.selection.kind == Selection::Kind::Mesh && !core.desc.mesh) {
        core.selection = Selection{};
    }
    core.mark_scene_dirty();
}

void undo_begin(ViewerCore& core) {
    if (core.edit_active) return;
    core.pending_before = core.idle_stash;   // true pre-edit state
    core.edit_active = true;
}

void undo_commit(ViewerCore& core, const char* what) {
    if (!core.edit_active) return;
    core.edit_active = false;
    if (states_equal(core.pending_before, capture_state(core))) return;
    core.undo_stack.push_back(std::move(core.pending_before));
    if (core.undo_stack.size() > 64) {
        core.undo_stack.erase(core.undo_stack.begin());
    }
    core.redo_stack.clear();
    std::printf("edit: %s (undo depth %zu)\n", what, core.undo_stack.size());
    std::fflush(stdout);
}

void perform_undo(ViewerCore& core) {
    if (core.edit_active) undo_commit(core, "edit");   // finish in-flight
    if (core.undo_stack.empty()) {
        std::printf("undo: nothing to undo\n");
        std::fflush(stdout);
        return;
    }
    core.redo_stack.push_back(capture_state(core));
    restore_state(core, core.undo_stack.back());
    core.undo_stack.pop_back();
    core.idle_stash = capture_state(core);
    std::printf("undo (%zu left, %zu redoable)\n", core.undo_stack.size(),
                core.redo_stack.size());
    std::fflush(stdout);
}

void perform_redo(ViewerCore& core) {
    if (core.redo_stack.empty()) {
        std::printf("redo: nothing to redo\n");
        std::fflush(stdout);
        return;
    }
    core.undo_stack.push_back(capture_state(core));
    restore_state(core, core.redo_stack.back());
    core.redo_stack.pop_back();
    core.idle_stash = capture_state(core);
    std::printf("redo (%zu left, %zu undoable)\n", core.redo_stack.size(),
                core.undo_stack.size());
    std::fflush(stdout);
}

// Call right after any scene-mutating ImGui widget: begins a coalesced
// edit on activation; the commit happens centrally when everything idles.
void undo_track(ViewerCore& core) {
    if (ImGui::IsItemActivated()) undo_begin(core);
}

// ---- Phase 5: scene save/load glue --------------------------------------

SceneSnapshot snapshot_from_core(const ViewerCore& core) {
    SceneSnapshot s;
    s.spheres = core.desc.spheres;
    if (core.desc.mesh) {
        s.mesh_source = core.desc.mesh_source_path;
        s.mesh_name = core.desc.mesh_name;
        std::memcpy(s.mesh_model, core.mesh_model, sizeof s.mesh_model);
    }
    s.env_source = core.desc.env_source_path;
    s.env_intensity = core.desc.env_intensity;
    s.env_yaw_deg = core.desc.env_yaw_deg;
    s.cam_pos = core.fly.pos;
    s.cam_yaw = core.fly.yaw;
    s.cam_pitch = core.fly.pitch;
    s.move_speed = core.fly.move_speed;
    s.vfov_deg = core.settings.vfov_deg;
    s.final_target_spp = core.settings.final_target_spp;
    s.max_depth = core.settings.max_depth;
    s.gpu_passes_per_tick = core.settings.gpu_passes_per_tick;
    s.clamp_indirect = core.settings.clamp_indirect;
    return s;
}

// Rebuilds the live scene from a loaded snapshot. Mesh import happens
// FIRST so a failure leaves the current scene untouched. Everything flows
// through the same paths as interactive edits: apply_sphere_edit,
// apply_mesh_transform (re-bake + build_bvh — the normal startup path),
// and ultimately mark_scene_dirty().
bool apply_snapshot(ViewerCore& core, SceneSnapshot&& snap,
                    std::string& error) {
    namespace fs = std::filesystem;

    std::shared_ptr<const MeshData> imported;
    bool mesh_same = false;
    if (!snap.mesh_source.empty()) {
        if (core.desc.mesh) {
            std::error_code ec;
            mesh_same = fs::weakly_canonical(snap.mesh_source, ec) ==
                        fs::weakly_canonical(core.desc.mesh_source_path, ec);
        }
        if (!mesh_same) {
            std::string err;
            imported = load_glb(snap.mesh_source, MeshPlacement{}, err);
            if (!imported) {
                error = "mesh import failed: " + err;
                return false;
            }
        }
    }

    // Commit point — no failures past here.
    core.desc.spheres = std::move(snap.spheres);
    apply_sphere_edit(core);

    if (snap.mesh_source.empty()) {
        if (core.desc.mesh) {
            core.desc.mesh.reset();
            core.desc.mesh_source_path.clear();
            core.mesh_object_tris.clear();
            core.mesh_tris.clear();
            core.mesh_nodes.clear();
            if (core.gpu) core.gpu->set_mesh(nullptr);
        }
    } else {
        if (!mesh_same) {
            core.desc.mesh = imported;
            core.desc.mesh_source_path = snap.mesh_source;
            core.mesh_object_tris = imported->tris;
            if (core.gpu) core.gpu->set_mesh(imported.get());
        }
        core.desc.mesh_name = snap.mesh_name;
        std::memcpy(core.mesh_model, snap.mesh_model, sizeof core.mesh_model);
        apply_mesh_transform(core);
    }

    // Environment: import if the source changed; clear if absent.
    {
        bool env_changed = false;
        if (snap.env_source.empty()) {
            env_changed = core.desc.env != nullptr;
            core.desc.env.reset();
            core.desc.env_source_path.clear();
        } else {
            namespace fs = std::filesystem;
            std::error_code ec;
            const bool same =
                core.desc.env &&
                fs::weakly_canonical(snap.env_source, ec) ==
                    fs::weakly_canonical(core.desc.env_source_path, ec);
            if (!same) {
                std::string err;
                auto env = load_hdr(snap.env_source, err);
                if (env) {
                    core.desc.env = env;
                    core.desc.env_source_path = snap.env_source;
                    env_changed = true;
                } else {
                    std::fprintf(stderr, "scene env: %s\n", err.c_str());
                }
            }
        }
        core.desc.env_intensity = snap.env_intensity;
        core.desc.env_yaw_deg = snap.env_yaw_deg;
        apply_env(core, env_changed);
    }

    core.fly.pos = snap.cam_pos;
    core.fly.yaw = snap.cam_yaw;
    core.fly.pitch = snap.cam_pitch;
    core.fly.move_speed = snap.move_speed;
    core.settings.vfov_deg = snap.vfov_deg;
    core.settings.final_target_spp = snap.final_target_spp;
    core.settings.max_depth = snap.max_depth;
    core.settings.gpu_passes_per_tick = snap.gpu_passes_per_tick;
    core.settings.clamp_indirect = snap.clamp_indirect;
    if (core.gpu) {
        core.gpu->set_max_depth(snap.max_depth);
        core.gpu->set_clamp_indirect(snap.clamp_indirect);
    }

    core.selection = Selection{};   // stale indices must not survive a load
    core.mark_scene_dirty();
    return true;
}

void print_selection(const ViewerCore& core) {
    const Selection& sel = core.selection;
    switch (sel.kind) {
        case Selection::Kind::None:
            std::printf("selection: none\n");
            break;
        case Selection::Kind::Mesh:
            std::printf("selection: mesh (glTF model), hit at %.2f\n",
                        sel.distance);
            break;
        case Selection::Kind::Sphere: {
            const SphereData& sd = core.desc.spheres[sel.index];
            std::printf("selection: sphere #%d%s — %s, r=%.2f, "
                        "center (%.2f, %.2f, %.2f), hit at %.2f\n",
                        sel.index, sel.index == 0 ? " (ground)" : "",
                        mat_type_name(sd.mat), sd.radius, sd.center.x,
                        sd.center.y, sd.center.z, sel.distance);
            break;
        }
    }
    std::fflush(stdout);
}

void render_thread_main(ViewerCore& core) {
    // Leave one core for the UI thread so event handling stays snappy.
    const unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    ProgressiveRenderer renderer(core.scene, core.settings, hw - 1,
                                 env_lookup(core.desc));

    const auto settle =
        std::chrono::duration<float, std::milli>(core.settings.settle_ms);
    std::uint64_t accum_gen = 0;   // generation the accumulator was built for

    while (!core.quit) {
        FlyCamera snap;
        std::uint64_t gen;
        Clock::time_point changed;
        {
            std::lock_guard lk(core.camera_mutex);
            gen = core.generation;
            snap = core.camera_snapshot;
            changed = core.last_change;
        }

        // Hysteresis: stay in preview until the camera has been still for
        // settle_ms, so resolution doesn't flicker between mouse events.
        // FINAL mode bypasses it — always full res.
        const bool final_mode = core.final_mode;
        const bool moving = !final_mode && (Clock::now() - changed) < settle;
        const int w = moving ? core.settings.width / core.settings.preview_divisor
                             : core.settings.width;
        const int h = moving ? core.settings.height / core.settings.preview_divisor
                             : core.settings.height;

        if (gen != accum_gen || w != renderer.width()) {
            renderer.reset(w, h);
            accum_gen = gen;
        }

        const int target = final_mode
            ? core.settings.final_target_spp
            : (moving ? INT_MAX : core.settings.max_accum_passes);
        if (renderer.passes() < target) {
            // Aspect always from full-res settings so preview framing matches.
            renderer.render_pass(snap.make_camera(core.settings.vfov_deg,
                                                  core.settings.aspect()));
            std::lock_guard lk(core.frame_mutex);
            core.published.assign(renderer.rgba().begin(), renderer.rgba().end());
            core.pub_w = w;
            core.pub_h = h;
            core.pub_passes = renderer.passes();
            core.fresh = true;
        } else {
            // FINAL mode: export once when the target is reached, BEFORE
            // parking (the park only wakes on dirty/save/quit).
            if (final_mode && !core.final_saved.exchange(true)) {
                const std::string name = timestamped_png_name();
                if (renderer.save_png(name)) {
                    std::printf("FINAL: saved %s (%dx%d @ %d spp)\n",
                                name.c_str(), renderer.width(),
                                renderer.height(), renderer.passes());
                }
            }
            // Converged: park until the camera moves, a save arrives, or quit.
            std::unique_lock lk(core.camera_mutex);
            core.camera_cv.wait(lk, [&] {
                return core.quit || core.save_requested ||
                       core.generation != accum_gen;
            });
        }

        if (core.save_requested.exchange(false)) {
            const std::string name = timestamped_png_name();
            if (renderer.save_png(name)) {
                std::printf("saved %s (%dx%d @ %d spp)\n", name.c_str(),
                            renderer.width(), renderer.height(),
                            renderer.passes());
            }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------

@interface PTView : NSView
@property(nonatomic, assign) ViewerCore* core;
@end

@implementation PTView {
    BOOL cursorHidden_;
    float dragAccum_;      // points moved since mouseDown — click vs drag
    BOOL pickCandidate_;   // press began on the viewport, not the UI
}

- (BOOL)acceptsFirstResponder { return YES; }
// Let the first click on an unfocused window start a drag instead of being
// swallowed by window activation.
- (BOOL)acceptsFirstMouse:(NSEvent*)event { return YES; }

// GPU backend: the view is backed by a CAMetalLayer the resolve kernel
// writes into. Requires `core` to be set before wantsLayer = YES.
- (CALayer*)makeBackingLayer {
    if (!self.core || !self.core->use_gpu) return [super makeBackingLayer];

    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = (__bridge id<MTLDevice>)self.core->gpu->metal_device();
    // Plain BGRA8 (not _sRGB): the resolve kernel gamma-encodes manually,
    // matching encode_channel() so screen == saved PNG. Compute-writing the
    // drawable needs framebufferOnly = NO (fine on Apple-family GPUs).
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = NO;
    // Pinned to render resolution; CA upscales to the Retina backing.
    // Chasing backingScaleFactor would 4x the ray count.
    layer.drawableSize = CGSizeMake(self.core->settings.width,
                                    self.core->settings.height);
    layer.maximumDrawableCount = 3;   // pacing semaphore is 2 — never blocks
    // Explicit sRGB tag: a nil colorspace means display-native
    // interpretation, which on P3 panels would render more saturated than
    // the CPU viewer's sRGB-tagged CGImages.
    static CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    layer.colorspace = srgb;
    layer.contentsGravity = kCAGravityResize;
    return layer;
}

// Physical key codes (layout-independent, so WASD stays positional on any
// keyboard): W=13 A=0 S=1 D=2 Q=12 E=14 F=3 P=35 Esc=53.
- (void)setKeyCode:(unsigned short)code pressed:(bool)down {
    switch (code) {
        case 13: self.core->input.fwd   = down; break;
        case 1:  self.core->input.back  = down; break;
        case 0:  self.core->input.left  = down; break;
        case 2:  self.core->input.right = down; break;
        case 12: self.core->input.down  = down; break;
        case 14: self.core->input.up    = down; break;
        default: break;
    }
}

- (void)keyDown:(NSEvent*)event {
    if (event.isARepeat) return;
    // Input arbitration: when a UI widget owns the keyboard, the camera
    // and hotkeys must not react.
    if (ui_wants_keyboard()) return;
    const bool shift =
        (event.modifierFlags & NSEventModifierFlagShift) != 0;
    if (event.keyCode == 32) {   // U — toggle the UI panel
        self.core->show_ui = !self.core->show_ui;
        return;
    }
    if (event.keyCode == 15) {   // R — toggle FINAL render mode (was F)
        set_final_mode(*self.core, !self.core->final_mode);
        return;
    }
    if (event.keyCode == 3) {    // F — frame the selected object
        frame_selected(*self.core);
        return;
    }
    if (event.keyCode == 9) {    // V — cycle fast-nav: off / solid / wire
        if (self.core->use_gpu) {
            self.core->fastnav = (self.core->fastnav + 1) % 3;
            const char* names[3] = {"off", "solid", "wireframe"};
            std::printf("fast-nav: %s\n", names[self.core->fastnav]);
            std::fflush(stdout);
        }
        return;
    }
    if (event.keyCode == 48) {   // Tab / Shift+Tab — cycle selection
        cycle_selection(*self.core, shift);
        return;
    }
    if (event.keyCode == 51) {   // Backspace — delete selected
        undo_begin(*self.core);
        delete_selected(*self.core);
        undo_commit(*self.core, "delete");
        return;
    }
    if (event.keyCode == 122 || (event.keyCode == 44 && shift)) {
        self.core->show_help = !self.core->show_help;   // F1 or ?
        return;
    }
    // 1/2/3 — gizmo mode (W/E/R would collide with WASD movement).
    if (event.keyCode == 18) { self.core->gizmo_op = 0; return; }   // translate
    if (event.keyCode == 19) { self.core->gizmo_op = 1; return; }   // rotate
    if (event.keyCode == 20) { self.core->gizmo_op = 2; return; }   // scale
    if (event.keyCode == 35) {   // P — save current accumulation as PNG
        self.core->save_requested = true;
        self.core->camera_cv.notify_one();
        return;
    }
    if (event.keyCode == 53) {   // Esc — close help overlay, else quit
        if (self.core->show_help) {
            self.core->show_help = false;
            return;
        }
        [NSApp terminate:nil];
        return;
    }
    // Swallow unhandled keys: forwarding to super beeps.
    [self setKeyCode:event.keyCode pressed:true];
}

- (void)keyUp:(NSEvent*)event {
    // Deliberately NOT gated on the UI: releasing a key must always clear
    // its bit, or a key held when the pointer enters the panel would stick.
    [self setKeyCode:event.keyCode pressed:false];
}

- (void)flagsChanged:(NSEvent*)event {
    self.core->input.boost =
        (event.modifierFlags & NSEventModifierFlagShift) != 0;
}

- (void)mouseDown:(NSEvent*)event {
    dragAccum_ = 0.0f;
    // A press on the UI panel is never a pick (Phase 2 arbitration).
    pickCandidate_ = !ui_wants_mouse();
    // Cursor hiding happens on the first actual drag, so plain clicks
    // (picks) don't flicker the cursor.
}

- (void)mouseUp:(NSEvent*)event {
    [self unhideCursor];   // never gated: always safe
    // A click on a gizmo handle is a manipulation, never a (de)select.
    const bool clicked = pickCandidate_ && dragAccum_ < 3.0f &&
                         !ui_wants_mouse() && !gizmo_busy();
    pickCandidate_ = NO;
    if (!clicked) return;

    // Pick ray through the cursor, using the SAME camera and ray function
    // as primary rays. Two 1:1 alignments make the mapping direct: the
    // view is 960x540 points over a 960x540 drawable, and an unflipped
    // NSView's y is bottom-up — exactly get_ray's t convention.
    const NSPoint p = [self convertPoint:event.locationInWindow fromView:nil];
    const float s = float(p.x) / float(self.bounds.size.width);
    const float t = float(p.y) / float(self.bounds.size.height);
    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f) return;

    const Camera cam = self.core->fly.make_camera(
        self.core->settings.vfov_deg, self.core->settings.aspect());
    self.core->selection = pick_scene(*self.core, cam.get_ray(s, t));
    print_selection(*self.core);
}

- (void)unhideCursor {
    if (cursorHidden_) {
        [NSCursor unhide];
        cursorHidden_ = NO;
    }
}

- (void)mouseDragged:(NSEvent*)event {
    if (ui_wants_mouse()) return;   // dragging a slider must not turn the camera
    if (gizmo_busy()) return;       // dragging the gizmo must not turn it either
    dragAccum_ += float(std::fabs(event.deltaX) + std::fabs(event.deltaY));
    if (!cursorHidden_) {
        [NSCursor hide];
        cursorHidden_ = YES;
    }
    // Raw HID deltas — keep flowing even when the pointer pins at a screen
    // edge, so no cursor-warping is needed.
    self.core->mouse_dx += float(event.deltaX);
    self.core->mouse_dy += float(event.deltaY);
}

- (void)scrollWheel:(NSEvent*)event {
    if (ui_wants_mouse()) return;   // scrolling the panel must not change speed
    const float k = std::pow(1.15f, float(event.scrollingDeltaY) * 0.05f);
    self.core->fly.move_speed =
        std::clamp(self.core->fly.move_speed * k, 0.05f, 50.0f);
}

@end

// ---------------------------------------------------------------------------

@interface PTAppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
- (instancetype)initWithCore:(ViewerCore*)core;
@end

@implementation PTAppDelegate {
    ViewerCore* core_;
    NSWindow* window_;
    PTView* view_;
    NSTimer* timer_;
    CGColorSpaceRef colorspace_;

    // GPU backend state — main-thread only, no locks anywhere.
    CAMetalLayer* metalLayer_;
    dispatch_semaphore_t gpuInflight_;
    // Session G: budget controller + instrumentation (zeroed in init).
    float nsPerPxPass_;   // EMA cost model from completed batches
    float gpuMsEma_;      // smoothed per-CB GPU time (display)
    float tickMsEma_;     // smoothed main-thread tick time (display)
    int lastSliceRows_;   // last submitted work shape (display)
    int lastSliceK_;
    bool savingPng_;      // async export in flight: pause new traces
    std::uint64_t gpuAppliedGen_;
    int gpuLastPasses_;
    float gpuRate_;   // smoothed passes/s for the title
    BOOL imguiInited_;

    // Phase 5: scene file UI state.
    char scenePath_[256];
    std::string sceneStatus_;
    // Session F: environment UI state.
    char envPath_[256];
    std::string envStatus_;
    // Phase 5.2: stats + rename buffer.
    Clock::time_point accumStart_;
    char nameBuf_[64];
    Selection lastNameSel_;   // resync nameBuf_ when the selection changes

    // imgui_impl_osx's hidden NSTextInputClient subview. Typed characters
    // only flow through ITS keyDown (via interpretKeyEvents/insertText) —
    // the event monitor alone can't deliver text. We arbitrate first-
    // responder between it and PTView based on io.WantTextInput.
    NSView* imguiTextInput_;
}

- (instancetype)initWithCore:(ViewerCore*)core {
    if ((self = [super init])) {
        core_ = core;
        // The RGBA bytes are already gamma-encoded; sRGB labels them so.
        colorspace_ = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        std::strncpy(scenePath_, "scene.json", sizeof scenePath_);
        std::strncpy(envPath_, "assets/kloofendal_puresky_2k.hdr",
                     sizeof envPath_);
    }
    return self;
}

- (void)dealloc {
    CGColorSpaceRelease(colorspace_);
}

- (void)applicationDidFinishLaunching:(NSNotification*)note {
    // Window content in points == render size in pixels; the layer upscales
    // to Retina backing. Chasing backingScaleFactor would 4x the ray count.
    const NSRect rect =
        NSMakeRect(0, 0, core_->settings.width, core_->settings.height);
    // GPU backend: resizable — the render resolution tracks the window
    // live (updateRenderSize). CPU reference backend keeps fixed dims.
    NSWindowStyleMask style = NSWindowStyleMaskTitled |
                              NSWindowStyleMaskClosable |
                              NSWindowStyleMaskMiniaturizable;
    if (core_->use_gpu) style |= NSWindowStyleMaskResizable;
    window_ = [[NSWindow alloc] initWithContentRect:rect
                                          styleMask:style
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    window_.contentMinSize = NSMakeSize(480, 270);
    window_.releasedWhenClosed = NO;   // ARC + default YES = over-release crash
    window_.title = @"pathtracer";
    window_.delegate = self;
    [window_ center];

    view_ = [[PTView alloc] initWithFrame:rect];
    view_.core = core_;   // must precede wantsLayer: makeBackingLayer reads it
    view_.wantsLayer = YES;
    view_.layer.contentsGravity = kCAGravityResize;
    view_.layer.magnificationFilter = kCAFilterLinear;   // smooth preview upscale
    view_.layer.backgroundColor = CGColorGetConstantColor(kCGColorBlack);
    window_.contentView = view_;

    [window_ makeKeyAndOrderFront:nil];
    [window_ makeFirstResponder:view_];
    if (@available(macOS 14.0, *)) {
        [NSApp activate];
    } else {
        [NSApp activateIgnoringOtherApps:YES];
    }

    if (core_->use_gpu) {
        metalLayer_ = (CAMetalLayer*)view_.layer;
        gpuInflight_ = dispatch_semaphore_create(2);
        nsPerPxPass_ = gpuMsEma_ = tickMsEma_ = 0.0f;
        lastSliceRows_ = lastSliceK_ = 0;
        savingPng_ = false;
        gpuAppliedGen_ = 0;   // != generation (1): first tick clears + starts
        accumStart_ = Clock::now();
        [self updateRenderSize];

        // Dear ImGui: OSX platform backend (installs its own event
        // monitors) + Metal renderer drawing into our drawable.
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplOSX_Init(view_);
        ImGui_ImplMetal_Init((__bridge id<MTLDevice>)core_->gpu->metal_device());
        imguiInited_ = YES;

        // Init added the text-input responder as a subview of our view;
        // find and remember it, then normalize focus back to PTView so
        // WASD works from the first frame.
        for (NSView* sub in view_.subviews) {
            if ([sub conformsToProtocol:@protocol(NSTextInputClient)]) {
                imguiTextInput_ = sub;
                break;
            }
        }
        [window_ makeFirstResponder:view_];
    } else {
        core_->render_thread = std::thread(render_thread_main, std::ref(*core_));
    }

    timer_ = [NSTimer timerWithTimeInterval:1.0 / 60.0
                                     target:self
                                   selector:@selector(tick:)
                                   userInfo:nil
                                    repeats:YES];
    // Common modes: keep firing during window drags and menu tracking.
    [[NSRunLoop mainRunLoop] addTimer:timer_ forMode:NSRunLoopCommonModes];
}

- (void)tick:(NSTimer*)timer {
    const auto now = Clock::now();
    float dt = std::chrono::duration<float>(now - core_->last_tick).count();
    core_->last_tick = now;
    dt = std::min(dt, 0.1f);   // don't lurch after a stall (window drag etc.)

    // Undo coalescing: roll the pre-edit stash while nothing is being
    // edited; commit one entry when an in-flight edit goes idle (this is
    // what turns a whole slider drag into a single undo step).
    if (imguiInited_) {
        const bool busy = ImGui::IsAnyItemActive() || ImGuizmo::IsUsing();
        if (core_->edit_active && !busy) {
            undo_commit(*core_, "edit");
        } else if (!core_->edit_active && !busy) {
            core_->idle_stash = capture_state(*core_);
        }
    }

    // Focus arbitration for text fields: while an ImGui text widget is
    // active, its NSTextInputClient responder must be first responder to
    // receive characters; the moment it deactivates, PTView takes back
    // the keyboard so WASD and hotkeys resume.
    if (imguiInited_ && imguiTextInput_) {
        const bool want_text = ImGui::GetIO().WantTextInput;
        NSResponder* fr = window_.firstResponder;
        if (want_text && fr != imguiTextInput_) {
            [window_ makeFirstResponder:imguiTextInput_];
        } else if (!want_text && fr == imguiTextInput_) {
            [window_ makeFirstResponder:view_];
        }
    }

    const bool look = core_->mouse_dx != 0.0f || core_->mouse_dy != 0.0f;
    if (core_->final_mode || gizmo_busy()) {
        // FINAL mode or gizmo interaction: camera stands down. Drop
        // pending input so resuming doesn't lurch from stale deltas.
        core_->mouse_dx = core_->mouse_dy = 0.0f;
    } else if (core_->input.any_movement() || look) {
        core_->fly.apply_move(core_->input, dt);
        core_->fly.apply_look(core_->mouse_dx, core_->mouse_dy);
        core_->mouse_dx = core_->mouse_dy = 0.0f;
        core_->mark_scene_dirty(/*camera_moved=*/true);
    }

    if (core_->use_gpu) {
        [self tickGPU:dt];
        return;
    }

    // CPU backend: blit the newest resolved frame, if any.
    CFDataRef data = nullptr;
    int w = 0, h = 0, passes = 0;
    {
        std::lock_guard lk(core_->frame_mutex);
        if (core_->fresh) {
            data = CFDataCreate(nullptr, core_->published.data(),
                                CFIndex(core_->published.size()));
            w = core_->pub_w;
            h = core_->pub_h;
            passes = core_->pub_passes;
            core_->fresh = false;
        }
    }
    if (!data) return;

    CGDataProviderRef provider = CGDataProviderCreateWithCFData(data);
    CGImageRef img = CGImageCreate(
        size_t(w), size_t(h), 8, 32, size_t(w) * 4, colorspace_,
        CGBitmapInfo(kCGImageAlphaNoneSkipLast) | kCGBitmapByteOrder32Big,
        provider, nullptr,
        false, kCGRenderingIntentDefault);
    [CATransaction begin];
    [CATransaction setDisableActions:YES];   // no implicit contents cross-fade
    view_.layer.contents = (__bridge id)img;
    [CATransaction commit];
    // The layer retains the image; releasing all three here is leak-free.
    CGImageRelease(img);
    CGDataProviderRelease(provider);
    CFRelease(data);

    if (core_->final_mode) {
        window_.title = [NSString
            stringWithFormat:@"pathtracer FINAL %d/%d spp%s", passes,
                             core_->settings.final_target_spp,
                             passes >= core_->settings.final_target_spp
                                 ? " — saved (F to unlock)"
                                 : ""];
    } else {
        window_.title = [NSString
            stringWithFormat:@"pathtracer — %dx%d @ %d spp", w, h, passes];
    }
}

// GPU backend tick: reset-on-change, then encode one batch if the GPU has
// a free slot. Non-blocking semaphore try: when the GPU is behind, the tick
// simply skips encoding — input stays responsive, convergence rate degrades
// gracefully.
- (void)tickGPU:(float)dt {
    const auto tick_t0 = Clock::now();
    struct TickTimer {   // EMA on every exit path
        PTAppDelegate* d;
        std::chrono::steady_clock::time_point t0;
        ~TickTimer() {
            const float ms = std::chrono::duration<float, std::milli>(
                                 Clock::now() - t0)
                                 .count();
            d->tickMsEma_ = d->tickMsEma_ > 0.0f
                                ? 0.9f * d->tickMsEma_ + 0.1f * ms
                                : ms;
        }
    } tick_timer{self, tick_t0};
    GpuRenderer& gpu = *core_->gpu;
    const auto now = Clock::now();

    // Throttled mesh re-bake, shared by the gizmo and the numeric TRS
    // fields: apply when the interaction settles, or every ~80ms during a
    // continuous drag/scrub.
    if (core_->mesh_apply_pending) {
        const bool interacting =
            ImGuizmo::IsUsing() ||
            (ImGui::GetCurrentContext() && ImGui::IsAnyItemActive());
        const float since_ms = std::chrono::duration<float, std::milli>(
            now - core_->last_mesh_apply).count();
        if (!interacting || since_ms > 80.0f) apply_mesh_transform(*core_);
    }

    const bool final_mode = core_->final_mode;
    const bool moving =
        !final_mode &&
        std::chrono::duration<float, std::milli>(now - core_->last_change)
                .count() < core_->settings.settle_ms;
    // Resolution scale. Below 1x: faster interactive movement. Above 1x:
    // supersampled rendering — at 2x the accumulation matches the Retina
    // drawable exactly (native-pixel path tracing), and PNG exports come
    // out at the scaled size. FINAL never drops below full res but does
    // follow the slider upward. The dims change flows through the same
    // reset machinery as everything else; no cost at 1x.
    const float scale = final_mode
        ? std::fmax(1.0f, core_->interactive_scale)
        : core_->interactive_scale;
    const int base_w = std::max(64, int(core_->settings.width * scale)) & ~1;
    const int base_h = std::max(36, int(core_->settings.height * scale)) & ~1;
    const int w = moving ? base_w / core_->settings.preview_divisor : base_w;
    const int h = moving ? base_h / core_->settings.preview_divisor : base_h;

    if (core_->generation != gpuAppliedGen_ || w != gpu.width()) {
        gpu.reset(w, h);   // marks a clear; encoded with the next batch
        gpuAppliedGen_ = core_->generation;
        gpuLastPasses_ = 0;
        accumStart_ = now;
    }

    const int target = final_mode ? core_->settings.final_target_spp
                                  : (moving ? INT_MAX
                                            : core_->settings.max_accum_passes);
    // Session E: shared camera/selection params for the raster preview and
    // the selection overlay — the SAME matrices ImGuizmo uses, so both
    // register with the traced image to sub-pixel.
    GpuRenderer::RasterParams rp{};
    mat_look_at(core_->fly.pos, core_->fly.pos + core_->fly.forward(),
                vec3(0.0f, 1.0f, 0.0f), rp.view);
    mat_perspective(core_->settings.vfov_deg, core_->settings.aspect(), 0.1f,
                    500.0f, rp.proj);
    rp.cam_pos[0] = core_->fly.pos.x;
    rp.cam_pos[1] = core_->fly.pos.y;
    rp.cam_pos[2] = core_->fly.pos.z;
    rp.wireframe = core_->fastnav == 2;
    if (!final_mode) {
        if (core_->selection.kind == Selection::Kind::Sphere) {
            rp.overlay_kind = 1;
            rp.overlay_index = core_->selection.index;
        } else if (core_->selection.kind == Selection::Kind::Mesh) {
            rp.overlay_kind = 2;
        }
    }
    const bool nav_active =
        core_->fastnav != 0 && moving && !final_mode && !core_->paused;

    // Encode every tick — with 0 passes once converged or while paused —
    // so the UI overlay stays live and responsive. A 0-pass frame is just
    // resolve + UI: trivial GPU cost. While fast-nav is active and the
    // camera moves, a rasterized frame replaces tracing entirely; movement
    // keeps bumping the generation, so on settle the tracer reconverges
    // from the current pose through the normal reset machinery.
    // Session G budget controller: learn the cost of a pixel-pass from
    // completed command buffers, then size each tick's work to a GPU-time
    // budget. Cheap frames run K full passes exactly as before; expensive
    // frames drop to one pass, then to a slice of rows — so no dispatch
    // monopolizes the GPU and the compositor/UI always get their slot.
    // The schedule changes; the accumulated image does not.
    {
        const float bms = gpu.last_batch_gpu_ms();
        const unsigned long long bpx = gpu.last_batch_px_passes();
        if (bms > 0.05f && bpx > 0) {
            const float ns = bms * 1.0e6f / float(bpx);
            nsPerPxPass_ =
                nsPerPxPass_ > 0.0f ? 0.8f * nsPerPxPass_ + 0.2f * ns : ns;
            gpuMsEma_ = gpuMsEma_ > 0.0f ? 0.8f * gpuMsEma_ + 0.2f * bms : bms;
        }
    }
    if (dispatch_semaphore_wait(gpuInflight_, DISPATCH_TIME_NOW) == 0) {
        dispatch_semaphore_t sem = gpuInflight_;
        const auto ui_enc = [self](void* cb, void* tex) {
            [self encodeUIOn:cb texture:tex];
        };
        if (nav_active) {
            gpu.encode_raster_frame(rp, (__bridge void*)metalLayer_,
                                    [sem] { dispatch_semaphore_signal(sem); },
                                    ui_enc);
        } else {
            GpuRenderer::TraceWork work{};   // passes == 0: present-only
            if (gpu.passes() < target && !core_->paused && !savingPng_) {
                const int kcap =
                    moving ? core_->settings.gpu_passes_per_tick_preview
                           : core_->settings.gpu_passes_per_tick;
                const float budget = final_mode
                                         ? core_->settings.gpu_budget_ms_final
                                         : core_->settings.gpu_budget_ms;
                const int cursor = gpu.partial_row();
                if (nsPerPxPass_ <= 0.0f) {
                    // First batch after launch: a modest probe to seed the
                    // cost model without risking a monster dispatch.
                    work.passes = 1;
                    work.row_start = cursor;
                    work.row_count =
                        std::min(h - cursor, std::max(32, h / 8));
                } else {
                    const double budget_px =
                        double(budget) * 1.0e6 / nsPerPxPass_;
                    const double full = double(w) * double(h);
                    if (cursor > 0) {
                        // Finish the in-progress pass first.
                        work.passes = 1;
                        work.row_start = cursor;
                        work.row_count = std::clamp(
                            int(budget_px / w), 32, h - cursor);
                    } else if (budget_px >= full) {
                        int k = int(std::min<double>(kcap, budget_px / full));
                        if (target != INT_MAX)
                            k = std::min(k, target - gpu.passes());
                        work.passes = std::max(1, k);
                    } else {
                        work.passes = 1;
                        work.row_start = 0;
                        work.row_count =
                            std::clamp(int(budget_px / w), 32, h);
                    }
                }
                lastSliceK_ = work.passes;
                lastSliceRows_ = work.row_count > 0 ? work.row_count : h;
            }
            const Camera cam = core_->fly.make_camera(
                core_->settings.vfov_deg, core_->settings.aspect());
            gpu.encode_frame(to_gpu_camera(cam), work,
                             (__bridge void*)metalLayer_,
                             [sem] { dispatch_semaphore_signal(sem); }, ui_enc,
                             rp.overlay_kind != 0 ? &rp : nullptr);
        }
    }

    // FINAL mode: export once when the target is reached (save_png waits
    // for the GPU, so the image includes every scheduled pass). Gated on
    // !reset_pending(): right after a reset, passes() still reports the
    // pre-reset count until a batch is encoded — without the gate, a
    // semaphore-starved entry tick could export the stale image.
    // Exports are ASYNC (Session G): the wait + readback + PNG encode run
    // on a background queue while the tick keeps presenting 0-pass frames
    // (savingPng_ pauses new trace work so the snapshot is race-free). The
    // main thread never calls waitUntilCompleted anymore.
    if (final_mode && !gpu.reset_pending() && gpu.partial_row() == 0 &&
        gpu.passes() >= core_->settings.final_target_spp &&
        !core_->final_saved.exchange(true)) {
        const std::string name = timestamped_png_name();
        const int sw = gpu.width(), sh = gpu.height(), sp = gpu.passes();
        savingPng_ = true;
        gpu.save_png_async(name, [self, name, sw, sh, sp](bool ok) {
            self->savingPng_ = false;
            if (ok)
                std::printf("FINAL: saved %s (%dx%d @ %d spp)\n",
                            name.c_str(), sw, sh, sp);
            std::fflush(stdout);
        });
    }

    if (core_->save_requested.exchange(false) && !savingPng_) {
        const std::string name = timestamped_png_name();
        const int sw = gpu.width(), sh = gpu.height(), sp = gpu.passes();
        savingPng_ = true;
        gpu.save_png_async(name, [self, name, sw, sh, sp](bool ok) {
            self->savingPng_ = false;
            if (ok)
                std::printf("saved %s (%dx%d @ %d spp)\n", name.c_str(), sw,
                            sh, sp);
            std::fflush(stdout);
        });
    }

    // Smoothed pass rate for the title (scheduled counts; ahead of the GPU
    // by at most 2 batches).
    const int passes = gpu.passes();
    if (passes > gpuLastPasses_ && dt > 0.0f) {
        const float inst = float(passes - gpuLastPasses_) / dt;
        gpuRate_ = gpuRate_ > 0.0f ? 0.9f * gpuRate_ + 0.1f * inst : inst;
    }
    gpuLastPasses_ = passes;
    if (final_mode) {
        window_.title = [NSString
            stringWithFormat:@"pathtracer [GPU] FINAL %d/%d spp%s", passes,
                             core_->settings.final_target_spp,
                             passes >= core_->settings.final_target_spp
                                 ? " — saved (F to unlock)"
                                 : ""];
    } else {
        window_.title = [NSString
            stringWithFormat:@"pathtracer [GPU] — %dx%d @ %d spp (%.0f passes/s)",
                             gpu.width(), gpu.height(), passes, gpuRate_];
    }
}

// Dear ImGui overlay: new-frame / build / render, drawn into the resolved
// drawable (loadAction Load) on the same command buffer, before present.
- (void)encodeUIOn:(void*)cmdBuf texture:(void*)texture {
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cmdBuf;
    id<MTLTexture> tex = (__bridge id<MTLTexture>)texture;

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = tex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    ImGui_ImplMetal_NewFrame(rpd);
    ImGui_ImplOSX_NewFrame(view_);
    // The drawable now matches the Retina backing (the resolve kernel
    // upscales the render), so ImGui keeps its native FramebufferScale —
    // 1.92's density-aware fonts rasterize crisp at 2x.
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
    if (core_->show_ui) [self buildUI];
    [self drawGizmo];   // tied to selection, not to panel visibility
    [self drawStatusBar];
    [self drawHelp];
    ImGui::Render();

    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:rpd];
    ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), cb, enc);
    [enc endEncoding];
}

- (void)buildUI {
    RenderSettings& s = core_->settings;
    GpuRenderer& gpu = *core_->gpu;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
    ImGui::Begin("Render");

    const bool final_mode = core_->final_mode;
    ImGui::Text("Mode: %s", final_mode ? "FINAL (camera locked)" : "Interactive");
    bool final_toggle = final_mode;
    if (ImGui::Checkbox("Final mode (R)", &final_toggle)) {
        set_final_mode(*core_, final_toggle);
    }
    if (final_mode) {
        char overlay[48];
        std::snprintf(overlay, sizeof overlay, "%d / %d spp", gpu.passes(),
                      s.final_target_spp);
        ImGui::ProgressBar(
            std::min(1.0f, float(gpu.passes()) /
                               float(std::max(1, s.final_target_spp))),
            ImVec2(-1.0f, 0.0f), overlay);
    } else {
        const float accum_s = std::chrono::duration<float>(
            Clock::now() - accumStart_).count();
        ImGui::Text("%d spp · %.0f passes/s · %.1fs", gpu.passes(), gpuRate_,
                    accum_s);
    }
    // Session G telemetry: per-CB GPU time vs budget, main-thread tick
    // time, and the shape of the last submitted slice.
    if (gpuMsEma_ > 0.0f) {
        ImGui::Text("GPU %.1f ms/batch · tick %.2f ms · %.1f ns/px·pass",
                    gpuMsEma_, tickMsEma_, nsPerPxPass_);
        if (lastSliceRows_ > 0 && lastSliceRows_ < gpu.height()) {
            ImGui::Text("slicing: %d rows/batch (%d batches/pass)",
                        lastSliceRows_,
                        (gpu.height() + lastSliceRows_ - 1) / lastSliceRows_);
        } else if (lastSliceK_ > 0) {
            ImGui::Text("full frame · %d pass%s/batch", lastSliceK_,
                        lastSliceK_ == 1 ? "" : "es");
        }
    }
    ImGui::Checkbox("Pause accumulation", &core_->paused);
    ImGui::BeginDisabled(core_->undo_stack.empty() && !core_->edit_active);
    if (ImGui::Button("Undo")) perform_undo(*core_);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(core_->redo_stack.empty());
    if (ImGui::Button("Redo")) perform_redo(*core_);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("cmd+Z / cmd+shift+Z");

    // ---- Outliner ----
    ImGui::SeparatorText("Outliner");
    if (ImGui::BeginChild("##outliner", ImVec2(0, 130),
                          ImGuiChildFlags_Borders)) {
        for (int i = 0; i < int(core_->desc.spheres.size()); ++i) {
            const Selection row{Selection::Kind::Sphere, i, 0.0f};
            char label[80];
            std::snprintf(label, sizeof label, "%s##obj%d",
                          display_name(*core_, row).c_str(), i);
            const bool is_sel =
                core_->selection.kind == Selection::Kind::Sphere &&
                core_->selection.index == i;
            if (ImGui::Selectable(label, is_sel)) core_->selection = row;
        }
        if (core_->desc.mesh) {
            const Selection row{Selection::Kind::Mesh, -1, 0.0f};
            const bool is_sel =
                core_->selection.kind == Selection::Kind::Mesh;
            char label[80];
            std::snprintf(label, sizeof label, "%s##objmesh",
                          display_name(*core_, row).c_str());
            if (ImGui::Selectable(label, is_sel)) core_->selection = row;
        }
    }
    ImGui::EndChild();
    ImGui::BeginDisabled(!can_duplicate(core_->selection));
    if (ImGui::Button("Duplicate")) {
        undo_begin(*core_);
        duplicate_selected(*core_);
        undo_commit(*core_, "duplicate");
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(core_->selection.kind == Selection::Kind::None);
    if (ImGui::Button("Delete (backspace)")) {
        undo_begin(*core_);
        delete_selected(*core_);
        undo_commit(*core_, "delete");
    }
    ImGui::EndDisabled();

    // ---- Lights (Session C part 2) ----
    // Lights ARE emissive spheres in this renderer (the sky is a kernel
    // constant), so they are already pickable and gizmo-draggable; this
    // list is a convenience selector, and their color/intensity is the
    // emissive control in the Material section above the gizmo.
    ImGui::SeparatorText("Lights");
    {
        int n_lights = 0;
        for (int i = 0; i < int(core_->desc.spheres.size()); ++i) {
            const SphereData& sd = core_->desc.spheres[i];
            const float strength = std::fmax(
                sd.mat.emission.x,
                std::fmax(sd.mat.emission.y, sd.mat.emission.z));
            if (strength <= 0.0f) continue;
            ++n_lights;
            const Selection row{Selection::Kind::Sphere, i, 0.0f};
            char label[96];
            std::snprintf(label, sizeof label, "%s  (%.1f)##light%d",
                          display_name(*core_, row).c_str(), strength, i);
            const bool is_sel =
                core_->selection.kind == Selection::Kind::Sphere &&
                core_->selection.index == i;
            if (ImGui::Selectable(label, is_sel)) core_->selection = row;
        }
        if (n_lights == 0) {
            ImGui::TextDisabled("none — raise any sphere's emissive strength");
        } else {
            ImGui::TextDisabled("select, then drag with the gizmo; edit "
                                "color/strength under Material");
        }
    }

    // ---- Selection ----
    ImGui::SeparatorText("Selection");
    const Selection& sel = core_->selection;
    if (sel.kind == Selection::Kind::None) {
        ImGui::TextDisabled("None — click an object or press Tab");
    } else {
        // Rename (persists through save/load; renames never reset accum).
        if (lastNameSel_.kind != sel.kind || lastNameSel_.index != sel.index) {
            const std::string& cur = sel.kind == Selection::Kind::Mesh
                ? core_->desc.mesh_name
                : core_->desc.spheres[sel.index].name;
            std::strncpy(nameBuf_, cur.c_str(), sizeof nameBuf_ - 1);
            nameBuf_[sizeof nameBuf_ - 1] = 0;
            lastNameSel_ = sel;
        }
        if (ImGui::InputText("name", nameBuf_, sizeof nameBuf_)) {
            if (sel.kind == Selection::Kind::Mesh)
                core_->desc.mesh_name = nameBuf_;
            else
                core_->desc.spheres[sel.index].name = nameBuf_;
        }
        undo_track(*core_);

        if (sel.kind == Selection::Kind::Mesh) {
            if (core_->desc.mesh)
                ImGui::Text("mesh · %zu triangles",
                            core_->desc.mesh->tris.size());
            // Numeric TRS on the native matrix. Decompose→euler is lossy
            // in representation (e.g. 270° may read back as -90°) but the
            // transform itself stays exact.
            float t[3], r[3], sc[3];
            ImGuizmo::DecomposeMatrixToComponents(core_->mesh_model, t, r, sc);
            bool edited = false;
            edited |= ImGui::DragFloat3("position", t, 0.02f);
            undo_track(*core_);
            edited |= ImGui::DragFloat3("rotation°", r, 0.5f);
            undo_track(*core_);
            edited |= ImGui::DragFloat3("scale", sc, 0.01f, 0.01f, 100.0f);
            undo_track(*core_);
            if (edited) {
                ImGuizmo::RecomposeMatrixFromComponents(t, r, sc,
                                                        core_->mesh_model);
                core_->mesh_apply_pending = true;   // throttled in tickGPU
            }
            if (ImGui::Button("Reset transform")) {
                undo_begin(*core_);
                const float ident[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                                         0, 0, 1, 0, 0, 0, 0, 1};
                std::memcpy(core_->mesh_model, ident, sizeof ident);
                core_->mesh_apply_pending = true;
                undo_commit(*core_, "reset mesh transform");
            }
            ImGui::SeparatorText("Material");
            ImGui::TextDisabled("from glTF textures (editable in a later\n"
                                "session)");
        } else {
            SphereData& sd = core_->desc.spheres[sel.index];
            ImGui::Text("sphere · %s", mat_type_name(sd.mat));
            bool edited = false;
            edited |= ImGui::DragFloat3("position", &sd.center.x, 0.02f);
            undo_track(*core_);
            edited |= ImGui::DragFloat("radius", &sd.radius, 0.01f, 0.01f,
                                       2000.0f);
            undo_track(*core_);
            if (edited) {
                sd.radius = std::fmax(0.01f, sd.radius);
                apply_sphere_edit(*core_);
            }
            const bool has_initial =
                sel.index < int(core_->initial_desc.spheres.size());
            ImGui::BeginDisabled(!has_initial);
            if (ImGui::Button("Reset transform") && has_initial) {
                undo_begin(*core_);
                const SphereData& init =
                    core_->initial_desc.spheres[sel.index];
                sd.center = init.center;
                sd.radius = init.radius;
                apply_sphere_edit(*core_);
                undo_commit(*core_, "reset sphere transform");
            }
            ImGui::EndDisabled();

            // ---- Material (Session C part 1) ----
            // Writes the SAME params the BSDF consumes — no reinterpreting
            // — and routes every change through apply_sphere_edit, i.e. the
            // one central reset. Mouse arbitration comes from the existing
            // WantCaptureMouse guards.
            ImGui::SeparatorText("Material");
            Material& m = sd.mat;
            bool mat_changed = false;
            float bc[3] = {m.base_color.x, m.base_color.y, m.base_color.z};
            if (ImGui::ColorEdit3("base color", bc)) {
                m.base_color = color(bc[0], bc[1], bc[2]);
                mat_changed = true;
            }
            undo_track(*core_);
            mat_changed |=
                ImGui::SliderFloat("metallic", &m.metallic, 0.0f, 1.0f);
            undo_track(*core_);
            mat_changed |=
                ImGui::SliderFloat("roughness", &m.roughness, 0.0f, 1.0f);
            undo_track(*core_);
            bool glass = m.transmission > 0.5f;
            if (ImGui::Checkbox("glass", &glass)) {
                m.transmission = glass ? 1.0f : 0.0f;
                mat_changed = true;
            }
            undo_track(*core_);
            if (glass) {
                ImGui::SameLine();
                ImGui::SetNextItemWidth(110);
                mat_changed |= ImGui::SliderFloat("ior", &m.ior, 1.0f, 2.5f);
                undo_track(*core_);
            }
            // Emission is stored as one HDR color; the UI splits it into
            // normalized color x strength (strength = max component).
            float strength = std::fmax(
                m.emission.x, std::fmax(m.emission.y, m.emission.z));
            float ec[3];
            if (strength > 0.0f) {
                ec[0] = m.emission.x / strength;
                ec[1] = m.emission.y / strength;
                ec[2] = m.emission.z / strength;
            } else {
                ec[0] = ec[1] = ec[2] = 1.0f;
            }
            bool emis_changed = ImGui::ColorEdit3("emissive", ec);
            undo_track(*core_);
            emis_changed |= ImGui::DragFloat("strength", &strength, 0.05f,
                                             0.0f, 1000.0f);
            undo_track(*core_);
            if (emis_changed) {
                strength = std::fmax(0.0f, strength);
                m.emission = color(ec[0], ec[1], ec[2]) * strength;
                mat_changed = true;
            }
            if (mat_changed) apply_sphere_edit(*core_);
        }

        // Gizmo controls.
        ImGui::RadioButton("Move (1)", &core_->gizmo_op, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Rotate (2)", &core_->gizmo_op, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Scale (3)", &core_->gizmo_op, 2);
        if (sel.kind == Selection::Kind::Sphere && core_->gizmo_op == 1) {
            ImGui::TextDisabled("(rotation has no effect on spheres)");
        }
        ImGui::Checkbox("Local space", &core_->gizmo_local);
        ImGui::SameLine();
        ImGui::Checkbox("Snap", &core_->snap_enabled);
        if (core_->snap_enabled) {
            ImGui::SetNextItemWidth(60);
            ImGui::DragFloat("grid", &core_->snap_translate, 0.05f, 0.01f,
                             10.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            ImGui::DragFloat("angle°", &core_->snap_rotate_deg, 1.0f, 1.0f,
                             90.0f);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(60);
            ImGui::DragFloat("step", &core_->snap_scale, 0.01f, 0.01f, 1.0f);
        }
    }

    // ---- Camera ----
    ImGui::SeparatorText("Camera");
    ImGui::SliderFloat("speed", &core_->fly.move_speed, 0.05f, 50.0f, "%.2f",
                       ImGuiSliderFlags_Logarithmic);
    if (ImGui::SliderFloat("fov°", &s.vfov_deg, 20.0f, 90.0f, "%.0f")) {
        core_->mark_scene_dirty();
    }
    if (ImGui::Button("Reset camera")) {
        core_->fly = core_->initial_fly;
        s.vfov_deg = core_->initial_vfov;
        core_->mark_scene_dirty(/*camera_moved=*/true);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("F frames selection · shift sprints");

    ImGui::SeparatorText("Render");
    ImGui::Text("Fast nav (V):");
    ImGui::SameLine();
    ImGui::RadioButton("Off", &core_->fastnav, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Solid", &core_->fastnav, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Wire", &core_->fastnav, 2);
    if (core_->fastnav != 0) {
        ImGui::TextDisabled("rasterized while moving; traces on settle");
    }
    if (ImGui::SliderFloat("resolution", &core_->interactive_scale, 0.25f,
                           4.0f, "%.2fx")) {
        core_->interactive_scale =
            std::clamp(core_->interactive_scale, 0.25f, 4.0f);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "%dx%d at this scale. 2.00x = native Retina pixels; above that\n"
            "renders larger than the display (shown downsampled) — FINAL\n"
            "and PNG exports keep the full size. 4x of a %dx%d window = 4K.",
            int(s.width * core_->interactive_scale) & ~1,
            int(s.height * core_->interactive_scale) & ~1, 960, 540);
    }
    // Every change goes through the ONE central reset hook — same pattern
    // future scene edits will use. No other accumulator-clear path exists.
    bool dirty = false;
    dirty |= ImGui::SliderInt("passes / tick", &s.gpu_passes_per_tick, 1, 32);
    ImGui::SliderFloat("GPU budget", &s.gpu_budget_ms, 4.0f, 33.0f,
                       "%.0f ms");   // scheduling only — never resets accum
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Interactivity vs throughput: max GPU time per\n"
                          "trace batch. Lower = smoother UI, higher = faster\n"
                          "convergence. FINAL mode uses %.0f ms.",
                          s.gpu_budget_ms_final);
    }
    dirty |= ImGui::SliderInt("final target spp", &s.final_target_spp, 64,
                              16384, "%d", ImGuiSliderFlags_Logarithmic);
    if (ImGui::SliderInt("max bounces", &s.max_depth, 1, 32)) {
        gpu.set_max_depth(s.max_depth);   // renderer holds its own copy
        dirty = true;
    }
    if (ImGui::SliderFloat("clamp indirect", &s.clamp_indirect, 0.0f, 100.0f,
                           s.clamp_indirect <= 0.0f ? "off" : "%.1f",
                           ImGuiSliderFlags_Logarithmic)) {
        gpu.set_clamp_indirect(s.clamp_indirect);
        dirty = true;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Firefly control: caps indirect bounce spikes\n"
                          "(HDRI suns). Direct light is never clamped.\n"
                          "0 = off (unbiased).");
    }
    if (dirty) {
        core_->final_saved = false;   // settings changed: re-arm the export
        core_->mark_scene_dirty();
    }

    // ---- Environment (Session F) ----
    ImGui::SeparatorText("Environment");
    ImGui::SetNextItemWidth(-60.0f);
    ImGui::InputText("hdr", envPath_, sizeof envPath_);
    if (ImGui::Button("Load HDRI")) {
        undo_begin(*core_);
        std::string err, resolved;
        // Reuse the launch-dir probing convention.
        namespace fs = std::filesystem;
        std::error_code ec;
        resolved = envPath_;
        if (fs::path(resolved).is_relative() && !fs::exists(resolved, ec)) {
            for (const char* prefix : {"../", "build/"}) {
                if (fs::exists(fs::path(prefix) / envPath_, ec)) {
                    resolved = (fs::path(prefix) / envPath_).string();
                    break;
                }
            }
        }
        auto env = load_hdr(resolved, err);
        if (env) {
            core_->desc.env = env;
            core_->desc.env_source_path = resolved;
            apply_env(*core_, true);
            undo_commit(*core_, "load HDRI");
            envStatus_ = "loaded " + resolved;
        } else {
            core_->edit_active = false;   // nothing changed
            envStatus_ = err;
        }
        std::printf("%s\n", envStatus_.c_str());
        std::fflush(stdout);
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!core_->desc.env);
    if (ImGui::Button("Clear")) {
        undo_begin(*core_);
        core_->desc.env.reset();
        core_->desc.env_source_path.clear();
        apply_env(*core_, true);
        undo_commit(*core_, "clear HDRI");
        envStatus_ = "gradient dome";
    }
    ImGui::EndDisabled();
    if (core_->desc.env) {
        bool env_edited = false;
        env_edited |= ImGui::SliderFloat(
            "intensity", &core_->desc.env_intensity, 0.05f, 8.0f, "%.2f",
            ImGuiSliderFlags_Logarithmic);
        undo_track(*core_);
        env_edited |= ImGui::SliderFloat("env yaw", &core_->desc.env_yaw_deg,
                                         -180.0f, 180.0f, "%.0f deg");
        undo_track(*core_);
        if (env_edited) apply_env(*core_, false);
    }
    if (!envStatus_.empty()) ImGui::TextWrapped("%s", envStatus_.c_str());

    ImGui::SeparatorText("Scene");
    ImGui::SetNextItemWidth(-60.0f);
    ImGui::InputText("file", scenePath_, sizeof scenePath_);
    if (ImGui::Button("Save")) {
        std::string err;
        if (save_scene(scenePath_, snapshot_from_core(*core_),
                       core_->desc.mesh_source_path,
                       core_->desc.env_source_path, err)) {
            std::error_code ec;
            sceneStatus_ = "saved " +
                std::filesystem::absolute(scenePath_, ec).string();
        } else {
            sceneStatus_ = err;
        }
        std::printf("%s\n", sceneStatus_.c_str());
        std::fflush(stdout);
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        std::string err, resolved;
        SceneSnapshot snap;
        undo_begin(*core_);
        if (load_scene(scenePath_, snap, err, &resolved) &&
            apply_snapshot(*core_, std::move(snap), err)) {
            undo_commit(*core_, "load scene");
            sceneStatus_ = "loaded " + resolved;
        } else {
            sceneStatus_ = err;
        }
        std::printf("%s\n", sceneStatus_.c_str());
        std::fflush(stdout);
    }
    if (!sceneStatus_.empty()) {
        ImGui::TextWrapped("%s", sceneStatus_.c_str());
    }

    ImGui::Separator();
    const float fps = ImGui::GetIO().Framerate;
    ImGui::Text("%.1f ms/frame (%.0f FPS)", fps > 0.0f ? 1000.0f / fps : 0.0f,
                fps);
    ImGui::TextDisabled("U hides this panel · ? shows shortcuts");
    ImGui::End();
}

- (void)drawStatusBar {
    const ImGuiIO& io = ImGui::GetIO();
    const float h = 24.0f;
    ImGui::SetNextWindowPos(ImVec2(0, io.DisplaySize.y - h));
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, h));
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::Begin("##statusbar", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoNav);
    ImGui::Text("%s%s  |  %.0f fps  |  %d spp%s  |  %s",
                core_->final_mode ? "FINAL" : "Interactive",
                core_->fastnav != 0 ? " [nav]" : "", io.Framerate,
                core_->gpu ? core_->gpu->passes() : 0,
                core_->paused ? " (paused)" : "",
                display_name(*core_, core_->selection).c_str());
    ImGui::End();
}

- (void)drawHelp {
    if (!core_->show_help) return;
    const ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(
        ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f),
        ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::Begin("Shortcuts", &core_->show_help,
                 ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoCollapse);
    static const char* rows[][2] = {
        {"W A S D / Q E", "move camera (hold Shift to sprint)"},
        {"left-drag", "look around"},
        {"scroll", "adjust move speed"},
        {"click", "select object (sky deselects)"},
        {"Tab / Shift+Tab", "cycle selection"},
        {"F", "frame selected object"},
        {"1 / 2 / 3", "gizmo: move / rotate / scale"},
        {"Backspace", "delete selected"},
        {"R", "toggle FINAL render mode"},
        {"P", "save PNG now"},
        {"U", "show/hide the panel"},
        {"? or F1", "this overlay"},
        {"Esc", "close overlay / quit"},
    };
    if (ImGui::BeginTable("##keys", 2)) {
        for (const auto& row : rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(row[0]);
            ImGui::TableSetColumnIndex(1);
            ImGui::TextDisabled("%s", row[1]);
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

// Phase 4: the transform gizmo. Drawn only with a live selection and not
// in FINAL mode (converging runs stay undisturbed). Every applied edit
// funnels through mark_scene_dirty() via the apply_* helpers.
- (void)drawGizmo {
    if (core_->final_mode) return;
    Selection& sel = core_->selection;
    if (sel.kind == Selection::Kind::None) return;

    ImGuiIO& io = ImGui::GetIO();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

    float view[16], proj[16];
    mat_look_at(core_->fly.pos, core_->fly.pos + core_->fly.forward(),
                vec3(0.0f, 1.0f, 0.0f), view);
    mat_perspective(core_->settings.vfov_deg, core_->settings.aspect(), 0.1f,
                    500.0f, proj);

    const ImGuizmo::OPERATION ops[3] = {ImGuizmo::TRANSLATE, ImGuizmo::ROTATE,
                                        ImGuizmo::SCALE};
    const ImGuizmo::OPERATION op = ops[core_->gizmo_op];
    const ImGuizmo::MODE mode =
        core_->gizmo_local ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
    // Per-op snap increment (translate uses all three lanes).
    const float inc = core_->gizmo_op == 0 ? core_->snap_translate
                    : core_->gizmo_op == 1 ? core_->snap_rotate_deg
                                           : core_->snap_scale;
    const float snap_vals[3] = {inc, inc, inc};
    const float* snap = core_->snap_enabled ? snap_vals : nullptr;

    if (sel.kind == Selection::Kind::Sphere) {
        SphereData& sd = core_->desc.spheres[sel.index];
        float t[3] = {sd.center.x, sd.center.y, sd.center.z};
        float r[3] = {0, 0, 0};
        float s[3] = {sd.radius, sd.radius, sd.radius};
        float model[16];
        ImGuizmo::RecomposeMatrixFromComponents(t, r, s, model);
        if (ImGuizmo::Manipulate(view, proj, op, mode, model, nullptr, snap)) {
            undo_begin(*core_);
            ImGuizmo::DecomposeMatrixToComponents(model, t, r, s);
            sd.center = point3(t[0], t[1], t[2]);
            // Spheres scale uniformly (mean of the axis handles); rotation
            // is a visual no-op on an analytic sphere.
            sd.radius = std::fmax(0.01f, (s[0] + s[1] + s[2]) / 3.0f);
            apply_sphere_edit(*core_);   // 4.6KB buffer swap — every frame is fine
        }
    } else {   // mesh — the throttled apply lives in tickGPU
        if (ImGuizmo::Manipulate(view, proj, op, mode, core_->mesh_model,
                                 nullptr, snap)) {
            undo_begin(*core_);
            core_->mesh_apply_pending = true;
        }
    }
}

// Window size drives the base render resolution (GPU backend). The
// drawable always matches the Retina backing (crisp UI); the render dims
// feed the same reset machinery as every other change, and the camera
// aspect, gizmo projection, and pick mapping all derive from them.
- (void)updateRenderSize {
    if (!core_->use_gpu || !metalLayer_) return;
    const NSSize sz = view_.bounds.size;
    const CGFloat bs = window_.backingScaleFactor;
    metalLayer_.drawableSize =
        CGSizeMake(std::fmax(sz.width * bs, 64.0),
                   std::fmax(sz.height * bs, 36.0));
    const int w = std::max(64, int(sz.width)) & ~1;
    const int h = std::max(36, int(sz.height)) & ~1;
    if (w != core_->settings.width || h != core_->settings.height) {
        core_->settings.width = w;
        core_->settings.height = h;
        core_->mark_scene_dirty();
    }
}

- (void)windowDidResize:(NSNotification*)note {
    [self updateRenderSize];
}

- (void)windowDidChangeBackingProperties:(NSNotification*)note {
    [self updateRenderSize];   // e.g. dragged to a non-Retina display
}

- (void)onUndo:(id)sender {
    perform_undo(*core_);
}
- (void)onRedo:(id)sender {
    perform_redo(*core_);
}

- (void)windowDidResignKey:(NSNotification*)note {
    // Cmd-Tab while holding W must not leave the camera flying forever.
    core_->input = InputState{};
    [view_ unhideCursor];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication*)app {
    return YES;
}

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication*)app {
    return YES;
}

- (void)applicationWillTerminate:(NSNotification*)note {
    [timer_ invalidate];
    core_->quit = true;
    core_->camera_cv.notify_all();
    if (core_->render_thread.joinable()) core_->render_thread.join();
    if (core_->gpu) core_->gpu->wait_idle();   // no in-flight UI draws
    if (imguiInited_) {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplOSX_Shutdown();
        ImGui::DestroyContext();
    }
}

@end

// ---------------------------------------------------------------------------

int run_viewer(const RenderSettings& settings, bool use_gpu,
               const SceneDesc& desc) {
    @autoreleasepool {
        auto core = std::make_unique<ViewerCore>();
        core->settings = settings;
        core->desc = desc;   // retained for CPU-side picking
        if (desc.mesh) {
            // Editable copies: object-space baseline + live world arrays.
            core->mesh_object_tris = desc.mesh->tris;
            core->mesh_tris = desc.mesh->tris;
            core->mesh_nodes = desc.mesh->nodes;
        }
        core->use_gpu = use_gpu;
        if (use_gpu) {
            std::string err;
            core->gpu = GpuRenderer::create(settings, flatten_scene(desc),
                                            desc.mesh.get(), err);
            if (core->gpu && desc.env) {
                core->gpu->set_env(desc.env.get());
                core->gpu->set_env_params(desc.env_intensity,
                                          desc.env_yaw_deg / 360.0f);
            }
            if (!core->gpu) {
                std::fprintf(stderr,
                             "GPU init failed (%s) — using CPU backend\n",
                             err.c_str());
                core->use_gpu = false;
            }
        }
        if (!core->use_gpu) core->scene = make_scene(desc);
        core->fly = FlyCamera::from_look_at(settings.cam_pos, settings.cam_look_at);
        core->camera_snapshot = core->fly;
        core->initial_desc = core->desc;   // per-object + camera reset baselines
        core->initial_fly = core->fly;
        core->initial_vfov = settings.vfov_deg;
        core->idle_stash = capture_state(*core);

        NSApplication* app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];

        // A bare CLI binary has no menu, and without one Cmd-Q doesn't exist.
        NSMenu* menubar = [[NSMenu alloc] init];
        NSMenuItem* appItem = [[NSMenuItem alloc] init];
        [menubar addItem:appItem];
        NSMenu* appMenu = [[NSMenu alloc] init];
        [appMenu addItemWithTitle:@"Quit pathtracer"
                           action:@selector(terminate:)
                    keyEquivalent:@"q"];
        appItem.submenu = appMenu;
        app.mainMenu = menubar;

        PTAppDelegate* delegate = [[PTAppDelegate alloc] initWithCore:core.get()];
        app.delegate = delegate;

        // Edit menu: native Cmd+Z / Cmd+Shift+Z routing for undo/redo.
        NSMenuItem* editItem = [[NSMenuItem alloc] init];
        [menubar addItem:editItem];
        NSMenu* editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
        NSMenuItem* undoItem = [editMenu addItemWithTitle:@"Undo"
                                                   action:@selector(onUndo:)
                                            keyEquivalent:@"z"];
        undoItem.target = delegate;
        NSMenuItem* redoItem = [editMenu addItemWithTitle:@"Redo"
                                                   action:@selector(onRedo:)
                                            keyEquivalent:@"Z"];
        redoItem.keyEquivalentModifierMask =
            NSEventModifierFlagCommand | NSEventModifierFlagShift;
        redoItem.target = delegate;
        editItem.submenu = editMenu;

        std::printf("backend: %s\n", core->use_gpu ? "GPU (Metal compute)"
                                                   : "CPU (reference)");
        std::printf("controls: WASD move · drag look · click select · "
                    "F frame · Tab cycle · R final mode (%d spp + PNG) · "
                    "P save PNG · U panel · ? shortcuts · Esc/Cmd-Q quit\n"
                    "NOTE: FINAL mode moved from F to R; F now frames the "
                    "selection.\n",
                    settings.final_target_spp);
        [app run];   // terminate: exits the process; cleanup runs in
                     // applicationWillTerminate, while `core` is still alive
    }
    return 0;
}
