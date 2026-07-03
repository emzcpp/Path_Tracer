#pragma once

// Phase 5: scene save/load. A SceneSnapshot is the full editable state —
// plain data, no UI or renderer types, so round-trips are testable
// headlessly. Geometry is referenced by identity, never serialized:
// spheres as primitive parameters, the mesh as a scene-file-relative path
// plus its Phase 4 model matrix (the loader re-imports through the normal
// load_glb + build_bvh path).

#include <string>
#include <vector>

#include "scene_setup.h"

struct SceneSnapshot {
    std::vector<SphereData> spheres;

    // Mesh reference; source empty = no mesh in the scene.
    std::string mesh_source;         // as stored: relative to the scene file
    std::string mesh_name;
    float mesh_model[16] = {1, 0, 0, 0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0, 0, 0, 1};

    // Camera pose.
    vec3 cam_pos{0.0f, 0.0f, 0.0f};
    float cam_yaw = 0.0f, cam_pitch = 0.0f;
    float move_speed = 2.0f;
    float vfov_deg = 35.0f;

    // Render settings worth restoring.
    int final_target_spp = 2048;
    int max_depth = 16;
    int gpu_passes_per_tick = 8;
};

// Write the snapshot as pretty JSON. The mesh source is stored relative to
// the scene file's directory when possible (portable scenes).
// `mesh_source_abs` is the current session's mesh path (may be relative to
// cwd); empty = no mesh. Returns false + `error` on I/O failure.
bool save_scene(const std::string& path, const SceneSnapshot& snap,
                const std::string& mesh_source_abs, std::string& error);

// Parse and validate a scene file. Relative paths are probed in the
// common launch locations (as given, then build/, then ../) since the app
// is started from the project root or build/ interchangeably; on success
// `resolved_path` (if non-null) receives the absolute file actually read,
// and `snap.mesh_source` holds a resolved mesh path. Fails gracefully:
// false + a clear message (with absolute paths) on missing/malformed files.
bool load_scene(const std::string& path, SceneSnapshot& snap,
                std::string& error, std::string* resolved_path = nullptr);
