#include "scene_io.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "json.hpp"

using nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr int kSceneVersion = 1;

json vec_to_json(const vec3& v) { return json::array({v.x, v.y, v.z}); }

bool json_to_vec(const json& j, vec3& out) {
    if (!j.is_array() || j.size() != 3 || !j[0].is_number()) return false;
    out = vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
    return true;
}

json material_to_json(const Material& m) {
    // Session B schema: metallic-roughness parameters, written directly.
    return json{{"baseColor", vec_to_json(m.base_color)},
                {"emission", vec_to_json(m.emission)},
                {"metallic", m.metallic},
                {"roughness", m.roughness},
                {"ior", m.ior},
                {"transmission", m.transmission}};
}

bool material_from_json(const json& j, Material& m, std::string& error) {
    if (!j.is_object()) {
        error = "material is not an object";
        return false;
    }
    // Legacy schema (pre-GGX files, e.g. saved before Session B): a type
    // tag plus albedo/fuzz. Convert to metallic-roughness equivalents.
    if (j.contains("type")) {
        const std::string type = j["type"].get<std::string>();
        color albedo(0.5f, 0.5f, 0.5f);
        if (j.contains("albedo")) json_to_vec(j["albedo"], albedo);
        const float fuzz = j.value("fuzz", 0.0f);
        const float ior = j.value("ior", 1.5f);
        if (type == "lambertian") {
            m = Material::lambertian(albedo);
        } else if (type == "metal") {
            // old fuzz ~ alpha; roughness is perceptual (alpha = r^2)
            m = Material::metal(albedo, std::sqrt(std::fmax(0.0f, fuzz)));
        } else if (type == "glass") {
            m = Material::dielectric(ior);
        } else if (type == "emissive") {
            color e(0.0f, 0.0f, 0.0f);
            if (j.contains("emission")) json_to_vec(j["emission"], e);
            m = Material::emissive(e);
        } else {
            error = "unknown material type '" + type + "'";
            return false;
        }
        if (j.contains("emission")) json_to_vec(j["emission"], m.emission);
        return true;
    }
    // Current schema.
    if (!j.contains("baseColor") || !json_to_vec(j["baseColor"], m.base_color)) {
        error = "material missing baseColor";
        return false;
    }
    if (j.contains("emission")) json_to_vec(j["emission"], m.emission);
    m.metallic = std::clamp(j.value("metallic", 0.0f), 0.0f, 1.0f);
    m.roughness = std::clamp(j.value("roughness", 1.0f), 0.0f, 1.0f);
    m.ior = j.value("ior", 1.5f);
    m.transmission = std::clamp(j.value("transmission", 0.0f), 0.0f, 1.0f);
    return true;
}

} // namespace

bool save_scene(const std::string& path, const SceneSnapshot& snap,
                const std::string& mesh_source_abs, std::string& error) {
    json objects = json::array();
    for (const SphereData& s : snap.spheres) {
        json o{{"type", "sphere"},
               {"center", vec_to_json(s.center)},
               {"radius", s.radius},
               {"material", material_to_json(s.mat)}};
        if (!s.name.empty()) o["name"] = s.name;
        objects.push_back(std::move(o));
    }
    if (!mesh_source_abs.empty()) {
        // Portable scenes: store the mesh path relative to the scene file.
        std::string source = mesh_source_abs;
        std::error_code ec;
        const fs::path scene_dir =
            fs::absolute(fs::path(path), ec).parent_path();
        const fs::path rel =
            fs::relative(fs::absolute(mesh_source_abs, ec), scene_dir, ec);
        if (!ec && !rel.empty()) source = rel.string();

        json model = json::array();
        for (int i = 0; i < 16; ++i) model.push_back(snap.mesh_model[i]);
        json o{{"type", "mesh"}, {"source", source}, {"model", model}};
        if (!snap.mesh_name.empty()) o["name"] = snap.mesh_name;
        objects.push_back(std::move(o));
    }

    const json doc{
        {"version", kSceneVersion},
        {"camera",
         {{"position", vec_to_json(snap.cam_pos)},
          {"yaw", snap.cam_yaw},
          {"pitch", snap.cam_pitch},
          {"move_speed", snap.move_speed},
          {"vfov_deg", snap.vfov_deg}}},
        {"render",
         {{"final_target_spp", snap.final_target_spp},
          {"max_depth", snap.max_depth},
          {"gpu_passes_per_tick", snap.gpu_passes_per_tick}}},
        {"objects", objects}};

    std::ofstream out(path);
    if (!out) {
        error = "cannot write " + path;
        return false;
    }
    out << doc.dump(2) << "\n";
    if (!out) {
        error = "write failed for " + path;
        return false;
    }
    return true;
}

bool load_scene(const std::string& path, SceneSnapshot& snap,
                std::string& error, std::string* resolved_path) {
    // Sessions get launched from the project root or from build/
    // interchangeably, so a bare filename saved in one session may sit one
    // directory over in the next. Probe the common spots before failing.
    std::error_code ec;
    fs::path resolved = path;
    if (fs::path(path).is_relative() && !fs::exists(resolved, ec)) {
        for (const char* prefix : {"build/", "../"}) {
            const fs::path candidate = fs::path(prefix) / path;
            if (fs::exists(candidate, ec)) {
                resolved = candidate;
                break;
            }
        }
    }

    std::ifstream in(resolved);
    if (!in) {
        error = "cannot open " + fs::absolute(fs::path(path), ec).string();
        if (fs::path(path).is_relative())
            error += " (also probed build/ and ../)";
        return false;
    }
    if (resolved_path)
        *resolved_path = fs::absolute(resolved, ec).string();
    const json doc = json::parse(in, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded() || !doc.is_object()) {
        error = "not valid JSON: " + path;
        return false;
    }
    if (doc.value("version", 0) != kSceneVersion) {
        error = "unsupported scene version";
        return false;
    }

    SceneSnapshot out;   // fill a fresh one; commit only on full success

    if (const auto cam = doc.find("camera"); cam != doc.end() && cam->is_object()) {
        if (cam->contains("position")) json_to_vec((*cam)["position"], out.cam_pos);
        out.cam_yaw = cam->value("yaw", 0.0f);
        out.cam_pitch = cam->value("pitch", 0.0f);
        out.move_speed = cam->value("move_speed", 2.0f);
        out.vfov_deg = cam->value("vfov_deg", 35.0f);
    } else {
        error = "missing camera";
        return false;
    }

    if (const auto r = doc.find("render"); r != doc.end() && r->is_object()) {
        out.final_target_spp = r->value("final_target_spp", 2048);
        out.max_depth = r->value("max_depth", 16);
        out.gpu_passes_per_tick = r->value("gpu_passes_per_tick", 8);
    }

    const auto objs = doc.find("objects");
    if (objs == doc.end() || !objs->is_array()) {
        error = "missing objects array";
        return false;
    }
    for (const json& o : *objs) {
        const std::string type = o.value("type", "");
        if (type == "sphere") {
            SphereData s{};
            if (!o.contains("center") || !json_to_vec(o["center"], s.center)) {
                error = "sphere missing center";
                return false;
            }
            s.radius = o.value("radius", 0.0f);
            if (s.radius <= 0.0f) {
                error = "sphere with non-positive radius";
                return false;
            }
            if (!o.contains("material") ||
                !material_from_json(o["material"], s.mat, error)) {
                return false;
            }
            s.name = o.value("name", "");
            out.spheres.push_back(s);
        } else if (type == "mesh") {
            if (!out.mesh_source.empty()) {
                error = "multiple meshes not supported";
                return false;
            }
            const std::string stored = o.value("source", "");
            if (stored.empty()) {
                error = "mesh missing source";
                return false;
            }
            out.mesh_name = o.value("name", "");
            const auto model = o.find("model");
            if (model == o.end() || !model->is_array() || model->size() != 16) {
                error = "mesh model must be 16 floats";
                return false;
            }
            for (int i = 0; i < 16; ++i) {
                if (!(*model)[i].is_number()) {
                    error = "mesh model must be 16 floats";
                    return false;
                }
                out.mesh_model[i] = (*model)[i].get<float>();
            }
            // Resolve: scene-file-relative, then cwd, then as-given.
            const fs::path scene_dir =
                fs::absolute(resolved, ec).parent_path();
            if (fs::exists(scene_dir / stored, ec)) {
                out.mesh_source = (scene_dir / stored).string();
            } else if (fs::exists(stored, ec)) {
                out.mesh_source = stored;
            } else {
                error = "mesh source not found: " + stored;
                return false;
            }
        } else {
            error = "unknown object type '" + type + "'";
            return false;
        }
    }
    if (out.spheres.empty() && out.mesh_source.empty()) {
        error = "scene has no objects";
        return false;
    }

    snap = std::move(out);
    return true;
}
