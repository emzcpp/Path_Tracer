#include "gltf_loader.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "vec3.h"

// JPEG + PNG decode, from memory or file — never stdio for embedded data.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_HDR
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include "stb_image.h"
#pragma clang diagnostic pop

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

namespace {

// Column-major 4x4 (cgltf convention) applied to points/vectors.
vec3 xform_point(const float* m, const vec3& p) {
    return {m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
            m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
            m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
}
vec3 xform_dir(const float* m, const vec3& p) {
    return {m[0] * p.x + m[4] * p.y + m[8] * p.z,
            m[1] * p.x + m[5] * p.y + m[9] * p.z,
            m[2] * p.x + m[6] * p.y + m[10] * p.z};
}

// Upper-3x3 inverse transpose (correct normal transform under non-uniform
// scale in the node hierarchy).
void normal_matrix(const float* m, float out[9]) {
    const float a = m[0], b = m[4], c = m[8];
    const float d = m[1], e = m[5], f = m[9];
    const float g = m[2], h = m[6], i = m[10];
    const float det =
        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const float k = det != 0.0f ? 1.0f / det : 0.0f;
    out[0] = (e * i - f * h) * k; out[1] = (c * h - b * i) * k; out[2] = (b * f - c * e) * k;
    out[3] = (f * g - d * i) * k; out[4] = (a * i - c * g) * k; out[5] = (c * d - a * f) * k;
    out[6] = (d * h - e * g) * k; out[7] = (b * g - a * h) * k; out[8] = (a * e - b * d) * k;
}
vec3 apply3(const float n[9], const vec3& v) {
    return {n[0] * v.x + n[3] * v.y + n[6] * v.z,
            n[1] * v.x + n[4] * v.y + n[7] * v.z,
            n[2] * v.x + n[5] * v.y + n[8] * v.z};
}

// Assembled asset-space geometry, before our placement transform.
struct Assembled {
    std::vector<vec3> pos, nrm, tan;
    std::vector<float> tan_w;
    std::vector<float> uv;               // 2 per vertex
    std::vector<std::uint32_t> idx;
    bool any_tangents = false;
};

bool read_attr(const cgltf_attribute& attr, int comps,
               std::vector<float>& out) {
    const cgltf_size n = attr.data->count * comps;
    const std::size_t base = out.size();
    out.resize(base + n);
    return cgltf_accessor_unpack_floats(attr.data, out.data() + base, n) == n;
}

void append_primitive(const cgltf_primitive& prim, const float* world,
                      Assembled& out) {
    if (prim.type != cgltf_primitive_type_triangles || !prim.indices) return;

    const cgltf_attribute* a_pos = nullptr;
    const cgltf_attribute* a_nrm = nullptr;
    const cgltf_attribute* a_uv = nullptr;
    const cgltf_attribute* a_tan = nullptr;
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
        const cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) a_pos = &a;
        else if (a.type == cgltf_attribute_type_normal) a_nrm = &a;
        else if (a.type == cgltf_attribute_type_texcoord && a.index == 0) a_uv = &a;
        else if (a.type == cgltf_attribute_type_tangent) a_tan = &a;
    }
    if (!a_pos || !a_nrm || !a_uv) return;   // need full shading data

    std::vector<float> pos, nrm, uv, tan;
    if (!read_attr(*a_pos, 3, pos) || !read_attr(*a_nrm, 3, nrm) ||
        !read_attr(*a_uv, 2, uv))
        return;
    const bool has_tan = a_tan && read_attr(*a_tan, 4, tan);

    float nmat[9];
    normal_matrix(world, nmat);

    const std::uint32_t vbase = std::uint32_t(out.pos.size());
    const cgltf_size nv = a_pos->data->count;
    for (cgltf_size i = 0; i < nv; ++i) {
        out.pos.push_back(
            xform_point(world, vec3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])));
        out.nrm.push_back(normalize(
            apply3(nmat, vec3(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]))));
        out.uv.push_back(uv[i * 2]);
        out.uv.push_back(uv[i * 2 + 1]);
        if (has_tan) {
            const vec3 t = xform_dir(
                world, vec3(tan[i * 4], tan[i * 4 + 1], tan[i * 4 + 2]));
            const float len = t.length();
            out.tan.push_back(len > 1e-8f ? t / len : vec3(1, 0, 0));
            out.tan_w.push_back(tan[i * 4 + 3]);
        } else {
            out.tan.push_back(vec3(0, 0, 0));   // computed later
            out.tan_w.push_back(0.0f);
        }
    }
    if (has_tan) out.any_tangents = true;

    for (cgltf_size i = 0; i < prim.indices->count; ++i) {
        out.idx.push_back(
            vbase + std::uint32_t(cgltf_accessor_read_index(prim.indices, i)));
    }
}

void walk_node(const cgltf_node* node, Assembled& out) {
    float world[16];
    cgltf_node_transform_world(node, world);
    if (node->mesh) {
        for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p) {
            append_primitive(node->mesh->primitives[p], world, out);
        }
    }
    for (cgltf_size c = 0; c < node->children_count; ++c) {
        walk_node(node->children[c], out);
    }
}

// Lengyel-style per-vertex tangents from UVs, for meshes without TANGENT
// data (DamagedHelmet ships none). Accumulate face tangents/bitangents at
// shared vertices, then Gram-Schmidt against the vertex normal and derive
// the handedness sign the same way glTF's .w encodes it.
void compute_tangents(Assembled& a) {
    std::vector<vec3> tacc(a.pos.size(), vec3(0, 0, 0));
    std::vector<vec3> bacc(a.pos.size(), vec3(0, 0, 0));
    for (std::size_t t = 0; t + 2 < a.idx.size(); t += 3) {
        const std::uint32_t i0 = a.idx[t], i1 = a.idx[t + 1], i2 = a.idx[t + 2];
        const vec3 dp1 = a.pos[i1] - a.pos[i0];
        const vec3 dp2 = a.pos[i2] - a.pos[i0];
        const float du1 = a.uv[i1 * 2] - a.uv[i0 * 2];
        const float dv1 = a.uv[i1 * 2 + 1] - a.uv[i0 * 2 + 1];
        const float du2 = a.uv[i2 * 2] - a.uv[i0 * 2];
        const float dv2 = a.uv[i2 * 2 + 1] - a.uv[i0 * 2 + 1];
        const float denom = du1 * dv2 - du2 * dv1;
        if (std::fabs(denom) < 1e-12f) continue;
        const float r = 1.0f / denom;
        const vec3 T = (dp1 * dv2 - dp2 * dv1) * r;
        const vec3 B = (dp2 * du1 - dp1 * du2) * r;
        for (std::uint32_t i : {i0, i1, i2}) {
            tacc[i] += T;
            bacc[i] += B;
        }
    }
    for (std::size_t i = 0; i < a.pos.size(); ++i) {
        const vec3& n = a.nrm[i];
        vec3 t = tacc[i] - n * dot(n, tacc[i]);
        const float len = t.length();
        if (len > 1e-8f) {
            t = t / len;
        } else {
            // Degenerate UVs: any tangent perpendicular to n.
            t = std::fabs(n.x) > 0.9f ? normalize(cross(n, vec3(0, 1, 0)))
                                      : normalize(cross(n, vec3(1, 0, 0)));
        }
        a.tan[i] = t;
        a.tan_w[i] = dot(cross(n, t), bacc[i]) < 0.0f ? -1.0f : 1.0f;
    }
}

Texture16 decode_image(const cgltf_image* image, const std::string& glb_dir,
                       bool srgb, const float factor[3]) {
    Texture16 empty;
    if (!image) return empty;

    int w = 0, h = 0, n = 0;
    unsigned char* rgb = nullptr;
    if (image->buffer_view && image->buffer_view->buffer->data) {
        const auto* bytes =
            static_cast<const unsigned char*>(image->buffer_view->buffer->data) +
            image->buffer_view->offset;
        rgb = stbi_load_from_memory(bytes, int(image->buffer_view->size), &w,
                                    &h, &n, 3);
    } else if (image->uri && std::strncmp(image->uri, "data:", 5) != 0) {
        const std::string p = glb_dir + "/" + image->uri;
        rgb = stbi_load(p.c_str(), &w, &h, &n, 3);
    }
    if (!rgb) return empty;
    Texture16 t = texture_from_rgb8(rgb, w, h, srgb);
    stbi_image_free(rgb);

    if (factor[0] != 1.0f || factor[1] != 1.0f || factor[2] != 1.0f) {
        for (std::size_t i = 0, count = std::size_t(w) * h; i < count; ++i)
            for (int c = 0; c < 3; ++c)
                t.texels[i * 4 + c] = std::uint16_t(
                    std::lrintf(t.texels[i * 4 + c] * factor[c]));
    }
    return t;
}

// Factor-only material channel: a 1x1 texture so the shading path stays
// uniform whether or not the asset ships an image.
Texture16 solid_texel(float r, float g, float b) {
    Texture16 t;
    t.w = t.h = 1;
    t.texels = {std::uint16_t(std::lrintf(r * 65535.0f)),
                std::uint16_t(std::lrintf(g * 65535.0f)),
                std::uint16_t(std::lrintf(b * 65535.0f)), 65535};
    return t;
}

} // namespace

std::shared_ptr<const MeshData> load_glb(const std::string& path,
                                         const MeshPlacement& placement,
                                         std::string& error) {
    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.c_str(), &data) !=
        cgltf_result_success) {
        error = "cgltf: cannot parse " + path;
        return nullptr;
    }
    // RAII-ish: single exit path frees.
    struct Free {
        cgltf_data* d;
        ~Free() { cgltf_free(d); }
    } freer{data};

    if (cgltf_load_buffers(&options, data, path.c_str()) !=
        cgltf_result_success) {
        error = "cgltf: cannot load buffers for " + path;
        return nullptr;
    }

    // ---- assemble all primitives through the node hierarchy ----
    Assembled a;
    const cgltf_scene* scene =
        data->scene ? data->scene
                    : (data->scenes_count ? &data->scenes[0] : nullptr);
    if (scene) {
        for (cgltf_size i = 0; i < scene->nodes_count; ++i)
            walk_node(scene->nodes[i], a);
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; ++i)
            if (!data->nodes[i].parent) walk_node(&data->nodes[i], a);
    }
    if (a.idx.empty()) {
        error = "no renderable triangles (need POSITION/NORMAL/TEXCOORD_0)";
        return nullptr;
    }
    if (!a.any_tangents) compute_tangents(a);

    auto out = std::make_shared<MeshData>();
    MeshData::Info& info = out->info;
    info.vert_count = a.pos.size();
    info.index_count = a.idx.size();

    // ---- placement: scale to target height, spin, ground-center ----
    const float yaw = placement.yaw_deg * 3.14159265358979f / 180.0f;
    const float cy = std::cos(yaw), sy = std::sin(yaw);
    const auto spin = [&](const vec3& v) {
        return vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
    };
    vec3 pre_mn(1e30f), pre_mx(-1e30f), rot_mn(1e30f), rot_mx(-1e30f);
    info.uv_min[0] = info.uv_min[1] = 1e30f;
    info.uv_max[0] = info.uv_max[1] = -1e30f;
    std::vector<vec3> wpos(a.pos.size()), wnrm(a.pos.size()),
        wtan(a.pos.size());
    for (std::size_t i = 0; i < a.pos.size(); ++i) {
        const vec3& p = a.pos[i];
        pre_mn = {std::fmin(pre_mn.x, p.x), std::fmin(pre_mn.y, p.y),
                  std::fmin(pre_mn.z, p.z)};
        pre_mx = {std::fmax(pre_mx.x, p.x), std::fmax(pre_mx.y, p.y),
                  std::fmax(pre_mx.z, p.z)};
        wpos[i] = spin(p);
        rot_mn = {std::fmin(rot_mn.x, wpos[i].x), std::fmin(rot_mn.y, wpos[i].y),
                  std::fmin(rot_mn.z, wpos[i].z)};
        rot_mx = {std::fmax(rot_mx.x, wpos[i].x), std::fmax(rot_mx.y, wpos[i].y),
                  std::fmax(rot_mx.z, wpos[i].z)};
        wnrm[i] = spin(a.nrm[i]);   // rotation: normals rotate directly
        wtan[i] = spin(a.tan[i]);
        info.uv_min[0] = std::fmin(info.uv_min[0], a.uv[i * 2]);
        info.uv_max[0] = std::fmax(info.uv_max[0], a.uv[i * 2]);
        info.uv_min[1] = std::fmin(info.uv_min[1], a.uv[i * 2 + 1]);
        info.uv_max[1] = std::fmax(info.uv_max[1], a.uv[i * 2 + 1]);
    }
    const float s = placement.target_height / (rot_mx.y - rot_mn.y);
    const vec3 T(-s * 0.5f * (rot_mn.x + rot_mx.x), -s * rot_mn.y,
                 -s * 0.5f * (rot_mn.z + rot_mx.z));
    for (vec3& p : wpos) p = s * p + T;

    info.pre_min[0] = pre_mn.x; info.pre_min[1] = pre_mn.y; info.pre_min[2] = pre_mn.z;
    info.pre_max[0] = pre_mx.x; info.pre_max[1] = pre_mx.y; info.pre_max[2] = pre_mx.z;
    const vec3 post_mn = s * rot_mn + T, post_mx = s * rot_mx + T;
    info.post_min[0] = post_mn.x; info.post_min[1] = post_mn.y; info.post_min[2] = post_mn.z;
    info.post_max[0] = post_mx.x; info.post_max[1] = post_mx.y; info.post_max[2] = post_mx.z;

    // ---- deindex into triangle records ----
    const auto p3 = [](const vec3& v) { return pt_float3{v.x, v.y, v.z}; };
    out->tris.reserve(a.idx.size() / 3);
    for (std::size_t t = 0; t + 2 < a.idx.size(); t += 3) {
        const std::uint32_t i0 = a.idx[t], i1 = a.idx[t + 1], i2 = a.idx[t + 2];
        GPUTriangle tri{};
        tri.p0 = p3(wpos[i0]);
        tri.e1 = p3(wpos[i1] - wpos[i0]);
        tri.e2 = p3(wpos[i2] - wpos[i0]);
        tri.n0 = p3(wnrm[i0]);
        tri.n1 = p3(wnrm[i1]);
        tri.n2 = p3(wnrm[i2]);
        tri.t0 = p3(wtan[i0]);
        tri.t1 = p3(wtan[i1]);
        tri.t2 = p3(wtan[i2]);
        tri.w0 = a.tan_w[i0];
        tri.w1 = a.tan_w[i1];
        tri.w2 = a.tan_w[i2];
        tri.u0 = a.uv[i0 * 2]; tri.v0 = a.uv[i0 * 2 + 1];
        tri.u1 = a.uv[i1 * 2]; tri.v1 = a.uv[i1 * 2 + 1];
        tri.u2 = a.uv[i2 * 2]; tri.v2 = a.uv[i2 * 2 + 1];
        out->tris.push_back(tri);
    }

    // ---- material: first primitive that has one ----
    const cgltf_material* mat = nullptr;
    for (cgltf_size m = 0; m < data->meshes_count && !mat; ++m)
        for (cgltf_size p = 0; p < data->meshes[m].primitives_count && !mat; ++p)
            mat = data->meshes[m].primitives[p].material;

    const std::string glb_dir =
        std::filesystem::path(path).parent_path().string();
    if (mat) {
        const auto& pbr = mat->pbr_metallic_roughness;
        const float base_f[3] = {pbr.base_color_factor[0],
                                 pbr.base_color_factor[1],
                                 pbr.base_color_factor[2]};
        out->base = decode_image(
            pbr.base_color_texture.texture ? pbr.base_color_texture.texture->image
                                           : nullptr,
            glb_dir, /*srgb=*/true, base_f);
        if (!out->base.valid()) {
            // Factor-only base color (already linear per spec).
            out->base = solid_texel(base_f[0], base_f[1], base_f[2]);
        }

        // glTF: G = roughness x factor, B = metallic x factor. LINEAR data.
        const float mr_f[3] = {1.0f, pbr.roughness_factor, pbr.metallic_factor};
        out->mr = decode_image(
            pbr.metallic_roughness_texture.texture
                ? pbr.metallic_roughness_texture.texture->image
                : nullptr,
            glb_dir, /*srgb=*/false, mr_f);
        if (!out->mr.valid())
            out->mr = solid_texel(1.0f, pbr.roughness_factor, pbr.metallic_factor);

        const float emis_f[3] = {mat->emissive_factor[0],
                                 mat->emissive_factor[1],
                                 mat->emissive_factor[2]};
        out->emissive = decode_image(
            mat->emissive_texture.texture ? mat->emissive_texture.texture->image
                                          : nullptr,
            glb_dir, /*srgb=*/true, emis_f);
        if (!out->emissive.valid() &&
            (emis_f[0] > 0.0f || emis_f[1] > 0.0f || emis_f[2] > 0.0f))
            out->emissive = solid_texel(emis_f[0], emis_f[1], emis_f[2]);

        // Normal map: LINEAR, never sRGB-decoded.
        const float one[3] = {1, 1, 1};
        out->normal = decode_image(
            mat->normal_texture.texture ? mat->normal_texture.texture->image
                                        : nullptr,
            glb_dir, /*srgb=*/false, one);
    } else {
        out->base = solid_texel(0.8f, 0.8f, 0.8f);
        out->mr = solid_texel(1.0f, 0.9f, 0.0f);
    }

    // ---- BVH (permutes tris into leaf order) ----
    info.bvh = build_bvh(out->tris, out->nodes);
    return out;
}

std::shared_ptr<const EnvMap> load_hdr(const std::string& path,
                                       std::string& error) {
    int w = 0, h = 0, n = 0;
    float* data = stbi_loadf(path.c_str(), &w, &h, &n, 3);
    if (!data) {
        error = "cannot load HDR: " + path;
        return nullptr;
    }
    auto env = std::make_shared<EnvMap>();
    env->w = w;
    env->h = h;
    env->texels.resize(std::size_t(w) * h * 4);
    // Radiance files can technically encode Inf/NaN; one bad texel would
    // poison the accumulator forever. Screen at load (host-only).
    std::size_t bad = 0;
    for (std::size_t i = 0, count = std::size_t(w) * h; i < count; ++i) {
        for (int c = 0; c < 3; ++c) {
            float v = data[i * 3 + c];   // linear radiance, stored untouched
            if (!std::isfinite(v)) {
                v = 0.0f;
                ++bad;
            }
            env->texels[i * 4 + c] = v;
        }
        env->texels[i * 4 + 3] = 1.0f;
    }
    if (bad > 0)
        std::fprintf(stderr, "load_hdr: sanitized %zu non-finite texels\n",
                     bad);
    stbi_image_free(data);

    // ---- Session H: build the luminance x sin(theta) sampling CDFs ----
    const float PI = 3.14159265358979f;
    const std::size_t texel_count = std::size_t(w) * h;
    double lum_sum = 0.0;
    for (std::size_t i = 0; i < texel_count; ++i) {
        lum_sum += 0.2126 * env->texels[i * 4] +
                   0.7152 * env->texels[i * 4 + 1] +
                   0.0722 * env->texels[i * 4 + 2];
    }
    const float floor_w =
        float(lum_sum / double(texel_count)) * 1e-3f + 1e-12f;

    env->row_cdf.resize(h);
    env->cond_cdf.resize(std::size_t(w) * h);
    std::vector<double> row_sum(h, 0.0);
    for (int y = 0; y < h; ++y) {
        const float sin_th = std::sin(PI * (float(y) + 0.5f) / float(h));
        double run = 0.0;
        float* crow = env->cond_cdf.data() + std::size_t(y) * w;
        for (int x = 0; x < w; ++x) {
            const std::size_t i = (std::size_t(y) * w + x) * 4;
            const float lum = 0.2126f * env->texels[i] +
                              0.7152f * env->texels[i + 1] +
                              0.0722f * env->texels[i + 2];
            run += double((lum + floor_w) * sin_th);
            crow[x] = float(run);
        }
        row_sum[y] = run;
        const float inv = run > 0.0 ? float(1.0 / run) : 0.0f;
        for (int x = 0; x < w; ++x) crow[x] *= inv;
        crow[w - 1] = 1.0f;   // exact top: binary search can't fall off
    }
    double total = 0.0;
    for (int y = 0; y < h; ++y) total += row_sum[y];
    double run = 0.0;
    for (int y = 0; y < h; ++y) {
        run += row_sum[y];
        env->row_cdf[y] = float(run / total);
    }
    env->row_cdf[h - 1] = 1.0f;
    return env;
}
