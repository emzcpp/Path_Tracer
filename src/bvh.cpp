#include "bvh.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "vec3.h"

namespace {

// Leaves at <= 4 tris; depth 30 FORCES a leaf, which turns the traversal
// stack bound (uint stack[32]) into a guarantee rather than a heuristic.
constexpr int kLeafSize = 4;
constexpr int kMaxDepth = 30;

struct AABB {
    vec3 mn{1e30f, 1e30f, 1e30f};
    vec3 mx{-1e30f, -1e30f, -1e30f};

    void grow(const vec3& p) {
        mn = {std::fmin(mn.x, p.x), std::fmin(mn.y, p.y), std::fmin(mn.z, p.z)};
        mx = {std::fmax(mx.x, p.x), std::fmax(mx.y, p.y), std::fmax(mx.z, p.z)};
    }
    void grow(const AABB& b) { grow(b.mn); grow(b.mx); }
};

vec3 c3v(const pt_float3& v) { return {v.x, v.y, v.z}; }

AABB tri_bounds(const GPUTriangle& t) {
    AABB b;
    const vec3 p0 = c3v(t.p0);
    b.grow(p0);
    b.grow(p0 + c3v(t.e1));
    b.grow(p0 + c3v(t.e2));
    return b;
}

struct Builder {
    const std::vector<GPUTriangle>& tris;
    std::vector<BVHNode>& nodes;
    std::vector<std::uint32_t> order;   // permutation being built
    std::vector<AABB> tbox;
    std::vector<vec3> centroid;
    BvhStats stats;

    Builder(const std::vector<GPUTriangle>& t, std::vector<BVHNode>& n)
        : tris(t), nodes(n) {
        order.resize(tris.size());
        tbox.resize(tris.size());
        centroid.resize(tris.size());
        for (std::uint32_t i = 0; i < tris.size(); ++i) {
            order[i] = i;
            tbox[i] = tri_bounds(tris[i]);
            centroid[i] = 0.5f * (tbox[i].mn + tbox[i].mx);
        }
    }

    static float axis_of(const vec3& v, int a) {
        return a == 0 ? v.x : (a == 1 ? v.y : v.z);
    }

    // Writes nodes[node_index] before recursing (the vector may grow, so
    // never hold a reference across a recursion).
    void build(std::uint32_t node_index, std::uint32_t first,
               std::uint32_t count, int depth) {
        stats.max_depth = std::max(stats.max_depth, depth);

        AABB bounds, cbounds;
        for (std::uint32_t i = first; i < first + count; ++i) {
            bounds.grow(tbox[order[i]]);
            cbounds.grow(centroid[order[i]]);
        }
        // Inflate: kills the zero-thickness-box NaN corner case in the slab
        // test. Build-time transform => no parity surface.
        vec3 pad{std::fmax(1e-4f, 1e-5f * (bounds.mx.x - bounds.mn.x)),
                 std::fmax(1e-4f, 1e-5f * (bounds.mx.y - bounds.mn.y)),
                 std::fmax(1e-4f, 1e-5f * (bounds.mx.z - bounds.mn.z))};
        bounds.mn -= pad;
        bounds.mx += pad;

        if (count <= kLeafSize || depth >= kMaxDepth) {
            nodes[node_index] = BVHNode{{bounds.mn.x, bounds.mn.y, bounds.mn.z},
                                        first,
                                        {bounds.mx.x, bounds.mx.y, bounds.mx.z},
                                        count};
            ++stats.leaf_count;
            stats.mean_leaf_tris += float(count);
            return;
        }

        // Midpoint split on the longest centroid axis; median fallback when
        // the midpoint partition is degenerate (all centroids on one side).
        const vec3 cext = cbounds.mx - cbounds.mn;
        const int axis = cext.x > cext.y ? (cext.x > cext.z ? 0 : 2)
                                         : (cext.y > cext.z ? 1 : 2);
        const float mid = 0.5f * (axis_of(cbounds.mn, axis) +
                                  axis_of(cbounds.mx, axis));

        auto* base = order.data();
        auto pred = [&](std::uint32_t t) {
            return axis_of(centroid[t], axis) < mid;
        };
        std::uint32_t* split =
            std::partition(base + first, base + first + count, pred);
        std::uint32_t lcount = std::uint32_t(split - (base + first));
        if (lcount == 0 || lcount == count) {
            std::nth_element(base + first, base + first + count / 2,
                             base + first + count,
                             [&](std::uint32_t a, std::uint32_t b) {
                                 return axis_of(centroid[a], axis) <
                                        axis_of(centroid[b], axis);
                             });
            lcount = count / 2;
        }

        // Children allocated adjacently: right = left + 1, so the node only
        // stores the left index.
        const auto left = std::uint32_t(nodes.size());
        nodes.emplace_back();
        nodes.emplace_back();
        nodes[node_index] = BVHNode{{bounds.mn.x, bounds.mn.y, bounds.mn.z},
                                    left,
                                    {bounds.mx.x, bounds.mx.y, bounds.mx.z},
                                    0};
        build(left, first, lcount, depth + 1);
        build(left + 1, first + lcount, count - lcount, depth + 1);
    }
};

} // namespace

BvhStats build_bvh(std::vector<GPUTriangle>& tris,
                   std::vector<BVHNode>& nodes) {
    nodes.clear();
    if (tris.empty()) return {};

    Builder b(tris, nodes);
    nodes.emplace_back();
    b.build(0, 0, std::uint32_t(tris.size()), 0);

    b.stats.node_count = int(nodes.size());
    b.stats.mean_leaf_tris /= float(std::max(1, b.stats.leaf_count));

    if (b.stats.max_depth >= kMaxDepth + 1) {
        std::fprintf(stderr, "bvh: depth %d exceeds traversal stack bound\n",
                     b.stats.max_depth);
        std::abort();
    }

    // Permute triangles into leaf order so leaves index the array directly.
    std::vector<GPUTriangle> permuted(tris.size());
    for (std::size_t i = 0; i < tris.size(); ++i) permuted[i] = tris[b.order[i]];
    tris = std::move(permuted);
    return b.stats;
}
