#pragma once

// BVH construction over GPUTriangle records. Built once on the host; the
// resulting flat node array (and the leaf-order-permuted triangle array)
// is consumed by BOTH backends bit-identically, so the build strategy has
// zero parity surface — only traversal code (duplicated C++/MSL) matters.

#include <cstdint>
#include <vector>

#include "kernel_types.h"

struct BvhStats {
    int node_count = 0;
    int max_depth = 0;        // hard-fails if it would exceed the traversal
    int leaf_count = 0;       //   stack guarantee (depth cap 30)
    float mean_leaf_tris = 0.0f;
};

// Reorders `tris` into leaf order and fills `nodes`. Leaf nodes index the
// permuted array directly via left_or_first.
// `tri_ids` (optional): a parallel per-triangle array — e.g. material
// indices — permuted into the same leaf order as the triangles.
BvhStats build_bvh(std::vector<GPUTriangle>& tris,
                   std::vector<BVHNode>& nodes,
                   std::vector<std::uint32_t>* tri_ids = nullptr);
