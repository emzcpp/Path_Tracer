#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "hittable.h"

// Linear list of hittables — closest hit wins. No acceleration structure;
// with a handful of spheres, a BVH would cost more than it saves. (It's the
// obvious next step once the scene grows, on CPU and GPU alike.)
class Scene : public Hittable {
public:
    void add(std::unique_ptr<Hittable> object) {
        objects_.push_back(std::move(object));
    }

    bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override {
        HitRecord temp;
        bool hit_anything = false;
        float closest = t_max;

        for (const auto& object : objects_) {
            if (object->hit(r, t_min, closest, temp)) {
                hit_anything = true;
                closest = temp.t;
                rec = temp;
            }
        }
        return hit_anything;
    }

    bool occluded(const Ray& r, float t_min, float t_max) const override {
        for (const auto& o : objects_) {
            if (o->occluded(r, t_min, t_max)) return true;
        }
        return false;
    }

private:
    std::vector<std::unique_ptr<Hittable>> objects_;
};
