#pragma once

// Bridge to the platform viewer so main.cpp never sees Objective-C.

struct RenderSettings;
struct SceneDesc;

// Opens the interactive window; returns the process exit code.
// use_gpu: Metal compute backend (falls back to CPU if unavailable);
// false = the CPU reference backend (--cpu).
int run_viewer(const RenderSettings& settings, bool use_gpu,
               const SceneDesc& desc);
