// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <functional>
#include "video_core/regs.h"
#include "video_core/common/shader_runtime_cache.h"
#include "video_core/common/shader_disk_cache.h"

namespace FileUtil {
class IOFile;
}

namespace Core {
class System;
}

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {

enum class LoadCallbackStage : u8 {
    Prepare = 0,
    Decompile = 1,
    Build = 2,
    Complete = 3,
};

using DiskLoadCallback = std::function<void(LoadCallbackStage, std::size_t, std::size_t)>;

// A class that manages and caches shaders and pipelines
class PipelineCache {
public:
    PipelineCache(Frontend::EmuWindow& emu_window, std::unique_ptr<BackendBase>& backend);
    ~PipelineCache() = default;

    // Loads backend specific shader binaries from disk
    void LoadDiskCache(const std::atomic_bool& stop_loading, const DiskLoadCallback& callback);

    bool UsePicaVertexShader(const Pica::Regs& config, Pica::Shader::ShaderSetup& setup);
    void UseTrivialVertexShader();

    void UseFixedGeometryShader(const Pica::Regs& regs);
    void UseTrivialGeometryShader();

    // Compiles and caches a fragment shader based on the current pica state
    void UseFragmentShader(const Pica::Regs& config);

private:
    Frontend::EmuWindow& emu_window;
    std::unique_ptr<BackendBase>& backend;
    std::unique_ptr<ShaderGeneratorBase> generator;

    // Keeps all the compiled graphics pipelines
    std::unordered_map<PipelineInfo, PipelineHandle> cached_pipelines;

    // Current shaders
    ShaderHandle current_vertex_shader;
    ShaderHandle current_geometry_shader;
    ShaderHandle current_fragment_shader;

    // Pica runtime shader caches
    PicaVertexShaders pica_vertex_shaders;
    FixedGeometryShaders fixed_geometry_shaders;
    FragmentShaders fragment_shaders;
    ShaderHandle trivial_vertex_shader;

    // Serializes shader binaries to disk
    ShaderDiskCache disk_cache;
};
} // namespace VideoCore
