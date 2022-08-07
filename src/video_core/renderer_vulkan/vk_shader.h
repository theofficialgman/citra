// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/common/shader.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

class Instance;

class Shader : public VideoCore::ShaderBase {
public:
    Shader(Instance& instance, ShaderStage stage, std::string_view name,
           std::string&& source);
    ~Shader() override;

    bool Compile(ShaderOptimization level) override;

    /// Returns the underlying vulkan shader module handle
    vk::ShaderModule GetHandle() const {
        return module;
    }

private:
    Instance& instance;
    vk::ShaderModule module;
};

} // namespace VideoCore::Vulkan
