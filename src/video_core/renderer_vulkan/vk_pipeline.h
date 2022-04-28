// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <utility>
#include <variant>
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

using Resource = std::variant<VKBuffer, VKTexture>;

/// Vulkan pipeline objects represent a collection of shader modules
class VKPipeline final : private NonCopyable {
public:
    /// Includes all required information to build a Vulkan pipeline object
    class Info : private NonCopyable {
        Info() = default;
        ~Info() = default;

        /// Assign a shader module to a specific stage
        void AddShaderModule(const vk::ShaderModule& module, vk::ShaderStageFlagBits stage);

        /// Add a texture or a buffer to the target descriptor set
        void AddResource(const Resource& resource, vk::DescriptorType type, vk::ShaderStageFlags stages, int set = 0);

    private:
        using ResourceInfo = std::pair<std::reference_wrapper<Resource>, vk::DescriptorSetLayoutBinding>;

        std::unordered_map<int, std::vector<ResourceInfo>> descriptor_sets;
        std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;
    };

    VKPipeline() = default;
    ~VKPipeline() = default;

    /// Create a new Vulkan pipeline object
    void Create(const Info& info);
    void Create(vk::PipelineLayoutCreateInfo layout_info);

private:
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
};

} // namespace OpenGL
