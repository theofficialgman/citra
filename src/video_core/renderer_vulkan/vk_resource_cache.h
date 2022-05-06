// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

struct RenderPassCacheKey {
    vk::Format color, depth;
    vk::SampleCountFlagBits samples;
};

constexpr u32 DESCRIPTOR_SET_LAYOUT_COUNT = 3;

/// Wrapper class that manages resource caching and storage.
/// It stores pipelines and renderpasses
class VKResourceCache {
public:
    VKResourceCache() = default;
    ~VKResourceCache();

    // Perform at startup, create descriptor layouts, compiles all static shaders.
    bool Initialize();
    void Shutdown();

    // Public interface.
    vk::PipelineCache GetPipelineCache() const { return pipeline_cache.get(); }
    vk::RenderPass GetRenderPass(vk::Format color_format, vk::Format depth_format,
                                 vk::SampleCountFlagBits multisamples,
                                 vk::AttachmentLoadOp load_op);

    auto& GetDescriptorLayouts() const { return descriptor_layouts; }

private:
    // Descriptor sets
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_SET_LAYOUT_COUNT> descriptor_layouts;
    vk::UniquePipelineLayout pipeline_layout;

    // Render pass cache
    std::unordered_map<RenderPassCacheKey, vk::UniqueRenderPass> renderpass_cache;

    vk::UniquePipelineCache pipeline_cache;
    std::string pipeline_cache_filename;
};

extern std::unique_ptr<VKResourceCache> g_vk_res_cache;

}  // namespace Vulkan
