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

using RenderPassCacheKey = std::tuple<vk::Format, vk::Format, u32, vk::AttachmentLoadOp>;

constexpr u32 MAX_TEXTURE_UPLOAD_BUFFER_SIZE = 32 * 1024 * 1024;
constexpr u32 DESCRIPTOR_SET_LAYOUT_COUNT = 3;

class VKResourceCache
{
public:
  VKResourceCache() = default;
  ~VKResourceCache();

  // Perform at startup, create descriptor layouts, compiles all static shaders.
  bool Initialize();
  void Shutdown();

  // Public interface.
  VKBuffer& GetTextureUploadBuffer() { return texture_upload_buffer; }
  vk::Sampler GetSampler(const SamplerInfo& info);
  vk::RenderPass GetRenderPass(vk::Format color_format, vk::Format depth_format, u32 multisamples, vk::AttachmentLoadOp load_op);
  vk::PipelineCache GetPipelineCache() const { return pipeline_cache.get(); }

private:
    // Dummy image for samplers that are unbound
    VKTexture dummy_texture;
    VKBuffer texture_upload_buffer;

    // Descriptor sets
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_SET_LAYOUT_COUNT> descriptor_layouts;
    vk::UniquePipelineLayout pipeline_layout;

    // Render pass cache
    std::unordered_map<RenderPassCacheKey, vk::RenderPass> render_pass_cache;
    std::unordered_map<SamplerInfo, vk::Sampler> sampler_cache;

    vk::UniquePipelineCache pipeline_cache;
    std::string pipeline_cache_filename;
};

extern std::unique_ptr<VKResourceCache> g_object_cache;

}  // namespace Vulkan
