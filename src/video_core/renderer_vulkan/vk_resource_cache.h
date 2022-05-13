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

constexpr u32 MAX_DYNAMIC_STATES = 8;
constexpr u32 MAX_ATTACHMENTS = 2;
constexpr u32 MAX_VERTEX_BUFFERS = 3;

class Pipeline {
public:
    Pipeline();
    ~Pipeline() = default;

    void Build();

    void SetShaderStage(vk::ShaderStageFlagBits stage, vk::ShaderModule module);

    void AddVertexBuffer(u32 binding, u32 stride, vk::VertexInputRate input_rate);
    void AddVertexAttribute(u32 location, u32 binding, VkFormat format, u32 offset);

    void SetPrimitiveTopology(vk::PrimitiveTopology topology, bool enable_primitive_restart = false);
    void SetRasterizationState(vk::PolygonMode polygon_mode, vk::CullModeFlags cull_mode,
                               vk::FrontFace front_face);

    void SetDepthState(bool depth_test, bool depth_write, vk::CompareOp compare_op);
    void SetStencilState(bool stencil_test, vk::StencilOpState front, vk::StencilOpState back);
    void SetNoDepthTestState();
    void SetNoStencilState();

    void AddDynamicState(vk::DynamicState state);
    void SetMultisamples(VkSampleCountFlagBits samples);

private:
    vk::GraphicsPipelineCreateInfo pipeline_info;
    std::array<vk::PipelineShaderStageCreateInfo, 3> shader_stages;

    vk::PipelineVertexInputStateCreateInfo vertex_input_state;
    vk::PipelineInputAssemblyStateCreateInfo input_assembly;
    vk::PipelineRasterizationStateCreateInfo rasterization_state;
    vk::PipelineDepthStencilStateCreateInfo depth_state;

    // Blending
    vk::PipelineColorBlendStateCreateInfo blend_state;
    std::array<vk::PipelineColorBlendAttachmentState, MAX_ATTACHMENTS> blend_attachments;
    std::array<vk::DynamicState, MAX_DYNAMIC_STATES> dynamic_state_values;

    VkPipelineViewportStateCreateInfo m_viewport_state;
    VkViewport m_viewport;
    VkRect2D m_scissor;

    VkPipelineDynamicStateCreateInfo m_dynamic_state;
    vk::PipelineMultisampleStateCreateInfo multisample_info;
};

extern std::unique_ptr<VKResourceCache> g_vk_res_cache;

}  // namespace Vulkan
