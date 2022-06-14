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

class PipelineLayoutBuilder {
public:
    PipelineLayoutBuilder();
    ~PipelineLayoutBuilder() = default;

    void Clear();
    vk::PipelineLayout Build();

    void AddDescriptorSet(vk::DescriptorSetLayout layout);
    void AddPushConstants(vk::ShaderStageFlags stages, u32 offset, u32 size);

private:
    static constexpr u32 MAX_SETS = 8;
    static constexpr u32 MAX_PUSH_CONSTANTS = 5;

    vk::PipelineLayoutCreateInfo pipeline_layout_info;
    std::array<vk::DescriptorSetLayout, MAX_SETS> sets;
    std::array<vk::PushConstantRange, MAX_PUSH_CONSTANTS> push_constants;
};

class PipelineBuilder {
public:
    PipelineBuilder();
    ~PipelineBuilder() = default;

    void Clear();
    vk::Pipeline Build();

    void SetPipelineLayout(vk::PipelineLayout layout);
    void AddVertexBuffer(u32 binding, u32 stride, vk::VertexInputRate input_rate,
                         const std::span<vk::VertexInputAttributeDescription> attributes);
    void SetShaderStage(vk::ShaderStageFlagBits stage, vk::ShaderModule module);

    void SetPrimitiveTopology(vk::PrimitiveTopology topology, bool enable_primitive_restart = false);
    void SetLineWidth(float width);
    void SetMultisamples(vk::SampleCountFlagBits samples, bool per_sample_shading);
    void SetRasterizationState(vk::PolygonMode polygon_mode, vk::CullModeFlags cull_mode,
                               vk::FrontFace front_face);

    void SetNoCullRasterizationState();
    void SetDepthState(bool depth_test, bool depth_write, vk::CompareOp compare_op);
    void SetStencilState(bool stencil_test, vk::StencilOpState front, vk::StencilOpState back);
    void SetNoDepthTestState();
    void SetNoStencilState();

    void SetBlendConstants(float r, float g, float b, float a);
    void SetNoBlendingState();
    void SetBlendLogicOp(vk::LogicOp logic_op);
    void SetBlendAttachment(bool blend_enable, vk::BlendFactor src_factor, vk::BlendFactor dst_factor,
                            vk::BlendOp op, vk::BlendFactor alpha_src_factor, vk::BlendFactor alpha_dst_factor,
                            vk::BlendOp alpha_op,vk::ColorComponentFlags write_mask);

    void SetViewport(float x, float y, float width, float height, float min_depth, float max_depth);
    void SetScissorRect(s32 x, s32 y, u32 width, u32 height);
    void SetDynamicStates(const std::span<vk::DynamicState> states);
    void SetRenderingFormats(vk::Format color, vk::Format depth_stencil = vk::Format::eUndefined);

private:
    static constexpr u32 MAX_DYNAMIC_STATES = 20;
    static constexpr u32 MAX_SHADER_STAGES = 3;
    static constexpr u32 MAX_VERTEX_BUFFERS = 8;
    static constexpr u32 MAX_VERTEX_ATTRIBUTES = 16;

    vk::GraphicsPipelineCreateInfo pipeline_info;
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;

    vk::PipelineVertexInputStateCreateInfo vertex_input_state;
    std::array<vk::VertexInputBindingDescription, MAX_VERTEX_BUFFERS> vertex_buffers;
    std::array<vk::VertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES> vertex_attributes;

    vk::PipelineInputAssemblyStateCreateInfo input_assembly;
    vk::PipelineRasterizationStateCreateInfo rasterization_state;
    vk::PipelineDepthStencilStateCreateInfo depth_state;

    // Blending
    vk::PipelineColorBlendStateCreateInfo blend_state;
    vk::PipelineColorBlendAttachmentState blend_attachment;
    vk::PipelineDynamicStateCreateInfo dynamic_info;
    std::array<vk::DynamicState, MAX_DYNAMIC_STATES> dynamic_states;

    vk::PipelineViewportStateCreateInfo viewport_state;
    vk::Viewport viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    vk::Rect2D scissor;

    // Multisampling
    vk::PipelineMultisampleStateCreateInfo multisample_info;
    vk::PipelineRenderingCreateInfo rendering_info;
    vk::Format color_format, depth_stencil_format;
};

}  // namespace Vulkan
