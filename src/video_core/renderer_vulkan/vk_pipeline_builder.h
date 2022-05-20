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

constexpr u32 MAX_DYNAMIC_STATES = 14;
constexpr u32 MAX_SHADER_STAGES = 3;

class PipelineBuilder {
public:
    PipelineBuilder();
    ~PipelineBuilder() = default;

    vk::Pipeline Build();

    void SetPipelineLayout(vk::PipelineLayout layout);
    void SetShaderStage(vk::ShaderStageFlagBits stage, vk::ShaderModule module);

    void SetPrimitiveTopology(vk::PrimitiveTopology topology, bool enable_primitive_restart = false);
    void SetLineWidth(float width);
    void SetMultisamples(u32 multisamples, bool per_sample_shading);
    void SetRasterizationState(vk::PolygonMode polygon_mode, vk::CullModeFlags cull_mode,
                               vk::FrontFace front_face);

    void SetDepthState(bool depth_test, bool depth_write, vk::CompareOp compare_op);
    void SetStencilState(bool stencil_test, vk::StencilOpState front, vk::StencilOpState back);
    void SetNoDepthTestState();
    void SetNoStencilState();

    void SetBlendConstants(float r, float g, float b, float a);
    void SetNoBlendingState();
    void SetBlendAttachment(bool blend_enable, vk::BlendFactor src_factor, vk::BlendFactor dst_factor,
                            vk::BlendOp op, vk::BlendFactor alpha_src_factor, vk::BlendFactor alpha_dst_factor,
                            vk::BlendOp alpha_op,vk::ColorComponentFlags write_mask);

    void SetViewport(float x, float y, float width, float height, float min_depth, float max_depth);
    void SetScissorRect(s32 x, s32 y, u32 width, u32 height);
    void AddDynamicState(vk::DynamicState state);
    void SetMultisamples(vk::SampleCountFlagBits samples);

private:
    vk::GraphicsPipelineCreateInfo pipeline_info;
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;

    vk::PipelineVertexInputStateCreateInfo vertex_input_state;
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
};

}  // namespace Vulkan
