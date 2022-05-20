// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_pipeline_builder.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_state.h"
#include <algorithm>
#include <array>
#include <type_traits>

namespace Vulkan {

PipelineBuilder::PipelineBuilder() {
    vertex_input_state = vk::PipelineVertexInputStateCreateInfo{
        {}, HardwareVertex::binding_desc, HardwareVertex::attribute_desc
    };

    /* Include all required pointers to the pipeline info structure */
    vk::GraphicsPipelineCreateInfo pipeline_info{
        {}, 0, shader_stages.data(), &vertex_input_state, &input_assembly, nullptr,
        &viewport_state, &rasterization_state, &multisample_info, &depth_state,
        &blend_state, &dynamic_info, nullptr, nullptr };
}

vk::Pipeline PipelineBuilder::Build() {
    auto& device = g_vk_instace->GetDevice();
    auto result = device.createGraphicsPipeline({}, pipeline_info);

    if (result.result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Failed to build vulkan pipeline!");
        UNREACHABLE();
    }

    return result.value;
}

void PipelineBuilder::SetPipelineLayout(vk::PipelineLayout layout) {
    pipeline_info.layout = layout;
}

void PipelineBuilder::SetShaderStage(vk::ShaderStageFlagBits stage, vk::ShaderModule module) {
    auto result = std::ranges::find_if(shader_stages.begin(), shader_stages.end(), [stage](const auto& info) {
       return info.stage == stage;
    });

    /* If the stage already exists, just replace the module */
    if (result != shader_stages.end()) {
        result->module = module;
    }
    else {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateFlags(), stage, module, "main");
        pipeline_info.stageCount++;
    }
}

void PipelineBuilder::SetPrimitiveTopology(vk::PrimitiveTopology topology, bool enable_primitive_restart) {
    input_assembly.topology = topology;
    input_assembly.primitiveRestartEnable = enable_primitive_restart;
    pipeline_info.pInputAssemblyState = &input_assembly;
}

void PipelineBuilder::SetRasterizationState(vk::PolygonMode polygon_mode, vk::CullModeFlags cull_mode,
                                            vk::FrontFace front_face) {
    rasterization_state.polygonMode = polygon_mode;
    rasterization_state.cullMode = cull_mode;
    rasterization_state.frontFace = front_face;
}

void PipelineBuilder::SetLineWidth(float width) {
    rasterization_state.lineWidth = width;
}

void PipelineBuilder::SetMultisamples(u32 multisamples, bool per_sample_shading) {
    multisample_info.rasterizationSamples = static_cast<vk::SampleCountFlagBits>(multisamples);
    multisample_info.sampleShadingEnable = per_sample_shading;
    multisample_info.minSampleShading = (multisamples > 1) ? 1.0f : 0.0f;
}

void PipelineBuilder::SetDepthState(bool depth_test, bool depth_write, vk::CompareOp compare_op) {
    depth_state.depthTestEnable = depth_test;
    depth_state.depthWriteEnable = depth_write;
    depth_state.depthCompareOp = compare_op;
}

void PipelineBuilder::SetStencilState(bool stencil_test, vk::StencilOpState front, vk::StencilOpState back) {
    depth_state.stencilTestEnable = stencil_test;
    depth_state.front = front;
    depth_state.back = back;
}

void PipelineBuilder::SetBlendConstants(float r, float g, float b, float a) {
    blend_state.blendConstants = std::array<float, 4>{r, g, b, a};
}

void PipelineBuilder::SetBlendAttachment(bool blend_enable, vk::BlendFactor src_factor, vk::BlendFactor dst_factor,
                                         vk::BlendOp op, vk::BlendFactor alpha_src_factor,
                                         vk::BlendFactor alpha_dst_factor, vk::BlendOp alpha_op,
                                         vk::ColorComponentFlags write_mask) {
    blend_attachment.blendEnable = blend_enable;
    blend_attachment.srcColorBlendFactor = src_factor;
    blend_attachment.dstColorBlendFactor = dst_factor;
    blend_attachment.colorBlendOp = op;
    blend_attachment.srcAlphaBlendFactor = alpha_src_factor;
    blend_attachment.dstAlphaBlendFactor = alpha_dst_factor;
    blend_attachment.alphaBlendOp = alpha_op;
    blend_attachment.colorWriteMask = write_mask;

    blend_state.attachmentCount = 1;
    blend_state.pAttachments = &blend_attachment;
}

void PipelineBuilder::SetNoBlendingState() {
    SetBlendAttachment(false, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne,
        vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
}

void PipelineBuilder::AddDynamicState(vk::DynamicState state) {
    if (dynamic_info.dynamicStateCount < MAX_DYNAMIC_STATES) {
        dynamic_states[dynamic_info.dynamicStateCount] = state;

        dynamic_info.dynamicStateCount++;
        dynamic_info.pDynamicStates = dynamic_states.data();
        return;
    }

    LOG_ERROR(Render_Vulkan, "Cannot include more dynamic states!");
    UNREACHABLE();
}

void PipelineBuilder::SetViewport(float x, float y, float width, float height, float min_depth, float max_depth) {
    viewport = vk::Viewport{ x, y, width, height, min_depth, max_depth };
    viewport_state.pViewports = &viewport;
    viewport_state.viewportCount = 1;
}

void PipelineBuilder::SetScissorRect(s32 x, s32 y, u32 width, u32 height) {
    scissor = vk::Rect2D{{x, y}, {width, height}};
    viewport_state.pScissors = &scissor;
    viewport_state.scissorCount = 1u;
}

}  // namespace Vulkan
