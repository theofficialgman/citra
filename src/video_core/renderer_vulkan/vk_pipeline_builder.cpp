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

PipelineLayoutBuilder::PipelineLayoutBuilder() {
    Clear();
}

void PipelineLayoutBuilder::Clear() {
    pipeline_layout_info = vk::PipelineLayoutCreateInfo{};
}

vk::PipelineLayout PipelineLayoutBuilder::Build() {
    auto device = g_vk_instace->GetDevice();

    auto result = device.createPipelineLayout(pipeline_layout_info);
    if (!result) {
        LOG_ERROR(Render_Vulkan, "Failed to create pipeline layout");
        return VK_NULL_HANDLE;
    }

    return result;
}

void PipelineLayoutBuilder::AddDescriptorSet(vk::DescriptorSetLayout layout) {
    assert(pipeline_layout_info.setLayoutCount < MAX_SETS);

    sets[pipeline_layout_info.setLayoutCount++] = layout;
    pipeline_layout_info.pSetLayouts = sets.data();
}

void PipelineLayoutBuilder::AddPushConstants(vk::ShaderStageFlags stages, u32 offset, u32 size) {
    assert(pipeline_layout_info.pushConstantRangeCount < MAX_PUSH_CONSTANTS);

    push_constants[pipeline_layout_info.pushConstantRangeCount++] = {stages, offset, size};
    pipeline_layout_info.pPushConstantRanges = push_constants.data();
}

PipelineBuilder::PipelineBuilder() {
    Clear();
}

void PipelineBuilder::Clear() {
    pipeline_info = vk::GraphicsPipelineCreateInfo{};
    shader_stages.clear();

    vertex_input_state = vk::PipelineVertexInputStateCreateInfo{};
    input_assembly = vk::PipelineInputAssemblyStateCreateInfo{};
    rasterization_state = vk::PipelineRasterizationStateCreateInfo{};
    depth_state = vk::PipelineDepthStencilStateCreateInfo{};

    blend_state = vk::PipelineColorBlendStateCreateInfo{};
    blend_attachment = vk::PipelineColorBlendAttachmentState{};
    dynamic_info = vk::PipelineDynamicStateCreateInfo{};
    dynamic_states.fill({});

    viewport_state = vk::PipelineViewportStateCreateInfo{};
    multisample_info = vk::PipelineMultisampleStateCreateInfo{};

    // Set defaults
    SetNoCullRasterizationState();
    SetNoDepthTestState();
    SetNoBlendingState();
    SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);

    // Have to be specified even if dynamic
    SetViewport(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f);
    SetScissorRect(0, 0, 1, 1);
    SetBlendConstants(1.0f, 1.0f, 1.0f, 1.0f);
    SetMultisamples(vk::SampleCountFlagBits::e1, false);
}

vk::Pipeline PipelineBuilder::Build() {
    auto device = g_vk_instace->GetDevice();

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

    pipeline_info.pStages = shader_stages.data();
}

void PipelineBuilder::AddVertexBuffer(u32 binding, u32 stride, vk::VertexInputRate input_rate,
                                      std::span<vk::VertexInputAttributeDescription> attributes) {
    // Copy attributes to private array
    auto loc = vertex_attributes.begin() + vertex_input_state.vertexAttributeDescriptionCount;
    std::copy(attributes.begin(), attributes.end(), loc);

    vertex_buffers[vertex_input_state.vertexBindingDescriptionCount++] = {binding, stride, input_rate};
    vertex_input_state.vertexAttributeDescriptionCount += attributes.size();

    vertex_input_state.pVertexBindingDescriptions = vertex_buffers.data();
    vertex_input_state.pVertexAttributeDescriptions = vertex_attributes.data();

    pipeline_info.pVertexInputState = &vertex_input_state;
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
    pipeline_info.pRasterizationState = &rasterization_state;
}

void PipelineBuilder::SetLineWidth(float width) {
    rasterization_state.lineWidth = width;
    pipeline_info.pRasterizationState = &rasterization_state;
}

void PipelineBuilder::SetMultisamples(vk::SampleCountFlagBits samples, bool per_sample_shading) {
    multisample_info.rasterizationSamples = samples;
    multisample_info.sampleShadingEnable = per_sample_shading;
    multisample_info.minSampleShading = (static_cast<u32>(samples) > 1) ? 1.0f : 0.0f;
    pipeline_info.pMultisampleState = &multisample_info;
}

void PipelineBuilder::SetNoCullRasterizationState() {
    SetRasterizationState(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eClockwise);
}

void PipelineBuilder::SetDepthState(bool depth_test, bool depth_write, vk::CompareOp compare_op) {
    depth_state.depthTestEnable = depth_test;
    depth_state.depthWriteEnable = depth_write;
    depth_state.depthCompareOp = compare_op;
    pipeline_info.pDepthStencilState = &depth_state;
}

void PipelineBuilder::SetStencilState(bool stencil_test, vk::StencilOpState front, vk::StencilOpState back) {
    depth_state.stencilTestEnable = stencil_test;
    depth_state.front = front;
    depth_state.back = back;
    pipeline_info.pDepthStencilState = &depth_state;
}

void PipelineBuilder::SetNoStencilState() {
    depth_state.stencilTestEnable = VK_FALSE;
    depth_state.front = vk::StencilOpState{};
    depth_state.back = vk::StencilOpState{};
}

void PipelineBuilder::SetNoDepthTestState() {
    SetDepthState(false, false, vk::CompareOp::eAlways);
}

void PipelineBuilder::SetBlendConstants(float r, float g, float b, float a) {
    blend_state.blendConstants = std::array<float, 4>{r, g, b, a};
    pipeline_info.pColorBlendState = &blend_state;
}

void PipelineBuilder::SetBlendLogicOp(vk::LogicOp logic_op)  {
    blend_state.logicOp = logic_op;
    blend_state.logicOpEnable = false;
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
    pipeline_info.pColorBlendState = &blend_state;
}

void PipelineBuilder::SetNoBlendingState() {
    SetBlendAttachment(false, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne,
        vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
}

void PipelineBuilder::SetDynamicStates(const std::span<vk::DynamicState> states) {
    if (states.size() > MAX_DYNAMIC_STATES) {
        LOG_ERROR(Render_Vulkan, "Cannot include more dynamic states!");
        UNREACHABLE();
    }

    // Copy the state data
    std::copy(states.begin(), states.end(), dynamic_states.begin());
    dynamic_info.dynamicStateCount = states.size();
    dynamic_info.pDynamicStates = dynamic_states.data();
    pipeline_info.pDynamicState = &dynamic_info;
    return;
}

void PipelineBuilder::SetRenderingFormats(vk::Format color, vk::Format depth_stencil) {
    color_format = color;
    depth_stencil_format = depth_stencil;

    auto IsStencil = [](vk::Format format) -> bool {
        switch (format) {
        case vk::Format::eD16UnormS8Uint:
        case vk::Format::eD24UnormS8Uint:
        case vk::Format::eD32SfloatS8Uint:
            return true;
        default:
            return false;
        };
    };

    const u32 color_attachment_count = color == vk::Format::eUndefined ? 0 : 1;
    rendering_info = vk::PipelineRenderingCreateInfo{0, color_attachment_count, &color_format, depth_stencil_format,
                        IsStencil(depth_stencil) ? depth_stencil : vk::Format::eUndefined};
    pipeline_info.pNext = &rendering_info;
}

void PipelineBuilder::SetViewport(float x, float y, float width, float height, float min_depth, float max_depth) {
    viewport = vk::Viewport{x, y, width, height, min_depth, max_depth};
    viewport_state.pViewports = &viewport;
    viewport_state.viewportCount = 1;
    pipeline_info.pViewportState = &viewport_state;
}

void PipelineBuilder::SetScissorRect(s32 x, s32 y, u32 width, u32 height) {
    scissor = vk::Rect2D{{x, y}, {width, height}};
    viewport_state.scissorCount = 1u;
    viewport_state.pScissors = &scissor;
    pipeline_info.pViewportState = &viewport_state;
}

}  // namespace Vulkan
