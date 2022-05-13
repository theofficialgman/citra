// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_resource_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include <algorithm>
#include <array>
#include <type_traits>

namespace Vulkan {

VKResourceCache::~VKResourceCache() {
    for (int i = 0; i < DESCRIPTOR_SET_LAYOUT_COUNT; i++) {
        g_vk_instace->GetDevice().destroyDescriptorSetLayout(descriptor_layouts[i]);
    }
}

bool VKResourceCache::Initialize() {
    // Define the descriptor sets we will be using
    std::array<vk::DescriptorSetLayoutBinding, 2> ubo_set = {{
        { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex |
          vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment }, // shader_data
        { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex } // pica_uniforms
    }};

    std::array<vk::DescriptorSetLayoutBinding, 4> texture_set = {{
        { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex0
        { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex1
        { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex2
        { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex_cube
    }};

    std::array<vk::DescriptorSetLayoutBinding, 3> lut_set = {{
        { 0, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment }, // texture_buffer_lut_lf
        { 1, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment }, // texture_buffer_lut_rg
        { 2, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment } // texture_buffer_lut_rgba
    }};

    // Create and store descriptor set layouts
    std::array<vk::DescriptorSetLayoutCreateInfo, DESCRIPTOR_SET_LAYOUT_COUNT> create_infos = {{
        { vk::DescriptorSetLayoutCreateFlags(), ubo_set },
        { vk::DescriptorSetLayoutCreateFlags(), texture_set },
        { vk::DescriptorSetLayoutCreateFlags(), lut_set }
    }};

    for (int i = 0; i < DESCRIPTOR_SET_LAYOUT_COUNT; i++) {
        descriptor_layouts[i] = g_vk_instace->GetDevice().createDescriptorSetLayout(create_infos[i]);
    }

    // Create the standard descriptor set layout
    vk::PipelineLayoutCreateInfo layout_info({}, descriptor_layouts);
    pipeline_layout = g_vk_instace->GetDevice().createPipelineLayoutUnique(layout_info);

    return true;
}

vk::RenderPass VKResourceCache::GetRenderPass(vk::Format color_format, vk::Format depth_format,
                                              vk::SampleCountFlagBits multisamples,
                                              vk::AttachmentLoadOp load_op) {
    // Search the cache if we can reuse an already created renderpass
    RenderPassCacheKey key = {
        .color = color_format,
        .depth = depth_format,
        .samples = multisamples
    };

    auto it = renderpass_cache.find(key);
    if (it != renderpass_cache.end()) {
        return it->second.get();
    }

    // Otherwise create a new one with the parameters provided
    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics);
    std::array<vk::AttachmentDescription, 2> attachments;
    std::array<vk::AttachmentReference, 2> references;
    u32 index = 0;

    if (color_format != vk::Format::eUndefined) {
        references[index] = vk::AttachmentReference{index, vk::ImageLayout::eColorAttachmentOptimal};
        attachments[index] =
        {
            {},
            color_format,
            multisamples,
            load_op,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eColorAttachmentOptimal
        };

        subpass.setColorAttachmentCount(1);
        subpass.setPColorAttachments(&references[index++]);
    }

    if (depth_format != vk::Format::eUndefined) {
        references[index] = vk::AttachmentReference{index, vk::ImageLayout::eDepthStencilAttachmentOptimal};
        attachments[index] =
        {
            {},
            depth_format,
            static_cast<vk::SampleCountFlagBits>(multisamples),
            load_op,
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,
            vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eDepthStencilAttachmentOptimal,
            vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        subpass.setPDepthStencilAttachment(&references[index++]);
    }

    std::array<vk::SubpassDescription, 1> subpasses = { subpass };
    vk::RenderPassCreateInfo renderpass_info({}, attachments, subpasses);

    auto renderpass = g_vk_instace->GetDevice().createRenderPassUnique(renderpass_info);
    vk::RenderPass handle = renderpass.get();

    renderpass_cache.emplace(key, std::move(renderpass));
    return handle;
}

Pipeline::Pipeline() { Clear(); }

void Pipeline::Clear()
{
    m_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    m_ci.pNext = nullptr;
    m_ci.flags = 0;
    m_ci.pSetLayouts = nullptr;
    m_ci.setLayoutCount = 0;
    m_ci.pPushConstantRanges = nullptr;
    m_ci.pushConstantRangeCount = 0;
}

void Pipeline::Build() {
    VkPipelineLayout layout;
    VkResult res = vkCreatePipelineLayout(device, &m_ci, nullptr, &layout);
    if (res != VK_SUCCESS)
    {
        LOG_VULKAN_ERROR(res, "vkCreatePipelineLayout() failed: ");
        return VK_NULL_HANDLE;
    }

    Clear();
    return layout;
}

void Pipeline::AddDescriptorSet(VkDescriptorSetLayout layout)
{
    pxAssert(m_ci.setLayoutCount < MAX_SETS);

    m_sets[m_ci.setLayoutCount] = layout;

    m_ci.setLayoutCount++;
    m_ci.pSetLayouts = m_sets.data();
}

void Pipeline::AddPushConstants(VkShaderStageFlags stages, u32 offset, u32 size)
{
    pxAssert(m_ci.pushConstantRangeCount < MAX_PUSH_CONSTANTS);

    VkPushConstantRange& r = m_push_constants[m_ci.pushConstantRangeCount];
    r.stageFlags = stages;
    r.offset = offset;
    r.size = size;

    m_ci.pushConstantRangeCount++;
    m_ci.pPushConstantRanges = m_push_constants.data();
}

GraphicsPipelineBuilder::GraphicsPipelineBuilder() { Clear(); }

void GraphicsPipelineBuilder::Clear()
{
    m_ci = {};
    m_ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    m_shader_stages = {};

    m_vertex_input_state = {};
    m_vertex_input_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    m_ci.pVertexInputState = &m_vertex_input_state;
    m_vertex_attributes = {};
    m_vertex_buffers = {};

    m_input_assembly = {};
    m_input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

    m_rasterization_state = {};
    m_rasterization_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    m_rasterization_state.lineWidth = 1.0f;
    m_depth_state = {};
    m_depth_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    m_blend_state = {};
    m_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    m_blend_attachments = {};

    m_viewport_state = {};
    m_viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    m_viewport = {};
    m_scissor = {};

    m_dynamic_state = {};
    m_dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    m_dynamic_state_values = {};

    m_multisample_state = {};
    m_multisample_state.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

    m_provoking_vertex = {};
    m_provoking_vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT;

    // set defaults
    SetNoCullRasterizationState();
    SetNoDepthTestState();
    SetNoBlendingState();
    SetPrimitiveTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    // have to be specified even if dynamic
    SetViewport(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f);
    SetScissorRect(0, 0, 1, 1);
    SetMultisamples(VK_SAMPLE_COUNT_1_BIT);
}

VkPipeline GraphicsPipelineBuilder::Create(VkDevice device, VkPipelineCache pipeline_cache, bool clear /* = true */)
{
    VkPipeline pipeline;
    VkResult res = vkCreateGraphicsPipelines(device, pipeline_cache, 1, &m_ci, nullptr, &pipeline);
    if (res != VK_SUCCESS)
    {
        LOG_VULKAN_ERROR(res, "vkCreateGraphicsPipelines() failed: ");
        return VK_NULL_HANDLE;
    }

    if (clear)
        Clear();

    return pipeline;
}

void GraphicsPipelineBuilder::SetShaderStage(
    VkShaderStageFlagBits stage, VkShaderModule module, const char* entry_point)
{
    pxAssert(m_ci.stageCount < MAX_SHADER_STAGES);

    u32 index = 0;
    for (; index < m_ci.stageCount; index++)
    {
        if (m_shader_stages[index].stage == stage)
            break;
    }
    if (index == m_ci.stageCount)
    {
        m_ci.stageCount++;
        m_ci.pStages = m_shader_stages.data();
    }

    VkPipelineShaderStageCreateInfo& s = m_shader_stages[index];
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    s.stage = stage;
    s.module = module;
    s.pName = entry_point;
}

void GraphicsPipelineBuilder::AddVertexBuffer(
    u32 binding, u32 stride, VkVertexInputRate input_rate /*= VK_VERTEX_INPUT_RATE_VERTEX*/)
{
    pxAssert(m_vertex_input_state.vertexAttributeDescriptionCount < MAX_VERTEX_BUFFERS);

    VkVertexInputBindingDescription& b = m_vertex_buffers[m_vertex_input_state.vertexBindingDescriptionCount];
    b.binding = binding;
    b.stride = stride;
    b.inputRate = input_rate;

    m_vertex_input_state.vertexBindingDescriptionCount++;
    m_vertex_input_state.pVertexBindingDescriptions = m_vertex_buffers.data();
    m_ci.pVertexInputState = &m_vertex_input_state;
}

void GraphicsPipelineBuilder::AddVertexAttribute(u32 location, u32 binding, VkFormat format, u32 offset)
{
    pxAssert(m_vertex_input_state.vertexAttributeDescriptionCount < MAX_VERTEX_BUFFERS);

    VkVertexInputAttributeDescription& a =
        m_vertex_attributes[m_vertex_input_state.vertexAttributeDescriptionCount];
    a.location = location;
    a.binding = binding;
    a.format = format;
    a.offset = offset;

    m_vertex_input_state.vertexAttributeDescriptionCount++;
    m_vertex_input_state.pVertexAttributeDescriptions = m_vertex_attributes.data();
    m_ci.pVertexInputState = &m_vertex_input_state;
}

void GraphicsPipelineBuilder::SetPrimitiveTopology(
    VkPrimitiveTopology topology, bool enable_primitive_restart /*= false*/)
{
    m_input_assembly.topology = topology;
    m_input_assembly.primitiveRestartEnable = enable_primitive_restart;

    m_ci.pInputAssemblyState = &m_input_assembly;
}

void GraphicsPipelineBuilder::SetRasterizationState(
    VkPolygonMode polygon_mode, VkCullModeFlags cull_mode, VkFrontFace front_face)
{
    m_rasterization_state.polygonMode = polygon_mode;
    m_rasterization_state.cullMode = cull_mode;
    m_rasterization_state.frontFace = front_face;

    m_ci.pRasterizationState = &m_rasterization_state;
}

void GraphicsPipelineBuilder::SetLineWidth(float width) { m_rasterization_state.lineWidth = width; }

void GraphicsPipelineBuilder::SetMultisamples(u32 multisamples, bool per_sample_shading)
{
    m_multisample_state.rasterizationSamples = static_cast<VkSampleCountFlagBits>(multisamples);
    m_multisample_state.sampleShadingEnable = per_sample_shading;
    m_multisample_state.minSampleShading = (multisamples > 1) ? 1.0f : 0.0f;
}

void GraphicsPipelineBuilder::SetNoCullRasterizationState()
{
    SetRasterizationState(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
}

void GraphicsPipelineBuilder::SetDepthState(bool depth_test, bool depth_write, VkCompareOp compare_op)
{
    m_depth_state.depthTestEnable = depth_test;
    m_depth_state.depthWriteEnable = depth_write;
    m_depth_state.depthCompareOp = compare_op;

    m_ci.pDepthStencilState = &m_depth_state;
}

void GraphicsPipelineBuilder::SetStencilState(
    bool stencil_test, const VkStencilOpState& front, const VkStencilOpState& back)
{
    m_depth_state.stencilTestEnable = stencil_test;
    m_depth_state.front = front;
    m_depth_state.back = back;
}

void GraphicsPipelineBuilder::SetNoStencilState()
{
    m_depth_state.stencilTestEnable = VK_FALSE;
    m_depth_state.front = {};
    m_depth_state.back = {};
}

void GraphicsPipelineBuilder::SetNoDepthTestState() { SetDepthState(false, false, VK_COMPARE_OP_ALWAYS); }

void GraphicsPipelineBuilder::SetBlendConstants(float r, float g, float b, float a)
{
    m_blend_state.blendConstants[0] = r;
    m_blend_state.blendConstants[1] = g;
    m_blend_state.blendConstants[2] = b;
    m_blend_state.blendConstants[3] = a;
    m_ci.pColorBlendState = &m_blend_state;
}

void GraphicsPipelineBuilder::AddBlendAttachment(bool blend_enable, VkBlendFactor src_factor,
    VkBlendFactor dst_factor, VkBlendOp op, VkBlendFactor alpha_src_factor, VkBlendFactor alpha_dst_factor,
    VkBlendOp alpha_op,
    VkColorComponentFlags
        write_mask /* = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT */)
{
    pxAssert(m_blend_state.attachmentCount < MAX_ATTACHMENTS);

    VkPipelineColorBlendAttachmentState& bs = m_blend_attachments[m_blend_state.attachmentCount];
    bs.blendEnable = blend_enable;
    bs.srcColorBlendFactor = src_factor;
    bs.dstColorBlendFactor = dst_factor;
    bs.colorBlendOp = op;
    bs.srcAlphaBlendFactor = alpha_src_factor;
    bs.dstAlphaBlendFactor = alpha_dst_factor;
    bs.alphaBlendOp = alpha_op;
    bs.colorWriteMask = write_mask;

    m_blend_state.attachmentCount++;
    m_blend_state.pAttachments = m_blend_attachments.data();
    m_ci.pColorBlendState = &m_blend_state;
}

void GraphicsPipelineBuilder::SetBlendAttachment(u32 attachment, bool blend_enable, VkBlendFactor src_factor,
    VkBlendFactor dst_factor, VkBlendOp op, VkBlendFactor alpha_src_factor, VkBlendFactor alpha_dst_factor,
    VkBlendOp alpha_op,
    VkColorComponentFlags
        write_mask /*= VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT*/)
{
    pxAssert(attachment < MAX_ATTACHMENTS);

    VkPipelineColorBlendAttachmentState& bs = m_blend_attachments[attachment];
    bs.blendEnable = blend_enable;
    bs.srcColorBlendFactor = src_factor;
    bs.dstColorBlendFactor = dst_factor;
    bs.colorBlendOp = op;
    bs.srcAlphaBlendFactor = alpha_src_factor;
    bs.dstAlphaBlendFactor = alpha_dst_factor;
    bs.alphaBlendOp = alpha_op;
    bs.colorWriteMask = write_mask;

    if (attachment >= m_blend_state.attachmentCount)
    {
        m_blend_state.attachmentCount = attachment + 1u;
        m_blend_state.pAttachments = m_blend_attachments.data();
        m_ci.pColorBlendState = &m_blend_state;
    }
}

void GraphicsPipelineBuilder::AddBlendFlags(u32 flags)
{
    m_blend_state.flags |= flags;
}

void GraphicsPipelineBuilder::ClearBlendAttachments()
{
    m_blend_attachments = {};
    m_blend_state.attachmentCount = 0;
}

void GraphicsPipelineBuilder::SetNoBlendingState()
{
    ClearBlendAttachments();
    SetBlendAttachment(0, false, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE,
        VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);
}

void GraphicsPipelineBuilder::AddDynamicState(VkDynamicState state)
{
    pxAssert(m_dynamic_state.dynamicStateCount < MAX_DYNAMIC_STATE);

    m_dynamic_state_values[m_dynamic_state.dynamicStateCount] = state;
    m_dynamic_state.dynamicStateCount++;
    m_dynamic_state.pDynamicStates = m_dynamic_state_values.data();
    m_ci.pDynamicState = &m_dynamic_state;
}

void GraphicsPipelineBuilder::SetDynamicViewportAndScissorState()
{
    AddDynamicState(VK_DYNAMIC_STATE_VIEWPORT);
    AddDynamicState(VK_DYNAMIC_STATE_SCISSOR);
}

void GraphicsPipelineBuilder::SetViewport(
    float x, float y, float width, float height, float min_depth, float max_depth)
{
    m_viewport.x = x;
    m_viewport.y = y;
    m_viewport.width = width;
    m_viewport.height = height;
    m_viewport.minDepth = min_depth;
    m_viewport.maxDepth = max_depth;

    m_viewport_state.pViewports = &m_viewport;
    m_viewport_state.viewportCount = 1u;
    m_ci.pViewportState = &m_viewport_state;
}

void GraphicsPipelineBuilder::SetScissorRect(s32 x, s32 y, u32 width, u32 height)
{
    m_scissor.offset.x = x;
    m_scissor.offset.y = y;
    m_scissor.extent.width = width;
    m_scissor.extent.height = height;

    m_viewport_state.pScissors = &m_scissor;
    m_viewport_state.scissorCount = 1u;
    m_ci.pViewportState = &m_viewport_state;
}

void GraphicsPipelineBuilder::SetMultisamples(VkSampleCountFlagBits samples)
{
    m_multisample_state.rasterizationSamples = samples;
    m_ci.pMultisampleState = &m_multisample_state;
}

void GraphicsPipelineBuilder::SetPipelineLayout(VkPipelineLayout layout) { m_ci.layout = layout; }

void GraphicsPipelineBuilder::SetRenderPass(VkRenderPass render_pass, u32 subpass)
{
    m_ci.renderPass = render_pass;
    m_ci.subpass = subpass;
}

void GraphicsPipelineBuilder::SetProvokingVertex(VkProvokingVertexModeEXT mode)
{
    Util::AddPointerToChain(&m_rasterization_state, &m_provoking_vertex);

    m_provoking_vertex.provokingVertexMode = mode;
}

}  // namespace Vulkan
