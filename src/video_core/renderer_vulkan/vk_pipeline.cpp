// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/logging/log.h"
#include "video_core/common/pool_manager.h"
#include "video_core/renderer_vulkan/pica_to_vulkan.h"
#include "video_core/renderer_vulkan/vk_pipeline.h"
#include "video_core/renderer_vulkan/vk_shader.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

namespace VideoCore::Vulkan {

vk::ShaderStageFlags ToVkStageFlags(BindingType type) {
    vk::ShaderStageFlags flags;
    switch (type) {
    case BindingType::Sampler:
    case BindingType::Texture:
    case BindingType::TexelBuffer:
        flags = vk::ShaderStageFlagBits::eFragment;
        break;
    case BindingType::StorageImage:
    case BindingType::Uniform:
    case BindingType::UniformDynamic:
        flags = vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eGeometry |
                vk::ShaderStageFlagBits::eCompute;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown descriptor type!");
    }

    return flags;
}

vk::DescriptorType ToVkDescriptorType(BindingType type) {
    switch (type) {
    case BindingType::Uniform:
        return vk::DescriptorType::eUniformBuffer;
    case BindingType::UniformDynamic:
        return vk::DescriptorType::eUniformBufferDynamic;
    case BindingType::TexelBuffer:
        return vk::DescriptorType::eUniformTexelBuffer;
    case BindingType::Texture:
        return vk::DescriptorType::eSampledImage;
    case BindingType::Sampler:
        return vk::DescriptorType::eSampler;
    case BindingType::StorageImage:
        return vk::DescriptorType::eStorageImage;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown descriptor type!");
        UNREACHABLE();
    }
}

u32 AttribBytes(VertexAttribute attrib) {
    switch (attrib.type) {
    case AttribType::Float:
        return sizeof(float) * attrib.size;
    case AttribType::Int:
        return sizeof(u32) * attrib.size;
    case AttribType::Short:
        return sizeof(u16) * attrib.size;
    case AttribType::Byte:
    case AttribType::Ubyte:
        return sizeof(u8) * attrib.size;
    }
}

vk::Format ToVkAttributeFormat(VertexAttribute attrib) {
    switch (attrib.type) {
    case AttribType::Float:
        switch (attrib.size) {
        case 1: return vk::Format::eR32Sfloat;
        case 2: return vk::Format::eR32G32Sfloat;
        case 3: return vk::Format::eR32G32B32Sfloat;
        case 4: return vk::Format::eR32G32B32A32Sfloat;
        }
    default:
        LOG_CRITICAL(Render_Vulkan, "Unimplemented vertex attribute format!");
        UNREACHABLE();
    }
}

vk::ShaderStageFlagBits ToVkShaderStage(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Vertex:
        return vk::ShaderStageFlagBits::eVertex;
    case ShaderStage::Fragment:
        return vk::ShaderStageFlagBits::eFragment;
    case ShaderStage::Geometry:
        return vk::ShaderStageFlagBits::eGeometry;
    case ShaderStage::Compute:
        return vk::ShaderStageFlagBits::eCompute;
    default:
        LOG_CRITICAL(Render_Vulkan, "Undefined shader stage!");
        UNREACHABLE();
    }
}

PipelineOwner::PipelineOwner(Instance& instance, PipelineLayoutInfo info) :
    instance(instance), set_layout_count(info.group_count) {

    // Used as temp storage for CreateDescriptorSet
    std::array<vk::DescriptorSetLayoutBinding, MAX_BINDINGS_IN_GROUP> set_bindings;
    std::array<vk::DescriptorUpdateTemplateEntry, MAX_BINDINGS_IN_GROUP> update_entries;

    vk::Device device = instance.GetDevice();
    for (u32 set = 0; set < set_layout_count; set++) {
        auto& group = info.binding_groups[set];

        u32 binding = 0;
        while (group.Value(binding) != BindingType::None) {
            const BindingType type = group.Value(binding);
            set_bindings[binding] = vk::DescriptorSetLayoutBinding{
                .binding = binding,
                .descriptorType = ToVkDescriptorType(type),
                .descriptorCount = 1,
                .stageFlags = ToVkStageFlags(type)
            };

            // Also create update template to speed up descriptor writes
            update_entries[binding] = vk::DescriptorUpdateTemplateEntry{
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = ToVkDescriptorType(type),
                .offset = binding * sizeof(DescriptorData),
                .stride = 0
            };

            binding++;
        }

        const vk::DescriptorSetLayoutCreateInfo layout_info = {
            .bindingCount = binding,
            .pBindings = set_bindings.data()
        };

        // Create descriptor set layout
        set_layouts[set] = device.createDescriptorSetLayout(layout_info);

        const vk::DescriptorUpdateTemplateCreateInfo template_info = {
            .descriptorUpdateEntryCount = binding,
            .pDescriptorUpdateEntries = update_entries.data(),
            .descriptorSetLayout = set_layouts[set]
        };

        // Create descriptor set update template
        update_templates[set] = device.createDescriptorUpdateTemplate(template_info);
    }

    // Create pipeline layout
    const vk::PushConstantRange range = {
        .stageFlags = vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = info.push_constant_block_size
    };

    bool push_constants = info.push_constant_block_size > 0;
    const u32 range_count = push_constants ? 1u : 0u;

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = set_layout_count,
        .pSetLayouts = set_layouts.data(),
        .pushConstantRangeCount = range_count,
        .pPushConstantRanges = &range
    };

    pipeline_layout = device.createPipelineLayout(layout_info);
}

PipelineOwner::~PipelineOwner() {
    vk::Device device = instance.GetDevice();
    device.destroyPipelineLayout(pipeline_layout);

    u32 i = 0;
    while (set_layouts[i] && update_templates[i]) {
        device.destroyDescriptorSetLayout(set_layouts[i]);
        device.destroyDescriptorUpdateTemplate(update_templates[i++]);
    }
}

Pipeline::Pipeline(Instance& instance, CommandScheduler& scheduler, PoolManager& pool_manager, PipelineOwner& owner,
                   PipelineType type, PipelineInfo info, vk::RenderPass renderpass, vk::PipelineCache cache) :
    PipelineBase(type, info), instance(instance), scheduler(scheduler), pool_manager(pool_manager), owner(owner) {

    vk::Device device = instance.GetDevice();

    u32 shader_count = 0;
    std::array<vk::PipelineShaderStageCreateInfo, MAX_SHADER_STAGES> shader_stages;
    for (int i = 0; i < info.shaders.size(); i++) {
        auto& shader = info.shaders[i];
        if (!shader.IsValid()) {
            continue;
        }

        Shader* vk_shader = static_cast<Shader*>(shader.Get());
        shader_stages[shader_count++] = vk::PipelineShaderStageCreateInfo{
            .stage = ToVkShaderStage(shader->GetStage()),
            .module = vk_shader->GetHandle(),
            .pName = "main"
        };
    }

    if (type == PipelineType::Graphics) {

        /**
         * Vulkan doesn't intuitively support fixed attributes. To avoid duplicating the data and increasing
         * data upload, when the fixed flag is true, we specify VK_VERTEX_INPUT_RATE_INSTANCE as the input rate.
         * Since 1 instance is all we render, the shader will always read the single attribute.
         */
        std::array<vk::VertexInputBindingDescription, MAX_VERTEX_BINDINGS> bindings;
        for (u32 i = 0; i < info.vertex_layout.binding_count; i++) {
            const auto& binding = info.vertex_layout.bindings[i];
            bindings[i] = vk::VertexInputBindingDescription{
                .binding = binding.binding,
                .stride = binding.stride,
                .inputRate = binding.fixed.Value() ? vk::VertexInputRate::eInstance
                                                   : vk::VertexInputRate::eVertex
            };
        }

        // Populate vertex attribute structures
        std::array<vk::VertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES> attributes;
        for (u32 i = 0; i < info.vertex_layout.attribute_count; i++) {
            const auto& attr = info.vertex_layout.attributes[i];
            attributes[i] = vk::VertexInputAttributeDescription{
                .location = attr.location,
                .binding = attr.binding,
                .format = ToVkAttributeFormat(attr),
                .offset = attr.offset
            };
        }

        const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
            .vertexBindingDescriptionCount = info.vertex_layout.binding_count,
            .pVertexBindingDescriptions = bindings.data(),
            .vertexAttributeDescriptionCount = info.vertex_layout.attribute_count,
            .pVertexAttributeDescriptions = attributes.data()
        };

        const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
            .topology = PicaToVK::PrimitiveTopology(info.rasterization.topology),
            .primitiveRestartEnable = false
        };

        const vk::PipelineRasterizationStateCreateInfo raster_state = {
            .depthClampEnable = false,
            .rasterizerDiscardEnable = false,
            .cullMode = PicaToVK::CullMode(info.rasterization.cull_mode),
            .frontFace = PicaToVK::FrontFace(info.rasterization.cull_mode),
            .depthBiasEnable = false,
            .lineWidth = 1.0f
        };

        const vk::PipelineMultisampleStateCreateInfo multisampling = {
            .rasterizationSamples  = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = false
        };

        const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
            .blendEnable = info.blending.blend_enable.Value(),
            .srcColorBlendFactor = PicaToVK::BlendFunc(info.blending.src_color_blend_factor),
            .dstColorBlendFactor = PicaToVK::BlendFunc(info.blending.dst_color_blend_factor),
            .colorBlendOp = PicaToVK::BlendEquation(info.blending.color_blend_eq),
            .srcAlphaBlendFactor = PicaToVK::BlendFunc(info.blending.src_alpha_blend_factor),
            .dstAlphaBlendFactor = PicaToVK::BlendFunc(info.blending.dst_alpha_blend_factor),
            .alphaBlendOp = PicaToVK::BlendEquation(info.blending.alpha_blend_eq),
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        const vk::PipelineColorBlendStateCreateInfo color_blending = {
            .logicOpEnable = info.blending.logic_op_enable.Value(),
            .logicOp = PicaToVK::LogicOp(info.blending.logic_op), // TODO
            .attachmentCount = 1,
            .pAttachments = &colorblend_attachment,
            .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f}
        };

        const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
        const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
        const vk::PipelineViewportStateCreateInfo viewport_info = {
            .viewportCount = 1,
            .pViewports = &placeholder_viewport,
            .scissorCount = 1,
            .pScissors = &placeholder_scissor,
        };

        const bool extended_dynamic_states = instance.IsExtendedDynamicStateSupported();
        const std::array dynamic_states = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eLineWidth,
            vk::DynamicState::eStencilCompareMask,
            vk::DynamicState::eStencilWriteMask,
            vk::DynamicState::eStencilReference,
            // VK_EXT_extended_dynamic_state
            vk::DynamicState::eCullModeEXT,
            vk::DynamicState::eDepthCompareOpEXT,
            vk::DynamicState::eDepthTestEnableEXT,
            vk::DynamicState::eDepthWriteEnableEXT,
            vk::DynamicState::eFrontFaceEXT,
            vk::DynamicState::ePrimitiveTopologyEXT,
            vk::DynamicState::eStencilOpEXT,
            vk::DynamicState::eStencilTestEnableEXT,
        };

        const vk::PipelineDynamicStateCreateInfo dynamic_info = {
            .dynamicStateCount = extended_dynamic_states ? 14u : 6u,
            .pDynamicStates = dynamic_states.data()
        };

        const vk::StencilOpState stencil_op_state = {
            .failOp = PicaToVK::StencilOp(info.depth_stencil.stencil_fail_op),
            .passOp = PicaToVK::StencilOp(info.depth_stencil.stencil_pass_op),
            .depthFailOp = PicaToVK::StencilOp(info.depth_stencil.stencil_depth_fail_op),
            .compareOp = PicaToVK::CompareFunc(info.depth_stencil.stencil_compare_op),
            .compareMask = info.depth_stencil.stencil_compare_mask,
            .writeMask = info.depth_stencil.stencil_write_mask,
            .reference = info.depth_stencil.stencil_reference
        };

        const vk::PipelineDepthStencilStateCreateInfo depth_info = {
            .depthTestEnable = static_cast<u32>(info.depth_stencil.depth_test_enable.Value()),
            .depthWriteEnable = static_cast<u32>(info.depth_stencil.depth_write_enable.Value()),
            .depthCompareOp = PicaToVK::CompareFunc(info.depth_stencil.depth_compare_op),
            .depthBoundsTestEnable = false,
            .stencilTestEnable = static_cast<u32>(info.depth_stencil.stencil_test_enable.Value()),
            .front = stencil_op_state,
            .back = stencil_op_state
        };

        const vk::GraphicsPipelineCreateInfo pipeline_info = {
            .stageCount = shader_count,
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_info,
            .layout = owner.GetLayout(),
            .renderPass = renderpass
        };

        if (auto result = device.createGraphicsPipeline(cache, pipeline_info);
                result.result == vk::Result::eSuccess) {
            pipeline = result.value;
        } else {
           LOG_CRITICAL(Render_Vulkan, "Graphics pipeline creation failed!");
           UNREACHABLE();
        }
    } else { // Compute pipeline
        ASSERT(shader_count == 1);
        const vk::ComputePipelineCreateInfo pipeline_info = {
            .stage = shader_stages[0],
            .layout = owner.GetLayout()
        };

        if (auto result = device.createComputePipeline(cache, pipeline_info); result.result == vk::Result::eSuccess) {
            pipeline = result.value;
        } else {
           LOG_CRITICAL(Render_Vulkan, "Compute pipeline creation failed!");
           UNREACHABLE();
        }

    }
}

Pipeline::~Pipeline() {
    vk::Device device = instance.GetDevice();
    device.destroyPipeline(pipeline);
}

void Pipeline::Free() {
    pool_manager.Free<Pipeline>(this);
}

void Pipeline::BindTexture(u32 group, u32 slot, TextureHandle handle) {
    Texture* texture = static_cast<Texture*>(handle.Get());

    // NOTE: To prevent validation errors when using the image without uploading
    // transition it now to VK_IMAGE_LAYOUT_SHADER_READONLY_OPTIMAL
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    texture->Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);

    const DescriptorData data = {
        .image_info = vk::DescriptorImageInfo{
            .imageView = texture->GetView(),
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        }
    };

    owner.SetBinding(group, slot, data);
}

void Pipeline::BindBuffer(u32 group, u32 slot, BufferHandle handle, u32 offset, u32 range, u32 view) {
    Buffer* buffer = static_cast<Buffer*>(handle.Get());

    // Texel buffers are bound with their views
    // TODO: Support variable binding range?
    if (buffer->GetUsage() == BufferUsage::Texel) {
        const DescriptorData data = {
            .buffer_view = buffer->GetView(view)
        };

        owner.SetBinding(group, slot, data);
    } else {
        const DescriptorData data = {
            .buffer_info = vk::DescriptorBufferInfo{
                .buffer = buffer->GetHandle(),
                .offset = offset,
                .range = (range == WHOLE_SIZE ? buffer->GetCapacity() : range)
            }
        };

        owner.SetBinding(group, slot, data);
    }
}

void Pipeline::BindSampler(u32 group, u32 slot, SamplerHandle handle) {
    Sampler* sampler = static_cast<Sampler*>(handle.Get());

    const DescriptorData data = {
        .image_info = vk::DescriptorImageInfo{
            .sampler = sampler->GetHandle()
        }
    };

    owner.SetBinding(group, slot, data);
}

void Pipeline::BindPushConstant(std::span<const std::byte> data) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.pushConstants(owner.GetLayout(),
                                 vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                                 0, data.size(), data.data());
}

// Viewport and scissor are always dynamic
void Pipeline::SetViewport(float x, float y, float width, float height) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setViewport(0, vk::Viewport{x, y, width, height, 0.f, 1.f});
}

void Pipeline::SetScissor(s32 x, s32 y, u32 width, u32 height) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setScissor(0, vk::Rect2D{{x, y}, {width, height}});
}

void Pipeline::ApplyDynamic(const PipelineInfo& info) {

}

} // namespace VideoCore::Vulkan
