// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <span>
#include <shaderc/shaderc.hpp>
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_resource_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

// Define bitwise operators for DirtyFlags enum
DirtyFlags operator |=(DirtyFlags lhs, DirtyFlags rhs) {
    return static_cast<DirtyFlags> (
        static_cast<unsigned>(lhs) |
        static_cast<unsigned>(rhs)
    );
}

bool operator &(DirtyFlags lhs, DirtyFlags rhs) {
    return static_cast<u32>(lhs) &
           static_cast<u32>(rhs);
}

bool operator >(BindingID lhs, BindingID rhs) {
    return static_cast<u32>(lhs) &
           static_cast<u32>(rhs);
}

bool operator <(BindingID lhs, BindingID rhs) {
    return static_cast<u32>(lhs) &
           static_cast<u32>(rhs);
}

void VulkanState::Create() {
    // Create a dummy texture which can be used in place of a real binding.
    VKTexture::Info info = {
        .width = 1,
        .height = 1,
        .format = vk::Format::eR8G8B8A8Unorm,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D
    };

    dummy_texture.Create(info);
    dummy_texture.Transition(vk::ImageLayout::eShaderReadOnlyOptimal);

    // Create descriptor pool
    // TODO: Choose sizes more wisely
    const std::array<vk::DescriptorPoolSize, 3> pool_sizes{{
        { vk::DescriptorType::eUniformBuffer, 32 },
        { vk::DescriptorType::eCombinedImageSampler, 32 },
        { vk::DescriptorType::eStorageTexelBuffer, 32 },
    }};

    auto& device = g_vk_instace->GetDevice();
    vk::DescriptorPoolCreateInfo pool_create_info({}, 1024, pool_sizes);
    desc_pool = device.createDescriptorPoolUnique(pool_create_info);

    // Create descriptor sets
    auto& layouts = g_vk_res_cache->GetDescriptorLayouts();
    vk::DescriptorSetAllocateInfo alloc_info(desc_pool.get(), layouts);
    descriptor_sets = device.allocateDescriptorSetsUnique(alloc_info);

    // Create texture sampler
    auto props = g_vk_instace->GetPhysicalDevice().getProperties();
    vk::SamplerCreateInfo sampler_info{
        {}, vk::Filter::eNearest, vk::Filter::eNearest,
        vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        {}, true, props.limits.maxSamplerAnisotropy,
        false, vk::CompareOp::eAlways, {}, {},
        vk::BorderColor::eIntOpaqueBlack, false
    };
    sampler = g_vk_instace->GetDevice().createSamplerUnique(sampler_info);


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

    // Compile trivial vertex shader
    auto source = GenerateTrivialVertexShader(true);
    MakeShader(source.code, vk::ShaderStageFlagBits::eVertex);

    dirty_flags |= DirtyFlags::All;
}

void VulkanState::SetVertexBuffer(VKBuffer* buffer, vk::DeviceSize offset) {
    if (vertex_buffer == buffer) {
        return;
    }

    vertex_buffer = buffer;
    vertex_offset = offset;
    dirty_flags |= DirtyFlags::VertexBuffer;
}

void VulkanState::SetUniformBuffer(BindingID id, VKBuffer* buffer, u32 offset, u32 size) {
    assert(id < BindingID::Tex0);
    u32 index = static_cast<u32>(id);

    auto& binding = bindings[index];
    auto old_buffer = std::get<VKBuffer*>(binding.resource);
    if (old_buffer != buffer) {
        binding.resource = buffer;
        dirty_flags |= DirtyFlags::Uniform;
        binding.dirty = true;
    }
}

void VulkanState::SetTexture(BindingID id, VKTexture* image) {
    assert(id > BindingID::PicaUniform && id < BindingID::LutLF);
    u32 index = static_cast<u32>(id);

    auto& binding = bindings[index];
    auto old_image = std::get<VKTexture*>(binding.resource);
    if (old_image != image) {
        binding.resource = image;
        dirty_flags |= DirtyFlags::Texture;
        binding.dirty = true;
    }
}

void VulkanState::SetTexelBuffer(BindingID id, VKBuffer* buffer, vk::Format view_format) {
    assert(id > BindingID::TexCube);
    u32 index = static_cast<u32>(id);

    auto& binding = bindings[index];
    auto old_buffer = std::get<VKBuffer*>(binding.resource);
    if (old_buffer != buffer) {
        auto& device = g_vk_instace->GetDevice();

        binding.resource = buffer;
        binding.buffer_view = device.createBufferViewUnique({{}, buffer->GetBuffer(), view_format});
        dirty_flags |= DirtyFlags::TexelBuffer;
        binding.dirty = true;
    }
}

void VulkanState::UnbindTexture(VKTexture* image) {
    for (auto i = u32(BindingID::Tex0); i <= u32(BindingID::TexCube); i++) {
        auto current_image = std::get<VKTexture*>(bindings[i].resource);
        if (current_image == image) {
            UnbindTexture(i);
        }
    }
}

void VulkanState::UnbindTexture(u32 index) {
    bindings[index].resource = &dummy_texture;
    dirty_flags |= DirtyFlags::Texture;
}

void VulkanState::PushRenderTargets(VKTexture* color, VKTexture* depth_stencil) {
    color_attachment = color;
    depth_attachment = depth_stencil;
}

void VulkanState::SetRenderArea(vk::Rect2D new_render_area) {
    render_area = new_render_area;
}

void VulkanState::BeginRendering() {
    if (rendering) {
        return;
    }

    // Make sure attachments are in optimal layout
    color_attachment->Transition(vk::ImageLayout::eColorAttachmentOptimal);
    depth_attachment->Transition(vk::ImageLayout::eDepthStencilAttachmentOptimal);

    // Begin rendering
    vk::RenderingAttachmentInfoKHR color_info(color_attachment->GetView(), color_attachment->GetLayout());
    vk::RenderingAttachmentInfoKHR depth_stencil_info(depth_attachment->GetView(), depth_attachment->GetLayout());

    vk::RenderingInfo render_info
    (
        {}, render_area, 1, {},
        color_info,
        &depth_stencil_info,
        &depth_stencil_info
    );

    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.beginRendering(render_info);
    rendering = true;
}

void VulkanState::EndRendering() {
    if (!rendering) {
        return;
    }

    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.endRendering();
    rendering = false;
}

void VulkanState::SetViewport(vk::Viewport new_viewport) {
    if (new_viewport != viewport) {
        viewport = new_viewport;
        dirty_flags |= DirtyFlags::Viewport;
    }
}

void VulkanState::SetScissor(vk::Rect2D new_scissor) {
    if (new_scissor != scissor) {
        scissor = new_scissor;
        dirty_flags |= DirtyFlags::Scissor;
    }
}

void VulkanState::SetCullMode(vk::CullModeFlags flags) {
    if (cull_mode != flags) {
        cull_mode = flags;
        dirty_flags |= DirtyFlags::CullMode;
    }
}

void VulkanState::SetFrontFace(vk::FrontFace face) {
    if (front_face != face) {
        front_face = face;
        dirty_flags |= DirtyFlags::FrontFace;
    }
}

void VulkanState::SetLogicOp(vk::LogicOp new_logic_op) {
    if (logic_op != new_logic_op) {
        logic_op = new_logic_op;
        dirty_flags |= DirtyFlags::LogicOp;
    }
}

void VulkanState::SetColorMask(bool red, bool green, bool blue, bool alpha) {
    auto mask = static_cast<vk::ColorComponentFlags>(red | (green << 1) | (blue << 2) | (alpha << 3));
    pipeline_key.blend_config.setColorWriteMask(mask);
}

void VulkanState::SetBlendEnable(bool enable) {
    pipeline_key.blend_config.setBlendEnable(enable);
}

void VulkanState::SetBlendCostants(float red, float green, float blue, float alpha) {
    std::array<float, 4> color = { red, green, blue, alpha };
    if (color != blend_constants) {
        blend_constants = color;
        dirty_flags = DirtyFlags::BlendConsts;
    }
}

void VulkanState::SetStencilWrite(u32 mask) {
    if (mask != stencil_write_mask) {
        stencil_write_mask = mask;
        dirty_flags |= DirtyFlags::StencilMask;
    }
}

void VulkanState::SetStencilInput(u32 mask) {
    if (mask != stencil_input_mask) {
        stencil_input_mask = mask;
        dirty_flags |= DirtyFlags::StencilMask;
    }
}

void VulkanState::SetStencilTest(bool enable, vk::StencilOp fail, vk::StencilOp pass, vk::StencilOp depth_fail,
                               vk::CompareOp compare, u32 ref) {
    stencil_enabled = enable;
    stencil_ref = ref;
    fail_op = fail;
    pass_op = pass;
    depth_fail_op = depth_fail;
    compare_op = compare;
    dirty_flags |= DirtyFlags::Stencil;
}

void VulkanState::SetDepthWrite(bool enable) {
    if (enable != depth_writes) {
        depth_writes = enable;
        dirty_flags |= DirtyFlags::DepthWrite;
    }
}

void VulkanState::SetDepthTest(bool enable, vk::CompareOp compare) {
    depth_enabled = enable;
    test_func = compare;
    dirty_flags |= DirtyFlags::DepthTest;
}

void VulkanState::SetBlendOp(vk::BlendOp rgb_op, vk::BlendOp alpha_op, vk::BlendFactor src_color,
                             vk::BlendFactor dst_color, vk::BlendFactor src_alpha, vk::BlendFactor dst_alpha) {
    auto& blend = pipeline_key.blend_config;
    blend.setColorBlendOp(rgb_op);
    blend.setAlphaBlendOp(alpha_op);
    blend.setSrcColorBlendFactor(src_color);
    blend.setDstColorBlendFactor(dst_color);
    blend.setSrcAlphaBlendFactor(src_alpha);
    blend.setDstAlphaBlendFactor(dst_alpha);
}

void VulkanState::SetFragmentShader(const Pica::Regs& regs) {
    vk::Pipeline pipeline;
    pipeline_key.fragment_config = PicaFSConfig::BuildFromRegs(regs);
    auto it1 = pipelines.find(pipeline_key);

    do {
        // Try to use an already complete pipeline
        if (it1 != pipelines.end()) {
            pipeline = it1->second.get();
            break;
        }

        // Maybe the shader has been compiled but the pipeline state changed?
        auto shader = fragment_shaders.find(pipeline_key.fragment_config);
        if (shader != fragment_shaders.end()) {
            pipeline = MakePipeline(shader->second.get());
            break;
        }

        // Re-compile shader module and create new pipeline
        auto result = GenerateFragmentShader(pipeline_key.fragment_config);
        auto module = MakeShader(result.code, vk::ShaderStageFlagBits::eFragment);
        pipeline = MakePipeline(module);
    } while (false);

    // Bind the pipeline
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
}

vk::ShaderModule VulkanState::MakeShader(const std::string& source, vk::ShaderStageFlagBits stage) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    shaderc_shader_kind kind{};
    std::string name{};
    switch (stage) {
    case vk::ShaderStageFlagBits::eVertex:
        kind = shaderc_glsl_vertex_shader;
        name = "vertex shader";
        break;
    case vk::ShaderStageFlagBits::eFragment:
        kind = shaderc_glsl_fragment_shader;
        name = "fragment shader";
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown shader stage");
        UNREACHABLE();
    }

    auto shader_module = compiler.CompileGlslToSpv(source.data(), kind, name.c_str(), options);
    if (shader_module.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_CRITICAL(Render_Vulkan, shader_module.GetErrorMessage().c_str());
    }

    auto shader_code = std::vector<u32>{shader_module.cbegin(), shader_module.cend()};
    vk::ShaderModuleCreateInfo shader_info{{}, shader_code};

    auto& device = g_vk_instace->GetDevice();
    auto shader = device.createShaderModuleUnique(shader_info);

    if (stage == vk::ShaderStageFlagBits::eFragment) {
        auto handle = shader.get();
        fragment_shaders[pipeline_key.fragment_config] = std::move(shader);
        return handle;
    }
    else if (stage == vk::ShaderStageFlagBits::eVertex) {
        trivial_vertex_shader = std::move(shader);
        return trivial_vertex_shader.get();
    }

    UNREACHABLE();
}

vk::Pipeline VulkanState::MakePipeline(vk::ShaderModule fragment) {
    std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages {{
        { {}, vk::ShaderStageFlagBits::eVertex, trivial_vertex_shader.get(), "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, fragment, "main" }
    }};

    vk::PipelineVertexInputStateCreateInfo vertex_input_info{
        {}, HardwareVertex::binding_desc, HardwareVertex::attribute_desc
    };

    vk::PipelineInputAssemblyStateCreateInfo input_assembly{{}, vk::PrimitiveTopology::eTriangleList, false};
    vk::PipelineRasterizationStateCreateInfo rasterizer{
        {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
        vk::FrontFace::eClockwise, false
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{{}, vk::SampleCountFlagBits::e1};
    vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, pipeline_key.blend_config};

    // Enable every required dynamic state
    std::array<vk::DynamicState, 14> dynamic_states{
        vk::DynamicState::eDepthCompareOp, vk::DynamicState::eLineWidth,
        vk::DynamicState::eDepthTestEnable, vk::DynamicState::eColorWriteEnableEXT,
        vk::DynamicState::eStencilTestEnable, vk::DynamicState::eStencilOp,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eCullMode, vk::DynamicState::eBlendConstants,
        vk::DynamicState::eViewport, vk::DynamicState::eScissor,
        vk::DynamicState::eLogicOpEXT, vk::DynamicState::eFrontFace
    };

    vk::PipelineDynamicStateCreateInfo dynamic_info{{}, dynamic_states};

    vk::PipelineDepthStencilStateCreateInfo depth_info{
        {}, true, true, vk::CompareOp::eGreaterOrEqual, false, true
    };

    vk::GraphicsPipelineCreateInfo pipeline_info{
        {}, shader_stages, &vertex_input_info, &input_assembly, nullptr, nullptr,
        &rasterizer, &multisampling, &depth_info, &color_blending, &dynamic_info,
        pipeline_layout.get(), nullptr
    };

    auto& device = g_vk_instace->GetDevice();
    auto result = device.createGraphicsPipelineUnique(nullptr, pipeline_info);

    if (result.result == vk::Result::eSuccess) {
        auto handle = result.value.get();
        pipelines[pipeline_key] = std::move(result.value);
        return handle;
    }

    return VK_NULL_HANDLE;
}

void VulkanState::Apply() {
    // Update resources in descriptor sets if changed
    UpdateDescriptorSet();

    // Start rendering if not already started
    BeginRendering();

    // Re-apply dynamic parts of the pipeline
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    if (dirty_flags & DirtyFlags::VertexBuffer) {
        command_buffer.bindVertexBuffers(0, vertex_buffer->GetBuffer(), vertex_offset);
    }

    if (dirty_flags & DirtyFlags::IndexBuffer) {
        command_buffer.bindIndexBuffer(index_buffer->GetBuffer(), index_offset, vk::IndexType::eUint16);
    }

    if (dirty_flags & DirtyFlags::Viewport) {
        command_buffer.setViewport(0, viewport);
    }

    if (dirty_flags & DirtyFlags::Scissor) {
        command_buffer.setScissor(0, scissor);
    }

    dirty_flags = DirtyFlags::None;
}

void VulkanState::UpdateDescriptorSet() {
    std::vector<vk::WriteDescriptorSet> writes;
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    std::vector<vk::DescriptorImageInfo> image_infos;

    auto& device = g_vk_instace->GetDevice();

    // Check if any resource has been updated
    if (dirty_flags & DirtyFlags::Uniform) {
        for (int i = 0; i < 2; i++) {
            if (bindings[i].dirty) {
                auto buffer = std::get<VKBuffer*>(bindings[i].resource);
                buffer_infos.emplace_back(buffer->GetBuffer(), 0, VK_WHOLE_SIZE);
                writes.emplace_back(descriptor_sets[i].get(), i, 0, 1, vk::DescriptorType::eUniformBuffer,
                                    nullptr, &buffer_infos.back(), nullptr);
                bindings[i].dirty = false;
            }
        }
    }

    if (dirty_flags & DirtyFlags::Texture) {
        for (int i = 2; i < 6; i++) {
            if (bindings[i].dirty) {
                auto texture = std::get<VKTexture*>(bindings[i].resource);
                image_infos.emplace_back(sampler.get(), texture->GetView(), vk::ImageLayout::eShaderReadOnlyOptimal);
                writes.emplace_back(descriptor_sets[i].get(), i, 0, 1, vk::DescriptorType::eCombinedImageSampler,
                                    &image_infos.back());
                bindings[i].dirty = false;
            }
        }
    }

    if (dirty_flags & DirtyFlags::TexelBuffer) {
        for (int i = 6; i < 9; i++) {
            if (bindings[i].dirty) {
                auto buffer = std::get<VKBuffer*>(bindings[i].resource);
                buffer_infos.emplace_back(buffer->GetBuffer(), 0, VK_WHOLE_SIZE);
                writes.emplace_back(descriptor_sets[i].get(), i, 0, 1, vk::DescriptorType::eStorageTexelBuffer,
                                    nullptr, &buffer_infos.back(), &bindings[i].buffer_view.get());
                bindings[i].dirty = false;
            }
        }
    }

    if (!writes.empty()) {
        device.updateDescriptorSets(writes, {});
    }
}

}  // namespace Vulkan
