// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <span>
#include <shaderc/shaderc.hpp>
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

std::unique_ptr<VulkanState> s_vulkan_state{};

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

DescriptorUpdater::DescriptorUpdater() {
    Clear();
}

void DescriptorUpdater::Clear() {
    writes = {};
    write_count = 0;
}

void DescriptorUpdater::Update() {
    assert(write_count > 0);

    auto& device = g_vk_instace->GetDevice();
    device.updateDescriptorSets(writes, {});
}

void DescriptorUpdater::SetDescriptorSet(vk::DescriptorSet _set) {
    set = _set;
}

void DescriptorUpdater::AddCombinedImageSamplerDescriptorWrite(u32 binding, vk::Sampler sampler,
                                                               const VKTexture& image) {
    assert(write_count < MAX_WRITES && image_info_count < MAX_IMAGE_INFOS);

    auto& info = image_infos[image_info_count++];
    info = vk::DescriptorImageInfo{sampler, image.GetView(), image.GetLayout()};

    writes[write_count++] = vk::WriteDescriptorSet{
        set, binding, 0, 1, vk::DescriptorType::eCombinedImageSampler, &info
    };
}

void DescriptorUpdater::AddBufferDescriptorWrite(u32 binding, vk::DescriptorType buffer_type,
                                                 u32 offset, u32 size, const VKBuffer& buffer,
                                                 const vk::BufferView& view) {
    assert(write_count < MAX_WRITES && buffer_info_count < MAX_BUFFER_INFOS);

    auto& info = buffer_infos[buffer_info_count++];
    info = vk::DescriptorBufferInfo{buffer.GetBuffer(), offset, size};

    writes[write_count++] = vk::WriteDescriptorSet{
        set, binding, 0, 1, buffer_type, nullptr, &info, &view
    };
}

VulkanState::VulkanState() {
    // Create a dummy texture which can be used in place of a real binding.
    VKTexture::Info info = {
        .width = 1,
        .height = 1,
        .format = vk::Format::eR8G8B8A8Unorm,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D
    };

    placeholder.Create(info);
    placeholder.Transition(vk::ImageLayout::eShaderReadOnlyOptimal);

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

    auto& device = g_vk_instace->GetDevice();
    sampler = device.createSamplerUnique(sampler_info);

    // Compile trivial vertex shader
    auto source = GenerateTrivialVertexShader(true);
    trivial_vertex_shader = vk::UniqueShaderModule{CompileShader(source.code, vk::ShaderStageFlagBits::eVertex)};

    // Configure descriptor sets and pipeline builder
    ConfigureDescriptorSets();
    ConfigurePipeline();

    dirty_flags |= DirtyFlags::All;
}

void VulkanState::Create() {
    if (!s_vulkan_state) {
        s_vulkan_state = std::make_unique<VulkanState>();
    }
}

VulkanState& VulkanState::Get() {
    assert(s_vulkan_state);
    return *s_vulkan_state;
}

void VulkanState::SetVertexBuffer(VKBuffer* buffer, vk::DeviceSize offset) {
    if (vertex_buffer == buffer) {
        return;
    }

    vertex_buffer = buffer;
    vertex_offset = offset;
    dirty_flags |= DirtyFlags::VertexBuffer;
}

void VulkanState::SetUniformBuffer(BindingID id, u32 offset, u32 size, const VKBuffer& buffer) {
    assert(id < BindingID::Tex0);
    u32 binding = static_cast<u32>(id);

    updater.SetDescriptorSet(g_vk_task_scheduler->GetDescriptorSet(0));
    updater.AddBufferDescriptorWrite(binding, vk::DescriptorType::eUniformBuffer, offset, size, buffer);
}

void VulkanState::SetTexture(BindingID id, const VKTexture& image) {
    assert(id > BindingID::PicaUniform && id < BindingID::LutLF);
    u32 binding = static_cast<u32>(id);

    updater.SetDescriptorSet(g_vk_task_scheduler->GetDescriptorSet(1));
    updater.AddCombinedImageSamplerDescriptorWrite(binding, sampler.get(), image);
}

void VulkanState::SetTexelBuffer(BindingID id, u32 offset, u32 size, const VKBuffer& buffer, u32 view_index) {
    assert(id > BindingID::TexCube);
    u32 binding = static_cast<u32>(id);

    updater.SetDescriptorSet(g_vk_task_scheduler->GetDescriptorSet(2));
    updater.AddBufferDescriptorWrite(binding, vk::DescriptorType::eStorageTexelBuffer,
                                     offset, size, buffer, buffer.GetView(view_index));
}

void VulkanState::UnbindTexture(const VKTexture& image) {
    updater.SetDescriptorSet(g_vk_task_scheduler->GetDescriptorSet(1));
    for (auto i = u32(BindingID::Tex0); i <= u32(BindingID::TexCube); i++) {
        auto view = updater.GetResource<vk::ImageView>(i);
        if (view == image.GetView()) {
            updater.AddCombinedImageSamplerDescriptorWrite(i, sampler.get(), placeholder);
        }
    }
}

void VulkanState::UnbindTexture(u32 index) {
    updater.SetDescriptorSet(g_vk_task_scheduler->GetDescriptorSet(1));
    updater.AddCombinedImageSamplerDescriptorWrite(index, sampler.get(), placeholder);
}

void VulkanState::BeginRendering(Attachment color, Attachment depth_stencil) {
    if (rendering) {
        return;
    }

    // Make sure attachments are in optimal layout
    vk::RenderingInfo render_info{{}, color.image->GetArea(), 1, {}};
    std::array<vk::RenderingAttachmentInfo, 2> infos{};

    if (color.image) {
        color.image->Transition(vk::ImageLayout::eColorAttachmentOptimal);

        infos[0] = vk::RenderingAttachmentInfo{
            color.image->GetView(), color.image->GetLayout(), {}, {}, {},
            color.load_op, color.store_op, color.clear_color
        };

        render_info.colorAttachmentCount = 1;
        render_info.pColorAttachments = &infos[0];
    }

    if (depth_stencil.image) {
        depth_stencil.image->Transition(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        infos[1] = vk::RenderingAttachmentInfo{
                depth_stencil.image->GetView(), depth_stencil.image->GetLayout(), {}, {}, {},
                depth_stencil.load_op, depth_stencil.store_op, depth_stencil.clear_color
        };

        render_info.pDepthAttachment = &infos[1];
        render_info.pStencilAttachment = &infos[1];
    }

    // Begin rendering
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
    stencil_op = compare;
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
    depth_op = compare;
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
    pipeline_key.fragment_config = PicaFSConfig::BuildFromRegs(regs);
    auto it1 = pipelines.find(pipeline_key);

    // Try to use an already complete pipeline
    vk::Pipeline pipeline;
    if (it1 != pipelines.end()) {
        pipeline = it1->second.get();
    }
    else {
        // Maybe the shader has been compiled but the pipeline state changed?
        auto shader = fragment_shaders.find(pipeline_key.fragment_config);
        if (shader != fragment_shaders.end()) {
            builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, shader->second.get());
            pipeline = builder.Build();
        }
        else {
            // Re-compile shader module and create new pipeline
            auto result = GenerateFragmentShader(pipeline_key.fragment_config);
            auto module = CompileShader(result.code, vk::ShaderStageFlagBits::eFragment);
            fragment_shaders.emplace(pipeline_key.fragment_config, vk::UniqueShaderModule{module});

            builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, shader->second.get());
            pipeline = builder.Build();
        }

        // Cache the resulted pipeline
        pipelines.emplace(pipeline_key, vk::UniquePipeline{pipeline});
    }

    // Bind the pipeline
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
}

vk::ShaderModule VulkanState::CompileShader(const std::string& source, vk::ShaderStageFlagBits stage) {
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
    auto shader = device.createShaderModule(shader_info);
    return shader;
}

void VulkanState::Apply() {
    // Update resources in descriptor sets if changed
    updater.Update();

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

    if (dirty_flags & DirtyFlags::DepthTest) {
        command_buffer.setDepthTestEnable(depth_enabled);
        command_buffer.setDepthCompareOp(depth_op);
    }

    if (dirty_flags & DirtyFlags::Stencil) {
        command_buffer.setStencilTestEnable(stencil_enabled);
        command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, stencil_ref);
        command_buffer.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, fail_op, pass_op,
                                    depth_fail_op, stencil_op);
    }

    if (dirty_flags & DirtyFlags::LogicOp) {
        command_buffer.setLogicOpEXT(logic_op);
    }

    if (dirty_flags & DirtyFlags::CullMode) {
        command_buffer.setCullMode(cull_mode);
    }

    if (dirty_flags & DirtyFlags::FrontFace) {
        command_buffer.setFrontFace(front_face);
    }

    if (dirty_flags & DirtyFlags::BlendConsts) {
        command_buffer.setBlendConstants(blend_constants.data());
    }

    if (dirty_flags & DirtyFlags::StencilMask) {
        command_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, stencil_write_mask);
        command_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, stencil_input_mask);
    }

    if (dirty_flags & DirtyFlags::DepthWrite) {
        command_buffer.setDepthWriteEnable(depth_writes);
    }

    dirty_flags = DirtyFlags::None;
}

void VulkanState::ConfigureDescriptorSets() {
    // Draw descriptor sets
    std::array<vk::DescriptorSetLayoutBinding, 2> ubo_set{{
        { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex |
          vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment }, // shader_data
        { 1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex } // pica_uniforms
    }};

    std::array<vk::DescriptorSetLayoutBinding, 4> texture_set{{
        { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex0
        { 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex1
        { 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex2
        { 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }, // tex_cube
    }};

    std::array<vk::DescriptorSetLayoutBinding, 3> lut_set{{
        { 0, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment }, // texture_buffer_lut_lf
        { 1, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment }, // texture_buffer_lut_rg
        { 2, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment } // texture_buffer_lut_rgba
    }};

    // Create and store descriptor set layouts
    std::array<vk::DescriptorSetLayoutCreateInfo, DESCRIPTOR_SET_LAYOUT_COUNT> create_infos{{
        { vk::DescriptorSetLayoutCreateFlags(), ubo_set },
        { vk::DescriptorSetLayoutCreateFlags(), texture_set },
        { vk::DescriptorSetLayoutCreateFlags(), lut_set }
    }};

    auto& device = g_vk_instace->GetDevice();
    for (int i = 0; i < DESCRIPTOR_SET_LAYOUT_COUNT; i++) {
        descriptor_layouts[i] = device.createDescriptorSetLayout(create_infos[i]);
    }

    // Create the standard descriptor set layout
    vk::PipelineLayoutCreateInfo layout_info({}, descriptor_layouts);
    pipeline_layout = device.createPipelineLayoutUnique(layout_info);
}

void VulkanState::ConfigurePipeline() {
    builder.SetPipelineLayout(pipeline_layout.get());

    // Set rasterization state
    builder.SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
    builder.SetLineWidth(1.0f);
    builder.SetRasterizationState(vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
                                  vk::FrontFace::eClockwise);

    // Set depth, stencil tests and blending
    builder.SetNoDepthTestState();
    builder.SetNoStencilState();
    builder.SetBlendConstants(1.f, 1.f, 1.f, 1.f);
    builder.SetBlendAttachment(true, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                               vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                               vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);


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

    builder.SetDynamicStates(dynamic_states);

    // Configure vertex buffer
    auto attributes = HardwareVertex::attribute_desc;
    builder.AddVertexBuffer(0, sizeof(HardwareVertex), vk::VertexInputRate::eVertex, attributes);

    // Add trivial vertex shader
    builder.SetShaderStage(vk::ShaderStageFlagBits::eVertex, trivial_vertex_shader.get());
}

}  // namespace Vulkan
