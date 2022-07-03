// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

std::unique_ptr<VulkanState> s_vulkan_state;

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

void DescriptorUpdater::Reset() {
    write_count = 0;
    buffer_count = 0;
    image_count = 0;
}

void DescriptorUpdater::Update() {
    assert(write_count > 0);

    auto device = g_vk_instace->GetDevice();
    device.updateDescriptorSets(write_count, writes.data(), 0, nullptr);

    Reset();
}

void DescriptorUpdater::PushTextureArrayUpdate(vk::DescriptorSet set, u32 binding, vk::Sampler sampler,
                                               std::span<vk::ImageView> views) {
    ASSERT(image_count < MAX_UPDATES);

    u32 start = image_count;
    for (auto& view : views) {
        image_infos[image_count++] = {sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal};
    }

    writes[write_count++] = vk::WriteDescriptorSet{set, binding, 0, static_cast<u32>(views.size()),
                             vk::DescriptorType::eCombinedImageSampler,
                             image_infos.data() + start};
}

void DescriptorUpdater::PushCombinedImageSamplerUpdate(vk::DescriptorSet set, u32 binding,
                                                       vk::Sampler sampler, vk::ImageView view) {
    ASSERT(image_count < MAX_UPDATES);

    image_infos[image_count] = {sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal};

    writes[write_count++] = vk::WriteDescriptorSet{set, binding, 0, 1,
                             vk::DescriptorType::eCombinedImageSampler,
                             &image_infos[image_count++]};
}

void DescriptorUpdater::PushBufferUpdate(vk::DescriptorSet set, u32 binding,
                                         vk::DescriptorType buffer_type, u32 offset, u32 size,
                                         vk::Buffer buffer, const vk::BufferView& view) {
    ASSERT(buffer_count < MAX_UPDATES);

    buffer_infos[buffer_count] = vk::DescriptorBufferInfo{buffer, offset, size};

    writes[write_count++] = vk::WriteDescriptorSet{set, binding, 0, 1, buffer_type, nullptr,
                             &buffer_infos[buffer_count++],
                             view ? &view : nullptr};
}

VulkanState::VulkanState(const std::shared_ptr<Swapchain>& swapchain) : swapchain(swapchain) {
    // Create a placeholder texture which can be used in place of a real binding.
    Texture::Info info{
        .width = 1,
        .height = 1,
        .format = vk::Format::eR8G8B8A8Unorm,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D,
        .usage = vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferDst
    };

    placeholder.Create(info);

    // Create texture sampler
    auto props = g_vk_instace->GetPhysicalDevice().getProperties();
    vk::SamplerCreateInfo sampler_info{
        {}, vk::Filter::eLinear,
        vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge,
        {}, true, props.limits.maxSamplerAnisotropy,
        false, vk::CompareOp::eAlways, {}, {},
        vk::BorderColor::eIntOpaqueBlack, false
    };

    // TODO: Sampler cache
    auto device = g_vk_instace->GetDevice();
    render_sampler = device.createSampler(sampler_info);
    present_sampler = device.createSampler(sampler_info);

    // Unbind all texture units
    present_view = placeholder.GetView();
    for (int i = 0; i < 4; i++) {
        render_views[i] = placeholder.GetView();
    }

    // Configure descriptor sets and pipeline builders
    BuildDescriptorLayouts();
    ConfigureRenderPipeline();
    ConfigurePresentPipeline();
}

VulkanState::~VulkanState() {
    auto device = g_vk_instace->GetDevice();
    device.waitIdle();

    // Destroy vertex shader
    device.destroyShaderModule(render_vertex_shader);
    device.destroyShaderModule(present_vertex_shader);
    device.destroyShaderModule(present_fragment_shader);

    // Destroy pipeline layouts
    device.destroyPipelineLayout(render_pipeline_layout);
    device.destroyPipelineLayout(present_pipeline_layout);

    // Destroy descriptor layouts
    for (auto& layout : descriptor_layouts) {
        device.destroyDescriptorSetLayout(layout);
    }

    // Destroy samplers
    device.destroySampler(render_sampler);
    device.destroySampler(present_sampler);

    // Destroy shaders
    for (auto& shader : render_fragment_shaders) {
        device.destroyShaderModule(shader.second);
    }

    // Destroy pipelines
    for (auto& pipeline : render_pipelines) {
        device.destroyPipeline(pipeline.second);
    }

    device.destroyPipeline(present_pipeline);
}

void VulkanState::Create(const std::shared_ptr<Swapchain>& swapchain) {
    if (!s_vulkan_state) {
        s_vulkan_state = std::make_unique<VulkanState>(swapchain);
    }
}

VulkanState& VulkanState::Get() {
    assert(s_vulkan_state);
    return *s_vulkan_state;
}

void VulkanState::SetVertexBuffer(const Buffer& buffer, vk::DeviceSize offset) {
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    cmdbuffer.bindVertexBuffers(0, buffer.GetBuffer(), offset);
}

void VulkanState::SetUniformBuffer(u32 binding, u32 offset, u32 size, const Buffer& buffer) {
    auto& set = descriptor_sets[0];
    updater.PushBufferUpdate(set, binding,
            vk::DescriptorType::eUniformBuffer,
            offset, size, buffer.GetBuffer());
    descriptors_dirty = true;
}

void VulkanState::SetTexture(u32 binding, const Texture& image) {
    auto& set = descriptor_sets[1];
    updater.PushCombinedImageSamplerUpdate(set, binding, render_sampler, image.GetView());
    render_views[binding] = image.GetView();
    descriptors_dirty = true;
}

void VulkanState::SetTexelBuffer(u32 binding, u32 offset, u32 size, const Buffer& buffer, u32 view_index) {
    auto& set = descriptor_sets[2];
    updater.PushBufferUpdate(set, binding,
            vk::DescriptorType::eUniformTexelBuffer,
            offset, size, buffer.GetBuffer(),
            buffer.GetView(view_index));
    descriptors_dirty = true;
}

void VulkanState::SetPresentTextures(vk::ImageView view0, vk::ImageView view1, vk::ImageView view2) {
    auto& set = descriptor_sets[3];

    std::array views{view0, view1, view2};
    updater.PushTextureArrayUpdate(set, 0, present_sampler, views);
    descriptors_dirty = true;
}

void VulkanState::SetPresentData(DrawInfo data) {
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    cmdbuffer.pushConstants(present_pipeline_layout, vk::ShaderStageFlagBits::eFragment |
                            vk::ShaderStageFlagBits::eVertex, 0, sizeof(data), &data);

}

void VulkanState::SetPlaceholderColor(u8 red, u8 green, u8 blue, u8 alpha) {
    std::array<u8, 4> color{red, green, blue, alpha};
    placeholder.Upload(0, 0, 1, placeholder.GetArea(), color);
}

void VulkanState::UnbindTexture(const Texture& image) {
    for (int i = 0; i < 4; i++) {
        if (render_views[i] == image.GetView()) {
            render_views[i] = placeholder.GetView();
            updater.PushCombinedImageSamplerUpdate(descriptor_sets[1], i,
                    render_sampler, render_views[i]);
            descriptors_dirty = true;
        }
    }

    if (present_view == image.GetView()) {
        present_view = placeholder.GetView();
        updater.PushCombinedImageSamplerUpdate(descriptor_sets[3], 0,
                render_sampler, present_view);
        descriptors_dirty = true;
    }
}

void VulkanState::UnbindTexture(u32 unit) {
    render_views[unit] = placeholder.GetView();
    updater.PushCombinedImageSamplerUpdate(descriptor_sets[1], unit,
            render_sampler, render_views[unit]);
    descriptors_dirty = true;
}

void VulkanState::BeginRendering(Texture* color, Texture* depth, bool update_pipeline_formats,
                    vk::ClearColorValue color_clear, vk::AttachmentLoadOp color_load_op,
                    vk::AttachmentStoreOp color_store_op, vk::ClearDepthStencilValue depth_clear,
                    vk::AttachmentLoadOp depth_load_op, vk::AttachmentStoreOp depth_store_op,
                    vk::AttachmentLoadOp stencil_load_op, vk::AttachmentStoreOp stencil_store_op) {
    // Make sure to exit previous render context
    EndRendering();

    // Make sure attachments are in optimal layout
    vk::RenderingInfo render_info{{}, {}, 1, {}};
    std::array<vk::RenderingAttachmentInfo, 3> infos{};

    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    if (color != nullptr) {
        color->Transition(cmdbuffer, vk::ImageLayout::eColorAttachmentOptimal);

        infos[0] = vk::RenderingAttachmentInfo{
            color->GetView(), color->GetLayout(), {}, {}, {},
            color_load_op, color_store_op, color_clear
        };

        render_info.colorAttachmentCount = 1;
        render_info.pColorAttachments = &infos[0];
        render_info.renderArea = color->GetArea();
    }

    if (depth != nullptr) {
        depth->Transition(cmdbuffer, vk::ImageLayout::eDepthStencilAttachmentOptimal);

        infos[1] = vk::RenderingAttachmentInfo{
            depth->GetView(), depth->GetLayout(), {}, {}, {},
            depth_load_op, depth_store_op, depth_clear
        };

        render_info.pDepthAttachment = &infos[1];


        if (IsStencil(depth->GetFormat())) {
            infos[2] = vk::RenderingAttachmentInfo{
                depth->GetView(), depth->GetLayout(), {}, {}, {},
                stencil_load_op, stencil_store_op, depth_clear
            };

            render_info.pStencilAttachment = &infos[2];
        }
    }

    if (update_pipeline_formats) {
        render_pipeline_key.color = color != nullptr ?
                    color->GetFormat() :
                    vk::Format::eUndefined;
        render_pipeline_key.depth_stencil = depth != nullptr ?
                    depth->GetFormat() :
                    vk::Format::eUndefined;
    }

    // Begin rendering
    cmdbuffer.beginRendering(render_info);
    rendering = true;
}

void VulkanState::EndRendering() {
    if (!rendering) {
        return;
    }

    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    cmdbuffer.endRendering();
    rendering = false;
}

void VulkanState::SetViewport(vk::Viewport new_viewport) {
    if (new_viewport != viewport) {
        viewport = new_viewport;
        dirty_flags.set(DynamicStateFlags::Viewport);
    }
}

void VulkanState::SetScissor(vk::Rect2D new_scissor) {
    if (new_scissor != scissor) {
        scissor = new_scissor;
        dirty_flags.set(DynamicStateFlags::Scissor);
    }
}

void VulkanState::SetCullMode(vk::CullModeFlags flags) {
    if (cull_mode != flags) {
        cull_mode = flags;
        dirty_flags.set(DynamicStateFlags::CullMode);
    }
}

void VulkanState::SetFrontFace(vk::FrontFace face) {
    if (front_face != face) {
        front_face = face;
        dirty_flags.set(DynamicStateFlags::FrontFace);
    }
}

void VulkanState::SetColorMask(vk::ColorComponentFlags mask) {
    render_pipeline_key.blend_config.colorWriteMask = mask;
}

void VulkanState::SetLogicOp(vk::LogicOp logic_op) {
    render_pipeline_key.blend_logic_op = logic_op;
}

void VulkanState::SetBlendEnable(bool enable) {
    render_pipeline_key.blend_config.blendEnable = enable;
}

void VulkanState::SetBlendCostants(float red, float green, float blue, float alpha) {
    std::array<float, 4> color{red, green, blue, alpha};
    if (color != blend_constants) {
        blend_constants = color;
        dirty_flags.set(DynamicStateFlags::BlendConstants);
    }
}

void VulkanState::SetBlendOp(vk::BlendOp rgb_op, vk::BlendOp alpha_op, vk::BlendFactor src_color,
                             vk::BlendFactor dst_color, vk::BlendFactor src_alpha, vk::BlendFactor dst_alpha) {
    auto& blend = render_pipeline_key.blend_config;
    blend.colorBlendOp = rgb_op;
    blend.alphaBlendOp = alpha_op;
    blend.srcColorBlendFactor = src_color;
    blend.dstColorBlendFactor = dst_color;
    blend.srcAlphaBlendFactor = src_alpha;
    blend.dstAlphaBlendFactor = dst_alpha;
}

void VulkanState::SetStencilWrite(u32 mask) {
    if (mask != stencil_write_mask) {
        stencil_write_mask = mask;
        dirty_flags.set(DynamicStateFlags::StencilMask);
    }
}

void VulkanState::SetStencilInput(u32 mask) {
    if (mask != stencil_input_mask) {
        stencil_input_mask = mask;
        dirty_flags.set(DynamicStateFlags::StencilMask);
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
    dirty_flags.set(DynamicStateFlags::StencilTest);
}

void VulkanState::SetDepthWrite(bool enable) {
    if (enable != depth_writes) {
        depth_writes = enable;
        dirty_flags.set(DynamicStateFlags::DepthWrite);
    }
}

void VulkanState::SetDepthTest(bool enable, vk::CompareOp compare) {
    depth_enabled = enable;
    depth_op = compare;
    dirty_flags.set(DynamicStateFlags::DepthTest);
}


void VulkanState::InitDescriptorSets() {
    auto pool = g_vk_task_scheduler->GetDescriptorPool();
    auto device = g_vk_instace->GetDevice();

    // Allocate new sets
    vk::DescriptorSetAllocateInfo allocate_info{pool, descriptor_layouts};
    auto sets = device.allocateDescriptorSets(allocate_info);

    // Update them if the previous sets are valid
    u32 copy_count = 0;
    std::array<vk::CopyDescriptorSet, 10> copies;

    // Copy only valid descriptors
    std::array<u32, 4> binding_count{2, 4, 3, 1};
    for (int i = 0; i < descriptor_sets.size(); i++) {
        if (descriptor_sets[i]) {
            for (u32 binding = 0; binding < binding_count[i]; binding++) {
                copies[copy_count++] = {descriptor_sets[i], binding, 0, sets[i], binding, 0, 1};
            }
        }
    }

    if (copy_count < 10) {
        // Some descriptors weren't copied and thus need manual updating
        descriptors_dirty = true;
    }

    device.updateDescriptorSets(0, nullptr, copy_count, copies.data());
    std::copy_n(sets.begin(), descriptor_sets.size(), descriptor_sets.begin());
}

void VulkanState::ApplyRenderState(const Pica::Regs& regs) {
    // Update any pending texture units
    if (descriptors_dirty) {
        updater.Update();
        descriptors_dirty = false;
    }

    // Bind an appropriate render pipeline
    render_pipeline_key.fragment_config = PicaFSConfig::BuildFromRegs(regs);
    auto result = render_pipelines.find(render_pipeline_key);

    // Try to use an already complete pipeline
    vk::Pipeline pipeline;
    if (result != render_pipelines.end()) {
        pipeline = result->second;
    }
    else {
        // Maybe the shader has been compiled but the pipeline state changed?
        auto shader = render_fragment_shaders.find(render_pipeline_key.fragment_config);
        if (shader != render_fragment_shaders.end()) {
            render_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, shader->second);
        }
        else {
            // Re-compile shader module and create new pipeline
            auto code = GenerateFragmentShader(render_pipeline_key.fragment_config);
            auto module = CompileShader(code, vk::ShaderStageFlagBits::eFragment);
            render_fragment_shaders.emplace(render_pipeline_key.fragment_config, module);
            render_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, module);
        }

        // Update pipeline builder
        auto& att = render_pipeline_key.blend_config;
        render_pipeline_builder.SetRenderingFormats(render_pipeline_key.color, render_pipeline_key.depth_stencil);
        render_pipeline_builder.SetBlendLogicOp(render_pipeline_key.blend_logic_op);
        render_pipeline_builder.SetBlendAttachment(att.blendEnable, att.srcColorBlendFactor, att.dstColorBlendFactor,
                                                   att.colorBlendOp, att.srcAlphaBlendFactor, att.dstAlphaBlendFactor,
                                                   att.alphaBlendOp, att.colorWriteMask);
        // Cache the resulted pipeline
        pipeline = render_pipeline_builder.Build();
        render_pipelines.emplace(render_pipeline_key, pipeline);
    }

    // Bind the render pipeline
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    // Force set all dynamic state for new pipeline
    dirty_flags.set();

    ApplyCommonState(true);

    // Bind render descriptor sets
    if (descriptor_sets[1]) {
        cmdbuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, render_pipeline_layout,
                                     0, 3, descriptor_sets.data(), 0, nullptr);
        return;
    }

    LOG_CRITICAL(Render_Vulkan, "Texture unit descriptor set unallocated!");
    UNREACHABLE();
}

void VulkanState::ApplyPresentState() {
    // Update present texture if it was reallocated by the renderer
    if (descriptors_dirty) {
        updater.Update();
        descriptors_dirty = false;
    }

    // Bind present pipeline and descriptors
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipeline);

    ApplyCommonState(false);

    if (descriptor_sets[3]) {
        cmdbuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, present_pipeline_layout,
                                     0, 1, &descriptor_sets[3], 0, nullptr);
        return;
    }

    LOG_CRITICAL(Render_Vulkan, "Present descriptor set unallocated!");
    UNREACHABLE();
}

void VulkanState::ApplyCommonState(bool extended) {
    // Re-apply dynamic parts of the pipeline
    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();
    if (dirty_flags.test(DynamicStateFlags::Viewport)) {
        cmdbuffer.setViewport(0, viewport);
    }

    if (dirty_flags.test(DynamicStateFlags::Scissor)) {
        cmdbuffer.setScissor(0, scissor);
    }

    if (dirty_flags.test(DynamicStateFlags::DepthTest) && extended) {
        cmdbuffer.setDepthTestEnable(depth_enabled);
        cmdbuffer.setDepthCompareOp(depth_op);
    }

    if (dirty_flags.test(DynamicStateFlags::StencilTest) && extended) {
        cmdbuffer.setStencilTestEnable(stencil_enabled);
        cmdbuffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, stencil_ref);
        cmdbuffer.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, fail_op, pass_op,
                                    depth_fail_op, stencil_op);
    }

    if (dirty_flags.test(DynamicStateFlags::CullMode) && extended) {
        cmdbuffer.setCullMode(cull_mode);
    }

    if (dirty_flags.test(DynamicStateFlags::FrontFace) && extended) {
        cmdbuffer.setFrontFace(front_face);
    }

    if (dirty_flags.test(DynamicStateFlags::BlendConstants) && extended) {
        cmdbuffer.setBlendConstants(blend_constants.data());
    }

    if (dirty_flags.test(DynamicStateFlags::StencilMask) && extended) {
        cmdbuffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, stencil_write_mask);
        cmdbuffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, stencil_input_mask);
    }

    if (dirty_flags.test(DynamicStateFlags::DepthWrite) && extended) {
        cmdbuffer.setDepthWriteEnable(depth_writes);
    }

    dirty_flags.reset();
}

void VulkanState::BuildDescriptorLayouts() {
    // Render descriptor layouts
    std::array<vk::DescriptorSetLayoutBinding, 2> ubo_set{{
        {0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex |
          vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eFragment}, // shader_data
        {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex} // pica_uniforms
    }};
    std::array<vk::DescriptorSetLayoutBinding, 4> texture_set{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}, // tex0
        {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}, // tex1
        {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}, // tex2
        {3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}, // tex_cube
    }};
    std::array<vk::DescriptorSetLayoutBinding, 3> lut_set{{
        {0, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // texture_buffer_lut_lf
        {1, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // texture_buffer_lut_rg
        {2, vk::DescriptorType::eUniformTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment} // texture_buffer_lut_rgba
    }};
    std::array<vk::DescriptorSetLayoutBinding, 1> present_set{{
       {0, vk::DescriptorType::eCombinedImageSampler, 3, vk::ShaderStageFlagBits::eFragment}
    }};

    std::array<vk::DescriptorSetLayoutCreateInfo, DESCRIPTOR_SET_COUNT> create_infos{{
            { {}, ubo_set }, { {}, texture_set }, { {}, lut_set }, { {}, present_set }
    }};

    // Create the descriptor set layouts
    auto device = g_vk_instace->GetDevice();
    for (int i = 0; i < DESCRIPTOR_SET_COUNT; i++) {
        descriptor_layouts[i] = device.createDescriptorSetLayout(create_infos[i]);
    }
}

void VulkanState::ConfigureRenderPipeline() {
    // Make render pipeline layout
    PipelineLayoutBuilder lbuilder;
    lbuilder.AddDescriptorSet(descriptor_layouts[0]);
    lbuilder.AddDescriptorSet(descriptor_layouts[1]);
    lbuilder.AddDescriptorSet(descriptor_layouts[2]);
    render_pipeline_layout = lbuilder.Build();

    // Set rasterization state
    render_pipeline_builder.Clear();
    render_pipeline_builder.SetPipelineLayout(render_pipeline_layout);
    render_pipeline_builder.SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleList);
    render_pipeline_builder.SetLineWidth(1.0f);
    render_pipeline_builder.SetNoCullRasterizationState();
    render_pipeline_builder.SetRenderingFormats(render_pipeline_key.color, render_pipeline_key.depth_stencil);

    // Set depth, stencil tests and blending
    render_pipeline_builder.SetNoDepthTestState();
    render_pipeline_builder.SetNoStencilState();
    render_pipeline_builder.SetBlendConstants(1.f, 1.f, 1.f, 1.f);
    render_pipeline_builder.SetBlendAttachment(true, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                               vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
                               vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                               vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

    // Enable every required dynamic state
    std::array dynamic_states{
        vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthTestEnable, vk::DynamicState::eStencilTestEnable,
        vk::DynamicState::eStencilOp,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilReference, vk::DynamicState::eDepthWriteEnable,
        vk::DynamicState::eCullMode, vk::DynamicState::eBlendConstants,
        vk::DynamicState::eViewport, vk::DynamicState::eScissor,
        vk::DynamicState::eFrontFace
    };

    render_pipeline_builder.SetDynamicStates(dynamic_states);

    // Configure vertex buffer
    auto attributes = HardwareVertex::attribute_desc;
    render_pipeline_builder.AddVertexBuffer(0, sizeof(HardwareVertex), vk::VertexInputRate::eVertex, attributes);

    // Add trivial vertex shader
    auto code = GenerateTrivialVertexShader(true);
    render_vertex_shader = CompileShader(code, vk::ShaderStageFlagBits::eVertex);
    render_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eVertex, render_vertex_shader);
}

void VulkanState::ConfigurePresentPipeline() {
    // Make present pipeline layout
    PipelineLayoutBuilder lbuilder;
    lbuilder.AddDescriptorSet(descriptor_layouts[3]);
    lbuilder.AddPushConstants(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(DrawInfo));
    present_pipeline_layout = lbuilder.Build();

    // Set rasterization state
    present_pipeline_builder.Clear();
    present_pipeline_builder.SetPipelineLayout(present_pipeline_layout);
    present_pipeline_builder.SetPrimitiveTopology(vk::PrimitiveTopology::eTriangleStrip);
    present_pipeline_builder.SetLineWidth(1.0f);
    present_pipeline_builder.SetNoCullRasterizationState();
    present_pipeline_builder.SetRenderingFormats(vk::Format::eB8G8R8A8Unorm);

    // Set depth, stencil tests and blending
    present_pipeline_builder.SetNoDepthTestState();
    present_pipeline_builder.SetNoStencilState();
    present_pipeline_builder.SetNoBlendingState();

    // Enable every required dynamic state
    std::array dynamic_states{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    present_pipeline_builder.SetDynamicStates(dynamic_states);

    // Configure vertex buffer
    auto attributes = ScreenRectVertex::attribute_desc;
    present_pipeline_builder.AddVertexBuffer(0, sizeof(ScreenRectVertex), vk::VertexInputRate::eVertex, attributes);

    // Configure shader stages
    auto vertex_code = GetPresentVertexShader();
    present_vertex_shader = CompileShader(vertex_code, vk::ShaderStageFlagBits::eVertex);
    present_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eVertex, present_vertex_shader);

    auto fragment_code = GetPresentFragmentShader();
    present_fragment_shader = CompileShader(fragment_code, vk::ShaderStageFlagBits::eFragment);
    present_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, present_fragment_shader);

    present_pipeline = present_pipeline_builder.Build();
}

}  // namespace Vulkan
