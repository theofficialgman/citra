// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <span>
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

std::unique_ptr<VulkanState> s_vulkan_state{};

void DescriptorUpdater::Update() {
    assert(update_count > 0);

    auto device = g_vk_instace->GetDevice();
    device.updateDescriptorSets(update_count, writes.data(), 0, nullptr);

    Reset();
}

void DescriptorUpdater::PushCombinedImageSamplerUpdate(vk::DescriptorSet set, u32 binding,
                                                       vk::Sampler sampler, vk::ImageView view) {
    assert(update_count < MAX_DESCRIPTORS);

    auto& info = update_queue[update_count];
    info.image_info = vk::DescriptorImageInfo{sampler, view, vk::ImageLayout::eShaderReadOnlyOptimal};

    writes[update_count++] = vk::WriteDescriptorSet{
        set, binding, 0, 1,
        vk::DescriptorType::eCombinedImageSampler,
        &info.image_info
    };
}

void DescriptorUpdater::PushBufferUpdate(vk::DescriptorSet set, u32 binding,
                                         vk::DescriptorType buffer_type, u32 offset, u32 size,
                                         vk::Buffer buffer, const vk::BufferView& view) {
    assert(update_count < MAX_DESCRIPTORS);

    auto& info = update_queue[update_count];
    info.buffer_info = vk::DescriptorBufferInfo{buffer, offset, size};
    info.buffer_view = view;

    writes[update_count++] = vk::WriteDescriptorSet{
        set, binding, 0, 1,
        buffer_type, nullptr,
        &info.buffer_info, &info.buffer_view
    };
}

VulkanState::VulkanState() {
    // Create a placeholder texture which can be used in place of a real binding.
    VKTexture::Info info = {
        .width = 1,
        .height = 1,
        .format = vk::Format::eR8G8B8A8Srgb,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D
    };

    placeholder.Create(info);

    // Create texture sampler
    auto props = g_vk_instace->GetPhysicalDevice().getProperties();
    vk::SamplerCreateInfo sampler_info{
        {}, vk::Filter::eNearest,
        vk::Filter::eNearest,
        vk::SamplerMipmapMode::eNearest,
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

    // Destroy vertex shader
    device.destroyShaderModule(render_vertex_shader);

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

void VulkanState::SetVertexBuffer(const VKBuffer& buffer, vk::DeviceSize offset) {
    auto cmdbuffer = g_vk_task_scheduler->GetCommandBuffer();
    cmdbuffer.bindVertexBuffers(0, buffer.GetBuffer(), offset);
}

void VulkanState::SetUniformBuffer(u32 binding, u32 offset, u32 size, const VKBuffer& buffer) {
    auto& set = descriptor_sets[0];
    updater.PushBufferUpdate(set, binding,
            vk::DescriptorType::eUniformBuffer,
            offset, size, buffer.GetBuffer());
    descriptors_dirty = true;
}

void VulkanState::SetTexture(u32 binding, const VKTexture& image) {
    auto& set = descriptor_sets[1];
    updater.PushCombinedImageSamplerUpdate(set, binding, render_sampler, image.GetView());
    render_views[binding] = image.GetView();
    descriptors_dirty = true;
}

void VulkanState::SetTexelBuffer(u32 binding, u32 offset, u32 size, const VKBuffer& buffer, u32 view_index) {
    auto& set = descriptor_sets[2];
    updater.PushBufferUpdate(set, binding,
            vk::DescriptorType::eStorageTexelBuffer,
            offset, size, buffer.GetBuffer(),
            buffer.GetView(view_index));
    descriptors_dirty = true;
}

void VulkanState::SetPresentTexture(const VKTexture& image) {
    auto& set = descriptor_sets[3];
    updater.PushCombinedImageSamplerUpdate(set, 0, present_sampler, image.GetView());
    present_view = image.GetView();
    descriptors_dirty = true;
}

void VulkanState::SetPresentData(DrawInfo data) {
    present_data = data;
}

void VulkanState::SetPlaceholderColor(u8 red, u8 green, u8 blue, u8 alpha) {
    std::array<u8, 4> color{red, green, blue, alpha};
    placeholder.Upload(0, 0, 1, placeholder.GetArea(), color);
}

void VulkanState::UnbindTexture(const VKTexture& image) {
    for (int i = 0; i < 4; i++) {
        if (render_views[i] == image.GetView()) {
            render_views[i] = placeholder.GetView();
            descriptors_dirty = true;
        }
    }

    if (present_view == image.GetView()) {
        present_view = placeholder.GetView();
        descriptors_dirty = true;
    }
}

void VulkanState::UnbindTexture(u32 unit) {
    render_views[unit] = placeholder.GetView();
    descriptors_dirty = true;
}

void VulkanState::BeginRendering(OptRef<VKTexture> color, OptRef<VKTexture> depth,
                    vk::ClearColorValue color_clear, vk::AttachmentLoadOp color_load_op,
                    vk::AttachmentStoreOp color_store_op, vk::ClearDepthStencilValue depth_clear,
                    vk::AttachmentLoadOp depth_load_op, vk::AttachmentStoreOp depth_store_op,
                    vk::AttachmentLoadOp stencil_load_op, vk::AttachmentStoreOp stencil_store_op) {
    // Make sure to exit previous render context
    EndRendering();

    // Make sure attachments are in optimal layout
    vk::RenderingInfo render_info{{}, color->get().GetArea(), 1, {}};
    std::array<vk::RenderingAttachmentInfo, 3> infos{};

    if (color.has_value()) {
        auto& image = color->get();
        image.Transition(vk::ImageLayout::eColorAttachmentOptimal);

        infos[0] = vk::RenderingAttachmentInfo{
            image.GetView(), image.GetLayout(), {}, {}, {},
            color_load_op, color_store_op, color_clear
        };

        render_info.colorAttachmentCount = 1;
        render_info.pColorAttachments = &infos[0];
    }

    if (depth.has_value()) {
        auto& image = depth->get();
        image.Transition(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        infos[1] = vk::RenderingAttachmentInfo{
            image.GetView(), image.GetLayout(), {}, {}, {},
            depth_load_op, depth_store_op, depth_clear
        };

        infos[2] = vk::RenderingAttachmentInfo{
            image.GetView(), image.GetLayout(), {}, {}, {},
            stencil_load_op, stencil_store_op, depth_clear
        };

        render_info.pDepthAttachment = &infos[1];
        render_info.pStencilAttachment = &infos[2];
    }

    // Begin rendering
    auto cmdbuffer = g_vk_task_scheduler->GetCommandBuffer();
    cmdbuffer.beginRendering(render_info);
    rendering = true;
}

void VulkanState::EndRendering() {
    if (!rendering) {
        return;
    }

    auto cmdbuffer = g_vk_task_scheduler->GetCommandBuffer();
    cmdbuffer.endRendering();
    rendering = false;
}

void VulkanState::SetColorMask(bool red, bool green, bool blue, bool alpha) {
    auto mask = static_cast<vk::ColorComponentFlags>(red | (green << 1) | (blue << 2) | (alpha << 3));
    render_pipeline_key.blend_config.colorWriteMask = mask;
}

void VulkanState::SetLogicOp(vk::LogicOp logic_op) {
    render_pipeline_key.blend_logic_op = logic_op;
}

void VulkanState::SetBlendEnable(bool enable) {
    render_pipeline_key.blend_config.blendEnable = enable;
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

void VulkanState::InitDescriptorSets() {
    auto pool = g_vk_task_scheduler->GetDescriptorPool();
    auto device = g_vk_instace->GetDevice();

    // Allocate new sets
    vk::DescriptorSetAllocateInfo allocate_info{pool, descriptor_layouts};
    auto sets = device.allocateDescriptorSets(allocate_info);

    // Update them if the previous sets are valid
    auto result = std::ranges::find_if(descriptor_sets, [](vk::DescriptorSet set) { return bool(set); });
    if (result != descriptor_sets.end()) {
        std::array<vk::CopyDescriptorSet, 10> copies{{
            {descriptor_sets[0], 0, 0, sets[0], 0, 0}, // shader_data
            {descriptor_sets[0], 1, 0, sets[0], 1, 0}, // pica_uniforms
            {descriptor_sets[1], 0, 0, sets[1], 0, 0}, // tex0
            {descriptor_sets[1], 1, 0, sets[1], 1, 0}, // tex1
            {descriptor_sets[1], 2, 0, sets[1], 2, 0}, // tex2
            {descriptor_sets[1], 3, 0, sets[1], 3, 0}, // tex_cube
            {descriptor_sets[2], 0, 0, sets[2], 0, 0}, // texture_buffer_lut_lf
            {descriptor_sets[2], 1, 0, sets[2], 1, 0}, // texture_buffer_lut_rg
            {descriptor_sets[2], 2, 0, sets[2], 2, 0}, // texture_buffer_lut_rgba
            {descriptor_sets[3], 0, 0, sets[3], 0, 0}
        }};

        device.updateDescriptorSets({}, copies);
    }

    std::copy_n(sets.begin(), 4, descriptor_sets.begin());
}

void VulkanState::ApplyRenderState(const Pica::Regs& regs) {
    // Update any pending texture units
    if (descriptors_dirty) {
        updater.Update();
        descriptors_dirty = false;
    }

    // Bind an appropriate render pipeline
    render_pipeline_key.fragment_config = PicaFSConfig::BuildFromRegs(regs);
    auto it1 = render_pipelines.find(render_pipeline_key);

    // Try to use an already complete pipeline
    vk::Pipeline pipeline;
    if (it1 != render_pipelines.end()) {
        pipeline = it1->second.get();
    }
    else {
        // Maybe the shader has been compiled but the pipeline state changed?
        auto shader = render_fragment_shaders.find(render_pipeline_key.fragment_config);
        if (shader != render_fragment_shaders.end()) {
            render_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, shader->second.get());
        }
        else {
            // Re-compile shader module and create new pipeline
            auto code = GenerateFragmentShader(render_pipeline_key.fragment_config);
            auto module = CompileShader(code, vk::ShaderStageFlagBits::eFragment);
            render_fragment_shaders.emplace(render_pipeline_key.fragment_config, vk::UniqueShaderModule{module});

            render_pipeline_builder.SetShaderStage(vk::ShaderStageFlagBits::eFragment, shader->second.get());
        }

        // Update pipeline builder
        auto& att = render_pipeline_key.blend_config;
        render_pipeline_builder.SetBlendLogicOp(render_pipeline_key.blend_logic_op);
        render_pipeline_builder.SetBlendAttachment(att.blendEnable, att.srcColorBlendFactor, att.dstColorBlendFactor, att.colorBlendOp,
                                                   att.srcAlphaBlendFactor, att.dstAlphaBlendFactor, att.alphaBlendOp, att.colorWriteMask);

        // Cache the resulted pipeline
        pipeline = render_pipeline_builder.Build();
        render_pipelines.emplace(render_pipeline_key, vk::UniquePipeline{pipeline});
    }

    // Bind the render pipeline
    auto cmdbuffer = g_vk_task_scheduler->GetCommandBuffer();
    cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

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
    auto cmdbuffer = g_vk_task_scheduler->GetCommandBuffer();
    cmdbuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipeline.get());
    cmdbuffer.pushConstants(present_pipeline_layout, vk::ShaderStageFlagBits::eFragment |
                            vk::ShaderStageFlagBits::eVertex, 0, sizeof(present_data), &present_data);

    if (descriptor_sets[3]) {
        cmdbuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, present_pipeline_layout,
                                     0, 1, &descriptor_sets[3], 0, nullptr);
        return;
    }

    LOG_CRITICAL(Render_Vulkan, "Present descriptor set unallocated!");
    UNREACHABLE();
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
        {0, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // texture_buffer_lut_lf
        {1, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment}, // texture_buffer_lut_rg
        {2, vk::DescriptorType::eStorageTexelBuffer, 1, vk::ShaderStageFlagBits::eFragment} // texture_buffer_lut_rgba
    }};
    std::array<vk::DescriptorSetLayoutBinding, 1> present_set{{
       {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}
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

    // Set depth, stencil tests and blending
    render_pipeline_builder.SetNoDepthTestState();
    render_pipeline_builder.SetNoStencilState();
    render_pipeline_builder.SetBlendConstants(1.f, 1.f, 1.f, 1.f);
    render_pipeline_builder.SetBlendAttachment(true, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
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

    render_pipeline_builder.SetDynamicStates(dynamic_states);

    // Configure vertex buffer
    auto attributes = HardwareVertex::attribute_desc;
    render_pipeline_builder.AddVertexBuffer(0, sizeof(HardwareVertex), vk::VertexInputRate::eVertex, attributes);

    // Add trivial vertex shader
    auto code = GenerateTrivialVertexShader(true);
    std::cout << code << '\n';
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
    render_pipeline_builder.SetLineWidth(1.0f);
    render_pipeline_builder.SetNoCullRasterizationState();

    // Set depth, stencil tests and blending
    present_pipeline_builder.SetNoDepthTestState();
    present_pipeline_builder.SetNoStencilState();
    present_pipeline_builder.SetNoBlendingState();

    // Enable every required dynamic state
    std::array<vk::DynamicState, 3> dynamic_states{
        vk::DynamicState::eLineWidth,
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

    present_pipeline = vk::UniquePipeline{present_pipeline_builder.Build()};
}

}  // namespace Vulkan
