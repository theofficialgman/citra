// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "core/core.h"
#include "common/object_pool.h"
#include "video_core/renderer_vulkan/vk_backend.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_framebuffer.h"

namespace VideoCore::Vulkan {

constexpr vk::PipelineBindPoint ToVkPipelineBindPoint(PipelineType type) {
    switch (type) {
    case PipelineType::Graphics:
        return vk::PipelineBindPoint::eGraphics;
    case PipelineType::Compute:
        return vk::PipelineBindPoint::eCompute;
    }
}

constexpr vk::Rect2D ToVkRect2D(Rect2D rect) {
    return vk::Rect2D{
        .offset = vk::Offset2D{rect.x, rect.y},
        .extent = vk::Extent2D{rect.width, rect.height}
    };
}

constexpr vk::IndexType ToVkIndexType(AttribType type) {
    switch (type) {
    case AttribType::Short:
        return vk::IndexType::eUint16;
    case AttribType::Int:
        return vk::IndexType::eUint32;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown index type {}!", type);
        UNREACHABLE();
    }
}

Backend::Backend(Frontend::EmuWindow& window) : BackendBase(window),
    instance(window), swapchain(instance, instance.GetSurface()),
    scheduler(instance) {

    // TODO: Properly report GPU hardware
    auto& telemetry_session = Core::System::GetInstance().TelemetrySession();
    constexpr auto user_system = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(user_system, "GPU_Vendor", "NVIDIA");
    telemetry_session.AddField(user_system, "GPU_Model", "GTX 1650");
    telemetry_session.AddField(user_system, "GPU_Vulkan_Version", "Vulkan 1.3");

    // Pre-create all needed renderpasses by the renderer
    constexpr std::array color_formats = {
        vk::Format::eUndefined,
        vk::Format::eR8G8B8A8Unorm,
        vk::Format::eR8G8B8Unorm,
        vk::Format::eR5G5B5A1UnormPack16,
        vk::Format::eR5G6B5UnormPack16,
        vk::Format::eR4G4B4A4UnormPack16
    };

    constexpr std::array depth_stencil_formats = {
        vk::Format::eUndefined,
        vk::Format::eD16Unorm,
        vk::Format::eX8D24UnormPack32,
        vk::Format::eD24UnormS8Uint,
    };

    // Create all required renderpasses
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            if (color == 0 && depth == 0) continue;

            u32 index = color * MAX_COLOR_FORMATS + depth;
            renderpass_cache[index] = CreateRenderPass(color_formats[color], depth_stencil_formats[depth]);
        }
    }

    constexpr std::array pool_sizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1024},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 2048},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1024}
    };

    const vk::DescriptorPoolCreateInfo pool_info = {
        .maxSets = 2048,
        .poolSizeCount = pool_sizes.size(),
        .pPoolSizes = pool_sizes.data()
    };

    // Create descriptor pools
    vk::Device device = instance.GetDevice();
    for (u32 pool = 0; pool < SCHEDULER_COMMAND_COUNT; pool++) {
        descriptor_pools[pool] = device.createDescriptorPool(pool_info);
    }
}

Backend::~Backend() {
    vk::Device device = instance.GetDevice();
    for (auto& renderpass : renderpass_cache) {
        device.destroyRenderPass(renderpass);
    }
}

/**
 * To avoid many small heap allocations during handle creation, each resource has a dedicated pool
 * associated with it that batch allocates memory.
 */
BufferHandle Backend::CreateBuffer(BufferInfo info) {
    static ObjectPool<Buffer> buffer_pool;
    return BufferHandle{buffer_pool.Allocate(instance, scheduler, info)};
}

FramebufferHandle Backend::CreateFramebuffer(FramebufferInfo info) {
    static ObjectPool<Framebuffer> framebuffer_pool;

    // Get renderpass
    TextureFormat color = info.color.IsValid() ? info.color->GetFormat() : TextureFormat::Undefined;
    TextureFormat depth = info.depth_stencil.IsValid() ? info.depth_stencil->GetFormat() : TextureFormat::Undefined;
    vk::RenderPass renderpass = GetRenderPass(color, depth);

    return FramebufferHandle{framebuffer_pool.Allocate(instance, info, renderpass)};
}

TextureHandle Backend::CreateTexture(TextureInfo info) {
    static ObjectPool<Texture> texture_pool;
    return TextureHandle{texture_pool.Allocate(instance, scheduler, info)};
}

PipelineHandle Backend::CreatePipeline(PipelineType type, PipelineInfo info) {
    static ObjectPool<Pipeline> pipeline_pool;

    // Get renderpass
    vk::RenderPass renderpass = GetRenderPass(info.color_attachment, info.depth_attachment);

            // Find a pipeline layout first
    if (auto iter = pipeline_layouts.find(info.layout); iter != pipeline_layouts.end()) {
        PipelineLayout& layout = iter->second;

        return PipelineHandle{pipeline_pool.Allocate(instance, layout, type, info, renderpass, cache)};
    }

    // Create the layout
    auto result = pipeline_layouts.emplace(info.layout, PipelineLayout{instance, info.layout});
    return PipelineHandle{pipeline_pool.Allocate(instance, result.first->second, type, info, renderpass, cache)};
}

SamplerHandle Backend::CreateSampler(SamplerInfo info) {
    static ObjectPool<Sampler> sampler_pool;
    return SamplerHandle{sampler_pool.Allocate(info)};
}

void Backend::Draw(PipelineHandle pipeline_handle, FramebufferHandle draw_framebuffer,
                   BufferHandle vertex_buffer,
                   u32 base_vertex, u32 num_vertices) {
    // Bind descriptor sets
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    BindDescriptorSets(pipeline_handle);

    // Bind vertex buffer
    const Buffer* vertex = static_cast<const Buffer*>(vertex_buffer.Get());
    command_buffer.bindVertexBuffers(0, vertex->GetHandle(), vertex->GetBindOffset());

    // Begin renderpass
    const Framebuffer* framebuffer = static_cast<const Framebuffer*>(draw_framebuffer.Get());
    const vk::RenderPassBeginInfo renderpass_begin = {
        .renderPass = framebuffer->GetRenderpass(),
        .framebuffer = framebuffer->GetHandle(),
        .renderArea = ToVkRect2D(framebuffer->GetDrawRectangle()),
        .clearValueCount = 0,
        .pClearValues = nullptr
    };

    command_buffer.beginRenderPass(renderpass_begin, vk::SubpassContents::eInline);

    // Bind pipeline
    const Pipeline* pipeline = static_cast<const Pipeline*>(pipeline_handle.Get());
    command_buffer.bindPipeline(ToVkPipelineBindPoint(pipeline->GetType()), pipeline->GetHandle());

    // Submit draw
    command_buffer.draw(num_vertices, 1, base_vertex, 0);

    // End renderpass
    command_buffer.endRenderPass();
}

void Backend::DrawIndexed(PipelineHandle pipeline_handle, FramebufferHandle draw_framebuffer,
                          BufferHandle vertex_buffer, BufferHandle index_buffer, AttribType index_type,
                          u32 base_index, u32 num_indices, u32 base_vertex) {
    // Bind descriptor sets
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    BindDescriptorSets(pipeline_handle);

    // Bind vertex buffer
    const Buffer* vertex = static_cast<const Buffer*>(vertex_buffer.Get());
    command_buffer.bindVertexBuffers(0, vertex->GetHandle(), vertex->GetBindOffset());

    // Bind index buffer
    const Buffer* index = static_cast<const Buffer*>(index_buffer.Get());
    command_buffer.bindIndexBuffer(index->GetHandle(), index->GetBindOffset(), ToVkIndexType(index_type));

    // Begin renderpass
    const Framebuffer* framebuffer = static_cast<const Framebuffer*>(draw_framebuffer.Get());
    const vk::RenderPassBeginInfo renderpass_begin = {
        .renderPass = framebuffer->GetRenderpass(),
        .framebuffer = framebuffer->GetHandle(),
        .renderArea = ToVkRect2D(framebuffer->GetDrawRectangle()),
        .clearValueCount = 0,
        .pClearValues = nullptr
    };

    command_buffer.beginRenderPass(renderpass_begin, vk::SubpassContents::eInline);

    // Bind pipeline
    const Pipeline* pipeline = static_cast<const Pipeline*>(pipeline_handle.Get());
    command_buffer.bindPipeline(ToVkPipelineBindPoint(pipeline->GetType()), pipeline->GetHandle());

    // Submit draw
    command_buffer.drawIndexed(num_indices, 1, base_index, base_vertex, 0);

    // End renderpass
    command_buffer.endRenderPass();

}


vk::RenderPass Backend::CreateRenderPass(vk::Format color, vk::Format depth) const {
    // Define attachments
    const std::array attachments = {
        vk::AttachmentDescription{
            .format = color,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal
        },
        vk::AttachmentDescription{
            .format = depth,
            .initialLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        }
    };

    // Our renderpasses only defines one color and depth attachment
    const vk::AttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = vk::ImageLayout::eColorAttachmentOptimal
    };

    const vk::AttachmentReference depth_attachment_ref = {
        .attachment = 1,
        .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };

    const vk::SubpassDependency subpass_dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                        vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                         vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        .dependencyFlags = vk::DependencyFlagBits::eByRegion
    };

    // We also require only one subpass
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = &depth_attachment_ref
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = 2,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &subpass_dependency
    };

    // Create the renderpass
    vk::Device device = instance.GetDevice();
    return device.createRenderPass(renderpass_info);
}

vk::RenderPass Backend::GetRenderPass(TextureFormat color, TextureFormat depth) const {
    u32 color_index = color != TextureFormat::Undefined ? static_cast<u32>(color) + 1 : 0;
    u32 depth_index = depth != TextureFormat::Undefined ? static_cast<u32>(depth) - 4 : 0;
    return renderpass_cache[color_index * MAX_COLOR_FORMATS + depth_index];
}

void Backend::BindDescriptorSets(PipelineHandle handle) {
    Pipeline* pipeline = static_cast<Pipeline*>(handle.Get());
    PipelineLayout& pipeline_layout = pipeline->GetOwner();

    // Allocate required descriptor sets
    // TODO: Maybe cache them?
    u32 pool_index = scheduler.GetCurrentSlotIndex();
    const vk::DescriptorSetAllocateInfo alloc_info = {
        .descriptorPool = descriptor_pools[pool_index],
        .descriptorSetCount = pipeline_layout.GetDescriptorSetLayoutCount(),
        .pSetLayouts = pipeline_layout.GetDescriptorSetLayouts()
    };

    vk::Device device = instance.GetDevice();
    auto descriptor_sets = device.allocateDescriptorSets(alloc_info);

    // Write data to the descriptor sets
    for (u32 set = 0; set < descriptor_sets.size(); set++) {
        device.updateDescriptorSetWithTemplate(descriptor_sets[set],
                                               pipeline_layout.GetUpdateTemplate(set),
                                               pipeline_layout.GetData(set));
    }

    // Bind the descriptor sets
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindDescriptorSets(ToVkPipelineBindPoint(handle->GetType()), pipeline_layout.GetLayout(),
                                      0, descriptor_sets, {});
}

} // namespace VideoCore::Vulkan
