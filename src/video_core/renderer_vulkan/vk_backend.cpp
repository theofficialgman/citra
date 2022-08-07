// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "core/core.h"
#include "common/object_pool.h"
#include "video_core/renderer_vulkan/vk_backend.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace VideoCore::Vulkan {

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
        vk::Format::eR8G8B8A8Unorm,
        vk::Format::eR8G8B8Unorm,
        vk::Format::eR5G5B5A1UnormPack16,
        vk::Format::eR5G6B5UnormPack16,
        vk::Format::eR4G4B4A4UnormPack16
    };

    constexpr std::array depth_stencil_formats = {
        vk::Format::eD16Unorm,
        vk::Format::eX8D24UnormPack32,
        vk::Format::eD24UnormS8Uint,
    };

    // Create all required renderpasses
    for (u32 color = 0; color < MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth < MAX_DEPTH_FORMATS; depth++) {
            u32 index = color * MAX_COLOR_FORMATS + depth;
            renderpass_cache[index] = CreateRenderPass(color_formats[color], depth_stencil_formats[depth]);
        }
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
    return IntrusivePtr<Buffer>{buffer_pool.Allocate(info)};
}

FramebufferHandle Backend::CreateFramebuffer(FramebufferInfo info) {
}

TextureHandle Backend::CreateTexture(TextureInfo info) {
    static ObjectPool<Texture> texture_pool;
    return IntrusivePtr<Texture>{texture_pool.Allocate(info)};
}

PipelineHandle Backend::CreatePipeline(PipelineType type, PipelineInfo info) {
    static ObjectPool<Pipeline> pipeline_pool;

    // Find a pipeline layout first
    if (auto iter = pipeline_layouts.find(info.layout); iter != pipeline_layouts.end()) {
        PipelineLayout& layout = iter->second;

        return IntrusivePtr<Pipeline>{pipeline_pool.Allocate(instance, layout, type, info, cache)};
    }

    // Create the layout
    auto result = pipeline_layouts.emplace(info.layout, PipelineLayout{instance, info.layout});
    return IntrusivePtr<Pipeline>{pipeline_pool.Allocate(instance, result.first->second, type, info, cache)};
}

SamplerHandle Backend::CreateSampler(SamplerInfo info) {
    static ObjectPool<Sampler> sampler_pool;
    return IntrusivePtr<Sampler>{sampler_pool.Allocate(info)};
}

void Backend::Draw(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                   BufferHandle vertex_buffer,
                   u32 base_vertex, u32 num_vertices) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();

    Buffer* vertex = static_cast<Buffer*>(vertex_buffer.Get());
    command_buffer.bindVertexBuffers(0, vertex->GetHandle(), {0});

    // Submit draw
    command_buffer.draw(num_vertices, 1, base_vertex, 0);
}

void Backend::DrawIndexed(PipelineHandle pipeline, FramebufferHandle draw_framebuffer,
                          BufferHandle vertex_buffer, BufferHandle index_buffer,
                          u32 base_index, u32 num_indices, u32 base_vertex) {

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

} // namespace VideoCore::Vulkan
