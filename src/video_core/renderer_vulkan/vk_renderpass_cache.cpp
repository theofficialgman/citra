// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace VideoCore::Vulkan {

static constexpr std::array color_formats = {
    vk::Format::eUndefined,
    vk::Format::eR8G8B8A8Unorm,
    vk::Format::eR8G8B8Unorm,
    vk::Format::eR5G5B5A1UnormPack16,
    vk::Format::eR5G6B5UnormPack16,
    vk::Format::eR4G4B4A4UnormPack16
};

static constexpr std::array depth_stencil_formats = {
    vk::Format::eUndefined,
    vk::Format::eD16Unorm,
    vk::Format::eX8D24UnormPack32,
    vk::Format::eD24UnormS8Uint,
};

RenderpassCache::RenderpassCache(Instance& instance) : instance(instance) {
    // Pre-create all needed renderpasses by the renderer
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            if (color == 0 && depth == 0) {
                continue;
             }

            vk::Format color_format = instance.GetFormatAlternative(color_formats[color]);
            vk::Format depth_stencil_format = instance.GetFormatAlternative(depth_stencil_formats[depth]);

            // Construct both load and clear pass
            cached_renderpasses[color][depth][0] = CreateRenderPass(color_format, depth_stencil_format,
                                                                    vk::AttachmentLoadOp::eLoad,
                                                                    vk::ImageLayout::eColorAttachmentOptimal,
                                                                    vk::ImageLayout::eColorAttachmentOptimal);
            cached_renderpasses[color][depth][1] = CreateRenderPass(color_format, depth_stencil_format,
                                                                    vk::AttachmentLoadOp::eClear,
                                                                    vk::ImageLayout::eColorAttachmentOptimal,
                                                                    vk::ImageLayout::eColorAttachmentOptimal);
        }
    }
}

RenderpassCache::~RenderpassCache() {
    vk::Device device = instance.GetDevice();
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            if (color == 0 && depth == 0) {
                continue;
             }

            auto& load_pass = cached_renderpasses[color][depth][0];
            auto& clear_pass = cached_renderpasses[color][depth][1];

            // Destroy renderpasses
            device.destroyRenderPass(load_pass);
            device.destroyRenderPass(clear_pass);
        }
    }

    device.destroyRenderPass(present_renderpass);
}

void RenderpassCache::CreatePresentRenderpass(vk::Format format) {
    if (!present_renderpass) {
        present_renderpass = CreateRenderPass(format, vk::Format::eUndefined,
                                              vk::AttachmentLoadOp::eClear,
                                              vk::ImageLayout::eColorAttachmentOptimal,
                                              vk::ImageLayout::eColorAttachmentOptimal);
    }
}

vk::RenderPass RenderpassCache::GetRenderpass(TextureFormat color, TextureFormat depth, bool is_clear) const {
    const u32 color_index = static_cast<u32>(color);
    const u32 depth_index = (depth == TextureFormat::Undefined ? 0 : (static_cast<u32>(depth) - MAX_COLOR_FORMATS));

    ASSERT(color_index <= MAX_COLOR_FORMATS && depth_index <= MAX_DEPTH_FORMATS);
    return cached_renderpasses[color_index][depth_index][is_clear];
}

vk::RenderPass RenderpassCache::CreateRenderPass(vk::Format color, vk::Format depth, vk::AttachmentLoadOp load_op,
                                                 vk::ImageLayout initial_layout, vk::ImageLayout final_layout) const {
    // Define attachments

    u32 attachment_count = 0;
    std::array<vk::AttachmentDescription, 2> attachments;

    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = color,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = initial_layout,
            .finalLayout = final_layout
        };

        color_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eColorAttachmentOptimal
       };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = depth,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eLoad,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        use_depth = true;
    }

    // We also require only one subpass
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr
    };

    // Create the renderpass
    vk::Device device = instance.GetDevice();
    return device.createRenderPass(renderpass_info);
}

} // namespace VideoCore::Vulkan
