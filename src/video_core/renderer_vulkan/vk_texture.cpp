// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_resource_cache.h"
#include "video_core/renderer_vulkan/vk_state.h"

namespace Vulkan {

VKTexture::~VKTexture() {
    // Make sure to unbind the texture before destroying it
    g_vk_state->UnbindTexture(this);

    auto deleter = [this]() {
        auto& device = g_vk_instace->GetDevice();

        if (texture) {
            if (cleanup_image) {
                device.destroyImage(texture);
            }

            device.destroyImageView(texture_view);
            device.freeMemory(texture_memory);
        }
    };

    // Schedule deletion of the texture after it's no longer used
    // by the GPU
    g_vk_task_scheduler->Schedule(deleter);
}

void VKTexture::Create(const Info& info, bool make_staging) {
    auto& device = g_vk_instace->GetDevice();
    texture_info = info;

    switch (texture_info.format)
    {
    case vk::Format::eR8G8B8A8Uint:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eR32Uint:
        channels = 4;
        break;
    case vk::Format::eR8G8B8Uint:
        channels = 3;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown texture format {}", texture_info.format);
    }

    // Make sure the texture size doesn't exceed the global staging buffer size
    u32 image_size = texture_info.width * texture_info.height * channels;
    assert(image_size <= MAX_TEXTURE_UPLOAD_BUFFER_SIZE);

    // Create the texture
    vk::ImageCreateFlags flags;
    if (info.view_type == vk::ImageViewType::eCube) {
        flags = vk::ImageCreateFlagBits::eCubeCompatible;
    }

    vk::ImageCreateInfo image_info
    (
        flags,
        info.type,
        texture_info.format,
        { texture_info.width, texture_info.height, 1 }, info.mipmap_levels, info.array_layers,
        static_cast<vk::SampleCountFlagBits>(info.multisamples),
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled
    );

    texture = device.createImage(image_info);

    // Create texture memory
    auto requirements = device.getImageMemoryRequirements(texture);
    auto memory_index = VKBuffer::FindMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo alloc_info(requirements.size, memory_index);

    texture_memory = device.allocateMemory(alloc_info);
    device.bindImageMemory(texture, texture_memory, 0);

    // Create texture view
    vk::ImageViewCreateInfo view_info({}, texture, info.view_type, texture_info.format, {},
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    texture_view = device.createImageView(view_info);

    // Create staging buffer
    if (make_staging) {
        staging.Create(image_size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                       vk::BufferUsageFlagBits::eTransferSrc);
    }
}

void VKTexture::Adopt(vk::Image image, vk::ImageViewCreateInfo view_info) {
    // Prevent image cleanup at object destruction
    cleanup_image = false;
    texture = image;

    // Create image view
    texture_view = g_vk_instace->GetDevice().createImageView(view_info);
}

void VKTexture::TransitionLayout(vk::ImageLayout new_layout, vk::CommandBuffer& command_buffer) {
    struct LayoutInfo {
        vk::ImageLayout layout;
        vk::AccessFlags access;
        vk::PipelineStageFlags stage;
    };

    // Get optimal transition settings for every image layout. Settings taken from Dolphin
    auto layout_info = [&](vk::ImageLayout layout) -> LayoutInfo {
        LayoutInfo info = { .layout = layout };
        switch (texture_layout) {
        case vk::ImageLayout::eUndefined:
            // Layout undefined therefore contents undefined, and we don't care what happens to it.
            info.access = vk::AccessFlagBits::eNone;
            info.stage = vk::PipelineStageFlagBits::eTopOfPipe;
            break;

        case vk::ImageLayout::ePreinitialized:
            // Image has been pre-initialized by the host, so ensure all writes have completed.
            info.access = vk::AccessFlagBits::eHostWrite;
            info.stage = vk::PipelineStageFlagBits::eHost;
            break;

        case vk::ImageLayout::eColorAttachmentOptimal:
            // Image was being used as a color attachment, so ensure all writes have completed.
            info.access = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
            info.stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            break;

        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            // Image was being used as a depthstencil attachment, so ensure all writes have completed.
            info.access = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            info.stage = vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests;
            break;

        case vk::ImageLayout::eShaderReadOnlyOptimal:
            // Image was being used as a shader resource, make sure all reads have finished.
            info.access = vk::AccessFlagBits::eShaderRead;
            info.stage = vk::PipelineStageFlagBits::eFragmentShader;
            break;

        case vk::ImageLayout::eTransferSrcOptimal:
            // Image was being used as a copy source, ensure all reads have finished.
            info.access = vk::AccessFlagBits::eTransferRead;
            info.stage = vk::PipelineStageFlagBits::eTransfer;
            break;

        case vk::ImageLayout::eTransferDstOptimal:
            // Image was being used as a copy destination, ensure all writes have finished.
            info.access = vk::AccessFlagBits::eTransferWrite;
            info.stage = vk::PipelineStageFlagBits::eTransfer;
            break;

        default:
          LOG_CRITICAL(Render_Vulkan, "Unhandled vulkan image layout {}\n", texture_layout);
          break;
        }

        return info;
    };

    // Submit pipeline barrier
    LayoutInfo source = layout_info(texture_layout), dst = layout_info(new_layout);
    vk::ImageMemoryBarrier barrier
    (
        source.access, dst.access,
        source.layout, dst.layout,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        texture,
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
    );

    std::array<vk::ImageMemoryBarrier, 1> barriers = { barrier };

    command_buffer.pipelineBarrier(source.stage, dst.stage, vk::DependencyFlagBits::eByRegion, {}, {}, barriers);

    vk::SubmitInfo submit_info({}, {}, {}, 1, &command_buffer);

    // Update texture layout
    texture_layout = new_layout;
}

void VKTexture::CopyPixels(std::span<u32> new_pixels) {
    if (!staging.GetHostPointer()) {
        LOG_ERROR(Render_Vulkan, "Cannot copy pixels without staging buffer!");
    }

    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();

    // Copy pixels to staging buffer
    std::memcpy(staging.GetHostPointer(),
                new_pixels.data(), new_pixels.size() * channels);

    vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), 0,
                               { texture_info.width, texture_info.height, 1 });
    std::array<vk::BufferImageCopy, 1> regions = { region };

    // Transition image to transfer format
    TransitionLayout(vk::ImageLayout::eTransferDstOptimal, command_buffer);

    command_buffer.copyBufferToImage(staging.GetBuffer(), texture, vk::ImageLayout::eTransferDstOptimal, regions);

    // Prepare for shader reads
    TransitionLayout(vk::ImageLayout::eShaderReadOnlyOptimal, command_buffer);
}

void VKTexture::BlitTo(Common::Rectangle<u32> srect, VKTexture& dest,
                       Common::Rectangle<u32> drect, SurfaceParams::SurfaceType type,
                       vk::CommandBuffer& command_buffer) {
    // Ensure textures are of the same dimention
    assert(texture_info.width == dest.texture_info.width &&
           texture_info.height == dest.texture_info.height);

    vk::ImageAspectFlags image_aspect;
    switch (type) {
    case SurfaceParams::SurfaceType::Color:
    case SurfaceParams::SurfaceType::Texture:
        image_aspect = vk::ImageAspectFlagBits::eColor;
        break;
    case SurfaceParams::SurfaceType::Depth:
        image_aspect = vk::ImageAspectFlagBits::eDepth;
        break;
    case SurfaceParams::SurfaceType::DepthStencil:
        image_aspect = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unhandled image blit aspect\n");
        UNREACHABLE();
    }

    // Define the region to blit
    vk::ImageSubresourceLayers layers(image_aspect, 0, 0, 1);

    std::array<vk::Offset3D, 2> src_offsets = { vk::Offset3D(srect.left, srect.bottom, 1), vk::Offset3D(srect.right, srect.top, 1) };
    std::array<vk::Offset3D, 2> dst_offsets = { vk::Offset3D(drect.left, drect.bottom, 1), vk::Offset3D(drect.right, drect.top, 1) };
    std::array<vk::ImageBlit, 1> regions = {{{layers, src_offsets, layers, dst_offsets}}};

    // Transition image layouts
    TransitionLayout(vk::ImageLayout::eTransferSrcOptimal, command_buffer);
    dest.TransitionLayout(vk::ImageLayout::eTransferDstOptimal, command_buffer);

    // Perform blit operation
    command_buffer.blitImage(texture, vk::ImageLayout::eTransferSrcOptimal, dest.texture,
                             vk::ImageLayout::eTransferDstOptimal, regions, vk::Filter::eNearest);
}

void VKTexture::Fill(Common::Rectangle<u32> region, glm::vec4 color) {

}

void VKTexture::Fill(Common::Rectangle<u32> region, glm::vec2 depth_stencil) {

}

VKFramebuffer::~VKFramebuffer() {
    auto deleter = [this]() {
        if (framebuffer) {
            auto& device = g_vk_instace->GetDevice();
            device.destroyFramebuffer(framebuffer);
        }
    };

    g_vk_task_scheduler->Schedule(deleter);
}

void VKFramebuffer::Create(const Info& info) {
    // Make sure that either attachment is valid
    assert(info.color || info.depth_stencil);
    attachments = { info.color, info.depth_stencil };

    auto rect = info.color ? info.color->GetRect() : info.depth_stencil->GetRect();
    auto color_format = info.color ? info.color->GetFormat() : vk::Format::eUndefined;
    auto depth_format = info.depth_stencil ? info.depth_stencil->GetFormat() : vk::Format::eUndefined;

    vk::FramebufferCreateInfo framebuffer_info
    (
        {},
        g_vk_res_cache->GetRenderPass(color_format, depth_format, 1, vk::AttachmentLoadOp::eLoad),
        {},
        rect.extent.width,
        rect.extent.height,
        1
    );

    if (info.color && info.depth_stencil) {
        std::array<vk::ImageView, 2> views = { info.color->GetView(), info.depth_stencil->GetView() };
        framebuffer_info.setAttachments(views);
    }
    else {
        auto valid = info.color ? info.color : info.depth_stencil;
        std::array<vk::ImageView, 1> view = { valid->GetView() };
        framebuffer_info.setAttachments(view);
    }

    framebuffer = g_vk_instace->GetDevice().createFramebuffer(framebuffer_info);
}

void VKFramebuffer::Prepare(vk::CommandBuffer& command_buffer) {
    // Transition attachments to their optimal formats for rendering
    if (attachments[Attachments::Color]) {
        attachments[Attachments::Color]->TransitionLayout(vk::ImageLayout::eColorAttachmentOptimal, command_buffer);
    }

    if (attachments[Attachments::DepthStencil]) {
        attachments[Attachments::DepthStencil]->TransitionLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal, command_buffer);
    }
}

}
