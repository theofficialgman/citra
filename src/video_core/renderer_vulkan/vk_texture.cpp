// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_resource_cache.h"

namespace Vulkan {

void VKTexture::Create(const Info& info)
{
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

    texture = device.createImageUnique(image_info);

    // Create texture memory
    auto requirements = device.getImageMemoryRequirements(texture.get());
    auto memory_index = VKBuffer::FindMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo alloc_info(requirements.size, memory_index);

    texture_memory = device.allocateMemoryUnique(alloc_info);
    device.bindImageMemory(texture.get(), texture_memory.get(), 0);

    // Create texture view
    vk::ImageViewCreateInfo view_info({}, texture.get(), info.view_type, texture_info.format, {},
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    texture_view = device.createImageViewUnique(view_info);
}

void VKTexture::TransitionLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout)
{
    auto& device = g_vk_instace->GetDevice();
    auto& queue = g_vk_instace->graphics_queue;

    vk::CommandBufferAllocateInfo alloc_info(g_vk_instace->command_pool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::CommandBuffer command_buffer = device.allocateCommandBuffers(alloc_info)[0];

    command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageMemoryBarrier barrier({}, {}, old_layout, new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture.get(),
                                   vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    std::array<vk::ImageMemoryBarrier, 1> barriers = { barrier };

    vk::PipelineStageFlags source_stage, destination_stage;
    if (old_layout == vk::ImageLayout::eUndefined && new_layout == vk::ImageLayout::eTransferDstOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eNone;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        source_stage = vk::PipelineStageFlagBits::eTopOfPipe;
        destination_stage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (old_layout == vk::ImageLayout::eTransferDstOptimal && new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        source_stage = vk::PipelineStageFlagBits::eTransfer;
        destination_stage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else {
        LOG_CRITICAL(Render_Vulkan, "Unsupported layout transition");
        UNREACHABLE();
    }

    command_buffer.pipelineBarrier(source_stage, destination_stage, vk::DependencyFlagBits::eByRegion, {}, {}, barriers);
    command_buffer.end();

    vk::SubmitInfo submit_info({}, {}, {}, 1, &command_buffer);
    queue.submit(submit_info, nullptr);
    queue.waitIdle();

    device.freeCommandBuffers(g_vk_instace->command_pool.get(), command_buffer);
}

void VKTexture::CopyPixels(std::span<u32> new_pixels)
{
    auto& device = g_vk_instace->GetDevice();
    auto& queue = g_vk_instace->graphics_queue;

    // Transition image to transfer format
    TransitionLayout(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

    // Copy pixels to staging buffer
    std::memcpy(g_vk_res_cache->GetTextureUploadBuffer().GetHostPointer(),
                new_pixels.data(), new_pixels.size() * channels);

    // Copy the staging buffer to the image
    vk::CommandBufferAllocateInfo alloc_info(g_vk_instace->command_pool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::CommandBuffer command_buffer = device.allocateCommandBuffers(alloc_info)[0];

    command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), 0,
                               { texture_info.width, texture_info.height, 1 });
    std::array<vk::BufferImageCopy, 1> regions = { region };

    auto& staging = g_vk_res_cache->GetTextureUploadBuffer();
    command_buffer.copyBufferToImage(staging.GetBuffer(), texture.get(), vk::ImageLayout::eTransferDstOptimal, regions);
    command_buffer.end();

    vk::SubmitInfo submit_info({}, {}, {}, 1, &command_buffer);
    queue.submit(submit_info, nullptr);

    /// NOTE: Remove this when the renderer starts working, otherwise it will be very slow
    queue.waitIdle();
    device.freeCommandBuffers(g_vk_instace->command_pool.get(), command_buffer);

    // Prepare for shader reads
    TransitionLayout(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void VKFramebuffer::Create(const Info& info)
{
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

    framebuffer = g_vk_instace->GetDevice().createFramebufferUnique(framebuffer_info);
}

void VKFramebuffer::Prepare()
{
    // Transition attachments to their optimal formats for rendering
    if (attachments[Attachments::Color]) {
        attachments[Attachments::Color]->TransitionLayout(vk::ImageLayout::eUndefined,
                                                          vk::ImageLayout::eColorAttachmentOptimal);
    }

    if (attachments[Attachments::DepthStencil]) {
        attachments[Attachments::DepthStencil]->TransitionLayout(vk::ImageLayout::eUndefined,
                                                                 vk::ImageLayout::eDepthStencilAttachmentOptimal);
    }
}

}
