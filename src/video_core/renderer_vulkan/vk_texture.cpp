// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

void VKTexture::Create(const Info& info)
{
    auto& device = g_vk_instace->GetDevice();
    format = info.format;
    width = info.width;
    height = info.height;

    switch (format)
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
        LOG_CRITICAL(Render_Vulkan, "Unknown texture format {}", format);
    }

    // Create staging memory buffer for pixel transfers
    u32 image_size = width * height * channels;
    staging.Create(image_size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                   vk::BufferUsageFlagBits::eTransferSrc);
    pixels = staging.memory;

    // Create the texture
    vk::ImageCreateFlags flags = info.view_type == vk::ImageViewType::eCube ? vk::ImageCreateFlagBits::eCubeCompatible : {};
    vk::ImageCreateInfo image_info
    (
        flags,
        info.type,
        format,
        { width, height, 1 }, info.mipmap_levels, info.array_layers,
        vk::SampleCountFlagBits::e1,
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
    vk::ImageViewCreateInfo view_info({}, texture.get(), info.view_type, format, {},
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    texture_view = device.createImageViewUnique(view_info);

    // Create texture sampler
    auto properties = g_vk_instace->GetPhysicalDevice().getProperties();
    vk::SamplerCreateInfo sampler_info
    (
        {},
        info.sampler_info.mag_filter,
        info.sampler_info.min_filter,
        info.sampler_info.mipmap_mode,
        info.sampler_info.wrapping[0], info.sampler_info.wrapping[1], info.sampler_info.wrapping[2],
        {},
        true,
        properties.limits.maxSamplerAnisotropy,
        false,
        vk::CompareOp::eAlways,
        {},
        {},
        vk::BorderColor::eIntOpaqueBlack,
        false
    );

    texture_sampler = device.createSamplerUnique(sampler_info);
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
    std::memcpy(pixels, new_pixels.data(), new_pixels.size() * channels);

    // Copy the staging buffer to the image
    vk::CommandBufferAllocateInfo alloc_info(g_vk_instace->command_pool.get(), vk::CommandBufferLevel::ePrimary, 1);
    vk::CommandBuffer command_buffer = device.allocateCommandBuffers(alloc_info)[0];

    command_buffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), {0}, {width,height,1});
    std::array<vk::BufferImageCopy, 1> regions = { region };

    command_buffer.copyBufferToImage(staging.buffer.get(), texture.get(), vk::ImageLayout::eTransferDstOptimal, regions);
    command_buffer.end();

    vk::SubmitInfo submit_info({}, {}, {}, 1, &command_buffer);
    queue.submit(submit_info, nullptr);
    queue.waitIdle();

    device.freeCommandBuffers(g_vk_instace->command_pool.get(), command_buffer);

    // Prepare for shader reads
    TransitionLayout(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

}
