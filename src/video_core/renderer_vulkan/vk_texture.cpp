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
    if (texture) {
        // Make sure to unbind the texture before destroying it
        g_vk_state->UnbindTexture(this);

        auto deleter = [this]() {
            auto& device = g_vk_instace->GetDevice();
            device.destroyImage(texture);
            device.destroyImageView(view);
            device.freeMemory(memory);
        };

        // Schedule deletion of the texture after it's no longer used
        // by the GPU
        g_vk_task_scheduler->Schedule(deleter);
    }
}

void VKTexture::Create(const VKTexture::Info& create_info) {
    auto& device = g_vk_instace->GetDevice();
    info = create_info;

    switch (info.format)
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
        LOG_CRITICAL(Render_Vulkan, "Unknown texture format {}", info.format);
    }

    // Create the texture
    image_size = info.width * info.height * channels;

    vk::ImageCreateFlags flags{};
    if (info.view_type == vk::ImageViewType::eCube) {
        flags = vk::ImageCreateFlagBits::eCubeCompatible;
    }

    vk::ImageCreateInfo image_info {
        flags, info.type, info.format,
        { info.width, info.height, 1 }, info.levels, info.layers,
        static_cast<vk::SampleCountFlagBits>(info.multisamples),
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled
    };

    texture = device.createImage(image_info);

    // Create texture memory
    auto requirements = device.getImageMemoryRequirements(texture);
    auto memory_index = VKBuffer::FindMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo alloc_info(requirements.size, memory_index);

    memory = device.allocateMemory(alloc_info);
    device.bindImageMemory(texture, memory, 0);

    // Create texture view
    vk::ImageViewCreateInfo view_info {
        {}, texture, info.view_type, info.format, {},
        {info.aspect, 0, info.levels, 0, info.levels}
    };

    view = device.createImageView(view_info);
}

void VKTexture::Transition(vk::ImageLayout new_layout) {
    if (new_layout == layout) {
        return;
    }

    struct LayoutInfo {
        vk::ImageLayout layout;
        vk::AccessFlags access;
        vk::PipelineStageFlags stage;
    };

    // Get optimal transition settings for every image layout. Settings taken from Dolphin
    auto layout_info = [&](vk::ImageLayout layout) -> LayoutInfo {
        LayoutInfo info = { .layout = layout };
        switch (layout) {
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
          LOG_CRITICAL(Render_Vulkan, "Unhandled vulkan image layout {}\n", layout);
          break;
        }

        return info;
    };

    // Submit pipeline barrier
    LayoutInfo source = layout_info(layout), dst = layout_info(new_layout);
    vk::ImageMemoryBarrier barrier {
        source.access, dst.access,
        source.layout, dst.layout,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        texture,
        vk::ImageSubresourceRange(info.aspect, 0, 1, 0, 1)
    };

    std::array<vk::ImageMemoryBarrier, 1> barriers{ barrier };
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.pipelineBarrier(source.stage, dst.stage, vk::DependencyFlagBits::eByRegion, {}, {}, barriers);
    layout = new_layout;
}

void VKTexture::Upload(u32 level, u32 layer, u32 row_length, vk::Rect2D region, std::span<u8> pixels) {
    auto [buffer, offset] = g_vk_task_scheduler->RequestStaging(pixels.size());
    if (!buffer) {
        LOG_ERROR(Render_Vulkan, "Cannot copy pixels without staging buffer!");
    }

    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();

    // Copy pixels to staging buffer
    std::memcpy(buffer, pixels.data(), pixels.size());

    vk::BufferImageCopy copy_region{
        offset, row_length, region.extent.height,
        {info.aspect, level, layer, 1},
        { region.offset.x, region.offset.y, 0 },
        { region.extent.width, region.extent.height, 1 }
    };

    // Transition image to transfer format
    Transition(vk::ImageLayout::eTransferDstOptimal);
    command_buffer.copyBufferToImage(g_vk_task_scheduler->GetStaging().GetBuffer(),
                                     texture, vk::ImageLayout::eTransferDstOptimal,
                                     copy_region);

    // Prepare for shader reads
    Transition(vk::ImageLayout::eShaderReadOnlyOptimal);
}

}
