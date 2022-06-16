// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <fstream>
#include <iostream>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_state.h"

namespace Vulkan {

static int BytesPerPixel(vk::Format format) {
    switch (format) {
    case vk::Format::eD32SfloatS8Uint:
        return 5;
    case vk::Format::eD32Sfloat:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eR8G8B8A8Uint:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eD24UnormS8Uint:
        return 4;
    case vk::Format::eR8G8B8Uint:
    case vk::Format::eR8G8B8Srgb:
        return 3;
    case vk::Format::eR5G6B5UnormPack16:
    case vk::Format::eR5G5B5A1UnormPack16:
    case vk::Format::eR4G4B4A4UnormPack16:
    case vk::Format::eD16Unorm:
        return 2;
    default:
        UNREACHABLE();
    }
}

vk::ImageAspectFlags GetImageAspect(vk::Format format) {
    vk::ImageAspectFlags flags;
    switch (format) {
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        flags = vk::ImageAspectFlagBits::eStencil | vk::ImageAspectFlagBits::eDepth;
        break;
    case vk::Format::eD16Unorm:
    case vk::Format::eD32Sfloat:
        flags = vk::ImageAspectFlagBits::eDepth;
        break;
    default:
        flags = vk::ImageAspectFlagBits::eColor;
    }

    return flags;
}

VKTexture::~VKTexture() {
    Destroy();
}

void VKTexture::Create(const Info& create_info) {
    auto device = g_vk_instace->GetDevice();
    info = create_info;

    // Emulate RGB8 format with RGBA8
    is_rgb = false;
    if (info.format == vk::Format::eR8G8B8Srgb) {
        is_rgb = true;
        info.format = vk::Format::eR8G8B8A8Srgb;
    }

    is_d24s8 = false;
    if (info.format == vk::Format::eD24UnormS8Uint) {
        is_d24s8 = true;
        info.format = vk::Format::eD32SfloatS8Uint;
    }

    std::cout << "New surface!\n";

    // Create the texture
    image_size = info.width * info.height * BytesPerPixel(info.format);
    aspect = GetImageAspect(info.format);

    vk::ImageCreateFlags flags{};
    if (info.view_type == vk::ImageViewType::eCube) {
        flags = vk::ImageCreateFlagBits::eCubeCompatible;
    }

    vk::ImageCreateInfo image_info {
        flags, info.type, info.format,
        { info.width, info.height, 1 }, info.levels, info.layers,
        static_cast<vk::SampleCountFlagBits>(info.multisamples),
        vk::ImageTiling::eOptimal, info.usage
    };

    texture = device.createImage(image_info);

    // Create texture memory
    auto requirements = device.getImageMemoryRequirements(texture);
    auto memory_index = VKBuffer::FindMemoryType(requirements.memoryTypeBits,
                                                 vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::MemoryAllocateInfo alloc_info(requirements.size, memory_index);

    memory = device.allocateMemory(alloc_info);
    device.bindImageMemory(texture, memory, 0);

    // Create texture view
    vk::ImageViewCreateInfo view_info {
        {}, texture, info.view_type, info.format, {},
        {aspect, 0, info.levels, 0, info.layers}
    };

    view = device.createImageView(view_info);
}

void VKTexture::Adopt(const Info& create_info, vk::Image image) {
    info = create_info;
    image_size = info.width * info.height * BytesPerPixel(info.format);
    aspect = GetImageAspect(info.format);
    texture = image;

    // Create texture view
    vk::ImageViewCreateInfo view_info {
        {}, texture, info.view_type, info.format, {},
        {aspect, 0, info.levels, 0, info.layers}
    };

    auto device = g_vk_instace->GetDevice();
    view = device.createImageView(view_info);
    adopted = true;
}

void VKTexture::Destroy() {
    if (texture && !adopted) {
        // Make sure to unbind the texture before destroying it
        auto& state = VulkanState::Get();
        state.UnbindTexture(*this);

        auto deleter = [this]() {
            auto device = g_vk_instace->GetDevice();
            if (texture) {
                std::cout << "Surface destroyed!\n";
                device.destroyImage(texture);
                device.destroyImageView(view);
                device.freeMemory(memory);
            }
        };

        // Schedule deletion of the texture after it's no longer used
        // by the GPU
        g_vk_task_scheduler->Schedule(deleter);
    }

    // If the image was adopted (probably from the swapchain) then only
    // destroy the view
    if (adopted) {
        g_vk_task_scheduler->Schedule([this](){
            auto device = g_vk_instace->GetDevice();
            device.destroyImageView(view);
        });
    }
}

void VKTexture::Transition(vk::CommandBuffer cmdbuffer, vk::ImageLayout new_layout) {
    Transition(cmdbuffer, new_layout, 0, info.levels, 0, info.layers);
}

void VKTexture::Transition(vk::CommandBuffer cmdbuffer, vk::ImageLayout new_layout,
                           u32 start_level, u32 level_count, u32 start_layer, u32 layer_count) {
    if (new_layout == layout) {
        return;
    }

    struct LayoutInfo {
        vk::ImageLayout layout;
        vk::AccessFlags access;
        vk::PipelineStageFlags stage;
    };

    // Get optimal transition settings for every image layout. Settings taken from Dolphin
    auto layout_info = [](vk::ImageLayout layout) -> LayoutInfo {
        LayoutInfo info{ .layout = layout };
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

        case vk::ImageLayout::ePresentSrcKHR:
            info.access = vk::AccessFlagBits::eNone;
            info.stage = vk::PipelineStageFlagBits::eBottomOfPipe;
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
            UNREACHABLE();
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
        vk::ImageSubresourceRange{aspect, start_level, level_count, start_layer, layer_count}
    };

    cmdbuffer.pipelineBarrier(source.stage, dst.stage, vk::DependencyFlagBits::eByRegion, {}, {}, barrier);
    layout = new_layout;
}

void VKTexture::OverrideImageLayout(vk::ImageLayout new_layout) {
    layout = new_layout;
}

void VKTexture::Upload(u32 level, u32 layer, u32 row_length, vk::Rect2D region, std::span<u8> pixels) {
    u32 request_size = is_rgb ? (pixels.size() / 3) * 4 :
                       (is_d24s8 ? (pixels.size() / 4) * 5 : pixels.size());
    auto [buffer, offset] = g_vk_task_scheduler->RequestStaging(request_size);
    if (!buffer) {
        LOG_ERROR(Render_Vulkan, "Cannot upload pixels without staging buffer!");
    }

    // Copy pixels to staging buffer
    auto& state = VulkanState::Get();
    state.EndRendering();

    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();

    // Automatically convert RGB to RGBA
    if (is_rgb) {
        auto data = RGBToRGBA(pixels);
        std::memcpy(buffer, data.data(), data.size());
    }
    else if (is_d24s8) {
        auto data = D24S8ToD32S8(pixels);
        std::memcpy(buffer, data.data(), data.size() * sizeof(data[0]));
    }
    else {
        std::memcpy(buffer, pixels.data(), pixels.size());
    }

    vk::BufferImageCopy copy_region{
        offset, row_length, region.extent.height,
        {aspect, level, layer, 1},
        {region.offset.x, region.offset.y, 0},
        {region.extent.width, region.extent.height, 1}
    };

    // Transition image to transfer format
    Transition(cmdbuffer, vk::ImageLayout::eTransferDstOptimal);

    cmdbuffer.copyBufferToImage(g_vk_task_scheduler->GetStaging().GetBuffer(),
                                     texture, vk::ImageLayout::eTransferDstOptimal,
                                     copy_region);

    // Prepare image for shader reads
    Transition(cmdbuffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void VKTexture::Download(u32 level, u32 layer, u32 row_length, vk::Rect2D region, std::span<u8> memory) {
    u32 request_size = is_rgb ? (memory.size() / 3) * 4 :
                       (is_d24s8 ? (memory.size() / 4) * 5 : memory.size());
    auto [buffer, offset] = g_vk_task_scheduler->RequestStaging(request_size);
    if (!buffer) {
        LOG_ERROR(Render_Vulkan, "Cannot download texture without staging buffer!");
    }

    auto& state = VulkanState::Get();
    state.EndRendering();

    auto cmdbuffer = g_vk_task_scheduler->GetRenderCommandBuffer();

    // Copy pixels to staging buffer
    vk::BufferImageCopy download_region{
        offset, row_length, region.extent.height,
        {aspect, level, layer, 1},
        {region.offset.x, region.offset.y, 0},
        {region.extent.width, region.extent.height, 1}
    };

    // Automatically convert RGB to RGBA
    if (is_rgb) {
        auto data = RGBAToRGB(memory);
        std::memcpy(buffer, data.data(), data.size());
    }
    else if (is_d24s8) {
        auto data = D32S8ToD24S8(memory);
        std::memcpy(buffer, data.data(), data.size() * sizeof(data[0]));
    }
    else {
        std::memcpy(buffer, memory.data(), memory.size());
    }

    // Transition image to transfer format
    auto old_layout = GetLayout();
    Transition(cmdbuffer, vk::ImageLayout::eTransferSrcOptimal);

    cmdbuffer.copyImageToBuffer(texture, vk::ImageLayout::eTransferSrcOptimal,
                                     g_vk_task_scheduler->GetStaging().GetBuffer(),
                                     download_region);

    // Wait for the data to be available
    // NOTE: This is really slow and should be reworked
    g_vk_task_scheduler->Submit(true);
    std::memcpy(memory.data(), buffer, memory.size_bytes());

    // Restore layout
    Transition(cmdbuffer, old_layout);
}

template <typename Out, typename In>
std::span<Out> SpanCast(std::span<In> span) {
    return std::span(reinterpret_cast<Out*>(span.data()), span.size_bytes() / sizeof(Out));
}

std::vector<u8> VKTexture::RGBToRGBA(std::span<u8> data) {
    ASSERT(data.size() % 3 == 0);

    u32 new_size = (data.size() / 3) * 4;
    std::vector<u8> rgba(new_size);

    u32 dst_pos = 0;
    for (u32 i = 0; i < data.size(); i += 3) {
        std::memcpy(rgba.data() + dst_pos, data.data() + i, 3);
        rgba[dst_pos + 3] = 255u;
        dst_pos += 4;
    }

    return rgba;
}

std::vector<u64> VKTexture::D24S8ToD32S8(std::span<u8> data) {
    ASSERT(data.size() % 4 == 0);

    std::vector<u64> d32s8;
    std::span<u32> d24s8 = SpanCast<u32>(data);

    d32s8.reserve(data.size() * 2);
    std::ranges::transform(d24s8, std::back_inserter(d32s8), [](u32 comp) -> u64 {
        // Convert normalized 24bit depth component to floating point
        float fdepth = static_cast<float>(comp & 0xFFFFFF) / 0xFFFFFF;
        u64 result = static_cast<u64>(comp) << 8;

        // Use std::memcpy to avoid the unsafe casting required to preserve the floating
        // point bits
        std::memcpy(&result, &fdepth, 4);
        return result;
    });

    return d32s8;
}

std::vector<u8> VKTexture::RGBAToRGB(std::span<u8> data) {
    ASSERT(data.size() % 4 == 0);

    u32 new_size = (data.size() / 4) * 3;
    std::vector<u8> rgb(new_size);

    u32 dst_pos = 0;
    for (u32 i = 0; i < data.size(); i += 4) {
        std::memcpy(rgb.data() + dst_pos, data.data() + i, 3);
        dst_pos += 3;
    }

    return rgb;
}

std::vector<u32> VKTexture::D32S8ToD24S8(std::span<u8> data) {
    ASSERT(data.size() % 8 == 0);

    std::vector<u32> d24s8;
    std::span<u64> d32s8 = SpanCast<u64>(data);

    d24s8.reserve(data.size() / 2);
    std::ranges::transform(d32s8, std::back_inserter(d24s8), [](u64 comp) -> u32 {
        // Convert floating point to 24bit normalized depth
        float fdepth = 0.f;
        u32 depth = comp & 0xFFFFFFFF;
        std::memcpy(&fdepth, &depth, 4);

        u32 stencil = (comp >> 32) & 0xFF;
        u64 result = static_cast<u32>(fdepth * 0xFFFFFF) | (stencil << 24);
        return result;
    });

    return d24s8;
}

} // namespace Vulkan
