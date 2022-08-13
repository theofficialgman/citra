// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/common/pool_manager.h"
#include "video_core/renderer_vulkan/pica_to_vulkan.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

namespace VideoCore::Vulkan {

inline vk::Format ToVkFormat(TextureFormat format) {
    switch (format) {
    case TextureFormat::RGBA8:
        return vk::Format::eR8G8B8A8Unorm;
    case TextureFormat::RGB8:
        return vk::Format::eR8G8B8Unorm;
    case TextureFormat::RGB5A1:
        return vk::Format::eR5G5B5A1UnormPack16;
    case TextureFormat::RGB565:
        return vk::Format::eR5G6B5UnormPack16;
    case TextureFormat::RGBA4:
        return vk::Format::eR4G4B4A4UnormPack16;
    case TextureFormat::D16:
        return vk::Format::eD16Unorm;
    case TextureFormat::D24:
        return vk::Format::eX8D24UnormPack32;
    case TextureFormat::D24S8:
        return vk::Format::eD24UnormS8Uint;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown texture format {}!", format);
        return vk::Format::eUndefined;
    }
}

inline vk::ImageType ToVkImageType(TextureType type) {
    switch (type) {
    case TextureType::Texture1D:
        return vk::ImageType::e1D;
    case TextureType::Texture2D:
        return vk::ImageType::e2D;
    case TextureType::Texture3D:
        return vk::ImageType::e3D;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown texture type {}!", type);
        return vk::ImageType::e2D;
    }
}

inline vk::ImageViewType ToVkImageViewType(TextureViewType view_type) {
    switch (view_type) {
    case TextureViewType::View1D:
        return vk::ImageViewType::e1D;
    case TextureViewType::View2D:
        return vk::ImageViewType::e2D;
    case TextureViewType::View3D:
        return vk::ImageViewType::e3D;
    case TextureViewType::ViewCube:
        return vk::ImageViewType::eCube;
    case TextureViewType::View1DArray:
        return vk::ImageViewType::e1DArray;
    case TextureViewType::View2DArray:
        return vk::ImageViewType::e2DArray;
    case TextureViewType::ViewCubeArray:
        return vk::ImageViewType::eCubeArray;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown texture view type {}!", view_type);
        return vk::ImageViewType::e2D;
    }
}

Texture::Texture(Instance& instance, CommandScheduler& scheduler, PoolManager& pool_manager,
                 const TextureInfo& info) : TextureBase(info), instance(instance), scheduler(scheduler),
    pool_manager(pool_manager) {

    // Convert the input format to another that supports attachments
    advertised_format = ToVkFormat(info.format);
    internal_format = instance.GetFormatAlternative(advertised_format);
    aspect = GetImageAspect(advertised_format);

    vk::Device device = instance.GetDevice();
    const vk::ImageCreateInfo image_info = {
        .flags = info.view_type == TextureViewType::ViewCube ?
                 vk::ImageCreateFlagBits::eCubeCompatible :
                 vk::ImageCreateFlags{},
        .imageType = ToVkImageType(info.type),
        .format = internal_format,
        .extent = {info.width, info.height, 1},
        .mipLevels = info.levels,
        .arrayLayers = info.view_type == TextureViewType::ViewCube ? 6u : 1u,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = GetImageUsage(aspect),
    };

    const VmaAllocationCreateInfo alloc_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    };

    VkImage unsafe_image = VK_NULL_HANDLE;
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);
    VmaAllocator allocator = instance.GetAllocator();

    // Allocate texture memory
    if (auto result = vmaCreateImage(allocator, &unsafe_image_info, &alloc_info, &unsafe_image, &allocation, nullptr);
            result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }
    image = vk::Image{unsafe_image};

    const vk::ImageViewCreateInfo view_info = {
        .image = image,
        .viewType = ToVkImageViewType(info.view_type),
        .format = internal_format,
        .subresourceRange = {aspect, 0, 1, 0, 1} // Should this include the levels?
    };

    // Create image view
    image_view = device.createImageView(view_info);
}

Texture::Texture(Instance& instance, CommandScheduler& scheduler, PoolManager& pool_manager,
                 vk::Image image, vk::Format format, const TextureInfo& info) : TextureBase(info),
    instance(instance), scheduler(scheduler), pool_manager(pool_manager), image(image), is_texture_owned(false) {

    advertised_format = internal_format = format;
    aspect = vk::ImageAspectFlagBits::eColor;

    const vk::ImageViewCreateInfo view_info = {
        .image = image,
        .viewType = ToVkImageViewType(info.view_type),
        .format = format,
        .subresourceRange = {aspect, 0, info.levels, 0, 1}
    };

    // Create image view
    vk::Device device = instance.GetDevice();
    image_view = device.createImageView(view_info);
}

Texture::~Texture() {
    if (image && is_texture_owned) {
        auto deleter = [image = image, allocation = allocation,
                        view = image_view](vk::Device device, VmaAllocator allocator) {
            device.destroyImageView(view);
            vmaDestroyImage(allocator, static_cast<VkImage>(image), allocation);
        };

        // Schedule deletion of the texture after it's no longer used by the GPU
        scheduler.Schedule(deleter);
    } else if (!is_texture_owned) {
        // If the texture is not owning, destroy the view immediately.
        // Synchronization is the caller's responsibility
        vk::Device device = instance.GetDevice();
        device.destroyImageView(image_view);
    }
}

void Texture::Free() {
    pool_manager.Free<Texture>(this);
}

void Texture::TransitionSubresource(vk::CommandBuffer command_buffer, vk::ImageLayout new_layout,
                                    u32 level, u32 level_count) {
    // Don't do anything if the image is already in the wanted layout
    if (new_layout == layout) {
        return;
    }

    struct LayoutInfo {
        vk::AccessFlags access;
        vk::PipelineStageFlags stage;
    };

    // Get optimal transition settings for every image layout. Settings taken from Dolphin
    auto GetLayoutInfo = [](vk::ImageLayout layout) -> LayoutInfo {
        LayoutInfo info;
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
            info.access = vk::AccessFlagBits::eColorAttachmentRead |
                    vk::AccessFlagBits::eColorAttachmentWrite;
            info.stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            // Image was being used as a depthstencil attachment, so ensure all writes have completed.
            info.access = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                    vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            info.stage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                    vk::PipelineStageFlagBits::eLateFragmentTests;
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

    LayoutInfo source = GetLayoutInfo(layout);
    LayoutInfo dest = GetLayoutInfo(new_layout);

    const vk::ImageMemoryBarrier barrier = {
        .srcAccessMask = source.access,
        .dstAccessMask = dest.access,
        .oldLayout = layout,
        .newLayout = new_layout,
        .image = image,
        .subresourceRange = {aspect, level, level_count, 0, 1}
    };

    // Submit pipeline barrier
    command_buffer.pipelineBarrier(source.stage, dest.stage,
                                   vk::DependencyFlagBits::eByRegion,
                                   {}, {}, barrier);

    // Update layouts
    layout = new_layout;
}

void Texture::Transition(vk::CommandBuffer command_buffer, vk::ImageLayout new_layout) {
    TransitionSubresource(command_buffer, new_layout, 0, info.levels);
}

void Texture::Upload(Rect2D rectangle, u32 stride, std::span<const u8> data, u32 level) {
    const u64 byte_count = data.size();
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();

    // If the adverised format supports blitting then use GPU accelerated
    // format conversion.
    if (internal_format != advertised_format &&
        instance.IsFormatSupported(advertised_format, vk::FormatFeatureFlagBits::eBlitSrc)) {
        // Creating a new staging texture for each upload/download is expensive
        // but this path should not be common. TODO: Profile this
        StagingTexture staging{instance, scheduler, info};

        const std::array offsets = {
            vk::Offset3D{rectangle.x, rectangle.y, 0},
            vk::Offset3D{static_cast<s32>(rectangle.x + rectangle.width),
                         static_cast<s32>(rectangle.y + rectangle.height), 0}
        };

        const vk::ImageBlit image_blit = {
            .srcSubresource = {
                .aspectMask = aspect,
                .mipLevel = level,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .srcOffsets = offsets,
            .dstSubresource = {
                .aspectMask = aspect,
                .mipLevel = level,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .dstOffsets = offsets
        };

        // Copy data to staging texture
        std::memcpy(staging.GetMappedPtr(), data.data(), byte_count);
        staging.Commit(byte_count);

        Transition(command_buffer, vk::ImageLayout::eTransferDstOptimal);

        // Blit
        command_buffer.blitImage(staging.GetHandle(), vk::ImageLayout::eGeneral,
                                 image, vk::ImageLayout::eTransferDstOptimal,
                                 image_blit, vk::Filter::eNearest);

    // Otherwise use normal staging buffer path with possible CPU conversion
    } else {
        Buffer& staging = scheduler.GetCommandUploadBuffer();
        const u64 staging_offset = staging.GetCurrentOffset();

        // Copy pixels to the staging buffer
        if (advertised_format == vk::Format::eR8G8B8Unorm) {
            const u32 new_byte_count = (byte_count / 3) * 4;
            auto slice = staging.Map(new_byte_count);

            u32 dst_offset = 0;
            for (u32 src_offset = 0; src_offset < byte_count; src_offset += 3) {
                slice[dst_offset] = data[src_offset];
                slice[dst_offset+1] = data[src_offset+1];
                slice[dst_offset+2] = data[src_offset+2];
                slice[dst_offset+3] = 255;
                dst_offset += 4;
            }

            staging.Commit(new_byte_count);
        } else {
            // TODO: Handle format convertions and depth/stencil uploads
            ASSERT(aspect == vk::ImageAspectFlagBits::eColor &&
                   advertised_format == internal_format);

            auto slice = staging.Map(byte_count);
            std::memcpy(slice.data(), data.data(), byte_count);
            staging.Commit(byte_count);
        }

        const vk::BufferImageCopy copy_region = {
            .bufferOffset = staging_offset,
            .bufferRowLength = stride,
            .bufferImageHeight = rectangle.height,
            .imageSubresource = {
                .aspectMask = aspect,
                .mipLevel = level,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {rectangle.x, rectangle.y, 0},
            .imageExtent = {rectangle.width, rectangle.height, 1}
        };

        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        Transition(command_buffer, vk::ImageLayout::eTransferDstOptimal);

        // Copy staging buffer to the texture
        command_buffer.copyBufferToImage(staging.GetHandle(), image,
                                         vk::ImageLayout::eTransferDstOptimal,
                                         copy_region);
    }

    Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture::Download(Rect2D rectangle, u32 stride, std::span<u8> data, u32 level) {
    const std::size_t byte_count = data.size();
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();

    // If the adverised format supports blitting use GPU accelerated format conversion.
    if (internal_format != advertised_format &&
        instance.IsFormatSupported(advertised_format, vk::FormatFeatureFlagBits::eBlitDst)) {
        // Creating a new staging texture for each upload/download is expensive
        // but this path should not be common. TODO: Profile this
        StagingTexture staging{instance, scheduler, info};

        const std::array offsets = {
            vk::Offset3D{rectangle.x, rectangle.y, 0},
            vk::Offset3D{static_cast<s32>(rectangle.x + rectangle.width),
                         static_cast<s32>(rectangle.y + rectangle.height), 0}
        };

        const vk::ImageBlit image_blit = {
            .srcSubresource = {
                .aspectMask = aspect,
                .mipLevel = level,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .srcOffsets = offsets,
            .dstSubresource = {
                .aspectMask = aspect,
                .mipLevel = level,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .dstOffsets = offsets
        };

        Transition(command_buffer, vk::ImageLayout::eTransferSrcOptimal);

        // Blit
        command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                                 staging.GetHandle(), vk::ImageLayout::eGeneral,
                                 image_blit, vk::Filter::eNearest);

        // TODO: Async downloads
        scheduler.Submit(true);

        // Copy data to the destination
        staging.Commit(byte_count);
        std::memcpy(data.data(), staging.GetMappedPtr(), byte_count);

    // Otherwise use normal staging buffer path with possible CPU conversion
    } else {
        Buffer& staging = scheduler.GetCommandUploadBuffer();
        const u64 staging_offset = staging.GetCurrentOffset();

        if (advertised_format == vk::Format::eD24UnormS8Uint) {
            ASSERT(16 * 1024 * 1024 - staging_offset >= data.size() * 2);
        } else {
            ASSERT(aspect == vk::ImageAspectFlagBits::eColor &&
               advertised_format == internal_format);
        }

        u32 region_count = 0;
        std::array<vk::BufferImageCopy, 2> copy_regions;

        vk::BufferImageCopy copy_region = {
            .bufferOffset = staging_offset,
            .bufferRowLength = stride,
            .bufferImageHeight = rectangle.height,
            .imageSubresource = {
                .aspectMask = aspect,
                .mipLevel = level,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {rectangle.x, rectangle.y, 0},
            .imageExtent = {rectangle.width, rectangle.height, 1}
        };

        if (aspect & vk::ImageAspectFlagBits::eColor) {
            copy_regions[region_count++] = copy_region;
        } else if (aspect & vk::ImageAspectFlagBits::eStencil) {
            // Depth aspect download
            copy_region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
            copy_regions[region_count++] = copy_region;

            // Stencil aspect download
            copy_region.bufferOffset += data.size();
            copy_region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
            copy_regions[region_count++] = copy_region;
        }

        Transition(command_buffer, vk::ImageLayout::eTransferSrcOptimal);

        // Copy pixel data to the staging buffer
        command_buffer.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal,
                                         staging.GetHandle(), region_count, copy_regions.data());

        // TODO: Async downloads
        scheduler.Submit(true);

        // Copy data to the destination
        if (advertised_format == vk::Format::eD24UnormS8Uint) {
            const u32 new_byte_count = data.size() + (data.size() / 4);
            auto memory = staging.Map(new_byte_count);

            u32 depth_offset = 0;
            u32 stencil_offset = data.size();
            for (u32 dst_offset = 0; dst_offset < byte_count; dst_offset += 4) {
                float depth;
                std::memcpy(&depth, memory.data() + depth_offset, sizeof(float));
                u32 depth_uint = depth * 0xFFFFFF;

                std::memcpy(data.data() + dst_offset, &depth_uint, 3);
                data[dst_offset+3] = memory[stencil_offset];

                depth_offset += 4;
                stencil_offset += 1;
            }
        } else {
            auto memory = staging.Map(byte_count);
            std::memcpy(data.data(), memory.data(), byte_count);
        }
    }

    Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture::BlitTo(TextureHandle dest, Common::Rectangle<u32> source_rect, Common::Rectangle<u32> dest_rect,
                     u32 src_level, u32 dest_level, u32 src_layer, u32 dest_layer) {

    Texture* dest_texture = static_cast<Texture*>(dest.Get());

    // Prepare images for transfer
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    Transition(command_buffer, vk::ImageLayout::eTransferSrcOptimal);
    dest_texture->Transition(command_buffer, vk::ImageLayout::eTransferDstOptimal);

    const std::array source_offsets = {
        vk::Offset3D{static_cast<s32>(source_rect.left), static_cast<s32>(source_rect.bottom), 0},
        vk::Offset3D{static_cast<s32>(source_rect.right), static_cast<s32>(source_rect.top), 1}
    };

    const std::array dest_offsets = {
        vk::Offset3D{static_cast<s32>(dest_rect.left), static_cast<s32>(dest_rect.bottom), 0},
        vk::Offset3D{static_cast<s32>(dest_rect.right), static_cast<s32>(dest_rect.top), 1}
    };

    const vk::ImageBlit blit_area = {
      .srcSubresource = {
            .aspectMask = aspect,
            .mipLevel = src_level,
            .baseArrayLayer = src_layer,
            .layerCount = 1
       },
      .srcOffsets = source_offsets,
      .dstSubresource = {
            .aspectMask = dest_texture->GetAspectFlags(),
            .mipLevel = dest_level,
            .baseArrayLayer = dest_layer,
            .layerCount = 1
       },
      .dstOffsets = dest_offsets
    };

    command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                             dest_texture->GetHandle(), vk::ImageLayout::eTransferDstOptimal,
                             blit_area, vk::Filter::eNearest);

    // Prepare for shader reads
    Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
    dest_texture->Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

// TODO: Use AMD single pass downsampler
void Texture::GenerateMipmaps() {
    s32 current_width = info.width;
    s32 current_height = info.height;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    for (u32 i = 1; i < info.levels; i++) {
        TransitionSubresource(command_buffer, vk::ImageLayout::eTransferSrcOptimal, i - 1);
        TransitionSubresource(command_buffer, vk::ImageLayout::eTransferDstOptimal, i);

        const std::array source_offsets = {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{current_width, current_height, 1}
        };

        const std::array dest_offsets = {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{current_width > 1 ? current_width / 2 : 1,
                         current_height > 1 ? current_height / 2 : 1, 1}
        };

        const vk::ImageBlit blit_area = {
          .srcSubresource = {
                .aspectMask = aspect,
                .mipLevel = i - 1,
                .baseArrayLayer = 0,
                .layerCount = 1
           },
          .srcOffsets = source_offsets,
          .dstSubresource = {
                .aspectMask = aspect,
                .mipLevel = i,
                .baseArrayLayer = 0,
                .layerCount = 1
           },
          .dstOffsets = dest_offsets
        };

        command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                                 image, vk::ImageLayout::eTransferDstOptimal,
                                 blit_area, vk::Filter::eLinear);
    }

    // Prepare for shader reads
    Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture::CopyFrom(TextureHandle source) {
    const vk::ImageCopy image_copy = {
        .srcSubresource = {
            .aspectMask = aspect,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .srcOffset = {0, 0, 0},
        .dstSubresource = {
            .aspectMask = aspect,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
        .dstOffset = {0, 0, 0},
        .extent = {source->GetWidth(), source->GetHeight(), 1}
    };

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    Texture* texture = static_cast<Texture*>(source.Get());

    // Transition images
    vk::ImageLayout old_layout = texture->GetLayout();
    texture->Transition(command_buffer, vk::ImageLayout::eTransferSrcOptimal);
    Transition(command_buffer, vk::ImageLayout::eTransferDstOptimal);

    // Perform copy
    command_buffer.copyImage(texture->GetHandle(), vk::ImageLayout::eTransferSrcOptimal,
                             image, vk::ImageLayout::eTransferDstOptimal, image_copy);

    // We need to preserve the old texture layout
    texture->Transition(command_buffer, old_layout);
    Transition(command_buffer, vk::ImageLayout::eShaderReadOnlyOptimal);
}

StagingTexture::StagingTexture(Instance& instance, CommandScheduler& scheduler, const TextureInfo& info) :
    TextureBase(info), instance(instance), scheduler(scheduler) {

    format = ToVkFormat(info.format);
    const vk::ImageCreateInfo image_info = {
        .flags = info.view_type == TextureViewType::ViewCube ?
                 vk::ImageCreateFlagBits::eCubeCompatible :
                 vk::ImageCreateFlags{},
        .imageType = ToVkImageType(info.type),
        .format = format,
        .extent = {info.width, info.height, 1},
        .mipLevels = 1,
        .arrayLayers = info.view_type == TextureViewType::ViewCube ? 6u : 1u,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eLinear,
        .usage = vk::ImageUsageFlagBits::eTransferSrc |
                 vk::ImageUsageFlagBits::eTransferDst,
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VkImage unsafe_image = VK_NULL_HANDLE;
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);
    VmaAllocator allocator = instance.GetAllocator();

    // Allocate texture memory
    if (auto result = vmaCreateImage(allocator, &unsafe_image_info, &alloc_create_info, &unsafe_image, &allocation, nullptr);
            result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Allocation of staging texture failed with error {}", result);
        UNREACHABLE();
    }

    image = vk::Image{unsafe_image};

    // Map memory
    vmaMapMemory(allocator, allocation, &mapped_ptr);

    // For staging textures the most conventient layout is VK_IMAGE_LAYOUT_GENERAL because it allows
    // for well defined host access and works with vkCmdBlitImage, thus eliminating the need for layout transitions
    const vk::ImageMemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = image,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, info.levels, 0, 1}
    };

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   vk::DependencyFlagBits::eByRegion,
                                   {}, {}, barrier);
}

StagingTexture::~StagingTexture() {
    if (image) {
        auto deleter = [allocation = allocation,
                        image = image](vk::Device device, VmaAllocator allocator) {
            vmaUnmapMemory(allocator, allocation);
            vmaDestroyImage(allocator, static_cast<VkImage>(image), allocation);
        };

        // Schedule deletion of the texture after it's no longer used by the GPU
        scheduler.Schedule(deleter);
    }
}

void StagingTexture::Commit(u32 size) {
    VmaAllocator allocator = instance.GetAllocator();
    vmaFlushAllocation(allocator, allocation, 0, size);
}

Sampler::Sampler(Instance& instance, PoolManager& pool_manager, SamplerInfo info) :
    SamplerBase(info), instance(instance), pool_manager(pool_manager) {

    auto properties = instance.GetPhysicalDevice().getProperties();
    const auto filtering = PicaToVK::TextureFilterMode(info.mag_filter,
                                                       info.min_filter,
                                                       info.mip_filter);
    const vk::SamplerCreateInfo sampler_info = {
        .magFilter = filtering.mag_filter,
        .minFilter = filtering.min_filter,
        .mipmapMode = filtering.mip_mode,
        .addressModeU = PicaToVK::WrapMode(info.wrap_s),
        .addressModeV = PicaToVK::WrapMode(info.wrap_t),
        .anisotropyEnable = true,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = false,
        .compareOp = vk::CompareOp::eAlways,
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = false
    };

    vk::Device device = instance.GetDevice();
    sampler = device.createSampler(sampler_info);
}

Sampler::~Sampler() {
    vk::Device device = instance.GetDevice();
    device.destroySampler(sampler);
}

void Sampler::Free() {
    pool_manager.Free<Sampler>(this);
}

} // namespace VideoCore::Vulkan
