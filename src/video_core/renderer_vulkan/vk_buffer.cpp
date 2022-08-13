// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/common/pool_manager.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace VideoCore::Vulkan {

inline vk::BufferUsageFlags ToVkBufferUsage(BufferUsage usage) {
    switch (usage) {
    case BufferUsage::Vertex:
        return vk::BufferUsageFlagBits::eVertexBuffer;
    case BufferUsage::Index:
        return vk::BufferUsageFlagBits::eIndexBuffer;
    case BufferUsage::Uniform:
        return vk::BufferUsageFlagBits::eUniformBuffer;
    case BufferUsage::Texel:
        return vk::BufferUsageFlagBits::eUniformTexelBuffer;
    case BufferUsage::Staging:
        return vk::BufferUsageFlagBits::eTransferSrc;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown buffer usage flag {}!", usage);
        UNREACHABLE();
    }
}

inline vk::Format ToVkViewFormat(ViewFormat format) {
    switch (format) {
    case ViewFormat::R32Float:
        return vk::Format::eR32Sfloat;
    case ViewFormat::R32G32Float:
        return vk::Format::eR32G32Sfloat;
    case ViewFormat::R32G32B32Float:
        return vk::Format::eR32G32B32Sfloat;
    case ViewFormat::R32G32B32A32Float:
        return vk::Format::eR32G32B32A32Sfloat;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown buffer view format {}!", format);
        UNREACHABLE();
    }
}

inline auto ToVkAccessStageFlags(BufferUsage usage) {
    std::pair<vk::AccessFlags, vk::PipelineStageFlags> result;
    switch (usage) {
    case BufferUsage::Vertex:
        result = std::make_pair(vk::AccessFlagBits::eVertexAttributeRead,
                                vk::PipelineStageFlagBits::eVertexInput);
        break;
    case BufferUsage::Index:
        result = std::make_pair(vk::AccessFlagBits::eIndexRead,
                                vk::PipelineStageFlagBits::eVertexInput);
    case BufferUsage::Uniform:
        result = std::make_pair(vk::AccessFlagBits::eUniformRead,
                                vk::PipelineStageFlagBits::eVertexShader |
                                vk::PipelineStageFlagBits::eFragmentShader);
    case BufferUsage::Texel:
        result = std::make_pair(vk::AccessFlagBits::eShaderRead,
                                vk::PipelineStageFlagBits::eFragmentShader);
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown BufferUsage flag!");
    }

    return result;
}

Buffer::Buffer(Instance& instance, CommandScheduler& scheduler, PoolManager& pool_manager, const BufferInfo& info)
    : BufferBase(info), instance(instance), scheduler(scheduler), pool_manager(pool_manager) {

    vk::BufferCreateInfo buffer_info = {
        .size = info.capacity,
        .usage = ToVkBufferUsage(info.usage) | vk::BufferUsageFlagBits::eTransferDst
    };

    VmaAllocationCreateInfo alloc_create_info = {
        .flags = info.usage == BufferUsage::Staging ?
                (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT) :
                VmaAllocationCreateFlags{},
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VkBuffer unsafe_buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationInfo alloc_info;
    VmaAllocator allocator = instance.GetAllocator();

    // Allocate texture memory
    vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_create_info,
                    &unsafe_buffer, &allocation, &alloc_info);
    buffer = vk::Buffer{unsafe_buffer};

    vk::Device device = instance.GetDevice();
    for (u32 view = 0; view < info.views.size(); view++) {
        if (info.views[view] == ViewFormat::Undefined) {
            view_count = view;
            break;
        }

        const vk::BufferViewCreateInfo view_info = {
            .buffer = buffer,
            .format = ToVkViewFormat(info.views[view]),
            .range = info.capacity
        };

        views[view] = device.createBufferView(view_info);
    }

    // Map memory
    if (info.usage == BufferUsage::Staging) {
        mapped_ptr = alloc_info.pMappedData;
    }
}

Buffer::~Buffer() {
    if (buffer) {
        auto deleter = [allocation = allocation,
                        buffer = buffer,
                        views = views](vk::Device device, VmaAllocator allocator) {
            vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);

            u32 view_index = 0;
            while (views[view_index]) {
                device.destroyBufferView(views[view_index++]);
            }
        };

        // Delete the buffer immediately if it's allocated in host memory
        if (info.usage == BufferUsage::Staging) {
            vk::Device device = instance.GetDevice();
            VmaAllocator allocator = instance.GetAllocator();
            deleter(device, allocator);
        } else {
            scheduler.Schedule(deleter);
        }
    }
}

void Buffer::Free() {
    pool_manager.Free<Buffer>(this);
}

std::span<u8> Buffer::Map(u32 size, u32 alignment) {
    ASSERT(size <= info.capacity && alignment <= info.capacity);

    if (alignment > 0) {
        buffer_offset = Common::AlignUp<std::size_t>(buffer_offset, alignment);
    }

    // If the buffer is full, invalidate it
    if (buffer_offset + size > info.capacity) {
        // When invalidating a GPU buffer insert a full pipeline barrier to ensure all reads
        // have finished before reclaiming it
        if (info.usage != BufferUsage::Staging) {
            auto [access_mask, stage_mask] = ToVkAccessStageFlags(info.usage);

            const vk::BufferMemoryBarrier buffer_barrier = {
                .srcAccessMask = access_mask,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = buffer,
                .offset = 0,
                .size = info.capacity
            };

            // Insert pipeline barrier
            vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
            command_buffer.pipelineBarrier(stage_mask, vk::PipelineStageFlagBits::eTransfer,
                                           vk::DependencyFlagBits::eByRegion, {}, buffer_barrier, {});
        }

        Invalidate();
    }

    if (info.usage == BufferUsage::Staging) {
        return std::span<u8>{reinterpret_cast<u8*>(mapped_ptr) + buffer_offset, size};
    } else {
        Buffer& staging = scheduler.GetCommandUploadBuffer();
        return staging.Map(size, alignment);
    }
}

void Buffer::Commit(u32 size) {
    VmaAllocator allocator = instance.GetAllocator();
    if (info.usage == BufferUsage::Staging && size > 0) {
        vmaFlushAllocation(allocator, allocation, buffer_offset, size);
    } else {
        vk::CommandBuffer command_buffer = scheduler.GetUploadCommandBuffer();
        Buffer& staging = scheduler.GetCommandUploadBuffer();

        const vk::BufferCopy copy_region = {
            .srcOffset = staging.GetCurrentOffset(),
            .dstOffset = buffer_offset,
            .size = size
        };

        // Copy staging buffer to device local buffer
        staging.Commit(size);
        command_buffer.copyBuffer(staging.GetHandle(), buffer, copy_region);

        auto [access_mask, stage_mask] = ToVkAccessStageFlags(info.usage);
        const vk::BufferMemoryBarrier buffer_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = access_mask,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = buffer_offset,
            .size = size
        };

        // Add a pipeline barrier for the region modified
        command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, stage_mask,
                                       vk::DependencyFlagBits::eByRegion, {}, buffer_barrier, {});

    }

    buffer_offset += size;
}

}
