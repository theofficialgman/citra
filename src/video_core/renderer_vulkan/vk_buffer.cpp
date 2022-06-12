// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include <cstring>

namespace Vulkan {

VKBuffer::~VKBuffer() {
    Destroy();
}

void VKBuffer::Create(const VKBuffer::Info& info) {
    auto device = g_vk_instace->GetDevice();
    buffer_info = info;

    vk::BufferCreateInfo bufferInfo({}, info.size, info.usage);
    buffer = device.createBuffer(bufferInfo);

    auto mem_requirements = device.getBufferMemoryRequirements(buffer);

    auto memory_type_index = FindMemoryType(mem_requirements.memoryTypeBits, info.properties);
    vk::MemoryAllocateInfo alloc_info(mem_requirements.size, memory_type_index);

    buffer_memory = device.allocateMemory(alloc_info);
    device.bindBufferMemory(buffer, buffer_memory, 0);

    // Optionally map the buffer to CPU memory
    if (info.properties & vk::MemoryPropertyFlagBits::eHostVisible) {
        host_ptr = device.mapMemory(buffer_memory, 0, info.size);
    }

    for (auto& format : info.view_formats) {
        if (format != vk::Format::eUndefined) {
            views[view_count++] = device.createBufferView({{}, buffer, format, 0, info.size});
        }
    }
}

void VKBuffer::Recreate() {
    Destroy();
    Create(buffer_info);
}

void VKBuffer::Destroy() {
    if (buffer) {
        if (host_ptr != nullptr) {
            g_vk_instace->GetDevice().unmapMemory(buffer_memory);
        }

        auto deleter = [this]() {
            auto device = g_vk_instace->GetDevice();
            device.destroyBuffer(buffer);
            device.freeMemory(buffer_memory);

            for (int i = 0; i < view_count; i++) {
                device.destroyBufferView(views[i]);
            }
        };

        g_vk_task_scheduler->Schedule(deleter);
    }
}

void VKBuffer::CopyBuffer(const VKBuffer& src_buffer, const VKBuffer& dst_buffer, vk::BufferCopy region, vk::AccessFlags access_to_block) {
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.copyBuffer(src_buffer.buffer, dst_buffer.buffer, region);

    vk::BufferMemoryBarrier barrier{
        vk::AccessFlagBits::eTransferWrite, access_to_block,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        dst_buffer.buffer, region.dstOffset, region.size
    };

    // Add a pipeline barrier for the region modified
    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eVertexShader |
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   vk::DependencyFlagBits::eByRegion,
                                   0, nullptr, 1, &barrier, 0, nullptr);
}

u32 VKBuffer::FindMemoryType(u32 type_filter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties mem_properties = g_vk_instace->GetPhysicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
    {
        auto flags = mem_properties.memoryTypes[i].propertyFlags;
        if ((type_filter & (1 << i)) && (flags & properties) == properties)
            return i;
    }

    LOG_CRITICAL(Render_Vulkan, "Failed to find suitable memory type.");
    UNREACHABLE();
}

std::tuple<u8*, u32, bool> StreamBuffer::Map(u32 size, u32 alignment) {
    ASSERT(size <= buffer_info.size);
    ASSERT(alignment <= buffer_info.size);

    if (alignment > 0) {
        buffer_pos = Common::AlignUp<std::size_t>(buffer_pos, alignment);
    }

    bool invalidate = false;
    if (buffer_pos + size > buffer_info.size) {
        buffer_pos = 0;
        invalidate = true;
    }

    auto [staging_ptr, staging_offset] = g_vk_task_scheduler->RequestStaging(size);
    mapped_chunk = vk::BufferCopy{staging_offset, buffer_pos, size};

    return std::make_tuple(staging_ptr + buffer_pos, buffer_pos, invalidate);
}

void StreamBuffer::Commit(u32 size) {
    auto& staging = g_vk_task_scheduler->GetStaging();
    mapped_chunk.size = size;

    VKBuffer::CopyBuffer(staging, *this, mapped_chunk);
    buffer_pos += size;
}

}
