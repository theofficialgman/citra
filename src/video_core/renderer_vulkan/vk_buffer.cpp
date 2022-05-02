// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include <cstring>

namespace Vulkan {

VKBuffer::~VKBuffer()
{
    if (memory != nullptr) {
        g_vk_instace->GetDevice().unmapMemory(buffer_memory.get());
    }
}

void VKBuffer::Create(uint32_t byte_count, vk::MemoryPropertyFlags properties, vk::BufferUsageFlags usage, vk::Format view_format)
{
    auto& device = g_vk_instace->GetDevice();
    size = byte_count;

    vk::BufferCreateInfo bufferInfo({}, byte_count, usage);
    buffer = device.createBufferUnique(bufferInfo);

    auto mem_requirements = device.getBufferMemoryRequirements(buffer.get());

    auto memory_type_index = FindMemoryType(mem_requirements.memoryTypeBits, properties);
    vk::MemoryAllocateInfo alloc_info(mem_requirements.size, memory_type_index);

    buffer_memory = device.allocateMemoryUnique(alloc_info);
    device.bindBufferMemory(buffer.get(), buffer_memory.get(), 0);

    // Optionally map the buffer to CPU memory
    if (properties & vk::MemoryPropertyFlagBits::eHostVisible)
        memory = device.mapMemory(buffer_memory.get(), 0, byte_count);

    // Create buffer view for texel buffers
    if (usage & vk::BufferUsageFlagBits::eStorageTexelBuffer || usage & vk::BufferUsageFlagBits::eUniformTexelBuffer)
    {
        vk::BufferViewCreateInfo view_info({}, buffer.get(), view_format, 0, byte_count);
        buffer_view = device.createBufferViewUnique(view_info);
    }
}

void VKBuffer::CopyBuffer(VKBuffer& src_buffer, VKBuffer& dst_buffer, const vk::BufferCopy& region)
{
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.copyBuffer(src_buffer.buffer.get(), dst_buffer.buffer.get(), region);
}

uint32_t VKBuffer::FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties)
{
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

}
