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

VKBuffer::~VKBuffer() {
    if (buffer) {
        if (memory != nullptr) {
            g_vk_instace->GetDevice().unmapMemory(buffer_memory);
        }

        auto deleter = [this]() {
            auto& device = g_vk_instace->GetDevice();
                device.destroyBuffer(buffer);
                device.freeMemory(buffer_memory);
        };

        g_vk_task_scheduler->Schedule(deleter);
    }
}

void VKBuffer::Create(const VKBuffer::Info& info) {
    auto& device = g_vk_instace->GetDevice();
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
        memory = device.mapMemory(buffer_memory, 0, info.size);
    }
}

void VKBuffer::CopyBuffer(VKBuffer* src_buffer, VKBuffer* dst_buffer, vk::BufferCopy region) {
    auto command_buffer = g_vk_task_scheduler->GetCommandBuffer();
    command_buffer.copyBuffer(src_buffer->buffer, dst_buffer->buffer, region);
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

}
