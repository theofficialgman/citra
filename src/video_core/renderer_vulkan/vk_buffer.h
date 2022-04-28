// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

/// Generic Vulkan buffer object used by almost every resource
class VKBuffer final : public NonCopyable {
public:
    VKBuffer() = default;
    VKBuffer(VKBuffer&&) = default;
    ~VKBuffer();

    /// Create a generic Vulkan buffer object
    void Create(uint32_t size, vk::MemoryPropertyFlags properties, vk::BufferUsageFlags usage, vk::Format view_format = vk::Format::eUndefined);

    /// Global utility functions used by other objects
    static uint32_t FindMemoryType(uint32_t type_filter, vk::MemoryPropertyFlags properties);
    static void CopyBuffer(VKBuffer& src_buffer, VKBuffer& dst_buffer, const vk::BufferCopy& region);

    /// Return a pointer to the mapped memory if the buffer is host mapped
    u8* GetHostPointer() { return reinterpret_cast<u8*>(memory); }
    vk::Buffer& GetBuffer() { return buffer.get(); }

private:
    void* memory = nullptr;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory buffer_memory;
    vk::UniqueBufferView buffer_view;
    uint32_t size = 0;
};

}
