// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

/// Generic Vulkan buffer object used by almost every resource
class VKBuffer final : public NonCopyable {
public:
    struct Info {
        u32 size;
        vk::MemoryPropertyFlags properties;
        vk::BufferUsageFlags usage;
    };

    VKBuffer() = default;
    VKBuffer(VKBuffer&&) = default;
    ~VKBuffer();

    /// Create a new Vulkan buffer object
    void Create(const Info& info);

    /// Global utility functions used by other objects
    static u32 FindMemoryType(u32 type_filter, vk::MemoryPropertyFlags properties);
    static void CopyBuffer(VKBuffer* src_buffer, VKBuffer* dst_buffer, vk::BufferCopy region);

    /// Return a pointer to the mapped memory if the buffer is host mapped
    u8* GetHostPointer() { return reinterpret_cast<u8*>(memory); }
    vk::Buffer& GetBuffer() { return buffer; }
    u32 GetSize() const { return buffer_info.size; }

private:
    Info buffer_info;
    void* memory = nullptr;
    vk::Buffer buffer;
    vk::DeviceMemory buffer_memory;
};

}
