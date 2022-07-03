// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <span>
#include <vulkan/vulkan.hpp>
#include "common/common_types.h"

namespace Vulkan {

constexpr u32 MAX_BUFFER_VIEWS = 5;
constexpr u32 MAX_COMMIT_CHUNKS = 6;

/// Generic Vulkan buffer object used by almost every resource
class Buffer : public NonCopyable {
public:
    struct Info {
        u32 size;
        vk::MemoryPropertyFlags properties;
        vk::BufferUsageFlags usage;
        std::array<vk::Format, MAX_BUFFER_VIEWS> view_formats{};
    };

    Buffer() = default;
    ~Buffer();

    /// Enable move operations
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    /// Create a new Vulkan buffer object
    void Create(const Info& info);
    void Recreate();
    void Destroy();

    /// Global utility functions used by other objects
    static u32 FindMemoryType(u32 type_filter, vk::MemoryPropertyFlags properties);

    /// Return a pointer to the mapped memory if the buffer is host mapped
    u8* GetHostPointer() const { return reinterpret_cast<u8*>(host_ptr); }
    const vk::BufferView& GetView(u32 i = 0) const { return views[i]; }
    const vk::Buffer& GetBuffer() const { return buffer; }
    u32 GetSize() const { return buffer_info.size; }

    void Upload(std::span<const std::byte> data, u32 offset,
                vk::AccessFlags access_to_block = vk::AccessFlagBits::eVertexAttributeRead,
                vk::PipelineStageFlags stage_to_block = vk::PipelineStageFlagBits::eVertexInput);

protected:
    Info buffer_info;
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    void* host_ptr = nullptr;
    std::array<vk::BufferView, MAX_BUFFER_VIEWS> views;
    u32 view_count{};
};

class StreamBuffer : public Buffer {
public:
    /*
     * Allocates a linear chunk of memory in the GPU buffer with at least "size" bytes
     * and the optional alignment requirement.
     * If the buffer is full, the whole buffer is reallocated which invalidates old chunks.
     * The return values are the pointer to the new chunk, the offset within the buffer,
     * and the invalidation flag for previous chunks.
     * The actual used size must be specified on unmapping the chunk.
     */
    std::tuple<u8*, u32, bool> Map(u32 size, u32 alignment = 0);
    void Commit(u32 size, vk::AccessFlags access_to_block = vk::AccessFlagBits::eUniformRead,
                vk::PipelineStageFlags stage_to_block = vk::PipelineStageFlagBits::eVertexShader |
                                                        vk::PipelineStageFlagBits::eFragmentShader);

private:
    u32 buffer_pos{};
    vk::BufferCopy mapped_chunk;
};

}
