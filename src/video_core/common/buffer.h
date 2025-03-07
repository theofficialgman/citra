// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <span>
#include "common/assert.h"
#include "common/hash.h"
#include "common/intrusive_ptr.h"

namespace VideoCore {

enum class BufferUsage : u8 {
    Undefined = 0,
    Vertex = 1,
    Index = 2,
    Uniform = 3,
    Texel = 4,
    Staging = 5
};

enum class ViewFormat : u8 {
    Undefined = 0,
    R32Float = 1,
    R32G32Float = 2,
    R32G32B32Float = 3,
    R32G32B32A32Float = 4,
};

constexpr u32 MAX_BUFFER_VIEWS = 3;

struct BufferInfo {
    u32 capacity = 0;
    BufferUsage usage = BufferUsage::Undefined;
    std::array<ViewFormat, MAX_BUFFER_VIEWS> views{};

    auto operator<=>(const BufferInfo& info) const = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(BufferInfo));
    }
};

static_assert(sizeof(BufferInfo) == 8, "BufferInfo not packed!");
static_assert(std::is_standard_layout_v<BufferInfo>, "BufferInfo is not a standard layout!");

struct BufferDeleter;

class BufferBase : public IntrusivePtrEnabled<BufferBase, BufferDeleter> {
public:
    BufferBase() = default;
    BufferBase(const BufferInfo& info) : info(info) {}
    virtual ~BufferBase() = default;

    // This method is called by BufferDeleter. Forward to the derived pool!
    virtual void Free() = 0;

    // Disable copy constructor
    BufferBase(const BufferBase&) = delete;
    BufferBase& operator=(const BufferBase&) = delete;

    // Allocates a linear chunk of memory in the GPU buffer with at least "size" bytes
    // and the optional alignment requirement.
    // The actual used size must be specified on unmapping the chunk.
    virtual std::span<u8> Map(u32 size, u32 alignment = 0) = 0;

    // Flushes write to buffer memory
    virtual void Commit(u32 size = 0) = 0;

    // Returns the size of the buffer in bytes
    u32 GetCapacity() const {
        return info.capacity;
    }

    // Returns the usage of the buffer
    BufferUsage GetUsage() const {
        return info.usage;
    }

    // Returns the starting offset of the currently mapped buffer slice
    u32 GetCurrentOffset() const {
        return buffer_offset;
    }

    // Returns whether the buffer was invalidated by the most recent Map call
    bool IsInvalid() const {
        return invalid;
    }

    // Invalidates the buffer
    void Invalidate() {
        buffer_offset = 0;
        invalid = true;
    }

protected:
    BufferInfo info{};
    u32 buffer_offset = 0;
    bool invalid = false;
};

// Foward pointer to its parent pool
struct BufferDeleter {
    void operator()(BufferBase* buffer) {
        buffer->Free();
    }
};

using BufferHandle = IntrusivePtr<BufferBase>;

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::BufferInfo> {
    std::size_t operator()(const VideoCore::BufferInfo& info) const noexcept {
        return info.Hash();
    }
};
} // namespace std
