// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/assert.h"
#include "video_core/common/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

class Instance;
class CommandScheduler;

class Buffer : public VideoCore::BufferBase {
public:
    Buffer(Instance& instance, CommandScheduler& scheduler, const BufferInfo& info);
    ~Buffer() override;

    std::span<u8> Map(u32 size, u32 alignment = 0) override;

    /// Flushes write to buffer memory
    void Commit(u32 size = 0) override;

    /// Returns the Vulkan buffer handle
    vk::Buffer GetHandle() const {
        return buffer;
    }

    /// Returns an immutable reference to the requested buffer view
    const vk::BufferView& GetView(u32 index = 0) const {
        ASSERT(index < view_count);
        return views[index];
    }

protected:
    Instance& instance;
    CommandScheduler& scheduler;

    // Vulkan buffer handle
    void* mapped_ptr = nullptr;
    vk::Buffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    std::array<vk::BufferView, MAX_BUFFER_VIEWS> views{};
    u32 view_count = 0;
};

}
