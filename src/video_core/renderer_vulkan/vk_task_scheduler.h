// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <array>
#include <functional>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

constexpr u32 SCHEDULER_COMMAND_COUNT = 4;

using Deleter = std::function<void(vk::Device, VmaAllocator)>;

class Buffer;
class Instance;

class CommandScheduler {
public:
    CommandScheduler(Instance& instance);
    ~CommandScheduler();

    /// Create and initialize the work scheduler
    bool Create();

    /// Block host until the current command completes execution
    void Synchronize();

    /// Defer operation until the current command completes execution
    void Schedule(Deleter&& func);

    /// Submits the current command to the graphics queue
    void Submit(bool wait_completion = false, vk::Semaphore wait = VK_NULL_HANDLE,
                vk::Semaphore signal = VK_NULL_HANDLE);

    /// Returns the command buffer used for early upload operations.
    /// This is useful for vertex/uniform buffer uploads that happen once per frame
    vk::CommandBuffer GetUploadCommandBuffer();

    /// Returns the command buffer used for rendering
    inline vk::CommandBuffer GetRenderCommandBuffer() const {
        const CommandSlot& command = commands[current_command];
        return command.render_command_buffer;
    }

    /// Returns the upload buffer of the active command slot
    inline Buffer& GetCommandUploadBuffer() {
        CommandSlot& command = commands[current_command];
        return *command.upload_buffer;
    }

    /// Returns the index of the current command slot
    inline u32 GetCurrentSlotIndex() const {
        return current_command;
    }

private:
    /// Activates the next command slot and optionally waits for its completion
    void SwitchSlot();

private:
    Instance& instance;
    u64 next_fence_counter = 1;
    u64 completed_fence_counter = 0;

    struct CommandSlot {
        bool use_upload_buffer = false;
        u64 fence_counter = 0;
        vk::CommandBuffer render_command_buffer, upload_command_buffer;
        vk::Fence fence = VK_NULL_HANDLE;
        std::unique_ptr<Buffer> upload_buffer;
        std::vector<Deleter> cleanups;
    };

    vk::CommandPool command_pool = VK_NULL_HANDLE;
    std::array<CommandSlot, SCHEDULER_COMMAND_COUNT> commands;
    u32 current_command = 0;
};

}  // namespace Vulkan
