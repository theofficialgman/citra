// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <array>
#include <functional>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
class PoolManager;
}

namespace VideoCore::Vulkan {

constexpr u32 SCHEDULER_COMMAND_COUNT = 4;

class Buffer;
class Instance;

class CommandScheduler {
public:
    CommandScheduler(Instance& instance, PoolManager& pool_manager);
    ~CommandScheduler();

    // Blocks the host until the current command completes execution
    void Synchronize();

    // Sets a function to be called when the command slot is switched
    void SetSwitchCallback(std::function<void(u32)> callback);

    // Defers operation until the current command completes execution
    void Schedule(std::function<void(vk::Device, VmaAllocator)>&& func);

    // Submits the current command to the graphics queue
    void Submit(bool wait_completion = false, bool begin_next = true,
                vk::Semaphore wait = VK_NULL_HANDLE, vk::Semaphore signal = VK_NULL_HANDLE);

    // Returns the command buffer used for early upload operations.
    // This is useful for vertex/uniform buffer uploads that happen once per frame
    vk::CommandBuffer GetUploadCommandBuffer();

    // Returns the command buffer used for rendering
    vk::CommandBuffer GetRenderCommandBuffer() const {
        const CommandSlot& command = commands[current_command];
        return command.render_command_buffer;
    }

    // Returns the upload buffer of the active command slot
    Buffer& GetCommandUploadBuffer() {
        CommandSlot& command = commands[current_command];
        return *static_cast<Buffer*>(command.upload_buffer.Get());
    }

    // Returns the index of the current command slot
    inline u32 GetCurrentSlotIndex() const {
        return current_command;
    }

private:
    // Activates the next command slot and optionally waits for its completion
    void SwitchSlot();

private:
    Instance& instance;
    PoolManager& pool_manager;
    u64 next_fence_counter = 1;
    u64 completed_fence_counter = 0;

    struct CommandSlot {
        bool use_upload_buffer = false;
        u64 fence_counter = 0;
        vk::Fence fence = VK_NULL_HANDLE;
        vk::CommandBuffer render_command_buffer;
        vk::CommandBuffer upload_command_buffer;
        BufferHandle upload_buffer;
        std::vector<std::function<void(vk::Device, VmaAllocator)>> cleanups;
    };

    vk::CommandPool command_pool = VK_NULL_HANDLE;
    std::array<CommandSlot, SCHEDULER_COMMAND_COUNT> commands;
    std::function<void(u32)> switch_callback;
    u32 current_command = 0;
};

}  // namespace Vulkan
