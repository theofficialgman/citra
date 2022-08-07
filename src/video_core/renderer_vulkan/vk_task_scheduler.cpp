// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_buffer.h"

namespace VideoCore::Vulkan {

// 16MB should be enough for a single frame
constexpr BufferInfo STAGING_INFO = {
    .capacity = 16 * 1024 * 1024,
    .usage = BufferUsage::Staging
};

CommandScheduler::CommandScheduler(Instance& instance) : instance(instance) {

}

CommandScheduler::~CommandScheduler() {
    // Destroy Vulkan resources
    vk::Device device = instance.GetDevice();
    VmaAllocator allocator = instance.GetAllocator();

    for (auto& command : commands) {
        device.destroyFence(command.fence);

        // Clean up any scheduled resources
        for (auto& func : command.cleanups) {
            func(device, allocator);
        }
    }

    device.destroyCommandPool(command_pool);
}

bool CommandScheduler::Create() {
    vk::Device device = instance.GetDevice();
    const vk::CommandPoolCreateInfo pool_info = {
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex()
    };

    // Create command pool
    command_pool = device.createCommandPool(pool_info);

    vk::CommandBufferAllocateInfo buffer_info = {
        .commandPool = command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 2 * SCHEDULER_COMMAND_COUNT
    };

    // Allocate all command buffers
    const auto command_buffers = device.allocateCommandBuffers(buffer_info);

    // Initialize command slots
    for (std::size_t i = 0; i < commands.size(); i++) {
        commands[i] = CommandSlot{
            .render_command_buffer = command_buffers[2 * i],
            .upload_command_buffer = command_buffers[2 * i + 1],
            .fence = device.createFence({}),
            .upload_buffer = std::make_unique<Buffer>(instance, *this, STAGING_INFO)
        };
    }

    return true;
}

void CommandScheduler::Synchronize() {
    // Don't synchronize the same command twicec
    CommandSlot& command = commands[current_command];
    if (command.fence_counter <= completed_fence_counter) {
        return;
    }

    // Wait for this command buffer to be completed.
    vk::Device device = instance.GetDevice();
    if (device.waitForFences(command.fence, true, UINT64_MAX) != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Waiting for fences failed!");
    }

    // Cleanup resources for command buffers that have completed along with the current one
    const u64 now_fence_counter = command.fence_counter;
    VmaAllocator allocator = instance.GetAllocator();
    for (CommandSlot& command : commands) {
        if (command.fence_counter < now_fence_counter &&
            command.fence_counter > completed_fence_counter) {
            for (auto& func: command.cleanups) {
                func(device, allocator);
            }

            command.cleanups.clear();
        }
    }

    completed_fence_counter = now_fence_counter;
}

void CommandScheduler::Submit(bool wait_completion,
                              vk::Semaphore wait_semaphore,
                              vk::Semaphore signal_semaphore) {
    const CommandSlot& command = commands[current_command];

    // End command buffers
    command.render_command_buffer.end();
    if (command.use_upload_buffer) {
        command.upload_command_buffer.end();
    }

    constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks{
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const u32 signal_semaphore_count = signal_semaphore ? 1u : 0u;
    const u32 wait_semaphore_count = wait_semaphore ? 1u : 0u;
    const u32 command_buffer_count = command.use_upload_buffer ? 2u : 1u;
    const std::array command_buffers = { command.render_command_buffer,
                                         command.upload_command_buffer };

    // Prepeare submit info
    const vk::SubmitInfo submit_info = {
        .waitSemaphoreCount = wait_semaphore_count,
        .pWaitSemaphores = &wait_semaphore,
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = command_buffer_count,
        .pCommandBuffers = command_buffers.data(),
        .signalSemaphoreCount = signal_semaphore_count,
        .pSignalSemaphores = &signal_semaphore,
    };

    // Submit the command buffer
    vk::Queue queue = instance.GetGraphicsQueue();
    queue.submit(submit_info, command.fence);

    // Block host until the GPU catches up
    if (wait_completion) {
        Synchronize();
    }

    // Switch to next cmdbuffer.
    SwitchSlot();
}

void CommandScheduler::Schedule(Deleter&& func) {
    auto& command = commands[current_command];
    command.cleanups.push_back(func);
}

vk::CommandBuffer CommandScheduler::GetUploadCommandBuffer() {
    CommandSlot& command = commands[current_command];
    if (!command.use_upload_buffer) {
        const vk::CommandBufferBeginInfo begin_info = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        command.upload_command_buffer.begin(begin_info);
        command.use_upload_buffer = true;
    }

    return command.upload_command_buffer;
}

void CommandScheduler::SwitchSlot() {
    current_command = (current_command + 1) % SCHEDULER_COMMAND_COUNT;
    CommandSlot& command = commands[current_command];

    // Wait for the GPU to finish with all resources for this command.
    Synchronize();

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    // Move to the next command buffer.
    vk::Device device = instance.GetDevice();
    device.resetFences(command.fence);
    command.render_command_buffer.begin(begin_info);
    command.fence_counter = next_fence_counter++;
    command.use_upload_buffer = false;
}

}  // namespace Vulkan
