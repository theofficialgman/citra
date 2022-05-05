// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "common/assert.h"
#include "common/thread.h"

namespace Vulkan {

VKTaskScheduler::VKTaskScheduler(VKSwapChain* swapchain) : swapchain(swapchain) {

}

VKTaskScheduler::~VKTaskScheduler() {
    // Sync the GPU before exiting
    SyncToGPU();
}

vk::CommandBuffer VKTaskScheduler::GetCommandBuffer() {
    return tasks[current_task].command_buffer;
}

bool VKTaskScheduler::Create() {
    auto device = g_vk_instace->GetDevice();

    // Create command pool
    vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        g_vk_instace->GetGraphicsQueueFamilyIndex());
    command_pool = device.createCommandPoolUnique(pool_info);

    // Create timeline semaphore for syncronization
    vk::SemaphoreTypeCreateInfo timeline_info(vk::SemaphoreType::eTimeline, 0);
    vk::SemaphoreCreateInfo semaphore_info({}, &timeline_info);

    timeline = device.createSemaphoreUnique(semaphore_info);

    // Initialize task structures
    for (auto& task : tasks) {
        // Create command buffers
        vk::CommandBufferAllocateInfo buffer_info
        (
            command_pool.get(),
            vk::CommandBufferLevel::ePrimary,
            1, task.command_buffer
        );

        task.command_buffer = device.allocateCommandBuffers(buffer_info)[0];
    }

    // Create present semaphore
    present_semaphore = device.createSemaphoreUnique({});

    // Activate the first task.
    BeginTask();

    return true;
}

void VKTaskScheduler::SyncToGPU(u64 task_index) {
    // No need to sync if the GPU already has finished the task
    if (tasks[task_index].task_id <= GetGPUTick()) {
        return;
    }

    auto old_gpu_tick = GetGPUTick();

    // Wait for the task to complete
    vk::SemaphoreWaitInfo wait_info({}, timeline.get(), tasks[task_index].task_id);
    auto result = g_vk_instace->GetDevice().waitSemaphores(wait_info, UINT64_MAX);

    if (result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Failed waiting for timeline semaphore!");
    }

    auto new_gpu_tick = GetGPUTick();

    // Delete all resources that can be freed now
    for (auto& task : tasks) {
        if (task.task_id > old_gpu_tick && task.task_id <= new_gpu_tick) {
            for (auto& deleter : task.cleanups) {
                deleter();
            }
        }
    }
}

void VKTaskScheduler::SyncToGPU() {
    SyncToGPU(current_task);
}

void VKTaskScheduler::Submit(bool present, bool wait_completion) {
    // End the current task recording.
    auto& task = tasks[current_task];
    task.command_buffer.end();

    // When the task completes the timeline will increment to the task id
    vk::TimelineSemaphoreSubmitInfo timeline_info({}, task.task_id);

    std::array<vk::Semaphore, 2> signal_semaphores = { timeline.get(), present_semaphore.get() };
    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit_info({}, wait_stage, task.command_buffer, signal_semaphores, &timeline_info);

    // Wait for new swapchain image
    if (present) {
        auto available = swapchain->AcquireNextImage();
        submit_info.setWaitSemaphores(available);
    }

    // Submit the command buffer
    g_vk_instace->GetGraphicsQueue().submit(submit_info);

    // Present the image when rendering has finished
    if (present) {
        swapchain->Present(present_semaphore.get());
    }

    // Block host until the GPU catches up
    if (wait_completion) {
        SyncToGPU();
    }

    // Switch to next cmdbuffer.
    BeginTask();
}

void VKTaskScheduler::Schedule(std::function<void()> func) {
    auto& task = tasks[current_task];
    task.cleanups.push_back(func);
}

void VKTaskScheduler::BeginTask() {
    // Move to the next command buffer.
    u32 next_task_index = (current_task + 1) % CONCURRENT_TASK_COUNT;
    auto& task = tasks[next_task_index];
    auto& device = g_vk_instace->GetDevice();

    // Wait for the GPU to finish with all resources for this task.
    SyncToGPU(next_task_index);

    // Reset command pools to beginning since we can re-use the memory now
    device.resetCommandPool(command_pool.get());

    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    // Enable commands to be recorded to the command buffer again.
    task.command_buffer.begin(begin_info);

    // Reset upload command buffer state
    current_task = next_task_index;
    task.task_id = current_task_id++;
}

std::unique_ptr<VKTaskScheduler> g_vk_task_scheduler;

}  // namespace Vulkan
