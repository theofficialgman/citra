// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "common/assert.h"
#include "common/thread.h"

namespace Vulkan {

VKTaskScheduler::~VKTaskScheduler() {
    SyncToGPU();
}

std::tuple<u8*, u32> VKTaskScheduler::RequestStaging(u32 size) {
    auto& task = tasks[current_task];
    if (size > STAGING_BUFFER_SIZE - task.current_offset) {
        // If we run out of space, allocate a new buffer.
        // The old one will be safely destroyed when the task finishes
        task.staging.Recreate();
        task.current_offset = 0;

        return std::make_tuple(task.staging.GetHostPointer(), 0);
    }

    u8* ptr = task.staging.GetHostPointer() + task.current_offset;
    task.current_offset += size;

    return std::make_tuple(ptr, task.current_offset - size);
}

bool VKTaskScheduler::Create() {
    auto device = g_vk_instace->GetDevice();

    // Create command pool
    vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        g_vk_instace->GetGraphicsQueueFamilyIndex());
    command_pool = device.createCommandPoolUnique(pool_info);

    // Create timeline semaphore for syncronization
    vk::SemaphoreTypeCreateInfo timeline_info{vk::SemaphoreType::eTimeline, 0};
    vk::SemaphoreCreateInfo semaphore_info{{}, &timeline_info};

    timeline = device.createSemaphoreUnique(semaphore_info);

    VKBuffer::Info staging_info{
        .size = STAGING_BUFFER_SIZE,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
        .usage = vk::BufferUsageFlagBits::eTransferSrc
    };

    // Should be enough for a single frame
    const vk::DescriptorPoolSize pool_size{vk::DescriptorType::eCombinedImageSampler, 64};
    vk::DescriptorPoolCreateInfo pool_create_info{{}, 1024, pool_size};

    // Create global descriptor pool
    global_pool = device.createDescriptorPoolUnique(pool_create_info);

    for (auto& task : tasks) {
        // Create command buffers
        vk::CommandBufferAllocateInfo buffer_info{command_pool.get(), vk::CommandBufferLevel::ePrimary, 1};
        task.command_buffer = device.allocateCommandBuffers(buffer_info)[0];

        // Create staging buffer
        task.staging.Create(staging_info);

        // Create descriptor pool
        task.pool = device.createDescriptorPoolUnique(pool_create_info);
    }

    return true;
}

void VKTaskScheduler::SyncToGPU(u64 task_index) {
    // No need to sync if the GPU already has finished the task
    if (tasks[task_index].task_id <= GetGPUTick()) {
        return;
    }

    auto last_completed_task_id = GetGPUTick();

    // Wait for the task to complete
    vk::SemaphoreWaitInfo wait_info({}, timeline.get(), tasks[task_index].task_id);
    auto result = g_vk_instace->GetDevice().waitSemaphores(wait_info, UINT64_MAX);

    if (result != vk::Result::eSuccess) {
        LOG_CRITICAL(Render_Vulkan, "Failed waiting for timeline semaphore!");
    }

    auto completed_task_id = GetGPUTick();

    // Delete all resources that can be freed now
    for (auto& task : tasks) {
        if (task.task_id > last_completed_task_id && task.task_id <= completed_task_id) {
            for (auto& func : task.cleanups) {
                func();
            }
        }
    }
}

void VKTaskScheduler::SyncToGPU() {
    SyncToGPU(current_task);
}

void VKTaskScheduler::Submit(bool wait_completion, bool present, VKSwapChain* swapchain) {
    // End the current task recording.
    auto& task = tasks[current_task];
    task.command_buffer.end();

    const u32 num_signal_semaphores = present ? 2U : 1U;
    const std::array signal_values{task.task_id, u64(0)};
    std::array signal_semaphores{timeline.get(), vk::Semaphore{}};

    const u32 num_wait_semaphores = present ? 2U : 1U;
    const std::array wait_values{task.task_id - 1, u64(1)};
    std::array wait_semaphores{timeline.get(), vk::Semaphore{}};

    // When the task completes the timeline will increment to the task id
    const vk::TimelineSemaphoreSubmitInfoKHR timeline_si{num_wait_semaphores, wait_values.data(),
                                                         num_signal_semaphores, signal_values.data()};

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks{
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::SubmitInfo submit_info{num_wait_semaphores, wait_semaphores.data(), wait_stage_masks.data(), 1,
                                     &task.command_buffer, num_signal_semaphores, signal_semaphores.data(),
                                     &timeline_si};
    // Wait for new swapchain image
    if (present) {
        signal_semaphores[1] = swapchain->GetRenderSemaphore();
        wait_semaphores[1] = swapchain->GetAvailableSemaphore();
    }

    // Submit the command buffer
    auto queue = g_vk_instace->GetGraphicsQueue();
    queue.submit(submit_info);

    // Present the image when rendering has finished
    if (present) {
        swapchain->Present();
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
    u32 next_task_index = (current_task + 1) % TASK_COUNT;
    auto& task = tasks[next_task_index];
    auto device = g_vk_instace->GetDevice();

    // Wait for the GPU to finish with all resources for this task.
    SyncToGPU(next_task_index);
    device.resetDescriptorPool(task.pool.get());
    task.command_buffer.begin({vk::CommandBufferUsageFlagBits::eSimultaneousUse});

    // Move to the next command buffer.
    current_task = next_task_index;
    task.task_id = current_task_id++;
    task.current_offset = 0;

    auto& state = VulkanState::Get();
    state.InitDescriptorSets();
}

std::unique_ptr<VKTaskScheduler> g_vk_task_scheduler;

}  // namespace Vulkan
