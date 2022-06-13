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

    // Destroy Vulkan resources
    auto device = g_vk_instace->GetDevice();
    device.destroyCommandPool(command_pool);
    device.destroySemaphore(timeline);

    for (auto& task : tasks) {
        device.destroyDescriptorPool(task.pool);
    }
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
    std::memset(ptr, 0, size);

    task.current_offset += size;
    return std::make_tuple(ptr, task.current_offset - size);
}

VKBuffer& VKTaskScheduler::GetStaging() {
    return tasks[current_task].staging;
}

bool VKTaskScheduler::Create() {
    auto device = g_vk_instace->GetDevice();

    // Create command pool
    vk::CommandPoolCreateInfo pool_info(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        g_vk_instace->GetGraphicsQueueFamilyIndex());
    command_pool = device.createCommandPool(pool_info);

    // Create timeline semaphore for syncronization
    vk::SemaphoreTypeCreateInfo timeline_info{vk::SemaphoreType::eTimeline, 0};
    vk::SemaphoreCreateInfo semaphore_info{{}, &timeline_info};

    timeline = device.createSemaphore(semaphore_info);

    VKBuffer::Info staging_info{
        .size = STAGING_BUFFER_SIZE,
        .properties = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent,
        .usage = vk::BufferUsageFlagBits::eTransferSrc
    };

    // Should be enough for a single frame
    const vk::DescriptorPoolSize pool_size{vk::DescriptorType::eCombinedImageSampler, 64};
    vk::DescriptorPoolCreateInfo pool_create_info{{}, 1024, pool_size};

    for (auto& task : tasks) {
        // Create command buffers
        vk::CommandBufferAllocateInfo buffer_info{command_pool, vk::CommandBufferLevel::ePrimary, 2};
        auto buffers = device.allocateCommandBuffers(buffer_info);
        std::ranges::copy_n(buffers.begin(), 2, task.command_buffers.begin());

        // Create staging buffer
        task.staging.Create(staging_info);

        // Create descriptor pool
        task.pool = device.createDescriptorPool(pool_create_info);
    }

    return true;
}

vk::CommandBuffer VKTaskScheduler::GetRenderCommandBuffer() const {
    const auto& task = tasks[current_task];
    return task.command_buffers[1];
}

vk::CommandBuffer VKTaskScheduler::GetUploadCommandBuffer() {
    auto& task = tasks[current_task];
    if (!task.use_upload_buffer) {
        auto& cmdbuffer = task.command_buffers[0];
        cmdbuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        task.use_upload_buffer = true;
    }

    return task.command_buffers[0];
}

vk::DescriptorPool VKTaskScheduler::GetDescriptorPool() const {
    const auto& task = tasks[current_task];
    return task.pool;
}

void VKTaskScheduler::SyncToGPU(u64 task_index) {
    // No need to sync if the GPU already has finished the task
    if (tasks[task_index].task_id <= GetGPUTick()) {
        return;
    }

    auto last_completed_task_id = GetGPUTick();

    // Wait for the task to complete
    vk::SemaphoreWaitInfo wait_info{{}, timeline, tasks[task_index].task_id};
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

u64 VKTaskScheduler::GetCPUTick() const {
    return current_task_id;
}

u64 VKTaskScheduler::GetGPUTick() const {
    auto device = g_vk_instace->GetDevice();
    return device.getSemaphoreCounterValue(timeline);
}

void VKTaskScheduler::Submit(bool wait_completion, bool present, VKSwapChain* swapchain) {
    // End the current task recording.
    auto& task = tasks[current_task];

    // End command buffers
    task.command_buffers[1].end();
    if (task.use_upload_buffer) {
        task.command_buffers[0].end();
    }

    const u32 num_signal_semaphores = present ? 2U : 1U;
    const std::array signal_values{task.task_id, u64(0)};
    std::array signal_semaphores{timeline, vk::Semaphore{}};

    const u32 num_wait_semaphores = present ? 2U : 1U;
    const std::array wait_values{task.task_id - 1, u64(1)};
    std::array wait_semaphores{timeline, vk::Semaphore{}};

    // When the task completes the timeline will increment to the task id
    const vk::TimelineSemaphoreSubmitInfoKHR timeline_si{num_wait_semaphores, wait_values.data(),
                                                         num_signal_semaphores, signal_values.data()};

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks{
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const u32 cmdbuffer_count = task.use_upload_buffer ? 2u : 1u;
    const vk::SubmitInfo submit_info{num_wait_semaphores, wait_semaphores.data(), wait_stage_masks.data(), cmdbuffer_count,
                                     &task.command_buffers[2 - cmdbuffer_count], num_signal_semaphores, signal_semaphores.data(),
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
    device.resetDescriptorPool(task.pool);
    task.command_buffers[1].begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Move to the next command buffer.
    current_task = next_task_index;
    task.task_id = current_task_id++;
    task.current_offset = 0;
    task.use_upload_buffer = false;

    auto& state = VulkanState::Get();
    state.InitDescriptorSets();
}

std::unique_ptr<VKTaskScheduler> g_vk_task_scheduler;

}  // namespace Vulkan
