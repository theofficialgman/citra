// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/blocking_loop.h"
#include "common/threadsafe_queue.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_buffer.h"

namespace Vulkan {

constexpr u32 CONCURRENT_TASK_COUNT = 2;
constexpr u32 STAGING_BUFFER_SIZE = 16 * 1024 * 1024;

class VKSwapChain;

/// Wrapper class around command buffer execution. Handles an arbitrary
/// number of tasks that can be submitted concurrently. This allows the host
/// to start recording the next frame while the GPU is working on the
/// current one. Larger values can be used with caution, as they can cause
/// frame latency if the CPU is too far ahead of the GPU
class VKTaskScheduler {
public:
    explicit VKTaskScheduler(VKSwapChain* swapchain);
    ~VKTaskScheduler();

    /// Create and initialize the work scheduler
    bool Create();

    /// Retrieve either of the current frame's command buffers
    vk::CommandBuffer GetCommandBuffer() const { return tasks[current_task].command_buffer; }
    VKBuffer& GetStaging() { return tasks[current_task].staging; }
    std::tuple<u8*, u32> RequestStaging(u32 size);

    /// Returns the task id that the CPU is recording
    u64 GetCPUTick() const { return current_task_id; }

    /// Returns the last known task id to have completed execution in the GPU
    u64 GetGPUTick() const { return g_vk_instace->GetDevice().getSemaphoreCounterValue(timeline.get()); }

    /// Make the host wait for the GPU to complete
    void SyncToGPU();
    void SyncToGPU(u64 task_index);

    /// Schedule a vulkan object for destruction when the GPU no longer uses it
    void Schedule(std::function<void()> func);

    /// Submit the current work batch and move to the next frame
    void Submit(bool present = true, bool wait_completion = false);

private:
    void BeginTask();

private:
    struct Task {
        u64 task_id{};
        std::vector<std::function<void()>> cleanups;
        vk::CommandBuffer command_buffer;
        VKBuffer staging;
        u32 current_offset{};
    };

    vk::UniqueSemaphore timeline;
    vk::UniqueCommandPool command_pool;
    u64 current_task_id = 1;

    // Each task contains unique resources
    std::array<Task, CONCURRENT_TASK_COUNT> tasks;
    u32 current_task = CONCURRENT_TASK_COUNT - 1;

    // Presentation semaphore
    vk::UniqueSemaphore present_semaphore;
    VKSwapChain* swapchain = nullptr;
};

extern std::unique_ptr<VKTaskScheduler> g_vk_task_scheduler;

}  // namespace Vulkan
