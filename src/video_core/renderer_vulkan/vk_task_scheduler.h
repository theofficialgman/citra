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
#include "common/threadsafe_queue.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_buffer.h"

namespace Vulkan {

constexpr u32 TASK_COUNT = 3;
constexpr u32 STAGING_BUFFER_SIZE = 16 * 1024 * 1024;

class VKSwapChain;

/// Wrapper class around command buffer execution. Handles an arbitrary
/// number of tasks that can be submitted concurrently. This allows the host
/// to start recording the next frame while the GPU is working on the
/// current one. Larger values can be used with caution, as they can cause
/// frame latency if the CPU is too far ahead of the GPU
class VKTaskScheduler {
public:
    VKTaskScheduler() = default;
    ~VKTaskScheduler();

    /// Create and initialize the work scheduler
    bool Create();

    /// Retrieve either of the current frame's command buffers
    vk::CommandBuffer GetCommandBuffer() const { return tasks[current_task].command_buffer; }
    vk::DescriptorPool GetDescriptorPool() const { return tasks[current_task].pool.get(); }

    /// Access the staging buffer of the current task
    std::tuple<u8*, u32> RequestStaging(u32 size);
    VKBuffer& GetStaging() { return tasks[current_task].staging; }

    /// Query and/or synchronization CPU and GPU
    u64 GetCPUTick() const { return current_task_id; }
    u64 GetGPUTick() const { return g_vk_instace->GetDevice().getSemaphoreCounterValue(timeline.get()); }
    void SyncToGPU();
    void SyncToGPU(u64 task_index);

    void Schedule(std::function<void()> func);
    void Submit(bool wait_completion = false, bool present = false, VKSwapChain* swapchain = nullptr);

    void BeginTask();

private:
    struct Task {
        u64 current_offset{}, task_id{};
        vk::CommandBuffer command_buffer;
        vk::UniqueDescriptorPool pool;
        std::vector<std::function<void()>> cleanups;
        VKBuffer staging;
    };

    vk::UniqueDescriptorPool global_pool;
    vk::UniqueSemaphore timeline;
    vk::UniqueCommandPool command_pool;
    u64 current_task_id = 1;

    // Each task contains unique resources
    std::array<Task, TASK_COUNT> tasks;
    u64 current_task = 0;
};

extern std::unique_ptr<VKTaskScheduler> g_vk_task_scheduler;

}  // namespace Vulkan
