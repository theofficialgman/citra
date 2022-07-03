// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "video_core/renderer_vulkan/vk_buffer.h"

namespace Vulkan {

constexpr u32 TASK_COUNT = 5;
constexpr u32 STAGING_BUFFER_SIZE = 16 * 1024 * 1024;

class Swapchain;

/// Wrapper class around command buffer execution. Handles an arbitrary
/// number of tasks that can be submitted concurrently. This allows the host
/// to start recording the next frame while the GPU is working on the
/// current one. Larger values can be used with caution, as they can cause
/// frame latency if the CPU is too far ahead of the GPU
class TaskScheduler {
public:
    TaskScheduler() = default;
    ~TaskScheduler();

    /// Create and initialize the work scheduler
    bool Create();

    /// Retrieve either of the current frame's command buffers
    vk::CommandBuffer GetRenderCommandBuffer() const;
    vk::CommandBuffer GetUploadCommandBuffer();
    vk::DescriptorPool GetDescriptorPool() const;

    /// Access the staging buffer of the current task
    std::tuple<u8*, u32> RequestStaging(u32 size);
    Buffer& GetStaging();

    /// Query and/or synchronization CPU and GPU
    u64 GetCPUTick() const;
    u64 GetGPUTick() const;
    void SyncToGPU();
    void SyncToGPU(u64 task_index);

    void Schedule(std::function<void()> func);
    void Submit(bool wait_completion = false, bool present = false, Swapchain* swapchain = nullptr);

    void BeginTask();

private:
    struct Task {
        bool use_upload_buffer = false;
        u64 current_offset = 0, task_id = 0;
        std::array<vk::CommandBuffer, 2> command_buffers;
        std::vector<std::function<void()>> cleanups;
        vk::DescriptorPool pool;
        Buffer staging;
    };

    vk::Semaphore timeline;
    vk::CommandPool command_pool;
    u64 current_task_id = 0;

    // Each task contains unique resources
    std::array<Task, TASK_COUNT> tasks;
    u64 current_task = -1;
};

extern std::unique_ptr<TaskScheduler> g_vk_task_scheduler;

}  // namespace Vulkan
