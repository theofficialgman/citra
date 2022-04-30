// Copyright 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

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
#include "common/semaphore.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

constexpr u32 COMMAND_BUFFER_COUNT = 2;

class VKCommandManager {
public:
    explicit VKCommandManager(bool use_threaded_submission);
    ~VKCommandManager();

    bool Initialize();

    // These command buffers are allocated per-frame. They are valid until the command buffer
    // is submitted, after that you should call these functions again.
    vk::CommandBuffer GetCurrentInitCommandBuffer() {
        frame_resources[current_frame].init_command_buffer_used = true;
        return frame_resources[current_frame].command_buffers[0];
    }

    vk::CommandBuffer GetCurrentCommandBuffer() const {
        return frame_resources[current_frame].command_buffers[1];
    }

    vk::DescriptorPool GetCurrentDescriptorPool() const {
        return frame_resources[current_frame].descriptor_pool;
    }

    // Allocates a descriptors set from the pool reserved for the current frame.
    vk::DescriptorSet AllocateDescriptorSet(vk::DescriptorSetLayout set_layout);

    // Fence "counters" are used to track which commands have been completed by the GPU.
    // If the last completed fence counter is greater or equal to N, it means that the work
    // associated counter N has been completed by the GPU. The value of N to associate with
    // commands can be retreived by calling GetCurrentFenceCounter().
    u64 GetCompletedFenceCounter() const { return completed_fence_counter; }

    // Gets the fence that will be signaled when the currently executing command buffer is
    // queued and executed. Do not wait for this fence before the buffer is executed.
    u64 GetCurrentFenceCounter() const { return frame_resources[current_frame].fence_counter; }

    // Returns the semaphore for the current command buffer, which can be used to ensure the
    // swap chain image is ready before the command buffer executes.
    vk::Semaphore GetCurrentCommandBufferSemaphore() {
      frame_resources[current_frame].semaphore_used = true;
      return frame_resources[current_frame].semaphore;
    }

    // Ensure that the worker thread has submitted any previous command buffers and is idle.
    void WaitForWorkerThreadIdle();

    // Wait for a fence to be completed.
    // Also invokes callbacks for completion.
    void WaitForFenceCounter(u64 fence_counter);

    void SubmitCommandBuffer(bool submit_on_worker_thread, bool wait_for_completion,
                             vk::SwapchainKHR present_swap_chain = VK_NULL_HANDLE,
                             u32 present_image_index = -1);

    // Was the last present submitted to the queue a failure? If so, we must recreate our swapchain.
    bool CheckLastPresentFail() { return last_present_failed.TestAndClear(); }
    vk::Result GetLastPresentResult() const { return last_present_result; }

    // Schedule a vulkan resource for destruction later on. This will occur when the command buffer
    // is next re-used, and the GPU has finished working with the specified resource.
    template <typename VulkanObject>
    void DestroyResource(VulkanObject object);

private:
    void BeginCommandBuffer();
    bool CreateCommandBuffers();
    void DestroyCommandBuffers();

    bool CreateSubmitThread();

    void WaitForCommandBufferCompletion(u32 command_buffer_index);
    void SubmitCommandBuffer(u32 command_buffer_index, vk::SwapchainKHR present_swap_chain,
                             u32 present_image_index);
private:
    struct FrameResources {
        // [0] - Init (upload) command buffer, [1] - draw command buffer
        std::vector<vk::CommandBuffer> command_buffers = {};
        std::vector<std::function<void()>> cleanup_resources;

        vk::CommandPool command_pool;
        vk::DescriptorPool descriptor_pool;
        vk::Fence fence;
        vk::Semaphore semaphore;
        u64 fence_counter = 0;
        bool init_command_buffer_used = false;
        bool semaphore_used = false;
    };

    struct PendingCommandBufferSubmit {
        vk::SwapchainKHR present_swap_chain;
        u32 present_image_index;
        u32 command_buffer_index;
    };

    u64 next_fence_counter = 1;
    u64 completed_fence_counter = 0;

    std::array<FrameResources, COMMAND_BUFFER_COUNT> frame_resources;
    u32 current_frame = 0;

    // Threaded command buffer execution
    // Semaphore determines when a command buffer can be queued
    Common::Semaphore submit_semaphore;
    std::thread submit_thread;
    std::unique_ptr<Common::BlockingLoop> submit_loop;
    std::deque<PendingCommandBufferSubmit> pending_submits;
    std::mutex pending_submit_lock;
    Common::Flag last_present_failed;
    vk::Semaphore present_semaphore;
    vk::Result last_present_result = vk::Result::eSuccess;
    bool use_threaded_submission = false;
};

template <typename VulkanObject>
void VKCommandManager::DestroyResource(VulkanObject object) {
    auto& resources = frame_resources[current_frame];
    auto deleter = [object]() { g_vk_instace->GetDevice().destroy(object); };
    resources.cleanup_resources.push_back(deleter);
}

extern std::unique_ptr<VKCommandManager> g_command_buffer_mgr;

}  // namespace Vulkan
