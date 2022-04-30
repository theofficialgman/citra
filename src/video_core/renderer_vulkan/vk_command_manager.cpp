// Copyright 2016 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_command_manager.h"
#include "common/assert.h"
#include "common/thread.h"

namespace Vulkan {

VKCommandManager::VKCommandManager(bool use_threaded_submission)
    : submit_semaphore(1, 1), use_threaded_submission(use_threaded_submission) {
}

VKCommandManager::~VKCommandManager() {
    // If the worker thread is enabled, stop and block until it exits.
    if (use_threaded_submission) {
        submit_loop->Stop();
        submit_thread.join();
    }

    DestroyCommandBuffers();
}

bool VKCommandManager::Initialize()
{
    if (!CreateCommandBuffers()) {
        return false;
    }

    if (use_threaded_submission && !CreateSubmitThread()) {
        return false;
    }

    return true;
}

bool VKCommandManager::CreateCommandBuffers() {
    static constexpr vk::SemaphoreCreateInfo semaphore_create_info;

    auto device = g_vk_instace->GetDevice();
    for (auto& resources : frame_resources) {
        resources.init_command_buffer_used = false;
        resources.semaphore_used = false;

        // Create command pool
        vk::CommandPoolCreateInfo pool_info({}, g_vk_instace->GetGraphicsQueueFamilyIndex());
        resources.command_pool = device.createCommandPool(pool_info);

        // Create command buffers
        vk::CommandBufferAllocateInfo buffer_info
        (
            resources.command_pool,
            vk::CommandBufferLevel::ePrimary,
            resources.command_buffers.size()
        );

        resources.command_buffers = device.allocateCommandBuffers(buffer_info);

        vk::FenceCreateInfo fence_info(vk::FenceCreateFlagBits::eSignaled);
        resources.fence = device.createFence(fence_info);

        // TODO: A better way to choose the number of descriptors.
        const std::array<vk::DescriptorPoolSize, 3> pool_sizes{{
            { vk::DescriptorType::eUniformBuffer, 32 },
            { vk::DescriptorType::eCombinedImageSampler, 64 },
            { vk::DescriptorType::eStorageTexelBuffer, 64 }
        }};

        const vk::DescriptorPoolCreateInfo pool_create_info({}, 2048, pool_sizes);
        resources.descriptor_pool = device.createDescriptorPool(pool_create_info);
    }

    // Create present semaphore
    present_semaphore = device.createSemaphore(semaphore_create_info);

    // Activate the first command buffer. ActivateCommandBuffer moves forward, so start with the last
    current_frame = static_cast<u32>(frame_resources.size()) - 1;
    BeginCommandBuffer();
    return true;
}

void VKCommandManager::DestroyCommandBuffers() {
    vk::Device device = g_vk_instace->GetDevice();

    for (auto& resources : frame_resources) {
        // Destroy command pool which also clears any allocated command buffers
        if (resources.command_pool) {
            device.destroyCommandPool(resources.command_pool);
        }

        // Destroy any pending objects.
        for (auto& it : resources.cleanup_resources)
          it();

        // Destroy remaining vulkan objects
        if (resources.semaphore) {
            device.destroySemaphore(resources.semaphore);
        }

        if (resources.fence) {
            device.destroyFence(resources.fence);
        }

        if (resources.descriptor_pool) {
            device.destroyDescriptorPool(resources.descriptor_pool);
        }
    }

    device.destroySemaphore(present_semaphore);
}

vk::DescriptorSet VKCommandManager::AllocateDescriptorSet(vk::DescriptorSetLayout set_layout) {
    vk::DescriptorSetAllocateInfo allocate_info(frame_resources[current_frame].descriptor_pool, set_layout);
    return g_vk_instace->GetDevice().allocateDescriptorSets(allocate_info)[0];
}

bool VKCommandManager::CreateSubmitThread() {
    submit_loop = std::make_unique<Common::BlockingLoop>();

    submit_thread = std::thread([this]() {
    Common::SetCurrentThreadName("Vulkan CommandBufferManager SubmitThread");

    submit_loop->Run([this]() {
        PendingCommandBufferSubmit submit;
        {
          std::lock_guard<std::mutex> guard(pending_submit_lock);
          if (pending_submits.empty())
          {
            submit_loop->AllowSleep();
            return;
          }

          submit = pending_submits.front();
          pending_submits.pop_front();
        }

        SubmitCommandBuffer(submit.command_buffer_index, submit.present_swap_chain,
                          submit.present_image_index);
    });
    });

    return true;
}

void VKCommandManager::WaitForWorkerThreadIdle()
{
  // Drain the semaphore, then allow another request in the future.
  submit_semaphore.Wait();
  submit_semaphore.Post();
}

void VKCommandManager::WaitForFenceCounter(u64 fence_counter) {
    if (completed_fence_counter >= fence_counter)
        return;

    // Find the first command buffer which covers this counter value.
    u32 index = (current_frame + 1) % COMMAND_BUFFER_COUNT;
    while (index != current_frame) {
        if (frame_resources[index].fence_counter >= fence_counter)
            break;

        index = (index + 1) % COMMAND_BUFFER_COUNT;
    }

    ASSERT(index != current_frame);
    WaitForCommandBufferCompletion(index);
}

void VKCommandManager::WaitForCommandBufferCompletion(u32 index) {
    // Ensure this command buffer has been submitted.
    WaitForWorkerThreadIdle();

    // Wait for this command buffer to be completed.
    auto result = g_vk_instace->GetDevice().waitForFences(frame_resources[index].fence,
                                                          VK_TRUE, UINT64_MAX);

    if (result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "vkWaitForFences failed");
    }

    // Clean up any resources for command buffers between the last known completed buffer and this
    // now-completed command buffer. If we use >2 buffers, this may be more than one buffer.
    const u64 now_completed_counter = frame_resources[index].fence_counter;
    u32 cleanup_index = (current_frame + 1) % COMMAND_BUFFER_COUNT;

    while (cleanup_index != current_frame) {
        auto& resources = frame_resources[cleanup_index];
        if (resources.fence_counter > now_completed_counter) {
            break;
        }

        if (resources.fence_counter > completed_fence_counter) {
            for (auto& it : resources.cleanup_resources)
                it();

            resources.cleanup_resources.clear();
        }

        cleanup_index = (cleanup_index + 1) % COMMAND_BUFFER_COUNT;
    }

    completed_fence_counter = now_completed_counter;
}

void VKCommandManager::SubmitCommandBuffer(bool submit_on_worker_thread, bool wait_for_completion,
                                           vk::SwapchainKHR present_swap_chain,
                                           u32 present_image_index) {

    // End the current command buffer.
    auto& resources = frame_resources[current_frame];
    for (auto& command_buffer : resources.command_buffers) {
        command_buffer.end();
    }

    // Grab the semaphore before submitting command buffer either on-thread or off-thread.
    // This prevents a race from occurring where a second command buffer is executed
    // before the worker thread has woken and executed the first one yet.
    submit_semaphore.Wait();

    // Submitting off-thread?
    if (use_threaded_submission && submit_on_worker_thread && !wait_for_completion) {

        // Push to the pending submit queue.
        {
            std::lock_guard<std::mutex> guard(pending_submit_lock);
            pending_submits.push_back({present_swap_chain, present_image_index, current_frame});
        }

        // Wake up the worker thread for a single iteration.
        submit_loop->Wakeup();
    }
    else {
        // Pass through to normal submission path.
        SubmitCommandBuffer(current_frame, present_swap_chain, present_image_index);

        if (wait_for_completion) {
            WaitForCommandBufferCompletion(current_frame);
        }
    }

    // Switch to next cmdbuffer.
    BeginCommandBuffer();
}

void VKCommandManager::SubmitCommandBuffer(u32 command_buffer_index,
                                               vk::SwapchainKHR swapchain,
                                               u32 present_image_index) {
    auto& resources = frame_resources[command_buffer_index];

    vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submit_info({}, wait_stage, resources.command_buffers);

    // If the init command buffer did not have any commands recorded, don't submit it.
    if (!resources.init_command_buffer_used) {
        submit_info.setCommandBuffers(resources.command_buffers[1]);
    }

    if (resources.semaphore_used) {
        submit_info.setSignalSemaphores(resources.semaphore);
    }

    submit_info.setSignalSemaphores(present_semaphore);
    g_vk_instace->GetGraphicsQueue().submit(submit_info, resources.fence);

    // Should have a signal semaphore.
    vk::PresentInfoKHR present_info(present_semaphore, swapchain, present_image_index);
    auto last_present_result = g_vk_instace->GetPresentQueue().presentKHR(present_info);
    if (last_present_result != vk::Result::eSuccess) {

        // eErrorOutOfDateKHR is not fatal, just means we need to recreate our swap chain.
        if (last_present_result != vk::Result::eErrorOutOfDateKHR &&
            last_present_result != vk::Result::eSuboptimalKHR)
        {
            LOG_ERROR(Render_Vulkan, "Present queue return error");
        }

      // Don't treat eSuboptimalKHR as fatal on Android. Android 10+ requires prerotation.
      // See https://twitter.com/Themaister/status/1207062674011574273
    #ifdef VK_USE_PLATFORANDROID_KHR
        if (last_present_result != VK_SUBOPTIMAL_KHR) {
            last_present_failed.Set();
        }
    #else
        last_present_failed.Set();
    #endif
    }

    // Command buffer has been queued, so permit the next one.
    submit_semaphore.Post();
}

void VKCommandManager::BeginCommandBuffer()
{
    // Move to the next command buffer.
    const u32 next_buffer_index = (current_frame + 1) % COMMAND_BUFFER_COUNT;
    auto& resources = frame_resources[next_buffer_index];
    auto& device = g_vk_instace->GetDevice();

    // Wait for the GPU to finish with all resources for this command buffer.
    if (resources.fence_counter > completed_fence_counter) {
        WaitForCommandBufferCompletion(next_buffer_index);
    }

    // Reset fence to unsignaled before starting.
    device.resetFences(resources.fence);

    // Reset command pools to beginning since we can re-use the memory now
    device.resetCommandPool(resources.command_pool);

    vk::CommandBufferBeginInfo begin_info(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    // Enable commands to be recorded to the two buffers again.
    for (auto command_buffer : resources.command_buffers) {
        command_buffer.begin(begin_info);
    }

    // Also can do the same for the descriptor pools
    device.resetDescriptorPool(resources.descriptor_pool);

    // Reset upload command buffer state
    resources.init_command_buffer_used = false;
    resources.semaphore_used = false;
    resources.fence_counter = next_fence_counter++;
    current_frame = next_buffer_index;
}

std::unique_ptr<VKCommandManager> g_command_buffer_mgr;

}  // namespace Vulkan
