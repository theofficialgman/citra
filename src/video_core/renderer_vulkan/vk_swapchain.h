// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/common/framebuffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
class PoolManager;
}

namespace VideoCore::Vulkan {

class Instance;
class Backend;
class CommandScheduler;
class Texture;
class RenderpassCache;

class Swapchain {
public:
    Swapchain(Instance& instance, CommandScheduler& scheduler, RenderpassCache& renderpass_cache,
              PoolManager& pool_manager, vk::SurfaceKHR surface);
    ~Swapchain();

    // Creates (or recreates) the swapchain with a given size.
    void Create(u32 width, u32 height, bool vsync_enabled);

    // Acquires the next image in the swapchain.
    void AcquireNextImage();

    // Presents the current image and move to the next one
    void Present();

    FramebufferHandle GetCurrentFramebuffer() const {
        return framebuffers[current_image];
    }

    // Returns current swapchain state
    vk::Extent2D GetExtent() const {
        return extent;
    }

    // Returns the swapchain surface
    vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    // Returns the swapchain format
    vk::SurfaceFormatKHR GetSurfaceFormat() const {
        return surface_format;
    }

    // Returns the Vulkan swapchain handle
    vk::SwapchainKHR GetHandle() const {
        return swapchain;
    }

    // Returns the semaphore that will be signaled when vkAcquireNextImageKHR completes
    vk::Semaphore GetAvailableSemaphore() const {
        return image_available;
    }

    // Returns the semaphore that will signal when the current image will be presented
    vk::Semaphore GetPresentSemaphore() const {
        return render_finished;
    }

    // Returns the current swapchain image
    vk::Image GetCurrentImage() {
        return vk_images[current_image];
    }

    // Returns true when the swapchain should be recreated
    bool NeedsRecreation() const {
        return is_suboptimal || is_outdated;
    }

private:
    void Configure(u32 width, u32 height);

private:
    Instance& instance;
    CommandScheduler& scheduler;
    RenderpassCache& renderpass_cache;
    PoolManager& pool_manager;
    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
    vk::SurfaceKHR surface = VK_NULL_HANDLE;

    // Swapchain properties
    vk::SurfaceFormatKHR surface_format;
    vk::PresentModeKHR present_mode;
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    u32 image_count;

    // Swapchain state
    std::vector<vk::Image> vk_images;
    std::vector<TextureHandle> textures;
    std::vector<FramebufferHandle> framebuffers;
    vk::Semaphore image_available, render_finished;
    u32 current_image = 0, current_frame = 0;
    bool vsync_enabled = false;
    bool is_outdated = true;
    bool is_suboptimal = true;
};

} // namespace Vulkan
