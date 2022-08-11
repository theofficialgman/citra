// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/common/framebuffer.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

class Instance;
class Backend;
class CommandScheduler;
class Texture;
class RenderpassCache;

class Swapchain {
public:
    Swapchain(Instance& instance, CommandScheduler& scheduler, RenderpassCache& renderpass_cache,
              Backend* backend, vk::SurfaceKHR surface);
    ~Swapchain();

    /// Creates (or recreates) the swapchain with a given size.
    void Create(u32 width, u32 height, bool vsync_enabled);

    /// Acquire the next image in the swapchain.
    void AcquireNextImage();

    /// Present the current image and move to the next one
    void Present();

    FramebufferHandle GetCurrentFramebuffer() const {
        return framebuffers[current_image];
    }

    /// Return current swapchain state
    inline vk::Extent2D GetExtent() const {
        return extent;
    }

    /// Return the swapchain surface
    inline vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    /// Return the swapchain format
    inline vk::SurfaceFormatKHR GetSurfaceFormat() const {
        return surface_format;
    }

    /// Return the Vulkan swapchain handle
    inline vk::SwapchainKHR GetHandle() const {
        return swapchain;
    }

    /// Return the semaphore that will be signaled when vkAcquireNextImageKHR completes
    inline vk::Semaphore GetAvailableSemaphore() const {
        return image_available;
    }

    /// Return the semaphore that will signal when the current image will be presented
    inline vk::Semaphore GetPresentSemaphore() const {
        return render_finished;
    }

    /// Return the current swapchain image
    inline vk::Image GetCurrentImage() {
        return vk_images[current_image];
    }

    /// Returns true when the swapchain should be recreated
    inline bool NeedsRecreation() const {
        return is_suboptimal || is_outdated;
    }

private:
    void Configure(u32 width, u32 height);

private:
    Backend* backend = nullptr;
    Instance& instance;
    CommandScheduler& scheduler;
    RenderpassCache& renderpass_cache;
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
