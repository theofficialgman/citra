// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string_view>
#include "core/frontend/emu_window.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

struct SwapChainImage {
    vk::Image image;
    vk::UniqueImageView image_view;
    vk::UniqueFramebuffer framebuffer;
};

struct SwapChainDetails {
    vk::SurfaceFormatKHR format;
    vk::PresentModeKHR present_mode;
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    u32 image_count;
};

class VKSwapChain {
public:
    VKSwapChain(vk::SurfaceKHR surface);
    ~VKSwapChain() = default;

    /// Creates (or recreates) the swapchain with a given size.
    bool Create(u32 width, u32 height, bool vsync_enabled);

    /// Acquire the next image in the swapchain.
    vk::Semaphore AcquireNextImage();
    void Present(vk::Semaphore render_semaphore);

    /// Returns true when the swapchain needs to be recreated.
    bool NeedsRecreation() const { return IsSubOptimal() || IsOutDated(); }
    bool IsOutDated() const { return is_outdated; }
    bool IsSubOptimal() const { return is_suboptimal; }
    bool IsVSyncEnabled() const { return vsync_enabled; }
    u32 GetCurrentImageIndex() const { return image_index; }

    /// Get current swapchain state
    vk::Extent2D GetSize() const { return details.extent; }
    vk::SurfaceKHR GetSurface() const { return surface; }
    vk::SurfaceFormatKHR GetSurfaceFormat() const { return details.format; }
    vk::SwapchainKHR GetSwapChain() const { return swapchain.get(); }

    /// Retrieve current texture and framebuffer
    vk::Image GetCurrentImage() { return swapchain_images[image_index].image; }
    vk::Framebuffer GetCurrentFramebuffer() { return swapchain_images[image_index].framebuffer.get(); }

private:
    void PopulateSwapchainDetails(vk::SurfaceKHR surface, u32 width, u32 height);
    void SetupImages();

private:
    SwapChainDetails details{};
    vk::SurfaceKHR surface;
    vk::UniqueSemaphore image_available;
    bool vsync_enabled = false;
    bool is_outdated = false, is_suboptimal = false;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<SwapChainImage> swapchain_images;
    u32 image_index = 0, frame_index = 0;
};

} // namespace Vulkan
