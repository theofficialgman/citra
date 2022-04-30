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
    VKTexture texture;
    VKFramebuffer framebuffer;
};

class VKSwapchain {
public:
    VKSwapchain(vk::SurfaceKHR surface);
    ~VKSwapchain() = default;

    /// Creates (or recreates) the swapchain with a given size.
    void Create(u32 width, u32 height, bool vsync_enabled);

    /// Acquires the next image in the swapchain, waits as needed.
    void AcquireNextImage();

    /// Returns true when the swapchain needs to be recreated.
    bool NeedsRecreation() const { return IsSubOptimal(); }
    bool IsOutDated() const { return is_outdated; }
    bool IsSubOptimal() const { return is_suboptimal; }
    bool IsVSyncEnabled() const { return vsync_enabled; }
    u32 GetCurrentImageIndex() const { return image_index; }

    /// Get current swapchain state
    vk::Extent2D GetSize() const { return extent; }
    vk::SurfaceKHR GetSurface() const { return surface; }
    vk::SurfaceFormatKHR GetSurfaceFormat() const { return surface_format; }
    vk::Format GetTextureFormat() const { return texture_format; }
    vk::SwapchainKHR GetSwapChain() const { return swapchain.get(); }
    vk::Image GetCurrentImage() const { return swapchain_images[image_index].image; }

    /// Retrieve current texture and framebuffer
    VKTexture& GetCurrentTexture() { return swapchain_images[image_index].texture; }
    VKFramebuffer& GetCurrentFramebuffer() { return swapchain_images[image_index].framebuffer; }

private:
    vk::SurfaceKHR surface;
    vk::SurfaceFormatKHR surface_format = {};
    vk::PresentModeKHR present_mode = vk::PresentModeKHR::eFifo;
    vk::Format texture_format = vk::Format::eUndefined;
    vk::Extent2D extent;
    bool vsync_enabled = false;
    bool is_outdated = false, is_suboptimal = false;

    vk::UniqueSwapchainKHR swapchain;
    std::vector<SwapChainImage> swapchain_images;
    u32 image_index = 0, frame_index = 0;
};

} // namespace Vulkan
