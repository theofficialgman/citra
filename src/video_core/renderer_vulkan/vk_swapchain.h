// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <string_view>
#include "core/frontend/emu_window.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Vulkan {

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
    ~VKSwapChain();

    /// Creates (or recreates) the swapchain with a given size.
    bool Create(u32 width, u32 height, bool vsync_enabled);

    /// Acquire the next image in the swapchain.
    void AcquireNextImage();
    void Present();

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
    vk::SwapchainKHR GetSwapChain() const { return swapchain; }
    const vk::Semaphore& GetAvailableSemaphore() const { return image_available.get(); }
    const vk::Semaphore& GetRenderSemaphore() const { return render_finished.get(); }
    VKTexture& GetCurrentImage() { return swapchain_images[image_index]; }

private:
    void PopulateSwapchainDetails(vk::SurfaceKHR surface, u32 width, u32 height);
    void SetupImages();

private:
    SwapChainDetails details{};
    vk::SurfaceKHR surface;
    vk::UniqueSemaphore image_available, render_finished;
    bool vsync_enabled{false}, is_outdated{true}, is_suboptimal{true};

    vk::SwapchainKHR swapchain{VK_NULL_HANDLE};
    std::vector<VKTexture> swapchain_images;
    u32 image_index{0}, frame_index{0};
};

} // namespace Vulkan
