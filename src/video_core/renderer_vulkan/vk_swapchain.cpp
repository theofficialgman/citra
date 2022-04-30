// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <vector>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

VKSwapChain::VKSwapChain(vk::SurfaceKHR surface_) : surface(surface_) {

}

bool VKSwapChain::Create(u32 width, u32 height, bool vsync_enabled) {
    is_outdated = false;
    is_suboptimal = false;

    // Fetch information about the provided surface
    PopulateSwapchainDetails(surface, width, height);

    // Now we can actually create the swapchain
    vk::SwapchainCreateInfoKHR swapchain_info
    (
        {},
        surface,
        details.image_count,
        details.format.format, details.format.colorSpace,
        details.extent, 1,
        vk::ImageUsageFlagBits::eColorAttachment,
        vk::SharingMode::eExclusive,
        0, nullptr,
        details.transform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        details.present_mode,
        VK_TRUE,
        swapchain.get()
    );

    std::array<u32, 2> indices = {
        g_vk_instace->GetGraphicsQueueFamilyIndex(),
        g_vk_instace->GetPresentQueueFamilyIndex(),
    };

    // For dedicated present queues, select concurrent sharing mode
    if (indices[0] != indices[1]) {
        swapchain_info.setImageSharingMode(vk::SharingMode::eConcurrent);
        swapchain_info.setQueueFamilyIndices(indices);
    }

    auto new_swapchain = g_vk_instace->GetDevice().createSwapchainKHRUnique(swapchain_info);

    // If an old swapchain exists, destroy it and move the new one to its place.
    // Synchronization is the responsibility of the caller, not us
    if (!!swapchain) {
        swapchain_images.clear();
        swapchain.swap(new_swapchain);
    }

    return true;
}

void VKSwapChain::AcquireNextImage(vk::Semaphore present_semaphore) {
    const auto result = g_vk_instace->GetDevice().acquireNextImageKHR(*swapchain,
                        std::numeric_limits<u64>::max(), present_semaphore,
                        VK_NULL_HANDLE, &image_index);

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
        is_suboptimal = true;
        break;
    case vk::Result::eErrorOutOfDateKHR:
        is_outdated = true;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "acquireNextImageKHR returned unknown result");
        break;
    }
}

void VKSwapChain::Present(vk::Semaphore render_semaphore) {
    const auto present_queue = g_vk_instace->GetPresentQueue();

    vk::PresentInfoKHR present_info(render_semaphore, swapchain.get(), image_index);
    vk::Result result = present_queue.presentKHR(present_info);

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
        LOG_DEBUG(Render_Vulkan, "Suboptimal swapchain");
        break;
    case vk::Result::eErrorOutOfDateKHR:
        is_outdated = true;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Swapchain presentation failed");
        break;
    }

    frame_index = (frame_index + 1) % swapchain_images.size();
}

void VKSwapChain::PopulateSwapchainDetails(vk::SurfaceKHR surface, u32 width, u32 height) {
    auto& gpu = g_vk_instace->GetPhysicalDevice();

    // Choose surface format
    auto formats = gpu.getSurfaceFormatsKHR(surface);
    details.format = formats[0];

    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        details.format = { vk::Format::eB8G8R8A8Unorm };
    }
    else {
        for (const auto& format : formats) {
            if (format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                format.format == vk::Format::eB8G8R8A8Unorm) {
                details.format = format;
                break;
            }
        }
    }

    // Checks if a particular mode is supported, if it is, returns that mode.
    auto modes = gpu.getSurfacePresentModesKHR(surface);
    auto ModePresent = [&modes](vk::PresentModeKHR check_mode) {
        auto it = std::find_if(modes.begin(), modes.end(), [check_mode](const auto& mode) {
                return check_mode == mode;
        });

        return it != modes.end();
    };

    // FIFO is guaranteed by the Vulkan standard to be available
    details.present_mode = vk::PresentModeKHR::eFifo;

    // Prefer Mailbox if present for lowest latency
    if (ModePresent(vk::PresentModeKHR::eMailbox)) {
        details.present_mode = vk::PresentModeKHR::eMailbox;
    }

    // Query surface extent
    auto capabilities = gpu.getSurfaceCapabilitiesKHR(surface);
    details.extent = capabilities.currentExtent;

    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        details.extent.width = std::clamp(width, capabilities.minImageExtent.width,
                                          capabilities.maxImageExtent.width);
        details.extent.height = std::clamp(height, capabilities.minImageExtent.height,
                                           capabilities.maxImageExtent.height);
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    details.image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        details.image_count = std::min(details.image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    details.transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & details.transform)) {
        details.transform = capabilities.currentTransform;
    }
}

void VKSwapChain::SetupImages() {
    // Get the swap chain images
    auto& device = g_vk_instace->GetDevice();
    auto images = device.getSwapchainImagesKHR(swapchain.get());

    // Create the swapchain buffers containing the image and imageview
    swapchain_images.resize(images.size());
    for (int i = 0; i < swapchain_images.size(); i++)
    {
        vk::ImageViewCreateInfo color_attachment_view
        (
            {},
            images[i],
            vk::ImageViewType::e2D,
            details.format.format,
            {},
            { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        );

        // Wrap swapchain images with VKTexture
        swapchain_images[i].image.Adopt(images[i], color_attachment_view);

        // Create framebuffer for each swapchain image
        VKFramebuffer::Info fb_info = {
            .color = &swapchain_images[i].image
        };

        swapchain_images[i].framebuffer.Create(fb_info);
    }
}

} // namespace Vulkan
