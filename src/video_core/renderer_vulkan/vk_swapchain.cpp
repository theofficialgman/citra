// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

Swapchain::Swapchain(vk::SurfaceKHR surface_) : surface(surface_) {

}

Swapchain::~Swapchain() {
    auto device = g_vk_instace->GetDevice();
    auto instance = g_vk_instace->GetInstance();
    device.waitIdle();

    device.destroySemaphore(render_finished);
    device.destroySemaphore(image_available);
    device.destroySwapchainKHR(swapchain);
    instance.destroySurfaceKHR(surface);
}

bool Swapchain::Create(u32 width, u32 height, bool vsync_enabled) {
    is_outdated = false;
    is_suboptimal = false;

    // Fetch information about the provided surface
    PopulateSwapchainDetails(surface, width, height);

    const std::array indices {
        g_vk_instace->GetGraphicsQueueFamilyIndex(),
        g_vk_instace->GetPresentQueueFamilyIndex(),
    };

    // Now we can actually create the swapchain
    vk::SwapchainCreateInfoKHR swapchain_info{{}, surface, details.image_count, details.format.format,
                details.format.colorSpace, details.extent, 1, vk::ImageUsageFlagBits::eColorAttachment,
                vk::SharingMode::eExclusive, 1, indices.data(), details.transform,
                vk::CompositeAlphaFlagBitsKHR::eOpaque, details.present_mode, true, swapchain};

    // For dedicated present queues, select concurrent sharing mode
    if (indices[0] != indices[1]) {
        swapchain_info.imageSharingMode = vk::SharingMode::eConcurrent;
        swapchain_info.queueFamilyIndexCount = 2;
    }

    auto device = g_vk_instace->GetDevice();
    auto new_swapchain = device.createSwapchainKHR(swapchain_info);

    // If an old swapchain exists, destroy it and move the new one to its place.
    if (swapchain) {
        device.destroy(swapchain);
    }
    swapchain = new_swapchain;

    // Create sync objects if not already created
    if (!image_available) {
        image_available = device.createSemaphore({});
    }

    if (!render_finished) {
        render_finished = device.createSemaphore({});
    }

    // Create framebuffer and image views
    swapchain_images.clear();
    SetupImages();

    return true;
}

// Wait for maximum of 1 second
constexpr u64 ACQUIRE_TIMEOUT = 1000000000;

void Swapchain::AcquireNextImage() {
    auto result = g_vk_instace->GetDevice().acquireNextImageKHR(swapchain, ACQUIRE_TIMEOUT,
                                                                image_available, VK_NULL_HANDLE,
                                                                &image_index);
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

void Swapchain::Present() {
    const auto present_queue = g_vk_instace->GetPresentQueue();

    vk::PresentInfoKHR present_info(render_finished, swapchain, image_index);
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

void Swapchain::PopulateSwapchainDetails(vk::SurfaceKHR surface, u32 width, u32 height) {
    auto gpu = g_vk_instace->GetPhysicalDevice();

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

void Swapchain::SetupImages() {
    // Get the swap chain images
    auto device = g_vk_instace->GetDevice();
    auto images = device.getSwapchainImagesKHR(swapchain);

    Texture::Info image_info{
        .width = details.extent.width,
        .height = details.extent.height,
        .format = details.format.format,
        .type = vk::ImageType::e2D,
        .view_type = vk::ImageViewType::e2D,
        .usage = vk::ImageUsageFlagBits::eColorAttachment
    };

    // Create the swapchain buffers containing the image and imageview
    swapchain_images.resize(images.size());
    for (int i = 0; i < swapchain_images.size(); i++) {
        // Wrap swapchain images with Texture
        swapchain_images[i].Adopt(image_info, images[i]);
    }
}

} // namespace Vulkan
