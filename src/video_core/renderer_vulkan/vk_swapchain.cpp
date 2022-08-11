// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <array>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_backend.h"
#include "video_core/renderer_vulkan/vk_framebuffer.h"
#include "video_core/renderer_vulkan/vk_texture.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"

namespace VideoCore::Vulkan {

Swapchain::Swapchain(Instance& instance, CommandScheduler& scheduler, RenderpassCache& renderpass_cache,
                     Backend* backend, vk::SurfaceKHR surface) :
    backend(backend), instance(instance), scheduler(scheduler),
    renderpass_cache(renderpass_cache), surface(surface) {

}

Swapchain::~Swapchain() {
    // Destroy swapchain resources
    vk::Device device = instance.GetDevice();
    device.destroySemaphore(render_finished);
    device.destroySemaphore(image_available);
    device.destroySwapchainKHR(swapchain);
}

void Swapchain::Create(u32 width, u32 height, bool vsync_enabled) {
    is_outdated = false;
    is_suboptimal = false;

    // Fetch information about the provided surface
    Configure(width, height);

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 2u : 1u;
    const vk::SharingMode sharing_mode = exclusive ? vk::SharingMode::eExclusive :
                                                     vk::SharingMode::eConcurrent;

    // Now we can actually create the swapchain
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices   = queue_family_indices.data(),
        .preTransform = transform,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = swapchain
    };

    vk::Device device = instance.GetDevice();
    vk::SwapchainKHR new_swapchain = device.createSwapchainKHR(swapchain_info);

    // If an old swapchain exists, destroy it and move the new one to its place.
    if (vk::SwapchainKHR old_swapchain = std::exchange(swapchain, new_swapchain); old_swapchain) {
        device.destroySwapchainKHR(old_swapchain);
    }

    // Create sync objects if not already created
    if (!image_available) {
        image_available = device.createSemaphore({});
    }

    if (!render_finished) {
        render_finished = device.createSemaphore({});
    }

    // Create the present renderpass
    renderpass_cache.CreatePresentRenderpass(surface_format.format);

    // Create framebuffer and image views
    vk_images = device.getSwapchainImagesKHR(swapchain);

    const TextureInfo image_info = {
        .width = static_cast<u16>(width),
        .height = static_cast<u16>(height),
        .levels = 1,
        .type = TextureType::Texture2D,
        .view_type = TextureViewType::View2D,
        .format = TextureFormat::PresentColor
    };

    // Wrap vulkan image handles with our texture wrapper
    textures.clear();
    textures.resize(vk_images.size());
    framebuffers.clear();
    framebuffers.resize(vk_images.size());
    for (int i = 0; i < vk_images.size(); i++) {
        textures[i] = MakeHandle<Texture>(instance, scheduler, vk_images[i],
                                          surface_format.format, image_info);

        const FramebufferInfo framebuffer_info = {
            .color = textures[i]
        };

        vk::RenderPass renderpass = renderpass_cache.GetPresentRenderpass();
        framebuffers[i] = MakeHandle<Framebuffer>(instance, scheduler, framebuffer_info,
                                                  renderpass, renderpass);
    }
}

// Wait for maximum of 1 second
constexpr u64 ACQUIRE_TIMEOUT = 1000000000;

void Swapchain::AcquireNextImage() {
    vk::Device device = instance.GetDevice();
    vk::Result result = device.acquireNextImageKHR(swapchain, ACQUIRE_TIMEOUT,
                                                   image_available, VK_NULL_HANDLE,
                                                   &current_image);
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
        LOG_ERROR(Render_Vulkan, "vkAcquireNextImageKHR returned unknown result");
        break;
    }
}

void Swapchain::Present() {
    const vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render_finished,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &current_image
    };

    vk::Queue present_queue = instance.GetPresentQueue();
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

    current_frame = (current_frame + 1) % vk_images.size();
}

void Swapchain::Configure(u32 width, u32 height) {
    vk::PhysicalDevice physical = instance.GetPhysicalDevice();

    // Choose surface format
    auto formats = physical.getSurfaceFormatsKHR(surface);
    surface_format = formats[0];

    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        surface_format = vk::SurfaceFormatKHR{
            .format = vk::Format::eB8G8R8A8Unorm
        };
    } else {
        auto iter = std::find_if(formats.begin(), formats.end(), [](vk::SurfaceFormatKHR format) -> bool {
            return format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                format.format == vk::Format::eB8G8R8A8Unorm;
        });

        if (iter == formats.end()) {
            LOG_CRITICAL(Render_Vulkan, "Unable to find required swapchain format!");
        }
    }

    // Checks if a particular mode is supported, if it is, returns that mode.
    auto modes = physical.getSurfacePresentModesKHR(surface);

    // FIFO is guaranteed by the Vulkan standard to be available
    present_mode = vk::PresentModeKHR::eFifo;

    auto iter = std::find_if(modes.begin(), modes.end(), [](vk::PresentModeKHR mode) {
        return vk::PresentModeKHR::eMailbox == mode;
    });

    // Prefer Mailbox if present for lowest latency
    if (iter != modes.end()) {
        present_mode = vk::PresentModeKHR::eMailbox;
    }

    // Query surface extent
    auto capabilities = physical.getSurfaceCapabilitiesKHR(surface);
    extent = capabilities.currentExtent;

    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::clamp(width, capabilities.minImageExtent.width,
                                          capabilities.maxImageExtent.width);
        extent.height = std::clamp(height, capabilities.minImageExtent.height,
                                           capabilities.maxImageExtent.height);
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & transform)) {
        transform = capabilities.currentTransform;
    }
}

} // namespace VideoCore::Vulkan
