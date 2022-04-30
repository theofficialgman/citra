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

struct SwapchainDetails {
    vk::SurfaceFormatKHR format;
    vk::PresentModeKHR present_mode;
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    u32 image_count;
};

SwapchainDetails PopulateSwapchainDetails(vk::SurfaceKHR surface, u32 width, u32 height) {
    SwapchainDetails details;
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

    // FIFO is guaranteed by the standard to be available
    details.present_mode = vk::PresentModeKHR::eFifo;

    // Prefer Mailbox if present for lowest latency
    if (ModePresent(vk::PresentModeKHR::eMailbox)) {
        details.present_mode = vk::PresentModeKHR::eMailbox;
    }

    // Query surface capabilities
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

    return details;
}

VKSwapchain::VKSwapchain(vk::SurfaceKHR surface_) : surface(surface_) {

}

void VKSwapchain::Create(u32 width, u32 height, bool vsync_enabled) {
    is_outdated = false;
    is_suboptimal = false;

    const auto gpu = g_vk_instace->GetPhysicalDevice();
    auto details = PopulateSwapchainDetails(surface, width, height);

    // Store the old/current swap chain when recreating for resize
    vk::SwapchainKHR old_swapchain = swapchain.get();

    // Now we can actually create the swap chain
    vk::SwapchainCreateInfoKHR swap_chain_info
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
        old_swapchain
    );

    std::array<uint32_t, 2> indices = {{
        g_vulkan_context->GetGraphicsQueueFamilyIndex(),
        g_vulkan_context->GetPresentQueueFamilyIndex(),
    }};
    if (g_vulkan_context->GetGraphicsQueueFamilyIndex() !=
        g_vulkan_context->GetPresentQueueFamilyIndex())
    {
      swap_chain_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      swap_chain_info.queueFamilyIndexCount = 2;
      swap_chain_info.pQueueFamilyIndices = indices.data();
    }

  #ifdef SUPPORTS_VULKAN_EXCLUSIVE_FULLSCREEN
    if (m_fullscreen_supported)
    {
      VkSurfaceFullScreenExclusiveInfoEXT fullscreen_support = {};
      swap_chain_info.pNext = &fullscreen_support;
      fullscreen_support.sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;
      fullscreen_support.fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT;

      auto platform_info = g_vulkan_context->GetPlatformExclusiveFullscreenInfo(m_wsi);
      fullscreen_support.pNext = &platform_info;

      res = vkCreateSwapchainKHR(g_vulkan_context->GetDevice(), &swap_chain_info, nullptr,
                                 &m_swap_chain);
      if (res != VK_SUCCESS)
      {
        // Try without exclusive fullscreen.
        WARN_LOG_FMT(VIDEO, "Failed to create exclusive fullscreen swapchain, trying without.");
        swap_chain_info.pNext = nullptr;
        g_Config.backend_info.bSupportsExclusiveFullscreen = false;
        g_ActiveConfig.backend_info.bSupportsExclusiveFullscreen = false;
        m_fullscreen_supported = false;
      }
    }
  #endif

    if (m_swap_chain == VK_NULL_HANDLE)
    {
      res = vkCreateSwapchainKHR(g_vulkan_context->GetDevice(), &swap_chain_info, nullptr,
                                 &m_swap_chain);
    }
    if (res != VK_SUCCESS)
    {
      LOG_VULKAN_ERROR(res, "vkCreateSwapchainKHR failed: ");
      return false;
    }

    // Now destroy the old swap chain, since it's been recreated.
    // We can do this immediately since all work should have been completed before calling resize.
    if (old_swap_chain != VK_NULL_HANDLE)
      vkDestroySwapchainKHR(g_vulkan_context->GetDevice(), old_swap_chain, nullptr);

    m_width = size.width;
    m_height = size.height;
    m_layers = image_layers;
    return true;
}

void VKSwapchain::AcquireNextImage() {
    const auto result = g_vk_instace->GetDevice().acquireNextImageKHR(*swapchain,
                        std::numeric_limits<u64>::max(), *present_semaphores[frame_index],
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

void VKSwapchain::Present(vk::Semaphore render_semaphore) {
    const auto present_queue{device.GetPresentQueue()};


    vk::PresentInfoKHR present_info(

    const VkPresentInfoKHR present_info{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = render_semaphore ? 1U : 0U,
        .pWaitSemaphores = &render_semaphore,
        .swapchainCount = 1,
        .pSwapchains = swapchain.address(),
        .pImageIndices = &image_index,
        .pResults = nullptr,
    };
    switch (const VkResult result = present_queue.Present(present_info)) {
    case VK_SUCCESS:
        break;
    case VK_SUBOPTIMAL_KHR:
        LOG_DEBUG(Render_Vulkan, "Suboptimal swapchain");
        break;
    case VK_ERROR_OUT_OF_DATE_KHR:
        is_outdated = true;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Failed to present with error {}", vk::ToString(result));
        break;
    }
    ++frame_index;
    if (frame_index >= image_count) {
        frame_index = 0;
    }
}

void VKSwapchain::CreateSwapchain(const VkSurfaceCapabilitiesKHR& capabilities, u32 width,
                                  u32 height, bool srgb) {
    const auto physical_device{device.GetPhysical()};
    const auto formats{physical_device.GetSurfaceFormatsKHR(surface)};
    const auto present_modes{physical_device.GetSurfacePresentModesKHR(surface)};

    const VkSurfaceFormatKHR surface_format{ChooseSwapSurfaceFormat(formats)};
    present_mode = ChooseSwapPresentMode(present_modes);

    u32 requested_image_count{capabilities.minImageCount + 1};
    if (capabilities.maxImageCount > 0 && requested_image_count > capabilities.maxImageCount) {
        requested_image_count = capabilities.maxImageCount;
    }
    VkSwapchainCreateInfoKHR swapchain_ci{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface,
        .minImageCount = requested_image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = {},
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_FALSE,
        .oldSwapchain = nullptr,
    };
    const u32 graphics_family{device.GetGraphicsFamily()};
    const u32 present_family{device.GetPresentFamily()};
    const std::array<u32, 2> queue_indices{graphics_family, present_family};
    if (graphics_family != present_family) {
        swapchain_ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_ci.queueFamilyIndexCount = static_cast<u32>(queue_indices.size());
        swapchain_ci.pQueueFamilyIndices = queue_indices.data();
    }
    static constexpr std::array view_formats{VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB};
    VkImageFormatListCreateInfo format_list{
        .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
        .pNext = nullptr,
        .viewFormatCount = static_cast<u32>(view_formats.size()),
        .pViewFormats = view_formats.data(),
    };
    if (device.IsKhrSwapchainMutableFormatEnabled()) {
        format_list.pNext = std::exchange(swapchain_ci.pNext, &format_list);
        swapchain_ci.flags |= VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
    }
    // Request the size again to reduce the possibility of a TOCTOU race condition.
    const auto updated_capabilities = physical_device.GetSurfaceCapabilitiesKHR(surface);
    swapchain_ci.imageExtent = ChooseSwapExtent(updated_capabilities, width, height);
    // Don't add code within this and the swapchain creation.
    swapchain = device.GetLogical().CreateSwapchainKHR(swapchain_ci);

    extent = swapchain_ci.imageExtent;
    current_srgb = srgb;
    current_fps_unlocked = Settings::values.disable_fps_limit.GetValue();

    images = swapchain.GetImages();
    image_count = static_cast<u32>(images.size());
    image_view_format = srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
}

void VKSwapchain::CreateImageViews() {
    VkImageViewCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = {},
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = image_view_format,
        .components =
            {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    };

    image_views.resize(image_count);
    for (std::size_t i = 0; i < image_count; i++) {
        ci.image = images[i];
        image_views[i] = device.GetLogical().CreateImageView(ci);
    }
}

void VKSwapchain::Destroy() {
    frame_index = 0;
    present_semaphores.clear();
    framebuffers.clear();
    image_views.clear();
    swapchain.reset();
}

bool VKSwapchain::NeedsPresentModeUpdate() const {
    // Mailbox present mode is the ideal for all scenarios. If it is not available,
    // A different present mode is needed to support unlocked FPS above the monitor's refresh rate.
    return present_mode != VK_PRESENT_MODE_MAILBOX_KHR && HasFpsUnlockChanged();
}

} // namespace Vulkan
