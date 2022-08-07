// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// Include vulkan-hpp header
#define VK_NO_PROTOTYPES 1
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

// Include Vulkan memory allocator
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1001000 // Vulkan 1.1
#include <vk_mem_alloc.h>

namespace VideoCore::Vulkan {

/// Returns the aligned byte size of each pixel in the specified format
constexpr float GetFormatSize(vk::Format format) {
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eD24UnormS8Uint:
        return 4;
    case vk::Format::eR8G8B8Unorm:
        return 3;
    case vk::Format::eR5G5B5A1UnormPack16:
    case vk::Format::eR5G6B5UnormPack16:
    case vk::Format::eR4G4B4A4UnormPack16:
    case vk::Format::eD16Unorm:
        return 2;
    default:
        return 0;
    };
}

/// Return the image aspect associated on the provided format
constexpr vk::ImageAspectFlags GetImageAspect(vk::Format format) {
    vk::ImageAspectFlags flags;
    switch (format) {
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32SfloatS8Uint:
        flags = vk::ImageAspectFlagBits::eStencil | vk::ImageAspectFlagBits::eDepth;
        break;
    case vk::Format::eD16Unorm:
    case vk::Format::eD32Sfloat:
        flags = vk::ImageAspectFlagBits::eDepth;
        break;
    default:
        flags = vk::ImageAspectFlagBits::eColor;
    }

    return flags;
}

/// Returns a bit mask with the required usage of a format with a particular aspect
constexpr vk::ImageUsageFlags GetImageUsage(vk::ImageAspectFlags aspect) {
    auto usage = vk::ImageUsageFlagBits::eSampled |
            vk::ImageUsageFlagBits::eTransferDst |
            vk::ImageUsageFlagBits::eTransferSrc;

    if (aspect & vk::ImageAspectFlagBits::eDepth) {
        return usage | vk::ImageUsageFlagBits::eDepthStencilAttachment;
    } else {
        return usage | vk::ImageUsageFlagBits::eColorAttachment;
    }
};

/// Returns a bit mask with the required features of a format with a particular aspect
constexpr vk::FormatFeatureFlags GetFormatFeatures(vk::ImageAspectFlags aspect) {
    auto usage = vk::FormatFeatureFlagBits::eSampledImage |
            vk::FormatFeatureFlagBits::eTransferDst |
            vk::FormatFeatureFlagBits::eTransferSrc |
            vk::FormatFeatureFlagBits::eBlitSrc |
            vk::FormatFeatureFlagBits::eBlitDst;

    if (aspect & vk::ImageAspectFlagBits::eDepth) {
        return usage | vk::FormatFeatureFlagBits::eDepthStencilAttachment;
    } else {
        return usage | vk::FormatFeatureFlagBits::eColorAttachment;
    }
};

}
