// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore::Vulkan {

/// The global Vulkan instance
class Instance {
public:
    Instance(Frontend::EmuWindow& window);
    ~Instance();

    /// Returns the Vulkan instance
    vk::Instance GetInstance() const {
        return instance;
    }

    /// Returns the Vulkan surface
    vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    /// Returns the current physical device
    vk::PhysicalDevice GetPhysicalDevice() const {
        return physical_device;
    }

    /// Returns the Vulkan device
    vk::Device GetDevice() const {
        return device;
    }

    VmaAllocator GetAllocator() const {
        return allocator;
    }

    /// Retrieve queue information
    u32 GetGraphicsQueueFamilyIndex() const {
        return graphics_queue_family_index;
    }

    u32 GetPresentQueueFamilyIndex() const {
        return present_queue_family_index;
    }

    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }

    vk::Queue GetPresentQueue() const {
        return present_queue;
    }

    /// Feature support
    bool IsDynamicRenderingSupported() const {
        return dynamic_rendering;
    }

    bool IsExtendedDynamicStateSupported() const {
        /*return extended_dynamic_state;*/return false;
    }

    bool IsPushDescriptorsSupported() const {
        return push_descriptors;
    }

    /// Returns the minimum required alignment for uniforms
    vk::DeviceSize UniformMinAlignment() const {
        return device_limits.minUniformBufferOffsetAlignment;
    }

    /// Returns true when the format supports the provided feature flags
    bool IsFormatSupported(vk::Format format, vk::FormatFeatureFlags usage) const;

    /// Returns the most compatible format that supports the provided feature flags
    vk::Format GetFormatAlternative(vk::Format format) const;

private:
    bool CreateDevice(bool validation_enabled);
    void CreateAllocator();

private:
    // Queue family indexes
    u32 present_queue_family_index = 0, graphics_queue_family_index = 0;
    vk::Queue present_queue, graphics_queue;

    // Core vulkan objects
    vk::Device device;
    vk::PhysicalDevice physical_device;
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDeviceLimits device_limits;
    VmaAllocator allocator;

    // Features per vulkan version
    bool dynamic_rendering = false;
    bool extended_dynamic_state = false;
    bool push_descriptors = false;
};

} // namespace VideoCore::Vulkan
