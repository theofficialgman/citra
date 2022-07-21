// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/// The global Vulkan instance
class Instance {
public:
    Instance() = default;
    ~Instance();

    /// Construct global Vulkan context
    bool Create(vk::Instance instance, vk::PhysicalDevice gpu,
                vk::SurfaceKHR surface, bool enable_validation_layer);

    vk::Device GetDevice() const { return device; }
    vk::PhysicalDevice GetPhysicalDevice() const { return physical_device; }
    vk::Instance GetInstance() const { return instance; }

    /// Retrieve queue information
    u32 GetGraphicsQueueFamilyIndex() const { return graphics_queue_family_index; }
    u32 GetPresentQueueFamilyIndex() const { return present_queue_family_index; }
    vk::Queue GetGraphicsQueue() const { return graphics_queue; }
    vk::Queue GetPresentQueue() const { return present_queue; }

    /// Feature support
    bool SupportsAnisotropicFiltering() const;
    u32 UniformMinAlignment() const { return static_cast<u32>(device_limits.minUniformBufferOffsetAlignment); }

private:
    bool CreateDevice(vk::SurfaceKHR surface, bool validation_enabled);
    bool FindExtensions();
    bool FindFeatures();

public:
    // Queue family indexes
    u32 present_queue_family_index{}, graphics_queue_family_index{};
    vk::Queue present_queue, graphics_queue;

    // Core vulkan objects
    vk::PhysicalDevice physical_device;
    vk::Instance instance;
    vk::Device device;

    // Extensions and features
    std::vector<const char*> extensions;
    vk::PhysicalDeviceFeatures2 features{};
    vk::PhysicalDeviceLimits device_limits;

    // Features per vulkan version
    vk::PhysicalDeviceFeatures vk_features{};
    vk::PhysicalDeviceVulkan13Features vk13_features{};
    vk::PhysicalDeviceVulkan12Features vk12_features{};
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state_features{};
    vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT dynamic_state2_features{};
    vk::PhysicalDeviceColorWriteEnableFeaturesEXT color_write_features{};
};

extern std::unique_ptr<Instance> g_vk_instace;

} // namespace Vulkan
