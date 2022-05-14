// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <unordered_map>
#include <memory>
#include "common/common_types.h"

namespace Vulkan {

/// The global Vulkan instance
class VKInstance
{
public:
    VKInstance() = default;
    ~VKInstance();

    /// Construct global Vulkan context
    bool Create(vk::Instance instance, vk::PhysicalDevice gpu,
                vk::SurfaceKHR surface, bool enable_validation_layer);

    vk::Device& GetDevice() { return device.get(); }
    vk::PhysicalDevice& GetPhysicalDevice() { return physical_device; }

    /// Retrieve queue information
    u32 GetGraphicsQueueFamilyIndex() const { return graphics_queue_family_index; }
    u32 GetPresentQueueFamilyIndex() const { return present_queue_family_index; }
    vk::Queue GetGraphicsQueue() { return graphics_queue; }
    vk::Queue GetPresentQueue() { return present_queue; }

    /// Feature support
    bool SupportsAnisotropicFiltering() const;
    u32 UniformMinAlignment() const { return device_limits.minUniformBufferOffsetAlignment; }

private:
    bool CreateDevice(vk::SurfaceKHR surface, bool validation_enabled);
    bool FindExtensions();
    bool FindFeatures();

public:
    // Queue family indexes
    u32 present_queue_family_index{}, graphics_queue_family_index{};
    vk::Queue present_queue, graphics_queue;

    // Core vulkan objects
    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;

    // Extensions and features
    std::vector<const char*> extensions;
    vk::PhysicalDeviceFeatures2 features{};
    vk::PhysicalDeviceLimits device_limits;

    // Features per vulkan version
    vk::PhysicalDeviceFeatures vk_features{};
    vk::PhysicalDeviceVulkan12Features vk12_features{};
    vk::PhysicalDeviceVulkan13Features vk13_features{};
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state{};
    vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT dynamic_state2{};
};

extern std::unique_ptr<VKInstance> g_vk_instace;

} // namespace Vulkan
