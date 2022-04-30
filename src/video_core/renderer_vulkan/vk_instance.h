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

// Using multiple command buffers prevents stalling
constexpr u32 COMMAND_BUFFER_COUNT = 3;

struct FrameResources
{
    vk::CommandPool command_pool;
    std::array<vk::CommandBuffer, COMMAND_BUFFER_COUNT> command_buffers = {};
    vk::DescriptorPool descriptor_pool;
    vk::Fence fence;
    vk::Semaphore semaphore;
    u64 fence_counter = 0;
    bool init_command_buffer_used = false;
    bool semaphore_used = false;

    std::vector<std::function<void()>> cleanup_resources;
};

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

    /// Feature support
    bool SupportsAnisotropicFiltering() const;

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
    std::vector<const char*> device_extensions;
    vk::PhysicalDeviceFeatures device_features{};
};

extern std::unique_ptr<VKInstance> g_vk_instace;

} // namespace Vulkan
