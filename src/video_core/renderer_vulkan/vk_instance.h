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

// If the size of this is too small, it ends up creating a soft cap on FPS as the renderer will have
// to wait on available presentation frames. There doesn't seem to be much of a downside to a larger
// number but 9 swap textures at 60FPS presentation allows for 800% speed so thats probably fine
#ifdef ANDROID
// Reduce the size of swap_chain, since the UI only allows upto 200% speed.
constexpr std::size_t SWAP_CHAIN_SIZE = 6;
#else
constexpr std::size_t SWAP_CHAIN_SIZE = 9;
#endif

/// The global Vulkan instance
class VKInstance
{
public:
    VKInstance() = default;
    ~VKInstance();

    /// Construct global Vulkan context
    void Create(vk::UniqueInstance instance, vk::PhysicalDevice gpu, vk::UniqueSurfaceKHR surface,
                bool enable_debug_reports, bool enable_validation_layer);

    vk::Device& GetDevice() { return device.get(); }
    vk::PhysicalDevice& GetPhysicalDevice() { return physical_device; }

    /// Get a valid command buffer for the current frame
    vk::CommandBuffer& GetCommandBuffer();

    /// Feature support
    bool SupportsAnisotropicFiltering() const;

private:
    void CreateDevices(int device_id = 0);
    void CreateRenderpass();
    void CreateCommandBuffers();

public:
    // Queue family indexes
    u32 queue_family = -1;

    // Core vulkan objects
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    vk::Queue graphics_queue;

    // Pipeline
    vk::UniqueDescriptorPool descriptor_pool;
    std::array<std::vector<vk::DescriptorSetLayout>, SWAP_CHAIN_SIZE> descriptor_layouts;
    std::array<std::vector<vk::DescriptorSet>, SWAP_CHAIN_SIZE> descriptor_sets;

    // Command buffer
    vk::UniqueCommandPool command_pool;
    std::vector<vk::UniqueCommandBuffer> command_buffers;
};

extern std::unique_ptr<VKInstance> g_vk_instace;

} // namespace Vulkan
