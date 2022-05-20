// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <fstream>
#include <array>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

VKInstance::~VKInstance() {

}

bool VKInstance::Create(vk::Instance instance, vk::PhysicalDevice physical_device,
                        vk::SurfaceKHR surface, bool enable_validation_layer) {
    this->instance = instance;
    this->physical_device = physical_device;

    // Get physical device limits
    device_limits = physical_device.getProperties().limits;

    // Determine required extensions and features
    if (!FindExtensions() || !FindFeatures())
        return false;

    // Create logical device
    return CreateDevice(surface, enable_validation_layer);
}

bool VKInstance::CreateDevice(vk::SurfaceKHR surface, bool validation_enabled) {
    // Can't create an instance without a valid surface
    if (!surface) {
        LOG_CRITICAL(Render_Vulkan, "Invalid surface provided during instance creation!");
        return false;
    }

    auto family_properties = physical_device.getQueueFamilyProperties();
    if (family_properties.empty()) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan physical device reported no queues.");
        return false;
    }

    // Search queue families for graphics and present queues
    graphics_queue_family_index = -1;
    present_queue_family_index = -1;
    for (int i = 0; i < family_properties.size(); i++) {
        // Check if queue supports graphics
        if (family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            graphics_queue_family_index = i;

            // If this queue also supports presentation we are finished
            if (physical_device.getSurfaceSupportKHR(i, surface)) {
                present_queue_family_index = i;
                break;
            }
        }

        // Check if queue supports presentation
        if (physical_device.getSurfaceSupportKHR(i, surface)) {
            present_queue_family_index = i;
        }
    }

    if (graphics_queue_family_index == -1 ||
        present_queue_family_index == -1) {
        LOG_CRITICAL(Render_Vulkan, "Unable to find graphics and/or present queues.");
        return false;
    }

    static constexpr float queue_priorities[] = { 1.0f };

    vk::DeviceCreateInfo device_info;
    device_info.setPEnabledExtensionNames(extensions);

    // Create queue create info structs
    if (graphics_queue_family_index != present_queue_family_index) {
        std::array<vk::DeviceQueueCreateInfo, 2> queue_infos = {
            vk::DeviceQueueCreateInfo({}, graphics_queue_family_index, 1, queue_priorities),
            vk::DeviceQueueCreateInfo({}, present_queue_family_index, 1, queue_priorities)
        };

        device_info.setQueueCreateInfos(queue_infos);
    }
    else {
        std::array<vk::DeviceQueueCreateInfo, 1> queue_infos = {
            vk::DeviceQueueCreateInfo({}, graphics_queue_family_index, 1, queue_priorities),
        };

        device_info.setQueueCreateInfos(queue_infos);
    }

    // Set device features
    device_info.setPEnabledFeatures(&vk_features);

    // Enable debug layer on debug builds
    if (validation_enabled) {
        std::array<const char*, 1> layer_names = { "VK_LAYER_KHRONOS_validation" };
        device_info.setPEnabledLayerNames(layer_names);
    }

    // Create logical device
    device = physical_device.createDeviceUnique(device_info);

    // Grab the graphics and present queues.
    graphics_queue = device->getQueue(graphics_queue_family_index, 0);
    present_queue = device->getQueue(present_queue_family_index, 0);

    return true;
}

bool VKInstance::FindFeatures()
{
    auto available = physical_device.getFeatures();

    // Not having geometry shaders or wide lines will cause issues with rendering.
    if (!available.geometryShader && !available.wideLines) {
        LOG_WARNING(Render_Vulkan, "Geometry shaders not availabe! Rendering will be limited");
    }

    // Enable some common features other emulators like Dolphin use
    vk_features.dualSrcBlend = available.dualSrcBlend;
    vk_features.geometryShader = available.geometryShader;
    vk_features.samplerAnisotropy = available.samplerAnisotropy;
    vk_features.logicOp = available.logicOp;
    vk_features.fragmentStoresAndAtomics = available.fragmentStoresAndAtomics;
    vk_features.sampleRateShading = available.sampleRateShading;
    vk_features.largePoints = available.largePoints;
    vk_features.shaderStorageImageMultisample = available.shaderStorageImageMultisample;
    vk_features.occlusionQueryPrecise = available.occlusionQueryPrecise;
    vk_features.shaderClipDistance = available.shaderClipDistance;
    vk_features.depthClamp = available.depthClamp;
    vk_features.textureCompressionBC = available.textureCompressionBC;

    // Enable newer Vulkan features
    vk12_features.timelineSemaphore = true;
    vk13_features.dynamicRendering = true;
    dynamic_state.extendedDynamicState = true;
    dynamic_state2.extendedDynamicState2 = true;
    dynamic_state2.extendedDynamicState2LogicOp = true;
    dynamic_state2.extendedDynamicState2PatchControlPoints = true;

    // Include features in device creation
    features.setFeatures(vk_features);
    features.setPNext(&vk12_features);
    vk12_features.setPNext(&vk13_features);
    vk13_features.setPNext(&dynamic_state);
    dynamic_state.setPNext(&dynamic_state2);

    return true;
}

bool VKInstance::FindExtensions()
{
    auto available = physical_device.enumerateDeviceExtensionProperties();
    if (available.empty()) {
        LOG_CRITICAL(Render_Vulkan, "No extensions supported by device.");
        return false;
    }

    // List available device extensions
    for (const auto& prop : available) {
        LOG_INFO(Render_Vulkan, "Vulkan extension: {}", prop.extensionName);
    }

    // Helper lambda for adding extensions
    auto AddExtension = [&](const char* name, bool required) {
        auto result = std::find_if(available.begin(), available.end(), [&](const auto& prop) {
            return !std::strcmp(name, prop.extensionName);
        });

        if (result != available.end()) {
            LOG_INFO(Render_Vulkan, "Enabling extension: {}", name);
            extensions.push_back(name);
            return true;
        }

        if (required) {
            LOG_ERROR(Render_Vulkan, "Unable to find required extension {}.", name);
        }

        return false;
    };

    // Add required extensions
    if (!AddExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, true) ||
        !AddExtension(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, true) ||
        !AddExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, true) ||
        !AddExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, true)) {
        return false;
    }

    return true;
}

} // namespace Vulkan
