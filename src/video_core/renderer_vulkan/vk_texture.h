// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <span>
#include "video_core/renderer_vulkan/vk_buffer.h"

namespace Vulkan {

struct SamplerInfo {
    std::array<vk::SamplerAddressMode, 3> wrapping = { vk::SamplerAddressMode::eClampToEdge };
    vk::Filter min_filter = vk::Filter::eLinear;
    vk::Filter mag_filter = vk::Filter::eLinear;
    vk::SamplerMipmapMode mipmap_mode = vk::SamplerMipmapMode::eLinear;
};

/// Vulkan texture object
class VKTexture final : public NonCopyable {
public:
    /// Information for the creation of the target texture
    struct Info {
        u32 width, height;
        vk::Format format;
        vk::ImageType type;
        vk::ImageViewType view_type;
        u32 mipmap_levels = 1;
        u32 array_layers = 1;
        SamplerInfo sampler_info = {};
    };

    VKTexture() = default;
    VKTexture(VKTexture&&) = default;
    ~VKTexture() = default;

    /// Create a new Vulkan texture object along with its sampler
    void Create(const Info& info);

    /// Copies CPU side pixel data to the GPU texture buffer
    void CopyPixels(std::span<u32> pixels);

private:
    /// Used to transition the image to an optimal layout during transfers
    void TransitionLayout(vk::ImageLayout old_layout, vk::ImageLayout new_layout);

private:
    // Texture buffer
    void* pixels = nullptr;
    uint32_t width = 0, height = 0, channels = 0;
    VKBuffer staging;

    // Texture objects
    vk::UniqueImage texture;
    vk::UniqueImageView texture_view;
    vk::UniqueDeviceMemory texture_memory;
    vk::UniqueSampler texture_sampler;
    vk::Format format;
};

/// Vulkan framebuffer object similar to an FBO in OpenGL
class VKFramebuffer final : public NonCopyable {
public:
    VKFramebuffer() = default;
    ~VKFramebuffer() = default;

    // Create Vulkan framebuffer object
    void Create(u32 width, u32 height, u32 layers, u32 samples);

    VkRect2D GetRect() const { return VkRect2D{{0, 0}, {width, height}}; }

private:
    u32 width, height;
    vk::UniqueFramebuffer framebuffer;
    vk::RenderPass load_renderpass;
    vk::RenderPass discard_renderpass;
    vk::RenderPass clear_renderpass;
};

}
