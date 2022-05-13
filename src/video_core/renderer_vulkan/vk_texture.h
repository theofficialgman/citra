// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <span>
#include <functional>
#include <glm/glm.hpp>
#include "common/math_util.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_surface_params.h"

namespace Vulkan {

struct SamplerInfo {
    std::array<vk::SamplerAddressMode, 3> wrapping{};
    vk::Filter min_filter{}, mag_filter{};
    vk::SamplerMipmapMode mipmap_mode{};
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
        vk::ImageUsageFlags usage;
        vk::ImageAspectFlags aspect;
        u32 multisamples = 1;
        u32 levels = 1, layers = 1;
        SamplerInfo sampler_info = {};
    };

    VKTexture() = default;
    ~VKTexture();

    /// Create a new Vulkan texture object
    void Create(const VKTexture::Info& info);

    /// Query objects
    bool IsValid() const { return texture; }
    vk::Image GetHandle() const { return texture; }
    vk::ImageView GetView() const { return view; }
    vk::Format GetFormat() const { return info.format; }
    vk::ImageLayout GetLayout() const { return layout; }
    u32 GetSamples() const { return info.multisamples; }

    /// Copies CPU side pixel data to the GPU texture buffer
    void Upload(u32 level, u32 layer, u32 row_length, vk::Rect2D region, std::span<u8> pixels);

    /// Used to transition the image to an optimal layout during transfers
    void Transition(vk::ImageLayout new_layout);

private:
    VKTexture::Info info{};
    vk::ImageLayout layout{};
    vk::Image texture;
    vk::ImageView view;
    vk::DeviceMemory memory;
    u32 channels{}, image_size{};
};

} // namespace Vulkan
