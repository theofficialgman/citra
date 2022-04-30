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
    std::array<vk::SamplerAddressMode, 3> wrapping = { vk::SamplerAddressMode::eClampToEdge };
    vk::Filter min_filter = vk::Filter::eLinear;
    vk::Filter mag_filter = vk::Filter::eLinear;
    vk::SamplerMipmapMode mipmap_mode = vk::SamplerMipmapMode::eLinear;
};

/// Vulkan texture object
class VKTexture final : public NonCopyable {
    friend class VKFramebuffer;
public:
    /// Information for the creation of the target texture
    struct Info {
        u32 width, height;
        vk::Format format;
        vk::ImageType type;
        vk::ImageViewType view_type;
        u32 mipmap_levels = 1;
        u32 array_layers = 1;
        u32 multisamples = 1;
        SamplerInfo sampler_info = {};
    };

    VKTexture() = default;
    VKTexture(VKTexture&&) = default;
    ~VKTexture() = default;

    /// Create a new Vulkan texture object along with its sampler
    void Create(const Info& info);
    bool IsValid() { return !!texture; }

    /// Copies CPU side pixel data to the GPU texture buffer
    void CopyPixels(std::span<u32> pixels);

    /// Get Vulkan objects
    vk::ImageView& GetView() { return texture_view.get(); }
    vk::Format GetFormat() const { return texture_info.format; }
    vk::Rect2D GetRect() const { return vk::Rect2D({}, { texture_info.width, texture_info.height }); }
    u32 GetSamples() const { return texture_info.multisamples; }

    /// Used to transition the image to an optimal layout during transfers
    void TransitionLayout(vk::ImageLayout new_layout, vk::CommandBuffer& command_buffer);

    /// Fill the texture with the values provided
    void Fill(Common::Rectangle<u32> region, glm::vec4 color);
    void Fill(Common::Rectangle<u32> region, glm::vec2 depth_stencil);

    /// Copy current texture to another with optionally performing format convesions
    void BlitTo(Common::Rectangle<u32> source_rect, VKTexture& dest,
                Common::Rectangle<u32> dst_rect, SurfaceParams::SurfaceType type,
                vk::CommandBuffer& command_buffer);

private:
    Info texture_info;
    vk::ImageLayout texture_layout = vk::ImageLayout::eUndefined;
    vk::UniqueImage texture;
    vk::UniqueImageView texture_view;
    vk::UniqueDeviceMemory texture_memory;
    u32 channels;
};

enum Attachments {
    Color = 0,
    DepthStencil = 1
};

/// Vulkan framebuffer object similar to an FBO in OpenGL
class VKFramebuffer final : public NonCopyable {
public:
    struct Info {
        VKTexture* color;
        VKTexture* depth_stencil;
    };

    VKFramebuffer() = default;
    ~VKFramebuffer() = default;

    /// Create Vulkan framebuffer object
    void Create(const Info& info);

    /// Configure frambuffer for rendering
    void Prepare(vk::CommandBuffer& command_buffer);

    vk::Rect2D GetRect() const { return vk::Rect2D({}, { width, height }); }

private:
    u32 width, height;
    vk::UniqueFramebuffer framebuffer;
    std::array<VKTexture*, 2> attachments;
};

} // namespace Vulkan
