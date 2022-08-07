// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/common/texture.h"

namespace VideoCore {

enum class MSAASamples : u32 {
    x1,
    x2,
    x4,
    x8
};

/**
 * Information about a framebuffer
 */
struct FramebufferInfo {
    TextureHandle color;
    TextureHandle depth_stencil;
    MSAASamples samples = MSAASamples::x1;
    Rect2D draw_rect{};

    /// Hashes the framebuffer object and returns a unique identifier
    const u64 Hash() const {
        // The only member IntrusivePtr has is a pointer to the
        // handle so it's fine hash it
        return Common::ComputeStructHash64(*this);
    }
};

/**
 * A framebuffer is a collection of render targets and their configuration
 */
class FramebufferBase : public IntrusivePtrEnabled<FramebufferBase> {
public:
    FramebufferBase(const FramebufferInfo& info) : info(info) {}
    virtual ~FramebufferBase() = default;

    /// Returns an immutable reference to the color attachment
    const TextureHandle& GetColorAttachment() const {
        return info.color;
    }

    /// Returns an immutable reference to the depth/stencil attachment
    const TextureHandle& GetDepthStencilAttachment() const {
        return info.depth_stencil;
    }

    /// Returns how many samples the framebuffer takes
    MSAASamples GetMSAASamples() const {
        return info.samples;
    }

    /// Returns the rendering area
    Rect2D GetDrawRectangle() const {
        return info.draw_rect;
    }

protected:
    FramebufferInfo info;
};

using FramebufferHandle = IntrusivePtr<FramebufferBase>;

} // namespace VideoCore
