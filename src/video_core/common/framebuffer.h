// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math.h"
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

    // Hashes the framebuffer object and returns a unique identifier
    const u64 Hash() const {
        // IntrusivePtr only has a pointer member so it's fine hash it
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

    // Disable copy constructor
    FramebufferBase(const FramebufferBase&) = delete;
    FramebufferBase& operator=(const FramebufferBase&) = delete;

    // Clears the attachments bound to the framebuffer
    virtual void DoClear(Common::Vec4f color, float depth, u8 stencil) = 0;

    // Returns an immutable reference to the color attachment
    TextureHandle GetColorAttachment() const {
        return info.color;
    }

    // Returns an immutable reference to the depth/stencil attachment
    TextureHandle GetDepthStencilAttachment() const {
        return info.depth_stencil;
    }

    // Sets the area of the framebuffer affected by draw operations
    void SetDrawRect(Rect2D rect) {
        draw_rect = rect;
    }

    // Returns how many samples the framebuffer takes
    MSAASamples GetMSAASamples() const {
        return info.samples;
    }

protected:
    Rect2D draw_rect;
    FramebufferInfo info;
};

using FramebufferHandle = IntrusivePtr<FramebufferBase>;

} // namespace VideoCore
