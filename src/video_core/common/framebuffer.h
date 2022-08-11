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

// States which operation to perform on the framebuffer attachment during rendering
enum class LoadOp : u8 {
    Load = 0,
    Clear = 1
};

/**
 * Information about a framebuffer
 */
struct FramebufferInfo {
    TextureHandle color;
    TextureHandle depth_stencil;
    MSAASamples samples = MSAASamples::x1;

    auto operator<=>(const FramebufferInfo& info) const = default;

    // Hashes the framebuffer object and returns a unique identifier
    const u64 Hash() const {
        // IntrusivePtr only has a pointer member so it's fine hash it
        return Common::ComputeHash64(this, sizeof(FramebufferInfo));
    }
};

// A framebuffer is a collection of render targets and their configuration
class FramebufferBase : public IntrusivePtrEnabled<FramebufferBase> {
public:
    FramebufferBase(const FramebufferInfo& info) : info(info) {}
    virtual ~FramebufferBase() = default;

    // Disable copy constructor
    FramebufferBase(const FramebufferBase&) = delete;
    FramebufferBase& operator=(const FramebufferBase&) = delete;

    // Clears the attachments bound to the framebuffer using the last stored clear value
    virtual void DoClear() = 0;

    // Returns an immutable reference to the color attachment
    TextureHandle GetColorAttachment() const {
        return info.color;
    }

    // Returns an immutable reference to the depth/stencil attachment
    TextureHandle GetDepthStencilAttachment() const {
        return info.depth_stencil;
    }

    // Sets the area of the framebuffer affected by draw operations
    void SetDrawRect(Common::Rectangle<u32> rect) {
        draw_rect = rect;
    }

    LoadOp GetLoadOp() const {
        return load_op;
    }

    void SetLoadOp(LoadOp op) {
        load_op = op;
    }

    void SetClearValues(Common::Vec4f color, float depth, u8 stencil) {
        clear_color_value = color;
        clear_depth_value = depth;
        clear_stencil_value = stencil;
    }

    // Returns the area of the framebuffer affected by draw operations
    Common::Rectangle<u32> GetDrawRect() const {
        return draw_rect;
    }

    // Returns how many samples the framebuffer takes
    MSAASamples GetMSAASamples() const {
        return info.samples;
    }

protected:
    LoadOp load_op = LoadOp::Load;
    Common::Vec4f clear_color_value{};
    float clear_depth_value = 0.f;
    u8 clear_stencil_value = 0;
    Common::Rectangle<u32> draw_rect;
    FramebufferInfo info;
};

using FramebufferHandle = IntrusivePtr<FramebufferBase>;

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::FramebufferInfo> {
    std::size_t operator()(const VideoCore::FramebufferInfo& info) const noexcept {
        return info.Hash();
    }
};
} // namespace std
