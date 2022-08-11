// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <span>
#include "common/hash.h"
#include "common/intrusive_ptr.h"
#include "common/math_util.h"
#include "video_core/regs_texturing.h"

namespace VideoCore {

constexpr u32 MAX_COLOR_FORMATS = 5;
constexpr u32 MAX_DEPTH_FORMATS = 3;

enum class TextureFormat : u8 {
    RGBA8 = 0,
    RGB8 = 1,
    RGB5A1 = 2,
    RGB565 = 3,
    RGBA4 = 4,
    D16 = 5,
    D24 = 6,
    D24S8 = 7,
    PresentColor = 8, // Backend specific swapchain format
    Undefined = 255
};

enum class TextureType : u8 {
    Texture1D = 0,
    Texture2D = 1,
    Texture3D = 2,
    Undefined = 255
};

enum class TextureViewType : u8 {
    View1D = 0,
    View2D = 1,
    View3D = 2,
    ViewCube = 3,
    View1DArray = 4,
    View2DArray = 5,
    ViewCubeArray = 6,
    Undefined = 255
};

/**
 * A rectangle describing part of a texture
 * @param x, y are the offset from the bottom left corner
 * @param width, height are the extent of the rectangle
 */
struct Rect2D {
    Rect2D() = default;
    Rect2D(s32 x, s32 y, u32 width, u32 height) :
        x(x), y(y), width(width), height(height) {}
    Rect2D(Common::Rectangle<u32> rect) :
        x(rect.left), y(rect.bottom), width(rect.GetWidth()), height(rect.GetHeight()) {}

    s32 x = 0;
    s32 y = 0;
    u32 width = 0;
    u32 height = 0;
};

/**
 * Information about a texture packed to 8 bytes
 */
struct TextureInfo {
    u16 width = 0;
    u16 height = 0;
    u8 levels = 0;
    TextureType type = TextureType::Undefined;
    TextureViewType view_type = TextureViewType::Undefined;
    TextureFormat format = TextureFormat::Undefined;

    auto operator<=>(const TextureInfo& info) const = default;

    void UpdateMipLevels() {
        levels = std::log2(std::max(width, height)) + 1;
    }

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(TextureInfo));
    }
};

static_assert(sizeof(TextureInfo) == 8, "TextureInfo not packed!");
static_assert(std::is_standard_layout_v<TextureInfo>, "TextureInfo is not a standard layout!");

class TextureBase;
using TextureHandle = IntrusivePtr<TextureBase>;

class TextureBase : public IntrusivePtrEnabled<TextureBase> {
public:
    TextureBase() = default;
    TextureBase(const TextureInfo& info) : info(info) {}
    virtual ~TextureBase() = default;

    // Disable copy constructor
    TextureBase(const TextureBase&) = delete;
    TextureBase& operator=(const TextureBase&) = delete;

    // Uploads pixel data to the GPU memory
    virtual void Upload(Rect2D rectangle, u32 stride, std::span<const u8> data,
                        u32 level = 0) {};

    // Downloads pixel data from GPU memory
    virtual void Download(Rect2D rectangle, u32 stride, std::span<u8> data,
                          u32 level = 0) {};

    // Copies the rectangle area specified to the destionation texture
    virtual void BlitTo(TextureHandle dest, Rect2D source_rect, Rect2D dest_rect,
                        u32 src_level = 0, u32 dest_level = 0,
                        u32 src_layer = 0, u32 dest_layer = 0) {};

    // Copies texture data from the source texture
    virtual void CopyFrom(TextureHandle source) {};

    // Generates all possible mipmaps from the texture
    virtual void GenerateMipmaps() {};

    // Returns the texture info structure
    TextureInfo GetInfo() const {
        return info;
    }

    // Returns the unique texture identifier
    const u64 GetHash() const {
        return info.Hash();
    }

    // Returns the width of the texture
    u16 GetWidth() const {
        return info.width;
    }

    // Returns the height of the texture
    u16 GetHeight() const {
        return info.height;
    }

    // Returns the number of mipmap levels allocated
    u16 GetMipLevels() const {
        return info.levels;
    }

    // Returns the pixel format
    TextureFormat GetFormat() const {
        return info.format;
    }

protected:
    TextureInfo info;
};

struct SamplerInfo {
    Pica::TextureFilter mag_filter;
    Pica::TextureFilter min_filter;
    Pica::TextureFilter mip_filter;
    Pica::WrapMode wrap_s;
    Pica::WrapMode wrap_t;
    u32 border_color = 0;
    u32 lod_min = 0;
    u32 lod_max = 0;
    s32 lod_bias = 0;

    auto operator<=>(const SamplerInfo& info) const = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(SamplerInfo));
    }
};

class SamplerBase : public IntrusivePtrEnabled<SamplerBase> {
public:
    SamplerBase(SamplerInfo info) : info(info) {}
    virtual ~SamplerBase() = default;

    // Disable copy constructor
    SamplerBase(const SamplerBase&) = delete;
    SamplerBase& operator=(const SamplerBase&) = delete;

protected:
    SamplerInfo info{};
};

using SamplerHandle = IntrusivePtr<SamplerBase>;

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::TextureInfo> {
    std::size_t operator()(const VideoCore::TextureInfo& info) const noexcept {
        return info.Hash();
    }
};

template <>
struct hash<VideoCore::SamplerInfo> {
    std::size_t operator()(const VideoCore::SamplerInfo& info) const noexcept {
        return info.Hash();
    }
};
} // namespace std
