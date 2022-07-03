// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <map>
#include <type_traits>
#include "common/common_types.h"
#include "common/math_util.h"
#include "video_core/renderer_vulkan/vk_surface_params.h"

namespace Vulkan {

class RasterizerCacheVulkan;

struct CachedSurface;
using Surface = std::shared_ptr<CachedSurface>;

struct PixelFormatPair {
    const SurfaceParams::PixelFormat dst_format, src_format;
    struct less {
        using is_transparent = void;
        constexpr bool operator()(PixelFormatPair lhs, PixelFormatPair rhs) const {
            return std::tie(lhs.dst_format, lhs.src_format) <
                   std::tie(rhs.dst_format, rhs.src_format);
        }
        constexpr bool operator()(SurfaceParams::PixelFormat lhs,
                                  PixelFormatPair rhs) const {
            return lhs < rhs.dst_format;
        }
        constexpr bool operator()(PixelFormatPair lhs,
                                  SurfaceParams::PixelFormat rhs) const {
            return lhs.dst_format < rhs;
        }
    };
};

class FormatReinterpreterBase {
public:
    virtual ~FormatReinterpreterBase() = default;

    virtual void Reinterpret(Surface src_surface, const Common::Rectangle<u32>& src_rect,
                             Surface dst_surface, const Common::Rectangle<u32>& dst_rect) = 0;
};

class FormatReinterpreterVulkan : NonCopyable {
    using ReinterpreterMap =
        std::map<PixelFormatPair, std::unique_ptr<FormatReinterpreterBase>, PixelFormatPair::less>;

public:
    explicit FormatReinterpreterVulkan();
    ~FormatReinterpreterVulkan() = default;

    std::pair<ReinterpreterMap::iterator, ReinterpreterMap::iterator> GetPossibleReinterpretations(
        SurfaceParams::PixelFormat dst_format);

private:
    ReinterpreterMap reinterpreters;
};

} // namespace Vulkan
