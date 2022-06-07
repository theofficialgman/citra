// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/scope_exit.h"
#include "video_core/renderer_vulkan/vk_format_reinterpreter.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"
#include "video_core/renderer_vulkan/vk_state.h"

namespace Vulkan {

using PixelFormat = SurfaceParams::PixelFormat;

FormatReinterpreterVulkan::FormatReinterpreterVulkan() {
}

std::pair<FormatReinterpreterVulkan::ReinterpreterMap::iterator,
          FormatReinterpreterVulkan::ReinterpreterMap::iterator>
FormatReinterpreterVulkan::GetPossibleReinterpretations(PixelFormat dst_format) {
    return reinterpreters.equal_range(dst_format);
}

} // namespace Vulkan
