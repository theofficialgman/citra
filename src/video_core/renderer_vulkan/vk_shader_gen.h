// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include "common/hash.h"
#include "video_core/regs.h"
#include "video_core/shader/shader.h"
#include "video_core/renderer_vulkan/vk_shader_state.h"

namespace Vulkan {

namespace ShaderDecompiler {
struct ProgramResult {
    std::string code;
};
}

/**
 * Generates the GLSL vertex shader program source code that accepts vertices from software shader
 * and directly passes them to the fragment shader.
 * @param separable_shader generates shader that can be used for separate shader object
 * @returns String of the shader source code
 */
ShaderDecompiler::ProgramResult GenerateTrivialVertexShader(bool separable_shader);

/**
 * Generates the GLSL fragment shader program source code for the current Pica state
 * @param config ShaderCacheKey object generated for the current Pica state, used for the shader
 *               configuration (NOTE: Use state in this struct only, not the Pica registers!)
 * @param separable_shader generates shader that can be used for separate shader object
 * @returns String of the shader source code
 */
ShaderDecompiler::ProgramResult GenerateFragmentShader(const PicaFSConfig& config);

} // namespace Vulkan
