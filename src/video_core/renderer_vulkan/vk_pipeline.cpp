// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include <shaderc/shaderc.hpp>

namespace Vulkan {

shaderc::Compiler compiler;

void VKPipeline::Info::AddShaderModule(const vk::ShaderModule& source, vk::ShaderStageFlagBits stage)
{
    shaderc_shader_kind shader_stage;
    std::string name;
    switch (stage)
    {
    case vk::ShaderStageFlagBits::eVertex:
        shader_stage = shaderc_glsl_vertex_shader;
        name = "Vertex shader";
        break;
    case vk::ShaderStageFlagBits::eCompute:
        shader_stage = shaderc_glsl_compute_shader;
        name = "Compute shader";
        break;
    case vk::ShaderStageFlagBits::eFragment:
        shader_stage = shaderc_glsl_fragment_shader;
        name = "Fragment shader";
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown vulkan shader stage {}", stage);
        UNREACHABLE();
    }

    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetAutoBindUniforms(true);
    options.SetAutoMapLocations(true);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    auto result = compiler.CompileGlslToSpv(source.c_str(), shader_stage, name.c_str(), options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        LOG_CRITICAL(Render_Vulkan, "Failed to compile GLSL shader with error: {}", result.GetErrorMessage());
        UNREACHABLE();
    }

    auto shader_code = std::vector<uint32_t>{ result.cbegin(), result.cend() };

    vk::ShaderModuleCreateInfo module_info({}, shader_code);
    auto module = g_vk_instace->GetDevice().createShaderModuleUnique(module_info);
    shader_stages.emplace_back(vk::PipelineShaderStageCreateFlags(), stage, module.get(), "main");

    return std::move(module);
}


} // namespace Vulkan
