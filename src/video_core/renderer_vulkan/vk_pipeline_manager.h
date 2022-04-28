// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <variant>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "video_core/rasterizer_interface.h"
#include "video_core/regs_lighting.h"
#include "video_core/renderer_vulkan/pica_to_vulkan.h"
#include "video_core/renderer_vulkan/vk_shader_state.h"
#include "video_core/renderer_vulkan/vk_texture.h"

namespace Core {
class System;
}

namespace Vulkan {

enum class UniformBindings : GLuint { Common, VS, GS };

struct LightSrc {
    alignas(16) glm::vec3 specular_0;
    alignas(16) glm::vec3 specular_1;
    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 ambient;
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 spot_direction; // negated
    float dist_atten_bias;
    float dist_atten_scale;
};

/// Uniform structure for the Uniform Buffer Object, all vectors must be 16-byte aligned
// NOTE: Always keep a vec4 at the end. The GL spec is not clear wether the alignment at
//       the end of a uniform block is included in UNIFORM_BLOCK_DATA_SIZE or not.
//       Not following that rule will cause problems on some AMD drivers.
struct UniformData {
    int framebuffer_scale;
    int alphatest_ref;
    float depth_scale;
    float depth_offset;
    float shadow_bias_constant;
    float shadow_bias_linear;
    int scissor_x1;
    int scissor_y1;
    int scissor_x2;
    int scissor_y2;
    int fog_lut_offset;
    int proctex_noise_lut_offset;
    int proctex_color_map_offset;
    int proctex_alpha_map_offset;
    int proctex_lut_offset;
    int proctex_diff_lut_offset;
    float proctex_bias;
    int shadow_texture_bias;
    alignas(16) glm::ivec4 lighting_lut_offset[Pica::LightingRegs::NumLightingSampler / 4];
    alignas(16) glm::vec3 fog_color;
    alignas(8) glm::vec2 proctex_noise_f;
    alignas(8) glm::vec2 proctex_noise_a;
    alignas(8) glm::vec2 proctex_noise_p;
    alignas(16) glm::vec3 lighting_global_ambient;
    LightSrc light_src[8];
    alignas(16) glm::vec4 const_color[6]; // A vec4 color for each of the six tev stages
    alignas(16) glm::vec4 tev_combiner_buffer_color;
    alignas(16) glm::vec4 clip_coef;
};

static_assert(sizeof(UniformData) == 0x4F0, "The size of the UniformData structure has changed, update the structure in the shader");
static_assert(sizeof(UniformData) < 16384, "UniformData structure must be less than 16kb as per the OpenGL spec");

/// Uniform struct for the Uniform Buffer Object that contains PICA vertex/geometry shader uniforms.
// NOTE: the same rule from UniformData also applies here.
struct PicaUniformsData {
    void SetFromRegs(const Pica::ShaderRegs& regs, const Pica::Shader::ShaderSetup& setup);

    struct BoolAligned {
        alignas(16) int b;
    };

    std::array<BoolAligned, 16> bools;
    alignas(16) std::array<glm::uvec4, 4> i;
    alignas(16) std::array<glm::vec4, 96> f;
};

struct VSUniformData {
    PicaUniformsData uniforms;
};

static_assert(sizeof(VSUniformData) == 1856, "The size of the VSUniformData structure has changed, update the structure in the shader");
static_assert(sizeof(VSUniformData) < 16384, "VSUniformData structure must be less than 16kb as per the Vulkan spec");

using Resource = std::variant<VKBuffer, VKTexture>;

/// Includes all required information to build a Vulkan pipeline object
class VKPipelineInfo : private NonCopyable {
    VKPipelineInfo() = default;
    ~VKPipelineInfo() = default;

    /// Assign a shader module to a specific stage
    void AddShaderModule(const vk::ShaderModule& module, vk::ShaderStageFlagBits stage);

    /// Add a texture or a buffer to the target descriptor set
    void AddResource(const Resource& resource, vk::DescriptorType type, vk::ShaderStageFlags stages, int set = 0);

private:
    using ResourceInfo = std::pair<std::reference_wrapper<Resource>, vk::DescriptorSetLayoutBinding>;

    std::unordered_map<int, std::vector<ResourceInfo>> descriptor_sets;
    std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;
};

/// A class that manages the storage and management of Vulkan pipeline objects.
class PipelineManager {
public:
    PipelineManager(Frontend::EmuWindow& emu_window);
    ~PipelineManager();

    /// Retrieves the Vulkan pipeline that maps to the current PICA state.
    /// If not present, it is compiled and cached
    vk::Pipeline GetPipeline(const Pica::Regs& config, Pica::Shader::ShaderSetup& setup);

private:
    std::unordered_map<VKPipelineCacheKey, vk::UniquePipeline> pipelines;
    vk::UniquePipelineCache pipeline_cache;

    Frontend::EmuWindow& emu_window;
};
} // namespace Vulkan
