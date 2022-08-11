// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include "common/vector_math.h"
#include "video_core/regs_lighting.h"
#include "video_core/regs_shader.h"
#include "video_core/shader/shader.h"

namespace VideoCore {

enum class UniformBindings : u32 {
    Common = 0,
    VertexShader = 1,
    GeometryShader = 2
};

struct LightSrc {
    alignas(16) Common::Vec3f specular_0;
    alignas(16) Common::Vec3f specular_1;
    alignas(16) Common::Vec3f diffuse;
    alignas(16) Common::Vec3f ambient;
    alignas(16) Common::Vec3f position;
    alignas(16) Common::Vec3f spot_direction; // negated
    float dist_atten_bias;
    float dist_atten_scale;
};

/**
 * Uniform structure for the Uniform Buffer Object, all vectors must be 16-byte aligned
 * NOTE: Always keep a vec4 at the end. The GL spec is not clear wether the alignment at
 *       the end of a uniform block is included in UNIFORM_BLOCK_DATA_SIZE or not.
 *       Not following that rule will cause problems on some AMD drivers.
 */
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
    alignas(16) Common::Vec4i lighting_lut_offset[Pica::LightingRegs::NumLightingSampler / 4];
    alignas(16) Common::Vec3f fog_color;
    alignas(8) Common::Vec2f proctex_noise_f;
    alignas(8) Common::Vec2f proctex_noise_a;
    alignas(8) Common::Vec2f proctex_noise_p;
    alignas(16) Common::Vec3f lighting_global_ambient;
    LightSrc light_src[8];
    alignas(16) Common::Vec4f const_color[6]; // A vec4 color for each of the six tev stages
    alignas(16) Common::Vec4f tev_combiner_buffer_color;
    alignas(16) Common::Vec4f clip_coef;
};

static_assert(sizeof(UniformData) == 0x4F0,
              "The size of the UniformData structure has changed, update the structure in the shader");

/**
 * Uniform struct for the Uniform Buffer Object that contains PICA vertex/geometry shader uniforms.
 * NOTE: the same rule from UniformData also applies here.
 */
struct PicaUniformsData {
    void SetFromRegs(const Pica::ShaderRegs& regs, const Pica::Shader::ShaderSetup& setup);

    struct BoolAligned {
        alignas(16) int b;
    };

    std::array<BoolAligned, 16> bools;
    alignas(16) std::array<Common::Vec4u, 4> i;
    alignas(16) std::array<Common::Vec4f, 96> f;
};

struct VSUniformData {
    PicaUniformsData uniforms;
};

static_assert(sizeof(VSUniformData) == 1856,
              "The size of the VSUniformData structure has changed, update the structure in the shader");


inline Common::Vec4f ColorRGBA8(const u32 color) {
    return Common::Vec4<u8>{(color >> 0 & 0xFF), (color >> 8 & 0xFF),
                         (color >> 16 & 0xFF), (color >> 24 & 0xFF)} / 255.0f;
}

inline Common::Vec3f LightColor(const Pica::LightingRegs::LightColor& color) {
    return Common::Vec3<u32>{color.r.Value(), color.g.Value(), color.b.Value()} / 255.0f;
}


} // namespace VideoCore
