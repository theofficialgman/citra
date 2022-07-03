// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <type_traits>
#include <glm/glm.hpp>
#include "common/hash.h"
#include "video_core/regs.h"
#include "video_core/shader/shader.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/* Render vertex attributes */
struct VertexBase {
    VertexBase() = default;
    VertexBase(const Pica::Shader::OutputVertex& v, bool flip_quaternion) {
        position[0] = v.pos.x.ToFloat32();
        position[1] = v.pos.y.ToFloat32();
        position[2] = v.pos.z.ToFloat32();
        position[3] = v.pos.w.ToFloat32();
        color[0] = v.color.x.ToFloat32();
        color[1] = v.color.y.ToFloat32();
        color[2] = v.color.z.ToFloat32();
        color[3] = v.color.w.ToFloat32();
        tex_coord0[0] = v.tc0.x.ToFloat32();
        tex_coord0[1] = v.tc0.y.ToFloat32();
        tex_coord1[0] = v.tc1.x.ToFloat32();
        tex_coord1[1] = v.tc1.y.ToFloat32();
        tex_coord2[0] = v.tc2.x.ToFloat32();
        tex_coord2[1] = v.tc2.y.ToFloat32();
        tex_coord0_w = v.tc0_w.ToFloat32();
        normquat[0] = v.quat.x.ToFloat32();
        normquat[1] = v.quat.y.ToFloat32();
        normquat[2] = v.quat.z.ToFloat32();
        normquat[3] = v.quat.w.ToFloat32();
        view[0] = v.view.x.ToFloat32();
        view[1] = v.view.y.ToFloat32();
        view[2] = v.view.z.ToFloat32();

        if (flip_quaternion) {
            normquat = -normquat;
        }
    }

    glm::vec4 position;
    glm::vec4 color;
    glm::vec2 tex_coord0;
    glm::vec2 tex_coord1;
    glm::vec2 tex_coord2;
    float tex_coord0_w;
    glm::vec4 normquat;
    glm::vec3 view;
};

/// Structure that the hardware rendered vertices are composed of
struct HardwareVertex : public VertexBase {
    HardwareVertex() = default;
    HardwareVertex(const Pica::Shader::OutputVertex& v, bool flip_quaternion) : VertexBase(v, flip_quaternion) {};
    static constexpr auto binding_desc = vk::VertexInputBindingDescription(0, sizeof(VertexBase));
    static constexpr std::array<vk::VertexInputAttributeDescription, 8> attribute_desc =
    {
          vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VertexBase, position)),
          vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VertexBase, color)),
          vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(VertexBase, tex_coord0)),
          vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32Sfloat, offsetof(VertexBase, tex_coord1)),
          vk::VertexInputAttributeDescription(4, 0, vk::Format::eR32G32Sfloat, offsetof(VertexBase, tex_coord2)),
          vk::VertexInputAttributeDescription(5, 0, vk::Format::eR32Sfloat, offsetof(VertexBase, tex_coord0_w)),
          vk::VertexInputAttributeDescription(6, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VertexBase, normquat)),
          vk::VertexInputAttributeDescription(7, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexBase, view)),
    };
};

/**
 * Vertex structure that the drawn screen rectangles are composed of.
 */

struct ScreenRectVertexBase {
    ScreenRectVertexBase() = default;
    ScreenRectVertexBase(float x, float y, float u, float v, float s) {
        position.x = x;
        position.y = y;
        tex_coord.x = u;
        tex_coord.y = v;
        tex_coord.z = s;
    }

    glm::vec2 position;
    glm::vec3 tex_coord;
};

struct ScreenRectVertex : public ScreenRectVertexBase {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v, float s) : ScreenRectVertexBase(x, y, u, v, s) {};
    static constexpr auto binding_desc = vk::VertexInputBindingDescription(0, sizeof(ScreenRectVertexBase));
    static constexpr std::array<vk::VertexInputAttributeDescription, 2> attribute_desc =
    {
          vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(ScreenRectVertexBase, position)),
          vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(ScreenRectVertexBase, tex_coord)),
    };
};

enum class ProgramType : u32 { VS, GS, FS };

enum Attributes {
    ATTRIBUTE_POSITION,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_TEXCOORD0,
    ATTRIBUTE_TEXCOORD1,
    ATTRIBUTE_TEXCOORD2,
    ATTRIBUTE_TEXCOORD0_W,
    ATTRIBUTE_NORMQUAT,
    ATTRIBUTE_VIEW,
};

// Doesn't include const_color because we don't sync it, see comment in BuildFromRegs()
struct TevStageConfigRaw {
    u32 sources_raw;
    u32 modifiers_raw;
    u32 ops_raw;
    u32 scales_raw;
    explicit operator Pica::TexturingRegs::TevStageConfig() const noexcept {
        Pica::TexturingRegs::TevStageConfig stage;
        stage.sources_raw = sources_raw;
        stage.modifiers_raw = modifiers_raw;
        stage.ops_raw = ops_raw;
        stage.const_color = 0;
        stage.scales_raw = scales_raw;
        return stage;
    }
};

struct PicaFSConfigState {
    Pica::FramebufferRegs::CompareFunc alpha_test_func;
    Pica::RasterizerRegs::ScissorMode scissor_test_mode;
    Pica::TexturingRegs::TextureConfig::TextureType texture0_type;
    bool texture2_use_coord1;
    std::array<TevStageConfigRaw, 6> tev_stages;
    u8 combiner_buffer_input;

    Pica::RasterizerRegs::DepthBuffering depthmap_enable;
    Pica::TexturingRegs::FogMode fog_mode;
    bool fog_flip;
    bool alphablend_enable;
    Pica::FramebufferRegs::LogicOp logic_op;

    struct {
        struct {
            unsigned num;
            bool directional;
            bool two_sided_diffuse;
            bool dist_atten_enable;
            bool spot_atten_enable;
            bool geometric_factor_0;
            bool geometric_factor_1;
            bool shadow_enable;
        } light[8];

        bool enable;
        unsigned src_num;
        Pica::LightingRegs::LightingBumpMode bump_mode;
        unsigned bump_selector;
        bool bump_renorm;
        bool clamp_highlights;

        Pica::LightingRegs::LightingConfig config;
        bool enable_primary_alpha;
        bool enable_secondary_alpha;

        bool enable_shadow;
        bool shadow_primary;
        bool shadow_secondary;
        bool shadow_invert;
        bool shadow_alpha;
        unsigned shadow_selector;

        struct {
            bool enable;
            bool abs_input;
            Pica::LightingRegs::LightingLutInput type;
            float scale;
        } lut_d0, lut_d1, lut_sp, lut_fr, lut_rr, lut_rg, lut_rb;
    } lighting;

    struct {
        bool enable;
        u32 coord;
        Pica::TexturingRegs::ProcTexClamp u_clamp, v_clamp;
        Pica::TexturingRegs::ProcTexCombiner color_combiner, alpha_combiner;
        bool separate_alpha;
        bool noise_enable;
        Pica::TexturingRegs::ProcTexShift u_shift, v_shift;
        u32 lut_width;
        u32 lut_offset0;
        u32 lut_offset1;
        u32 lut_offset2;
        u32 lut_offset3;
        u32 lod_min;
        u32 lod_max;
        Pica::TexturingRegs::ProcTexFilter lut_filter;
    } proctex;

    bool shadow_rendering;
    bool shadow_texture_orthographic;
};

/**
 * This struct contains all state used to generate the GLSL fragment shader that emulates the
 * current Pica register configuration. This struct is used as a cache key for generated GLSL shader
 * programs. The functions in gl_shader_gen.cpp should retrieve state from this struct only, not by
 * directly accessing Pica registers. This should reduce the risk of bugs in shader generation where
 * Pica state is not being captured in the shader cache key, thereby resulting in (what should be)
 * two separate shaders sharing the same key.
 */
struct PicaFSConfig : Common::HashableStruct<PicaFSConfigState> {

    /// Construct a PicaFSConfig with the given Pica register configuration.
    static PicaFSConfig BuildFromRegs(const Pica::Regs& regs);

    bool TevStageUpdatesCombinerBufferColor(unsigned stage_index) const {
        return (stage_index < 4) && (state.combiner_buffer_input & (1 << stage_index));
    }

    bool TevStageUpdatesCombinerBufferAlpha(unsigned stage_index) const {
        return (stage_index < 4) && ((state.combiner_buffer_input >> 4) & (1 << stage_index));
    }
};

/**
 * This struct contains common information to identify a GL vertex/geometry shader generated from
 * PICA vertex/geometry shader.
 */
struct PicaShaderConfigCommon {
    void Init(const Pica::ShaderRegs& regs, Pica::Shader::ShaderSetup& setup);

    u64 program_hash;
    u64 swizzle_hash;
    u32 main_offset;
    bool sanitize_mul;

    u32 num_outputs;

    // output_map[output register index] -> output attribute index
    std::array<u32, 16> output_map;
};

/**
 * This struct contains information to identify a GL vertex shader generated from PICA vertex
 * shader.
 */
struct PicaVSConfig : Common::HashableStruct<PicaShaderConfigCommon> {
    explicit PicaVSConfig(const Pica::ShaderRegs& regs, Pica::Shader::ShaderSetup& setup) {
        state.Init(regs, setup);
    }
    explicit PicaVSConfig(const PicaShaderConfigCommon& conf) {
        state = conf;
    }
};

struct PicaGSConfigCommonRaw {
    void Init(const Pica::Regs& regs);

    u32 vs_output_attributes;
    u32 gs_output_attributes;

    struct SemanticMap {
        u32 attribute_index;
        u32 component_index;
    };

    // semantic_maps[semantic name] -> GS output attribute index + component index
    std::array<SemanticMap, 24> semantic_maps;
};

/**
 * This struct contains information to identify a GL geometry shader generated from PICA no-geometry
 * shader pipeline
 */
struct PicaFixedGSConfig : Common::HashableStruct<PicaGSConfigCommonRaw> {
    explicit PicaFixedGSConfig(const Pica::Regs& regs) {
        state.Init(regs);
    }
};

struct PipelineCacheKey {
    vk::Format color, depth_stencil;
    vk::PipelineColorBlendAttachmentState blend_config;
    vk::LogicOp blend_logic_op;
    PicaFSConfig fragment_config;

    auto operator <=>(const PipelineCacheKey& other) const = default;

    u64 Hash() const {
        const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof(PipelineCacheKey));
        return static_cast<size_t>(hash);
    }
};

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::PicaFSConfig> {
    std::size_t operator()(const Vulkan::PicaFSConfig& k) const noexcept {
        return k.Hash();
    }
};

template <>
struct hash<Vulkan::PicaVSConfig> {
    std::size_t operator()(const Vulkan::PicaVSConfig& k) const noexcept {
        return k.Hash();
    }
};

template <>
struct hash<Vulkan::PicaFixedGSConfig> {
    std::size_t operator()(const Vulkan::PicaFixedGSConfig& k) const noexcept {
        return k.Hash();
    }
};

template <>
struct hash<Vulkan::PipelineCacheKey> {
    size_t operator()(const Vulkan::PipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std
