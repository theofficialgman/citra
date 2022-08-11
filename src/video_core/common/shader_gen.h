// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <string>
#include "common/hash.h"
#include "video_core/regs.h"
#include "video_core/shader/shader.h"

namespace VideoCore {

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

    auto operator<=>(const TevStageConfigRaw& other) const = default;

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

/**
 * This struct contains all state used to generate the GLSL fragment shader that emulates the
 * current Pica register configuration. This struct is used as a cache key for generated GLSL shader
 * programs. The functions in gl_shader_gen.cpp should retrieve state from this struct only, not by
 * directly accessing Pica registers. This should reduce the risk of bugs in shader generation where
 * Pica state is not being captured in the shader cache key, thereby resulting in (what should be)
 * two separate shaders sharing the same key.
 */
struct PicaFSConfig {
    explicit PicaFSConfig(const Pica::Regs& regs);

    /// Returns the hash of the VS config
    const u64 Hash() const noexcept {
        return Common::ComputeHash64(this, sizeof(PicaFSConfig));
    }

    bool operator==(const PicaFSConfig& other) const noexcept {
        return std::memcmp(this, &other, sizeof(PicaFSConfig)) == 0;
    };

    bool operator!=(const PicaFSConfig& other) const noexcept {
        return !(*this == other);
    };

    bool TevStageUpdatesCombinerBufferColor(unsigned stage_index) const {
        return (stage_index < 4) && (combiner_buffer_input & (1 << stage_index));
    }

    bool TevStageUpdatesCombinerBufferAlpha(unsigned stage_index) const {
        return (stage_index < 4) && ((combiner_buffer_input >> 4) & (1 << stage_index));
    }

    Pica::CompareFunc alpha_test_func;
    Pica::RasterizerRegs::ScissorMode scissor_test_mode;
    Pica::TexturingRegs::TextureConfig::TextureType texture0_type;
    std::array<TevStageConfigRaw, 6> tev_stages;
    bool texture2_use_coord1;
    u8 combiner_buffer_input;

    Pica::RasterizerRegs::DepthBuffering depthmap_enable;
    Pica::TexturingRegs::FogMode fog_mode;
    bool fog_flip;
    bool alphablend_enable;
    Pica::LogicOp logic_op;

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
 * This struct contains information to identify a host vertex shader generated from PICA vertex
 * shader.
 */
struct PicaVSConfig {
    explicit PicaVSConfig(const Pica::ShaderRegs& regs, Pica::Shader::ShaderSetup& setup);

    /// Returns the hash of the VS config
    const u64 Hash() const noexcept {
        return Common::ComputeHash64(this, sizeof(PicaVSConfig));
    }

    bool operator==(const PicaVSConfig& other) const noexcept {
        return std::memcmp(this, &other, sizeof(PicaVSConfig)) == 0;
    };

    bool operator!=(const PicaVSConfig& other) const noexcept {
        return !(*this == other);
    };

    u64 program_hash = 0;
    u64 swizzle_hash = 0;
    u32 main_offset = 0;
    bool sanitize_mul = false;

    // output_map[output register index] -> output attribute index
    u32 num_outputs = 0;
    std::array<u32, 16> output_map{};
};

/**
 * This struct contains information to identify a GL geometry shader generated from PICA no-geometry
 * shader pipeline
 */
struct PicaFixedGSConfig {
    explicit PicaFixedGSConfig(const Pica::Regs& regs);

    /// Returns the hash of the GS config
    const u64 Hash() const noexcept {
        return Common::ComputeHash64(this, sizeof(PicaFixedGSConfig));
    }

    bool operator==(const PicaFixedGSConfig& other) const noexcept {
        return std::memcmp(this, &other, sizeof(PicaFixedGSConfig)) == 0;
    };

    bool operator!=(const PicaFixedGSConfig& other) const noexcept {
        return !(*this == other);
    };

    u32 vs_output_attributes = 0;
    u32 gs_output_attributes = 0;

    struct SemanticMap {
        u32 attribute_index = 0;
        u32 component_index = 0;
    };

    // semantic_maps[semantic name] -> GS output attribute index + component index
    std::array<SemanticMap, 24> semantic_maps{};
};

/**
 * Generates backend specific shader modules using the Pica state configuration
 * @todo Be replaced with a unified shader compiler
 */
class ShaderGeneratorBase {
public:
    ShaderGeneratorBase() = default;
    virtual ~ShaderGeneratorBase() = default;

    /**
     * Generates the GLSL vertex shader program source code that accepts vertices from software shader
     * and directly passes them to the fragment shader.
     * @param separable_shader generates shader that can be used for separate shader object
     * @returns String of the shader source code
     */
    virtual std::string GenerateTrivialVertexShader() = 0;

    /**
     * Generates the GLSL vertex shader program source code for the given VS program
     * @returns String of the shader source code
     */
    virtual std::string GenerateVertexShader(const Pica::Shader::ShaderSetup& setup,
                                             const PicaVSConfig& config) = 0;

    /**
     * Generates the GLSL fixed geometry shader program source code for non-GS PICA pipeline
     * @returns String of the shader source code
     */
    virtual std::string GenerateFixedGeometryShader(const PicaFixedGSConfig& config) = 0;

    /**
     * Generates the GLSL fragment shader program source code for the current Pica state
     * @param config ShaderCacheKey object generated for the current Pica state, used for the shader
     *               configuration (NOTE: Use state in this struct only, not the Pica registers!)
     * @param separable_shader generates shader that can be used for separate shader object
     * @returns String of the shader source code
     */
    virtual std::string GenerateFragmentShader(const PicaFSConfig& config) = 0;
};

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::PicaFSConfig> {
    std::size_t operator()(const VideoCore::PicaFSConfig& k) const noexcept {
        return k.Hash();
    }
};

template <>
struct hash<VideoCore::PicaVSConfig> {
    std::size_t operator()(const VideoCore::PicaVSConfig& k) const noexcept {
        return k.Hash();
    }
};

template <>
struct hash<VideoCore::PicaFixedGSConfig> {
    std::size_t operator()(const VideoCore::PicaFixedGSConfig& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std
