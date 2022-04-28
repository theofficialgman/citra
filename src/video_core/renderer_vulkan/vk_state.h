// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <vulkan/vulkan.hpp>

namespace Vulkan {

namespace TextureUnits {

struct TextureUnit {
    GLint id;
    constexpr GLenum Enum() const {
        return static_cast<GLenum>(GL_TEXTURE0 + id);
    }
};

constexpr TextureUnit PicaTexture(int unit) {
    return TextureUnit{unit};
}

constexpr TextureUnit TextureCube{6};
constexpr TextureUnit TextureBufferLUT_LF{3};
constexpr TextureUnit TextureBufferLUT_RG{4};
constexpr TextureUnit TextureBufferLUT_RGBA{5};

} // namespace TextureUnits

namespace ImageUnits {
constexpr uint ShadowBuffer = 0;
constexpr uint ShadowTexturePX = 1;
constexpr uint ShadowTextureNX = 2;
constexpr uint ShadowTexturePY = 3;
constexpr uint ShadowTextureNY = 4;
constexpr uint ShadowTexturePZ = 5;
constexpr uint ShadowTextureNZ = 6;
} // namespace ImageUnits

class VulkanState {
public:
    struct Messenger {
        bool cull_state;
        bool depth_state;
        bool color_mask;
        bool stencil_state;
        bool logic_op;
        bool texture_state;
    };

    struct {
        bool enabled;
        vk::CullModeFlags mode;
        vk::FrontFace front_face;
    } cull;

    struct {
        bool test_enabled;
        vk::CompareOp test_func;
        bool write_mask;
    } depth;

    vk::ColorComponentFlags color_mask;

    struct {
        bool test_enabled;
        vk::CompareOp test_func;
        int test_ref;
        uint32_t test_mask, write_mask;
        vk::StencilOp action_stencil_fail;
        vk::StencilOp action_depth_fail;
        vk::StencilOp action_depth_pass;
    } stencil;

    vk::LogicOp logic_op;

    // 3 texture units - one for each that is used in PICA fragment shader emulation
    struct TextureUnit {
        uint texture_2d; // GL_TEXTURE_BINDING_2D
        uint sampler;    // GL_SAMPLER_BINDING
    };
    std::array<TextureUnit, 3> texture_units;

    struct {
        uint texture_cube; // GL_TEXTURE_BINDING_CUBE_MAP
        uint sampler;      // GL_SAMPLER_BINDING
    } texture_cube_unit;

    struct {
        uint texture_buffer; // GL_TEXTURE_BINDING_BUFFER
    } texture_buffer_lut_lf;

    struct {
        uint texture_buffer; // GL_TEXTURE_BINDING_BUFFER
    } texture_buffer_lut_rg;

    struct {
        uint texture_buffer; // GL_TEXTURE_BINDING_BUFFER
    } texture_buffer_lut_rgba;

    // GL_IMAGE_BINDING_NAME
    uint image_shadow_buffer;
    uint image_shadow_texture_px;
    uint image_shadow_texture_nx;
    uint image_shadow_texture_py;
    uint image_shadow_texture_ny;
    uint image_shadow_texture_pz;
    uint image_shadow_texture_nz;

    struct {
        uint read_framebuffer; // GL_READ_FRAMEBUFFER_BINDING
        uint draw_framebuffer; // GL_DRAW_FRAMEBUFFER_BINDING
        uint vertex_array;     // GL_VERTEX_ARRAY_BINDING
        uint vertex_buffer;    // GL_ARRAY_BUFFER_BINDING
        uint uniform_buffer;   // GL_UNIFORM_BUFFER_BINDING
        uint shader_program;   // GL_CURRENT_PROGRAM
        uint program_pipeline; // GL_PROGRAM_PIPELINE_BINDING
    } draw;

    struct {
        bool enabled; // GL_SCISSOR_TEST
        int x, y;
        std::size_t width, height;
    } scissor;

    struct {
        int x, y;
        std::size_t width, height;
    } viewport;

    std::array<bool, 2> clip_distance;

    VulkanState();

    /// Get the currently active OpenGL state
    static VulkanState GetCurState() {
        return cur_state;
    }

    /// Apply all dynamic state to the provided Vulkan command buffer
    void Apply(vk::CommandBuffer& command_buffer) const;

private:
    static VulkanState cur_state;
};

} // namespace OpenGL
