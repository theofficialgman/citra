// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include "common/bit_field.h"
#include "common/common_types.h"
#include "common/vector_math.h"
#include "core/hw/gpu.h"
#include "video_core/pica_state.h"
#include "video_core/pica_types.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/regs_framebuffer.h"
#include "video_core/regs_lighting.h"
#include "video_core/regs_rasterizer.h"
#include "video_core/regs_texturing.h"
#include "video_core/shader/shader.h"

namespace Frontend {
class EmuWindow;
}

namespace Vulkan {
class ShaderProgramManager;

class RasterizerVulkan : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerVulkan(Frontend::EmuWindow& emu_window);
    ~RasterizerVulkan() override;

    void LoadDiskResources(const std::atomic_bool& stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void AddTriangle(const Pica::Shader::OutputVertex& v0, const Pica::Shader::OutputVertex& v1,
                     const Pica::Shader::OutputVertex& v2) override;
    void DrawTriangles() override;
    void NotifyPicaRegisterChanged(u32 id) override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const GPU::Regs::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, OpenGL::ScreenInfo& screen_info) override;
    bool AccelerateDrawBatch(bool is_indexed) override;

    /// Syncs entire status to match PICA registers
    void SyncEntireState() override;

private:
    struct SamplerInfo {
        using TextureConfig = Pica::TexturingRegs::TextureConfig;

        OGLSampler sampler;

        /// Creates the sampler object, initializing its state so that it's in sync with the
        /// SamplerInfo struct.
        void Create();
        /// Syncs the sampler object with the config, updating any necessary state.
        void SyncWithConfig(const TextureConfig& config);

    private:
        TextureConfig::TextureFilter mag_filter;
        TextureConfig::TextureFilter min_filter;
        TextureConfig::TextureFilter mip_filter;
        TextureConfig::WrapMode wrap_s;
        TextureConfig::WrapMode wrap_t;
        u32 border_color;
        u32 lod_min;
        u32 lod_max;
        s32 lod_bias;

        // TODO(wwylele): remove this once mipmap for cube is implemented
        bool supress_mipmap_for_cube = false;
    };

    struct VertexInfo
    {
        VertexInfo() = default;
        VertexInfo(const Pica::Shader::OutputVertex& v, bool flip_quaternion) {
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
    struct HardwareVertex : public VertexInfo
    {
        HardwareVertex() = default;
        HardwareVertex(const Pica::Shader::OutputVertex& v, bool flip_quaternion) : VertexInfo(v, flip_quaternion) {};
        static constexpr auto binding_desc = vk::VertexInputBindingDescription(0, sizeof(VertexInfo));
        static constexpr std::array<vk::VertexInputAttributeDescription, 8> attribute_desc =
        {
              vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VertexInfo, position)),
              vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VertexInfo, color)),
              vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(VertexInfo, tex_coord0)),
              vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32Sfloat, offsetof(VertexInfo, tex_coord1)),
              vk::VertexInputAttributeDescription(4, 0, vk::Format::eR32G32Sfloat, offsetof(VertexInfo, tex_coord2)),
              vk::VertexInputAttributeDescription(5, 0, vk::Format::eR32Sfloat, offsetof(VertexInfo, tex_coord0_w)),
              vk::VertexInputAttributeDescription(6, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(VertexInfo, normquat)),
              vk::VertexInputAttributeDescription(7, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexInfo, view)),
        };
    };


    /// Syncs the clip enabled status to match the PICA register
    void SyncClipEnabled();

    /// Syncs the clip coefficients to match the PICA register
    void SyncClipCoef();

    /// Sets the OpenGL shader in accordance with the current PICA register state
    void SetShader();

    /// Syncs the cull mode to match the PICA register
    void SyncCullMode();

    /// Syncs the depth scale to match the PICA register
    void SyncDepthScale();

    /// Syncs the depth offset to match the PICA register
    void SyncDepthOffset();

    /// Syncs the blend enabled status to match the PICA register
    void SyncBlendEnabled();

    /// Syncs the blend functions to match the PICA register
    void SyncBlendFuncs();

    /// Syncs the blend color to match the PICA register
    void SyncBlendColor();

    /// Syncs the fog states to match the PICA register
    void SyncFogColor();

    /// Sync the procedural texture noise configuration to match the PICA register
    void SyncProcTexNoise();

    /// Sync the procedural texture bias configuration to match the PICA register
    void SyncProcTexBias();

    /// Syncs the alpha test states to match the PICA register
    void SyncAlphaTest();

    /// Syncs the logic op states to match the PICA register
    void SyncLogicOp();

    /// Syncs the color write mask to match the PICA register state
    void SyncColorWriteMask();

    /// Syncs the stencil write mask to match the PICA register state
    void SyncStencilWriteMask();

    /// Syncs the depth write mask to match the PICA register state
    void SyncDepthWriteMask();

    /// Syncs the stencil test states to match the PICA register
    void SyncStencilTest();

    /// Syncs the depth test states to match the PICA register
    void SyncDepthTest();

    /// Syncs the TEV combiner color buffer to match the PICA register
    void SyncCombinerColor();

    /// Syncs the TEV constant color to match the PICA register
    void SyncTevConstColor(std::size_t tev_index,
                           const Pica::TexturingRegs::TevStageConfig& tev_stage);

    /// Syncs the lighting global ambient color to match the PICA register
    void SyncGlobalAmbient();

    /// Syncs the specified light's specular 0 color to match the PICA register
    void SyncLightSpecular0(int light_index);

    /// Syncs the specified light's specular 1 color to match the PICA register
    void SyncLightSpecular1(int light_index);

    /// Syncs the specified light's diffuse color to match the PICA register
    void SyncLightDiffuse(int light_index);

    /// Syncs the specified light's ambient color to match the PICA register
    void SyncLightAmbient(int light_index);

    /// Syncs the specified light's position to match the PICA register
    void SyncLightPosition(int light_index);

    /// Syncs the specified spot light direcition to match the PICA register
    void SyncLightSpotDirection(int light_index);

    /// Syncs the specified light's distance attenuation bias to match the PICA register
    void SyncLightDistanceAttenuationBias(int light_index);

    /// Syncs the specified light's distance attenuation scale to match the PICA register
    void SyncLightDistanceAttenuationScale(int light_index);

    /// Syncs the shadow rendering bias to match the PICA register
    void SyncShadowBias();

    /// Syncs the shadow texture bias to match the PICA register
    void SyncShadowTextureBias();

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed);

    struct VertexArrayInfo {
        u32 vs_input_index_min;
        u32 vs_input_index_max;
        u32 vs_input_size;
    };

    /// Retrieve the range and the size of the input vertex
    VertexArrayInfo AnalyzeVertexArray(bool is_indexed);

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

    bool is_amd;

    OpenGLState state;
    GLuint default_texture;

    RasterizerCacheOpenGL res_cache;

    std::vector<HardwareVertex> vertex_batch;

    bool shader_dirty = true;

    struct {
        UniformData data;
        std::array<bool, Pica::LightingRegs::NumLightingSampler> lighting_lut_dirty;
        bool lighting_lut_dirty_any;
        bool fog_lut_dirty;
        bool proctex_noise_lut_dirty;
        bool proctex_color_map_dirty;
        bool proctex_alpha_map_dirty;
        bool proctex_lut_dirty;
        bool proctex_diff_lut_dirty;
        bool dirty;
    } uniform_block_data = {};

    std::unique_ptr<ShaderProgramManager> shader_program_manager;

    // They shall be big enough for about one frame.
    static constexpr std::size_t VERTEX_BUFFER_SIZE = 16 * 1024 * 1024;
    static constexpr std::size_t INDEX_BUFFER_SIZE = 1 * 1024 * 1024;
    static constexpr std::size_t UNIFORM_BUFFER_SIZE = 2 * 1024 * 1024;
    static constexpr std::size_t TEXTURE_BUFFER_SIZE = 1 * 1024 * 1024;

    OGLVertexArray hw_vao; // VAO for hardware shader / accelerate draw
    std::array<bool, 16> hw_vao_enabled_attributes{};

    std::array<SamplerInfo, 3> texture_samplers;
    OGLStreamBuffer vertex_buffer;
    OGLStreamBuffer uniform_buffer;
    OGLStreamBuffer index_buffer;
    OGLStreamBuffer texture_buffer;
    OGLStreamBuffer texture_lf_buffer;
    OGLFramebuffer framebuffer;
    GLint uniform_buffer_alignment;
    std::size_t uniform_size_aligned_vs;
    std::size_t uniform_size_aligned_fs;

    SamplerInfo texture_cube_sampler;

    OGLTexture texture_buffer_lut_lf;
    OGLTexture texture_buffer_lut_rg;
    OGLTexture texture_buffer_lut_rgba;

    std::array<std::array<GLvec2, 256>, Pica::LightingRegs::NumLightingSampler> lighting_lut_data{};
    std::array<GLvec2, 128> fog_lut_data{};
    std::array<GLvec2, 128> proctex_noise_lut_data{};
    std::array<GLvec2, 128> proctex_color_map_data{};
    std::array<GLvec2, 128> proctex_alpha_map_data{};
    std::array<GLvec4, 256> proctex_lut_data{};
    std::array<GLvec4, 256> proctex_diff_lut_data{};

    bool allow_shadow;
};

} // namespace OpenGL
