// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "video_core/common/rasterizer_cache.h"
#include "video_core/common/pica_uniforms.h"
#include "video_core/common/pipeline.h"

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class PipelineCache;

enum class LoadCallbackStage : u8;
using DiskLoadCallback = std::function<void(LoadCallbackStage, std::size_t, std::size_t)>;

// Structure that the hardware rendered vertices are composed of
struct HardwareVertex {
    HardwareVertex() = default;
    HardwareVertex(const Pica::Shader::OutputVertex& v, bool flip_quaternion);

    // Returns the pipeline vertex layout of the vertex used with software shaders
    constexpr static VertexLayout GetVertexLayout();

    Common::Vec4f position;
    Common::Vec4f color;
    Common::Vec2f tex_coord0;
    Common::Vec2f tex_coord1;
    Common::Vec2f tex_coord2;
    float tex_coord0_w;
    Common::Vec4f normquat;
    Common::Vec3f view;
};

class BackendBase;
struct ScreenInfo;

class Rasterizer {
public:
    explicit Rasterizer(Frontend::EmuWindow& emu_window, std::unique_ptr<BackendBase>& backend);
    ~Rasterizer();

    void LoadDiskResources(const std::atomic_bool& stop_loading, const DiskLoadCallback& callback);

    void AddTriangle(const Pica::Shader::OutputVertex& v0, const Pica::Shader::OutputVertex& v1,
                     const Pica::Shader::OutputVertex& v2);
    void DrawTriangles();
    void NotifyPicaRegisterChanged(u32 id);
    void FlushAll();
    void FlushRegion(PAddr addr, u32 size);
    void InvalidateRegion(PAddr addr, u32 size);
    void FlushAndInvalidateRegion(PAddr addr, u32 size);
    void ClearAll(bool flush);

    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config);
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config);
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config);
    bool AccelerateDisplay(const GPU::Regs::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed);

    /// Syncs entire status to match PICA registers
    void SyncEntireState();

private:
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
    void SyncTevConstColor(std::size_t tev_index, const Pica::TexturingRegs::TevStageConfig& tev_stage);

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
    void UploadUniforms(PipelineHandle pipeline, bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(PipelineHandle pipeline, FramebufferHandle framebuffer, bool is_indexed);

    struct VertexArrayInfo {
        u32 vs_input_index_min;
        u32 vs_input_index_max;
        u32 vs_input_size;
    };

    /// Retrieve the range and the size of the input vertex
    VertexArrayInfo AnalyzeVertexArray(bool is_indexed);

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray(u32 vs_input_size, u32 vs_input_index_min, u32 vs_input_index_max);

private:
    std::unique_ptr<BackendBase>& backend;
    RasterizerCache res_cache;
    std::vector<HardwareVertex> vertex_batch;
    bool shader_dirty = true;

    struct {
        UniformData data;
        std::array<bool, Pica::LightingRegs::NumLightingSampler> lighting_lut_dirty{true};
        bool lighting_lut_dirty_any = true;
        bool fog_lut_dirty = true;
        bool proctex_noise_lut_dirty = true;
        bool proctex_color_map_dirty = true;
        bool proctex_alpha_map_dirty = true;
        bool proctex_lut_dirty = true;
        bool proctex_diff_lut_dirty = true;
        bool dirty = true;
    } uniform_block_data{};

    // Pipeline information structure used to identify a rasterizer pipeline
    // The shader handles are automatically filled by the pipeline cache
    PipelineInfo raster_info{};
    std::unique_ptr<PipelineCache> pipeline_cache;

    // Clear texture for placeholder purposes
    TextureHandle clear_texture;

    // Uniform alignment
    std::array<bool, 16> hw_vao_enabled_attributes{};
    std::size_t uniform_buffer_alignment;
    std::size_t uniform_size_aligned_vs = 0;
    std::size_t uniform_size_aligned_fs = 0;

    // Rasterizer used buffers (vertex, index, uniform, lut)
    BufferHandle vertex_buffer, index_buffer, uniform_buffer;
    BufferHandle texel_buffer_lut_lf, texel_buffer_lut;

    // Pica lighting data
    std::array<std::array<Common::Vec2f, 256>, Pica::LightingRegs::NumLightingSampler> lighting_lut_data{};
    std::array<Common::Vec2f, 128> fog_lut_data{};
    std::array<Common::Vec2f, 128> proctex_noise_lut_data{};
    std::array<Common::Vec2f, 128> proctex_color_map_data{};
    std::array<Common::Vec2f, 128> proctex_alpha_map_data{};
    std::array<Common::Vec4f, 256> proctex_lut_data{};
    std::array<Common::Vec4f, 256> proctex_diff_lut_data{};

    // Texture unit sampler cache
    SamplerInfo texture_cube_sampler;
    std::array<SamplerInfo, 3> texture_samplers;
    std::unordered_map<SamplerInfo, SamplerHandle> sampler_cache;

    // TODO: Remove this
    bool allow_shadow = false;
};

} // namespace VideoCore
