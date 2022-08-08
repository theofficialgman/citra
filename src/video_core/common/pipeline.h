// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/bit_field.h"
#include "common/bit_field_array.h"
#include "common/hash.h"
#include "video_core/common/buffer.h"
#include "video_core/common/texture.h"
#include "video_core/common/shader.h"
#include "video_core/regs_framebuffer.h"
#include "video_core/regs_rasterizer.h"
#include "video_core/regs_pipeline.h"

namespace VideoCore {

constexpr u32 MAX_SHADER_STAGES = 3;
constexpr u32 MAX_VERTEX_ATTRIBUTES = 8;
constexpr u32 MAX_BINDINGS_IN_GROUP = 7;
constexpr u32 MAX_BINDING_GROUPS = 6;

enum class PipelineType : u8 {
    Compute = 0,
    Graphics = 1
};

enum class BindingType : u32 {
    None = 0,
    Uniform = 1,
    UniformDynamic = 2,
    TexelBuffer = 3,
    Texture = 4,
    Sampler = 5,
    StorageImage = 6
};

using BindingGroup = BitFieldArray<0, 3, MAX_BINDINGS_IN_GROUP, BindingType>;

/**
 * Describes all the resources used in the pipeline
 */
struct PipelineLayoutInfo {
    u8 group_count = 0;
    std::array<BindingGroup, MAX_BINDING_GROUPS> binding_groups{};
    u8 push_constant_block_size = 0;
};

/**
 * The pipeline state is tightly packed with bitfields to reduce
 * the overhead of hashing as much as possible
 */
union RasterizationState {
    u8 value = 0;
    BitField<0, 2, Pica::TriangleTopology> topology;
    BitField<4, 2, Pica::CullMode> cull_mode;
};

union DepthStencilState {
    u64 value = 0;
    BitField<0, 1, u64> depth_test_enable;
    BitField<1, 1, u64> depth_write_enable;
    BitField<2, 1, u64> stencil_test_enable;
    BitField<3, 3, Pica::CompareFunc> depth_compare_op;
    BitField<6, 3, Pica::StencilAction> stencil_fail_op;
    BitField<9, 3, Pica::StencilAction> stencil_pass_op;
    BitField<12, 3, Pica::StencilAction> stencil_depth_fail_op;
    BitField<15, 3, Pica::CompareFunc> stencil_compare_op;
    BitField<18, 8, u64> stencil_reference;
    BitField<26, 8, u64> stencil_compare_mask;
    BitField<34, 8, u64> stencil_write_mask;
};

union BlendState {
    u32 value = 0;
    BitField<0, 4, Pica::BlendFactor> src_color_blend_factor;
    BitField<4, 4, Pica::BlendFactor> dst_color_blend_factor;
    BitField<8, 3, Pica::BlendEquation> color_blend_eq;
    BitField<11, 4, Pica::BlendFactor> src_alpha_blend_factor;
    BitField<15, 4, Pica::BlendFactor> dst_alpha_blend_factor;
    BitField<19, 3, Pica::BlendEquation> alpha_blend_eq;
    BitField<22, 4, u32> color_write_mask;
};

enum class AttribType : u8 {
    Float = 0,
    Int = 1,
    Short = 2,
    Byte = 3
};

union VertexAttribute {
    u8 value = 0;
    BitField<0, 2, AttribType> type;
    BitField<2, 3, u8> components;
};

#pragma pack(1)
struct VertexLayout {
    u8 stride = 0;
    std::array<VertexAttribute, MAX_VERTEX_ATTRIBUTES> attributes;
};
#pragma pack()

/**
 * Information about a graphics/compute pipeline
 */
#pragma pack(1)
struct PipelineInfo {
    std::array<ShaderHandle, MAX_SHADER_STAGES> shaders{};
    VertexLayout vertex_layout{};
    PipelineLayoutInfo layout{};
    BlendState blending{};
    DepthStencilState depth_stencil{};
    RasterizationState rasterization{};
    TextureFormat color_attachment = TextureFormat::RGBA8;
    TextureFormat depth_attachment = TextureFormat::D24S8;

    const u64 Hash() const {
        return Common::ComputeStructHash64(*this);
    }
};
#pragma pack()

class PipelineBase : public IntrusivePtrEnabled<PipelineBase> {
public:
    PipelineBase(PipelineType type, PipelineInfo info) :
        type(type), info(info) {}
    virtual ~PipelineBase() = default;

    // Disable copy constructor
    PipelineBase(const PipelineBase&) = delete;
    PipelineBase& operator=(const PipelineBase&) = delete;

    // Binds the texture in the specified slot
    virtual void BindTexture(u32 group, u32 slot, TextureHandle handle) = 0;

    // Binds the texture in the specified slot
    virtual void BindBuffer(u32 group, u32 slot, BufferHandle handle, u32 view = 0) = 0;

    // Binds the sampler in the specified slot
    virtual void BindSampler(u32 group, u32 slot, SamplerHandle handle) = 0;

    PipelineType GetType() const {
        return type;
    }

    /// Sets the primitive topology
    void SetTopology(Pica::TriangleTopology topology) {
        info.rasterization.topology.Assign(topology);
    }

    /// Sets the culling mode
    void SetCullMode(Pica::CullMode mode) {
        info.rasterization.cull_mode.Assign(mode);
    }

    /// Configures the color blending function
    void SetColorBlendFunc(Pica::BlendFactor src_color_factor,
                           Pica::BlendFactor dst_color_factor,
                           Pica::BlendEquation color_eq) {
        info.blending.src_color_blend_factor.Assign(src_color_factor);
        info.blending.dst_color_blend_factor.Assign(dst_color_factor);
        info.blending.color_blend_eq.Assign(color_eq);
    }

    /// Configures the alpha blending function
    void SetAlphaBlendFunc(Pica::BlendFactor src_alpha_factor,
                           Pica::BlendFactor dst_alpha_factor,
                           Pica::BlendEquation alpha_eq) {
        info.blending.src_alpha_blend_factor.Assign(src_alpha_factor);
        info.blending.dst_alpha_blend_factor.Assign(dst_alpha_factor);
        info.blending.alpha_blend_eq.Assign(alpha_eq);
    }

    /// Sets the color write mask
    void SetColorWriteMask(u32 mask) {
        info.blending.color_write_mask.Assign(mask);
    }

    /// Configures the depth test
    void SetDepthTest(bool enable, Pica::CompareFunc compare_op) {
        info.depth_stencil.depth_test_enable.Assign(enable);
        info.depth_stencil.depth_compare_op.Assign(compare_op);
    }

    /// Enables or disables depth writes
    void SetDepthWrites(bool enable) {
        info.depth_stencil.depth_write_enable.Assign(enable);
    }

    /// Configures the stencil test
    void SetStencilTest(bool enable, Pica::StencilAction fail, Pica::StencilAction pass,
                        Pica::StencilAction depth_fail, Pica::CompareFunc compare, u32 ref) {
        info.depth_stencil.stencil_test_enable.Assign(enable);
        info.depth_stencil.stencil_fail_op.Assign(fail);
        info.depth_stencil.stencil_pass_op.Assign(pass);
        info.depth_stencil.stencil_depth_fail_op.Assign(depth_fail);
        info.depth_stencil.stencil_compare_op.Assign(compare);
        info.depth_stencil.stencil_reference.Assign(ref);
    }

    /// Selects the bits of the stencil values participating in the stencil test
    void SetStencilCompareMask(u32 mask) {
        info.depth_stencil.stencil_compare_mask.Assign(mask);
    }

    /// Selects the bits of the stencil values updated by the stencil test
    void SetStencilWriteMask(u32 mask) {
        info.depth_stencil.stencil_write_mask.Assign(mask);
    }

protected:
    PipelineType type = PipelineType::Graphics;
    PipelineInfo info{};
};

using PipelineHandle = IntrusivePtr<PipelineBase>;

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::PipelineInfo> {
    std::size_t operator()(const VideoCore::PipelineInfo& info) const noexcept {
        return info.Hash();
    }
};
} // namespace std
